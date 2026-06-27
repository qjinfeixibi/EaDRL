import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from copy import deepcopy
from torch.utils.checkpoint import checkpoint
from typing import NamedTuple
import math
import numpy as np

from matnet_v2_models.TFJSPModel_sub import TFJSP_Encoder, FFSP_Decoder, MLPCritic, FJSP_Decoder


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
                             

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return AttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
                glimpse_key=self.glimpse_key[:, key],                       
                glimpse_val=self.glimpse_val[:, key],                       
                                               
            )
        return super(AttentionModelFixed, self).__getitem__(key)
    


class TFJSPModel_matnet_v2(nn.Module):
    def __init__(self,
                embedding_dim_,
                hidden_dim,
                ope_feat_dim,
                ma_feat_dim,
                veh_feat_dim,
                n_encode_layers=2,
                tanh_clipping=10.,
                mask_inner=True,
                mask_logits=True,
                normalization='batch',
                n_heads=8,
                checkpoint_encoder=False,
                shrink_size=None,
                **model_paras
                ):
        super().__init__()
        
                               
        self.embedding_dim = embedding_dim_
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = "greedy"
        self.temp = 1.0
        self.normalization = normalization
        self.tanh_clipping = tanh_clipping
        self.ope_feat_dim = ope_feat_dim
        self.ma_feat_dim = ma_feat_dim
        self.veh_feat_dim = veh_feat_dim
        self.all_feat_dim = ope_feat_dim + ma_feat_dim + veh_feat_dim + 1                                 
        self.algo = None                      
        self.model_paras = model_paras

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits
        
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size
        
                                       
        self.W_placeholder = nn.Parameter(torch.Tensor(2 * self.embedding_dim))
        self.W_placeholder.data.uniform_(-1, 1)                                                 
        self.init_embed_opes = nn.Linear(self.ope_feat_dim, self.embedding_dim)
        self.init_embed_mas = nn.Linear(self.ma_feat_dim, self.embedding_dim)
        self.init_embed_vehs = nn.Linear(self.veh_feat_dim, self.embedding_dim)
        self.init_embed_proc = nn.Linear(1, self.embedding_dim)
        self.init_embed_trans = nn.Linear(1, self.embedding_dim)
        
        self.init_embed = nn.Linear(self.all_feat_dim, self.embedding_dim)

        self.ffsp_encoder = TFJSP_Encoder(**model_paras)
        self.ffsp_decoder = FFSP_Decoder(**model_paras)
        
        assert self.embedding_dim % self.n_heads == 0
        self.project_out = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        
             

    def init(self, state):
        self.batch_size = state.ope_ma_adj_batch.size(0)
        self.num_opes = state.ope_ma_adj_batch.size(1)
        self.num_mas = state.ope_ma_adj_batch.size(2)
        self.num_jobs = state.mask_job_finish_batch.size(1)
        self.num_vehs = state.mask_veh_procing_batch.size(1)
        
                                                           
        self.prev_embed = torch.zeros(size=(self.batch_size, 1, self.embedding_dim))
    
    
    
    def forward(self, state):
        batch_size = state.ope_ma_adj_batch.size(0)
        num_opes = state.ope_ma_adj_batch.size(1)
        num_jobs = state.mask_job_finish_batch.size(1)
        num_mas = state.ope_ma_adj_batch.size(2)
        num_vehs = state.mask_veh_procing_batch.size(1)
        
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)               

                                 
        embed_feat_ope_ma_veh, embed_feat_ope, embed_feat_ma,\
            embed_feat_veh, norm_proc, norm_trans = self._init_embed(state)
        embedded_row, embedded_col = self.ffsp_encoder(embed_feat_ope, embed_feat_ma, norm_proc)                                            
        
        
    def _init_embed(self, state):
        '''
        Output:
            embed_feat_ope_ma_veh: [B, n_opes + n_mas + n_vehs, D_emb]
            embed_feat_ope: [B, n_opes, D_emb]
            embed_feat_ma:  [B, n_mas, D_emb]
            embed_feat_veh: [B, n_vehs, D_emb]
            norm_proc: [B, n_opes, n_mas]
            norm_trans: [B, n_mas, n_mas]
        '''
        batch_size, num_opes, num_mas = state.ope_ma_adj_batch.size()
        num_vehs = state.mask_veh_procing_batch.size(1)
                               
        batch_idxes = state.batch_idxes
                   
        raw_opes = state.feat_opes_batch.transpose(1, 2)[batch_idxes]                              
        raw_mas = state.feat_mas_batch.transpose(1, 2)[batch_idxes]                          
        raw_vehs = state.feat_vehs_batch.transpose(1, 2)[batch_idxes]                              
        proc_time = state.proc_times_batch[batch_idxes]                     
        trans_time = state.trans_times_batch[batch_idxes]                      
                   
        nums_opes = state.nums_opes_batch[batch_idxes]
        features = self.get_normalized(raw_opes, raw_mas, raw_vehs, proc_time, trans_time, batch_idxes,\
            nums_opes, flag_sample=True, flag_train=True)
        norm_opes = (deepcopy(features[0]))
        norm_mas = (deepcopy(features[1]))
        norm_vehs = (deepcopy(features[2]))
        norm_proc = (deepcopy(features[3]))
        norm_trans = (deepcopy(features[4]))
        
                                     
                                                                                                                                              
                                                                                                                                            
                                                                                                                                              
                                                                           
                                                                            
        
                                                                                                                                                
                                                                                         
                                                                                                                                                
        
                                                     
        embed_feat_ope = self.init_embed_opes(norm_opes)                        
        embed_feat_ma = self.init_embed_mas(norm_mas)                      
        embed_feat_veh = self.init_embed_vehs(norm_vehs)                        
        embed_feat_proc = self.init_embed_proc(norm_proc[..., None])                               
        embed_feat_trans = self.init_embed_trans(norm_trans[..., None])                    
        embed_feat_ope_ma_veh = torch.cat([embed_feat_ope, embed_feat_ma, embed_feat_veh], dim=1)                                        
        
        return embed_feat_ope_ma_veh, embed_feat_ope, embed_feat_ma, embed_feat_veh,\
            norm_proc, norm_trans
        
        
        
        
    
    
    def feature_normalize(self, data):
        return (data - torch.mean(data)) / ((data.std() + 1e-5))

    def get_normalized(self, raw_opes, raw_mas, raw_vehs, proc_time, trans_time, batch_idxes, nums_opes, flag_sample=False, flag_train=False):
        '''
        :param raw_opes: Raw feature vectors of operation nodes
        :param raw_mas: Raw feature vectors of machines nodes
        :param raw_vehs:
        :param proc_time: Processing time
        :param trans_time:
        :param batch_idxes: Uncompleted instances
        :param nums_opes: The number of operations for each instance
        :param flag_sample: Flag for DRL-S
        :param flag_train: Flag for training
        :return: Normalized feats, including operations, machines and edges
        '''
        batch_size = batch_idxes.size(0)                                   
                                                          
                                                         
                                                                                              
        
                                                                                                                
        if not flag_sample and not flag_train:
            raise Exception('Not described here yet!')
                            
                           
                                         
                                                                                                   
                                                                                                 
                                                          
                                                                                
                                                                 
                                                                              
                                                       
                                                     
                                                                  
                                                                
                                        
                                                                                     
        else:
            mean_opes = torch.mean(raw_opes, dim=-2, keepdim=True)                                             
            mean_mas = torch.mean(raw_mas, dim=-2, keepdim=True)                                            
            mean_vehs = torch.mean(raw_vehs, dim=-2, keepdim=True)                                             
            std_opes = torch.std(raw_opes, dim=-2, keepdim=True)                                             
            std_mas = torch.std(raw_mas, dim=-2, keepdim=True)                                            
            std_vehs = torch.std(raw_vehs, dim=-2, keepdim=True)                                             
            
            proc_time_norm = self.feature_normalize(proc_time)                                                
            trans_time_norm = self.feature_normalize(trans_time)                                               
            
        return ((raw_opes - mean_opes) / (std_opes + 1e-5), (raw_mas - mean_mas) / (std_mas + 1e-5),\
            (raw_vehs - mean_vehs) / (std_vehs + 1e-5), proc_time_norm, trans_time_norm)
    
    
    
    def random_act(self, state):
        '''
        Output:
            action: [ope_idx, mas_idx, job_idx, veh_idx]: [4, B]
        '''
        batch_idxes = state.batch_idxes
        batch_size, num_opes, num_mas = state.ope_ma_adj_batch.size()
        num_jobs = state.mask_job_procing_batch.size(1)
        num_vehs = state.mask_veh_procing_batch.size(1)
        
        rand_act = torch.ones(size=(batch_size, num_mas * num_jobs * self.num_vehs), dtype=torch.float)
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch, state.end_ope_biases_batch, state.ope_step_batch)                
        
                                                                                    
                                                                               
        dummy_shape = torch.zeros(size=(batch_size, num_jobs, num_mas, self.num_vehs))

                                                                                                
        eligible_proc = state.ope_ma_adj_batch.gather(1, ope_step_batch[..., None].expand(-1, -1, state.ope_ma_adj_batch.size(-1)))                       
        eligible_proc = eligible_proc[..., None].expand_as(dummy_shape)                             
        
                                                     
                                      
        ma_eligible = ~state.mask_ma_procing_batch[:, None, :].expand(-1, num_jobs, -1)                           
        job_eligible = ~(state.mask_job_procing_batch + state.mask_job_finish_batch)[..., None].expand(-1, -1, num_mas)
        ope_ma_adj = state.ope_ma_adj_batch.gather(1, ope_step_batch[..., None].expand(-1, -1, num_mas)).bool()                       
        ope_ma_eligible = job_eligible & ma_eligible & ope_ma_adj                       
                                              
        veh_eligible = ~state.mask_veh_procing_batch[:, None, None, :].expand_as(dummy_shape)
        ope_ma_veh_eligible = ope_ma_eligible[..., None].expand_as(dummy_shape) & veh_eligible                              
        
                                                                                                                             
                                                                                                                                                              
                                                                                                                             
                                                                                                                   
        
        eligible_batch = ope_ma_veh_eligible.flatten(1).count_nonzero(dim=1)       
        is_ineligible = (eligible_batch == 0).sum(dim=0)
        if is_ineligible > 0:
            print("No eligible O-M-V pair!")
            return 
                                                                    
        mask = ope_ma_veh_eligible.permute(0, 3, 2, 1).flatten(1)                                                             
        rand_act[~mask] = float('-inf')
        rand_act_prob = F.softmax(rand_act, dim=1)                                
        dist = Categorical(rand_act_prob)
        elig_act_idx = dist.sample()          
                                                                                  
        veh, ma, job, ope = self.act_2_objects(elig_act_idx, ope_step_batch, batch_idxes, num_mas, num_jobs)
                                                                                             

        return torch.stack((ope, ma, job, veh), dim=0)
    
    def act_2_objects(self, action, ope_step_batch, batch_idxes, num_mas, num_jobs):
        '''
        Input
            action: [B,]
            ope_ste: tensor: [B, num_jobs]
        Output:
            veh_idx: scalar
            ma_idx: scalar
            job_idx: scalar
            ope_idx: scalar
        '''
        
        veh_idx = torch.div(action, (self.num_mas * self.num_jobs), rounding_mode='floor')
        ma_job = torch.remainder(action, (self.num_mas * self.num_jobs))
        ma_idx = torch.div(ma_job, self.num_jobs, rounding_mode='floor')
        job_idx = torch.remainder(ma_job, self.num_jobs)
        
                                                  
                                                
                                     
                                     
        
        ope_idx = ope_step_batch[batch_idxes, job_idx]
        return veh_idx, ma_idx, job_idx, ope_idx