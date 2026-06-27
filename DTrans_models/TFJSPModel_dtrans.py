from random import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from copy import deepcopy
from torch.utils.checkpoint import checkpoint
from typing import NamedTuple
import math
import numpy as np

from DTrans_models.encoder import TFJSP_Encoder_DTrans
from DHJS_models.decoder_types import TFJSP_Decoder_DHJS_V2, TFJSP_Decoder_DHJS_V3
from DHJS_models.decoder import TFJSP_Decoder_DHJS_Base
from DTrans_models.embedder import DTrans_embedder
from DTrans_models.decoder import TFJSP_Decoder_DTrans_V4, TFJSP_Decoder_DTrans_V5
from DTrans_models.nodecoder import TFJSP_NoDecoder_DTrans

from GTrans_models.critic import MLPCritic
class TFJSPModel_DTrans(nn.Module):
    '''
    This version improve training speed, where only selected job node, not operation nodes, computes nearest vehicle nodes
    
    '''
    def __init__(self,
                embedding_dim_,
                hidden_dim_,
                problem,
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
                consd_trans_time_mat=True,
                encoder_version=1,
                decoder_version=1,
                meta_rl=None,
                **model_paras
                ):
        '''
        Input:
            meta_rl: train_paras['meta_rl']
        '''
        super().__init__()
        
                               
        self.embedding_dim = embedding_dim_
        self.hidden_dim = hidden_dim_
        self.n_encode_layers = n_encode_layers
        self.decode_type = "greedy"
        self.temp = 1.0
        self.normalization = normalization
        self.tanh_clipping = tanh_clipping
        self.ope_feat_dim = ope_feat_dim
        self.ma_feat_dim = ma_feat_dim
        self.veh_feat_dim = veh_feat_dim
        self.job_embedding = model_paras['job_centric']
        
        self.all_feat_dim = ope_feat_dim + ma_feat_dim + veh_feat_dim + 1                                 
        
        self.model_paras = model_paras

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size
        self.proctime_per_ope_max = model_paras["proctime_per_ope_max"]
        self.transtime_btw_ma_max = model_paras["transtime_btw_ma_max"]
        self.device = model_paras['device']
        self.num_core = 3
        
        if meta_rl is not None:
            batch_size = meta_rl['minibatch']
        else:
            batch_size = model_paras['batch_size']
                                                                                                          
                                                                                                                                                   
        self.consd_trans_time_mat = consd_trans_time_mat     

                          
        self.embedder = DTrans_embedder(
            embedding_dim_, self.ope_feat_dim, self.ma_feat_dim, self.veh_feat_dim,
            **model_paras
        )
                         
        self.encoder_version = encoder_version
        
        if encoder_version == 0:
            self.encoder = None
        else:
            self.encoder = TFJSP_Encoder_DTrans(
                encoder_version, 
                self.proctime_per_ope_max+1,
                self.transtime_btw_ma_max+1,
                self.transtime_btw_ma_max+1,
                **model_paras
            )
                         
        self.decoder_version = decoder_version
        self.prev_embed = torch.zeros(size=(batch_size, 1, self.embedding_dim))
        if decoder_version == 1:
            decoder_fn = TFJSP_Decoder_DHJS_Base
            self.prev_embed = torch.zeros(size=(batch_size, 1, 3*self.embedding_dim))
        elif decoder_version == 2:
            decoder_fn = TFJSP_Decoder_DHJS_V2
        elif decoder_version == 3:
            decoder_fn = TFJSP_Decoder_DHJS_V3
            self.prev_embed = torch.zeros(size=(batch_size, 1, 3*self.embedding_dim))
        elif decoder_version == 4:
            decoder_fn = TFJSP_Decoder_DTrans_V4
        elif decoder_version == 5:
            decoder_fn = TFJSP_Decoder_DTrans_V5
        elif decoder_version == 0:       
            decoder_fn = TFJSP_NoDecoder_DTrans
        else:
            raise Exception('decoder version error!')
        self.decoder = decoder_fn(**model_paras)
        
        
        assert self.embedding_dim % self.n_heads == 0
                                                                                          
        self.project_out = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        
                                                           
        self.batch_core_adj_list = None

        self.critic = MLPCritic(
            input_dim=384,         
            hidden_dim=128,                
            output_dim=1,
            num_layers=4,
            dropout=0.1
        ).to(self.device)
    
    def init(self, state, dataset=None, loader=None):
        self.batch_size = state.ope_ma_adj_batch.size(0)
        self.num_opes = state.ope_ma_adj_batch.size(1)
        self.num_mas = state.ope_ma_adj_batch.size(2)
        self.num_jobs = state.mask_job_finish_batch.size(1)
        self.num_vehs = state.mask_veh_procing_batch.size(1)
        
    def act(self, state, baseline=False, return_state_vec=False):
        return self.forward(state, baseline, return_state_vec)
    
    def forward(self, state, baseline=False, return_state_vec=False):
        batch_size = state.ope_ma_adj_batch.size(0)
        num_opes = state.ope_ma_adj_batch.size(1)
        num_jobs = state.mask_job_finish_batch.size(1)
        num_mas = state.ope_ma_adj_batch.size(2)
        num_vehs = state.mask_veh_procing_batch.size(1)
        
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)               

                           
        embeddings = self.embedder.embedding(state, self.encoder_version)
        embed_feat_ope = embeddings[0]
        embed_feat_ma = embeddings[1]
        embed_feat_veh = embeddings[2]
        proc_time = embeddings[3]
        onload_trans_time = embeddings[4]
        offload_trans_time = embeddings[5]
        offload_trans_time_OV = embeddings[6]
        
                          
        if self.encoder_version == 0:               
            embedded_ope = embed_feat_ope
            embedded_ma = embed_feat_ma
            embedded_veh = embed_feat_veh
        else:
            embedded_ope, embedded_ma, embedded_veh = self.encoder(
                embed_feat_ope, embed_feat_ma, embed_feat_veh, 
                proc_time, offload_trans_time, onload_trans_time,
                offload_trans_time_OV
            )    

                          
        action, log_p, dic = self._get_action_with_decoder(
            state, embedded_ope, embedded_ma, embedded_veh, 
            offload_trans_time_OV, onload_trans_time, proc_time,
            baseline=baseline
        )
                                                   
        
                                    
                                                                     
                                                                  
                                                                
                                                           
            
                         
        pooled_ope = embedded_ope.mean(dim=1)           
        pooled_ma = embedded_ma.mean(dim=1)
        pooled_veh = embedded_veh.mean(dim=1)
        pooled_state = torch.cat([pooled_ope, pooled_ma, pooled_veh], dim=1)           
        value = self.critic(pooled_state)
                                                                                                    
        if return_state_vec:
            return action, log_p, value, dic
        else:
            return action, log_p  
    
    def _get_action_with_decoder(
        self, state, embedded_ope, embedded_ma, embedded_veh, 
        offload_trans_time_OV, onload_trans_time, proc_time,
        baseline
        ):
        '''
        Input:
            state:
            embedding: [B, n_nodes, D_emb]
        Output:
            action: [3, B]
            log_p: [B, 1]
        '''
        batch_size, num_opes, num_mas = state.ope_ma_adj_batch.size()
        num_jobs = state.mask_job_procing_batch.size(1)
        if self.job_embedding:
            num_opes_jobs = num_jobs
        else:
            num_opes_jobs = num_opes
        
                          
        mask, mask_ope_ma = self._get_mask_ope_ma(state)                     
        mask_veh = ~state.mask_veh_procing_batch                

                                     
        embedding = torch.cat([embedded_ope, embedded_ma, embedded_veh], dim=1)                      
        self.decoder.set_nodes_kv(embedding)
        self.decoder.set_ope_kv(embedded_ope)
        self.decoder.set_ma_kv(embedded_ma)
        self.decoder.set_veh_kv(embedded_veh)
        if self.decoder_version in [4, 5]:
            self.decoder.set_trans_time(offload_trans_time_OV, onload_trans_time, proc_time)
        
                         
        action, log_p, prev_embed, dic = self.decoder(
            embedding, None, self.prev_embed, state, mask, mask_ope_ma, mask_veh,
            training=self.training, eval_type=self.model_paras['eval_type'], baseline=baseline,
            job_embedding=self.job_embedding
        )               
        self.prev_embed = prev_embed
        
        
        return action, log_p, dic
        
    
    
    def _get_mask_ope_ma(self, state):
        '''
        Output:
            mask: [B, n_jobs, n_mas]
            mask_ope_ma: [B, n_opes, n_mas]
        '''
        batch_idxes = state.batch_idxes
        batch_size, num_opes, num_mas = state.ope_ma_adj_batch.size()
        num_jobs = state.mask_job_procing_batch.size(1)
        
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)               
        opes_appertain_batch = state.opes_appertain_batch                
                      
        mask_ma = ~state.mask_ma_procing_batch[batch_idxes]             
        
                                   
        eligible_proc = state.ope_ma_adj_batch[batch_idxes].gather(1,
                          ope_step_batch[..., None].expand(-1, -1, state.ope_ma_adj_batch.size(-1))[batch_idxes])                        
        dummy_shape = torch.zeros(size=(len(batch_idxes), num_jobs, num_mas))
        ma_eligible = ~state.mask_ma_procing_batch[batch_idxes].unsqueeze(1).expand_as(dummy_shape)                     
        job_eligible = ~(state.mask_job_procing_batch[batch_idxes] +
                         state.mask_job_finish_batch[batch_idxes])[:, :, None].expand_as(dummy_shape)                       
        
        eligible = job_eligible & ma_eligible & (eligible_proc == 1)

        if (~(eligible)).all():
            print("No eligible J-M pair!")
            return
        mask = eligible                      
        
                                
                                                        
        mask_ope_step = torch.full(size=(batch_size, num_opes), dtype=torch.bool, fill_value=False) 
        tmp_batch_idxes = batch_idxes.unsqueeze(-1).repeat(1, num_jobs)              
        mask_ope_step[tmp_batch_idxes, ope_step_batch] = True
        
                                                                       
        mask_job = torch.where(mask.sum(dim=-1) > torch.zeros(size=(batch_size, num_jobs)), True, False)               
        mask_ope_by_job = mask_job.gather(1, opes_appertain_batch)
        
        mask_ope = mask_ope_by_job & mask_ope_step               
        
                                        
        mask_ope_padd = mask_ope[:, :, None].expand(-1, -1, num_mas)                        
        mask_ma_padd = mask_ma[:, None, :].expand(-1, num_opes, -1)                     
        ope_ma_adj = state.ope_ma_adj_batch[batch_idxes]
        mask_ope_ma = mask_ope_padd & mask_ma_padd & (ope_ma_adj==1)                      
        
        if (~(eligible)).all():
            print("No eligible O-M pair!")
            return
        
        return  mask, mask_ope_ma
    
    def _make_heads(self, v, num_steps=None):
        '''
        Ex) v = glimpse_key_fixed [B, 1, n_opes + n_mas, D_emb] -> [B, 1, n_opes + n_mas, H, D_emb/H] -> [H, B, 1, n_opes + n_mas, D_emb/H]
        '''
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps
        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)                                                          
        )       
