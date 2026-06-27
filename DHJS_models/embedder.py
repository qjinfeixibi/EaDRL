from json import encoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np

from env.common_func import generate_trans_mat
from DHJS_models.utils import get_kcore_graph

class DHJS_embedder(nn.Module):
    def __init__(
            self,
            embedding_dim_,
            ope_feat_dim,
            ma_feat_dim,
            veh_feat_dim,
            **model_paras,
        ):
        super().__init__()
                
        self.init_embed_opes = nn.Linear(ope_feat_dim, embedding_dim_)
        self.init_embed_mas = nn.Linear(ma_feat_dim, embedding_dim_)
        self.init_embed_vehs = nn.Linear(veh_feat_dim, embedding_dim_)
        self.init_embed_proc = nn.Linear(1, embedding_dim_)
        self.init_embed_trans = nn.Linear(1, embedding_dim_)
        self.proctime_per_ope_max = model_paras["proctime_per_ope_max"]
        self.transtime_btw_ma_max = model_paras["transtime_btw_ma_max"]
        self.device = model_paras['device']
        self.job_centric = model_paras['job_centric']
        
        self.time_length = model_paras["time_length"]
        
        self.embed_feat_ope_list = []
        self.embed_feat_ma_list = []
        self.embed_feat_veh = []
        self.norm_proc_trans_time_list = []
        self.norm_offload_trans_time_list = []
        self.norm_trans_time = []
    
    def embedding(self, state,  encoder_version=0):
        '''
        Input:
            job_embedding: if True, we embed operation, otherwise job
        Output:
            embed_feat_job_ma: [B, n_jobs + n_mas + n_veh, D_emb]
            embed_feat_job: [B, n_jobs, D_emb]
            embed_feat_ma:  [B, n_mas, D_emb]
            embed_feat_veh: [B, n_vehs, D_emb]
            norm_proc_trans_time: [B, n_jobs, n_mas]
            oper_adj_batch: [B, n_opes, n_opes]
            mask_dyn_ope_ma_adj: [B, n_opes, n_mas]
            mask_ma_status: [B, n_mas]
        '''
        batch_size, num_opes, num_mas = state.ope_ma_adj_batch.size()
        num_jobs = state.mask_job_finish_batch.size(1)
        num_vehs = state.mask_veh_procing_batch.size(1)
        
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)               
                               
        batch_idxes = state.batch_idxes
                   
        ope_ma_adj_batch = deepcopy(state.ope_ma_adj_batch)
        ope_ma_adj_batch = torch.where(ope_ma_adj_batch == 1, True, False)
        raw_opes = deepcopy(state.feat_opes_batch.transpose(1, 2)[batch_idxes])                              
        raw_jobs = raw_opes.gather(1, ope_step_batch[:, :, None].expand(-1, -1, raw_opes.size(2)))                               
        raw_mas = deepcopy(state.feat_mas_batch.transpose(1, 2)[batch_idxes])                          
        raw_vehs = deepcopy(state.feat_vehs_batch.transpose(1, 2)[batch_idxes])                              
        proc_time = deepcopy(state.proc_times_batch[batch_idxes])                     
        trans_time = deepcopy(state.trans_times_batch[batch_idxes])                      
        proc_time_on_job = proc_time.gather(1, ope_step_batch[:, :, None].expand(-1, -1, proc_time.size(2)))                        
        
        
                                                                   
                                                                       
                                                 
        trans_time_OM_pair, offload_trans_time, onload_trans_time, MVpair_trans_time =\
            generate_trans_mat(trans_time, state, job_embedding=False)    
        trans_time_OM_pair = trans_time_OM_pair.min(-1)[0]                                  
                                          
        oper_adj_batch = state.ope_adj_batch.float()                        
                                                                              
        mask_dyn_ope_ma_adj = state.dyn_ope_ma_adj_batch.bool()                     
        mask_dyn_ope_veh_adj = state.dyn_ope_veh_adj_batch.bool()
        if encoder_version not in [9, 12]:
            proc_time[~mask_dyn_ope_ma_adj] = self.proctime_per_ope_max + 1
            onload_trans_time[~mask_dyn_ope_ma_adj] = self.transtime_btw_ma_max + 1
            offload_trans_time[~mask_dyn_ope_veh_adj] = self.transtime_btw_ma_max + 1
                                                 
        mask_ope_status, mask_ma_status, mask_veh_status = self._get_mask_node_status(state)                               
        if encoder_version in [2, 4]:
            mask_ope_status_ext = mask_ope_status[:, :, None].expand(-1, -1, raw_opes.size(2))                           
            raw_opes[~mask_ope_status_ext] = 0                                             
            mask_ma_status_ext = mask_ma_status[:, :, None].expand(-1, -1, raw_mas.size(2))                        
            raw_mas[~mask_ma_status_ext] = 0
            mask_veh_status_ext = mask_veh_status[:, :, None].expand(-1, -1, raw_vehs.size(2))                          
            raw_vehs[~mask_veh_status_ext] = 0
        
                                  
        proc_trans_time = proc_time + onload_trans_time                       
        if encoder_version in [7, 8, 9, 10, 12]:
            proc_trans_time = proc_time
        proc_trans_time = torch.where(proc_time==0, self.proctime_per_ope_max+1, proc_trans_time)
                                                            
        raw_opes_jobs = raw_opes

                   
        nums_opes = state.nums_opes_batch[batch_idxes]
        features = self.get_normalized(raw_opes_jobs, raw_mas, raw_vehs, proc_trans_time,\
            trans_time, offload_trans_time, MVpair_trans_time, onload_trans_time,\
            batch_idxes, nums_opes, flag_sample=True, flag_train=True)
    
        norm_opes = features[0]
                                                                                                                                      
        norm_mas = features[1]
        norm_vehs = features[2]
        norm_proc_trans_time = features[3]                      
        norm_offload_trans_time = features[4]                         
        norm_trans_time = features[5]                      
        norm_MV_pair_trans_time = features[6]                       
        norm_onload_trans_time = features[7]                        
                                                     
        embed_feat_ope = self.init_embed_opes(norm_opes)                                  
        embed_feat_ma = self.init_embed_mas(norm_mas)                      
        embed_feat_veh = self.init_embed_vehs(norm_vehs)                        
        
                             
        if self.job_centric:
            embed_feat_ope = embed_feat_ope.gather(1, ope_step_batch[:, :, None].expand(-1, -1, embed_feat_ope.size(2)))
            norm_proc_trans_time = norm_proc_trans_time.gather(1, ope_step_batch[:, :, None].expand(-1, -1, norm_proc_trans_time.size(2)))
            norm_onload_trans_time = norm_onload_trans_time.gather(1, ope_step_batch[:, :, None].expand(-1, -1, norm_onload_trans_time.size(2)))
                              
        batch_core_adj_list = None
        
        
        if encoder_version == 9:
            return embed_feat_ope, embed_feat_ma, embed_feat_veh,\
            proc_trans_time, offload_trans_time, trans_time,\
            oper_adj_batch, batch_core_adj_list, MVpair_trans_time, onload_trans_time,\
            mask_dyn_ope_ma_adj, mask_ma_status
        
        return embed_feat_ope, embed_feat_ma, embed_feat_veh,\
            norm_proc_trans_time, norm_offload_trans_time, norm_trans_time,\
            oper_adj_batch, batch_core_adj_list, norm_MV_pair_trans_time, norm_onload_trans_time,\
            mask_dyn_ope_ma_adj, mask_ma_status


    def _embed_list(self,
        embed_feat_ope, embed_feat_ma, embed_feat_veh,
        norm_proc_trans_time, norm_offload_trans_time, norm_trans_time
    ):
        self.embed_feat_ope_list.append(embed_feat_ope)
        self.embed_feat_ma_list.append(embed_feat_ma)
        self.embed_feat_veh.append(embed_feat_veh)
        self.norm_proc_trans_time_list.append(norm_proc_trans_time)
        self.norm_offload_trans_time_list.append(norm_offload_trans_time)
        self.norm_trans_time.append(norm_trans_time)
        
        if len(self.embed_feat_ope_list) > self.time_length:
            self.embed_feat_ope_list = self.embed_feat_ope_list[-self.time_length:]
            self.embed_feat_ma_list = self.embed_feat_ma_list[-self.time_length:]
            self.embed_feat_veh = self.embed_feat_veh[-self.time_length:]
            self.norm_proc_trans_time_list = self.norm_proc_trans_time_list[-self.time_length:]
            self.norm_offload_trans_time_list = self.norm_offload_trans_time_list[-self.time_length:]
            self.norm_trans_time = self.norm_trans_time[-self.time_length:]
        else:
            resd_len = self.time_length - len(self.embed_feat_ope_list)
            self.embed_feat_ope_list = [self.embed_feat_ope_list[0]] * resd_len + self.embed_feat_ope_list
            self.embed_feat_ma_list = [self.embed_feat_ma_list[0]] * resd_len + self.embed_feat_ma_list
            self.embed_feat_veh = [self.embed_feat_veh[0]] * resd_len + self.embed_feat_veh
            self.norm_proc_trans_time_list = [self.norm_proc_trans_time_list[0]] * resd_len + self.norm_proc_trans_time_list
            self.norm_offload_trans_time_list = [self.norm_offload_trans_time_list[0]] * resd_len + self.norm_offload_trans_time_list
            self.norm_trans_time = [self.norm_trans_time[0]] * resd_len + self.norm_trans_time
            
        return torch.stack(self.embed_feat_ope_list), torch.stack(self.embed_feat_ma_list),\
            torch.stack(self.embed_feat_veh), torch.stack(self.norm_proc_trans_time_list),\
                torch.stack(self.norm_offload_trans_time_list), torch.stack(self.norm_trans_time)

    def _get_oper_adj(self, state):
        '''
        Output:
            oper_adj = [B, n_opes, n_opes]
        '''
        batch_size, num_opes, num_mas = state.ope_ma_adj_batch.size()
        num_jobs = state.mask_job_finish_batch.size(1)
        num_vehs = state.mask_veh_procing_batch.size(1)
        
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)               
        ope_end_batch = state.end_ope_biases_batch               
        
        oper_adj = torch.zeros(size=(batch_size, num_opes, num_opes))
        
                                           
        mask_ope_status = ~state.ope_status                
        
        
        
        
        

    def _get_mask_node_status(self, state):
        '''
        Output:
            mask_ope_status: [B, n_opes]
            mask_ma_status: [B, n_mas]
            mask_veh_status: [B, n_vehs]
        '''
                                        
        mask_ope_status = ~state.ope_status                
                                                 
        mask_ma_status = ~state.mask_ma_procing_batch             
        mask_veh_status = ~state.mask_veh_procing_batch                
        
        return mask_ope_status, mask_ma_status, mask_veh_status
    

    def feature_normalize(self, data):
        return (data - torch.mean(data)) / ((data.std() + 1e-5))

    def get_normalized(self, 
            raw_opes, raw_mas, raw_vehs, proc_time, trans_time, 
            empty_trans_time, MVpair_trans_time, onload_trans_time,
            batch_idxes, nums_opes, 
            flag_sample=False, flag_train=False
        ):
        '''
        :param raw_opes: Raw feature vectors of operation nodes
        :param raw_mas: Raw feature vectors of machines nodes
        :param raw_vehs:
        :param proc_time: Processing time
        :param trans_time [B, n_mas, n_mas]:
        :param empty_trans_time [B, n_opes, n_vehs]:
        :param MVpair_trans_time [B, n_vehs, n_mas]:
        :param onload_trans_time [B, n_opes, n_mas]:
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
            empty_trans_time_norm = self.feature_normalize(empty_trans_time)                                              
            trans_time_norm = self.feature_normalize(trans_time)                                             
            MVpair_trans_time_norm = self.feature_normalize(MVpair_trans_time)                      
            onload_trans_time_norm = self.feature_normalize(onload_trans_time)                      
        return ((raw_opes - mean_opes) / (std_opes + 1e-5), (raw_mas - mean_mas) / (std_mas + 1e-5),\
            (raw_vehs - mean_vehs) / (std_vehs + 1e-5), proc_time_norm,\
            empty_trans_time_norm, trans_time_norm, MVpair_trans_time_norm, onload_trans_time_norm)
        
