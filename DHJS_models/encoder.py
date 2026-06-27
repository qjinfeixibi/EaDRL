from json import encoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Categorical
from copy import deepcopy

from DHJS_models.encoding_block import EncodingBlock_Base, EncodingBlock_Job, EncodingBlock_Traj,\
    EncodingBlock_JobAdj, CDN
from DHJS_models.encoder_cdn import EncoderLayer_CDN
from DHJS_models.encoder_veh import EncoderLayer_AugVeh
from DHJS_models.encoder_intent import EncoderLayer_intent


class TFJSP_Encoder_DHJS(nn.Module):
    def __init__(self, encoder_version, **model_params):
        super().__init__()
        encoder_layer_num = model_params['encoder_layer_num']
        self.encoder_version = encoder_version
        if encoder_version in [1, 2, 3, 4, 5]:
            self.layers = nn.ModuleList([EncoderLayer_Base(encoder_version, **model_params) for _ in range(encoder_layer_num)])
        elif encoder_version in [6]:
            self.layers = nn.ModuleList([EncoderLayer_CDN(encoder_version, **model_params) for _ in range(encoder_layer_num)])
        elif encoder_version in [7, 8, 12]:
            self.layers = nn.ModuleList([EncoderLayer_AugVeh(encoder_version, **model_params) for _ in range(encoder_layer_num)])
        elif encoder_version in [9, 10, 11]:
            self.layers = nn.ModuleList([EncoderLayer_intent(encoder_version, **model_params) for _ in range(encoder_layer_num)])
        else:
            raise Exception('encoder version error!')

    def init(self):
        '''
        encoder version 3
        '''
        for layer in self.layers:
            layer.init()
           

    def forward(self, job_emb, ma_emb, veh_emb, proc_time_mat, 
            offload_trans_time, trans_time_mat, 
            oper_adj_batch=None, batch_core_adj_list=None, MVpair_trans_time=None,
            onload_trans_time=None, mask_dyn_ope_ma_adj=None,
            mask_ma=None,
        ):
                                                    
                                                    
                                                   
        for layer in self.layers:
            job_emb, ma_emb, veh_emb = layer(
                job_emb, ma_emb, veh_emb, proc_time_mat, offload_trans_time, 
                trans_time_mat, oper_adj_batch, batch_core_adj_list, MVpair_trans_time,
                onload_trans_time, mask_dyn_ope_ma_adj, mask_ma
            )
        return job_emb, ma_emb, veh_emb

class EncoderLayer_Base(nn.Module):
    def __init__(self, encoder_version, **model_params):
        super().__init__()
        self.encoder_version = encoder_version
        self.model_params = model_params
        
        self.ma_encoding_block = EncodingBlock_Base(**model_params)
        self.veh_encoding_block = EncodingBlock_Base(**model_params)
        if encoder_version in [1, 2, 3, 4, 5]:
            self.job_encoding_block = EncodingBlock_Job(**model_params)
            
                                      
                                                                                          
                                                                
            
            if encoder_version == 5:
                self.diffusion_block = CDN(
                    model_params['embedding_dim'], model_params['hidden_dim'], 
                    model_params['embedding_dim'], diffusion_num=2, bias=True, rnn_type='GRU'
                )
                                    
                                                                            
        else:
            raise Exception('EncoderLayer_JobEnc error!')

                     
                                                            
                                                      
                                                  
                                                
                                                  
                
                                                                                                              
                                                                                                            
                                                                                                              
                                                                                                              
                                                                                                                  
                                                                                                        

    def forward(self, 
            ope_emb, ma_emb, veh_emb, 
            proc_time, offload_trans_time, trans_time, 
            oper_adj_batch=None, batch_core_adj_list=None, 
            MVpair_trans_time=None, onload_trans_time=None,
            mask_dyn_ope_ma_adj=None, mask_ma=None
        ):
        '''
        :params ope_emb: [B, n_opes, E]
        :params ma_emb: [B, n_mas, E]
        :params veh_emb: [B, n_vehs, E]
        :params proc_time: [B, n_opes, n_mas]
        :params offload_trans_time: [B, n_opes, n_vehs]
        :params trans_time: [B, n_mas, n_mas]
        :params oper_adj_batch: [B, n_opes, n_opes]
        :params batch_core_adj_list: [B, max_kcore, n_nodes, n_nodes]
        :params MVpair_trans_time: [B, n_vehs, n_mas]
        :params onload_trans_time [B, n_opes, n_mas]
        :params mask_dyn_ope_ma_adj [B, n_opes, n_mas]
        :param mask_ma [B, n_mas]
        
        Output:
            ope_emb_out: [B, n_opes, E]
            ma_emb_out: [B, n_mas, E]
            veh_emb: [B, n_vehs, E]
        '''
        num_opes = ope_emb.size(1)
        num_mas = ma_emb.size(1)
        num_vehs = veh_emb.size(1)
        
                                       
                                          
                                               
                                                     
                                                                                     
                                                              
                                                                      
                   
                                            
                                                                                    
                                                                                                                             
        
            
        ope_emb_out = self.job_encoding_block(ope_emb, ma_emb, veh_emb, proc_time, offload_trans_time, oper_adj_batch)                   
        ma_emb_out = self.ma_encoding_block(ma_emb, ope_emb, proc_time.transpose(1, 2))                  
        veh_emb_out = self.veh_encoding_block(veh_emb, ope_emb, offload_trans_time.transpose(1, 2))                 
        
        if self.encoder_version == 5:
            node_emb = torch.cat([ope_emb_out, ma_emb_out, veh_emb_out], dim=1).float()                      
            
            node_emb_out = self.diffusion_block(node_emb, batch_core_adj_list)
            ope_emb_out = node_emb_out[:, :num_opes, :]
            ma_emb_out = node_emb_out[:, num_opes:num_opes+num_mas, :]
            veh_emb_out = node_emb_out[:, num_opes+num_mas:num_opes+num_mas+num_vehs, :]
        
                                             
        return ope_emb_out, ma_emb_out, veh_emb_out

    def _embed_list(self,
        embed_feat_ope, embed_feat_ma, embed_feat_veh,
        norm_proc_trans_time, norm_offload_trans_time, norm_trans_time
    ):
        '''
        Time shift, and insert current embed_feat
        '''
        
        self.embed_feat_ope_list = self.embed_feat_ope_list.roll(1,0)
        self.embed_feat_ope_list[-1] = embed_feat_ope
        self.embed_feat_ma_list = self.embed_feat_ma_list.roll(1,0)
        self.embed_feat_ma_list[-1] = embed_feat_ma
        self.embed_feat_veh_list = self.embed_feat_veh_list.roll(1,0)
        self.embed_feat_veh_list[-1] = embed_feat_veh
        self.norm_proc_trans_time_list = self.norm_proc_trans_time_list.roll(1,0)
        self.norm_proc_trans_time_list[-1] = norm_proc_trans_time
        self.norm_offload_trans_time_list = self.norm_offload_trans_time_list.roll(1,0)
        self.norm_offload_trans_time_list[-1] = norm_offload_trans_time
        self.norm_trans_time_list = self.norm_trans_time_list.roll(1,0)
        self.norm_trans_time_list[-1] = norm_trans_time
        
        
        
                                                         
                                                       
                                                         
                                                                     
                                                                           
                                                           
        
                                                              
                                                                                     
                                                                                   
                                                                           
                                                                                                 
                                                                                                       
                                                                                       
               
                                                                         
                                                                                                            
                                                                                                         
                                                                                             
                                                                                                                              
                                                                                                                                       
                                                                                                               
            
                                                                                               
                                                                                              
                                                                                                        
        return self.embed_feat_ope_list[:], self.embed_feat_ma_list[:], self.embed_feat_veh_list[:],\
            self.norm_proc_trans_time_list[:], self.norm_offload_trans_time_list[:],\
                self.norm_trans_time_list[:]
                
class EncoderLayer_allNoes(nn.Module):
    def __init__(self, encoder_version, **model_params):
        super().__init__()
        
        if encoder_version == 6:
                                           
                        
            self.encoding_block = EncodingBlock_Base(**model_params)
        else:
            raise Exception('encoder version error!')
        
    
    def forward(self, ope_emb, ma_emb, veh_emb, proc_time_mat, empty_trans_time_mat, trans_time_mat, 
                oper_adj_batch=None, batch_core_adj_list=None):
        '''
        Input:
            ope_emb: [B, n_opes, E]
            ma_emb: [B, n_mas, E]
            veh_emb: [B, n_vehs, E]
            proc_time_mat: [B, n_opes, n_mas]
            empty_trans_time_mat: [B, n_opes, n_vehs]
            trans_time_mat: [B, n_mas, n_mas]
            oper_adj_batch: [B, n_opes, n_opes]
            batch_core_adj_list: [B, max_kcore, n_nodes, n_nodes]
        Output:
            ope_emb_out: [B, n_opes, E]
            ma_emb_out: [B, n_mas, E]
            veh_emb: [B, n_vehs, E]
        '''
        batch_size, num_opes, D_emb = ope_emb.size()
        num_mas = ma_emb.size(1)
        num_vehs = veh_emb.size(1)
        
        nodes_emb = torch.cat([ope_emb, ma_emb, veh_emb], dim=1)                         
        ope_adj = torch.cat([oper_adj_batch, proc_time_mat, empty_trans_time_mat], dim=-1)                                      
        proc_time_mat_trans = proc_time_mat.transpose(1, 2)                        
        zero_ma_adj = torch.zeros(size=(batch_size, num_mas, num_mas+num_vehs))
        ma_adj = torch.cat([proc_time_mat_trans, zero_ma_adj], dim=-1)                                  
        
        empty_trans_time_mat_trans = empty_trans_time_mat.transpose(1, 2)                        
        zero_veh_adj = torch.zeros(size=(batch_size, num_vehs, num_mas+num_vehs))
        veh_adj = torch.cat([empty_trans_time_mat_trans, zero_veh_adj], dim=-1)                                    
        
        nodes_adj = torch.cat([ope_adj, ma_adj, veh_adj], dim=1)                                                   
        
        
        nodes_emb_out = self.encoding_block(nodes_emb, nodes_emb, nodes_adj)                  
        
        ope_emb_out = nodes_emb_out[:, :num_opes, :]
        ma_emb_out = nodes_emb_out[:, num_opes:num_opes+num_mas, :]
        veh_emb_out = nodes_emb_out[:, num_opes+num_mas:num_opes+num_mas+num_vehs, :]
        return ope_emb_out, ma_emb_out, veh_emb_out
        
        
