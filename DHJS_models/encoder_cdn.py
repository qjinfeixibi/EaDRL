import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from DHJS_models.encoding_block import EncodingBlock_Base


class EncoderLayer_CDN(nn.Module):
    def __init__(self, encoder_version, **model_params):
        super().__init__()
        self.encoder_version = encoder_version
        self.model_params = model_params
        
                                       
                    
        self.encoding_block = EncodingBlock_Base(**model_params)
        self.diffusion_block = DiffusionBlock_CDN(
            model_params['embedding_dim'], 
            model_params['embedding_dim'], bias=True, rnn_type='GRU'
        )
        
    def forward(self, ope_emb, ma_emb, veh_emb, proc_time_mat, empty_trans_time_mat, trans_time_mat, 
            oper_adj_batch=None, batch_core_adj_mat=None,
            MVpair_trans_time=None, onload_trans_time=None,
            mask_dyn_ope_ma_adj=None, mask_ma=None,
        ):
        '''
        :param ope_emb: [B, n_opes, E]
        :param ma_emb: [B, n_mas, E]
        :param veh_emb: [B, n_vehs, E]
        :param proc_time_mat: [B, n_opes, n_mas]
        :param empty_trans_time_mat: [B, n_opes, n_vehs]
        :param trans_time_mat: [B, n_mas, n_mas]
        :param oper_adj_batch: [B, n_opes, n_opes]
        :param batch_core_adj_mat: [B, num_core, n_nodes, n_nodes]
        :param MVpair_trans_time: [B, n_vehs, n_mas]
        :param onload_trans_time [B, n_opes, n_mas]
        :param mask_dyn_ope_ma_adj [B, n_opes, n_mas]
        :param mask_ma [B, n_mas]
        
        : return ope_emb_out: [B, n_opes, E]
        : return ma_emb_out: [B, n_mas, E]
        : return veh_emb: [B, n_vehs, E]
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
        nodes_emb_final_out = self.diffusion_block(nodes_emb_out, batch_core_adj_mat)
        
        ope_emb_out = nodes_emb_final_out[:, :num_opes, :]
        ma_emb_out = nodes_emb_final_out[:, num_opes:num_opes+num_mas, :]
        veh_emb_out = nodes_emb_final_out[:, num_opes+num_mas:num_opes+num_mas+num_vehs, :]
        return ope_emb_out, ma_emb_out, veh_emb_out
        

class DiffusionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, diffusion_num, bias=True, rnn_type='GRU'):
        super().__init__()
        self.diffusion_num = diffusion_num
        
        if diffusion_num == 1:
            self.diffusion_list = nn.ModuleList()
            self.diffusion_list.append(DiffusionBlock_CDN(input_dim, output_dim, bias=bias, rnn_type=rnn_type))
        elif diffusion_num > 1:
            self.diffusion_list = nn.ModuleList()
            self.diffusion_list.append(DiffusionBlock_CDN(input_dim, hidden_dim, bias=bias, rnn_type=rnn_type))
            for i in range(diffusion_num - 2):
                self.diffusion_list.append(DiffusionBlock_CDN(hidden_dim, hidden_dim, bias=bias, rnn_type=rnn_type))
            self.diffusion_list.append(DiffusionBlock_CDN(hidden_dim, output_dim, bias=bias, rnn_type=rnn_type))
        else:
            raise ValueError("number of layers should be positive!")
    
    def forward(self, nodes_emb, adj_mat):
        '''
        : param nodes_emb: [B, n_nodes, D_emb]
        : param adj_mat: [B, n_core, n_nodes, n_nodes]
        
        '''
        for i in range(self.diffusion_num):
            nodes_emb = self.diffusion_list[i](nodes_emb, adj_mat)
        return nodes_emb

class DiffusionBlock_CDN(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, rnn_type='GRU'):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.rnn_type = rnn_type

        self.linear = nn.Linear(input_dim, output_dim)
                                                                     
        assert self.rnn_type in ['LSTM', 'GRU']
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=input_dim, hidden_size=output_dim, num_layers=1, bias=bias, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=input_dim, hidden_size=output_dim, num_layers=1, bias=bias, batch_first=True)
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, nodes_emb, adj_list):
        '''
        : param nodes_emb: [B, n_nodes, D_emb]
        : param adj_list: [B, n_core, n_nodes, n_nodes]
        
        : return output_batch: [B, n_nodes, D_out]
        '''
        batch_size, num_nodes, D_emb = nodes_emb.size()
        output_batch = []
        for batch_idx in range(batch_size):
            hx_list = []
            for i, adj in enumerate(adj_list[batch_idx]):
                if i == 0:
                    res = torch.mm(adj, nodes_emb[batch_idx])
                else:
                    res = hx_list[-1] + torch.mm(adj, nodes_emb[batch_idx])
                                       
                hx_list.append(res)
            hx_list = [F.relu(res) for res in hx_list]
            hx = torch.stack(hx_list, dim=0).transpose(0, 1)                              
                                                                                                       
            self.rnn.flatten_parameters()
            output, _ = self.rnn(hx)
            output = output.sum(dim=1)                       
                                                                               
            output = self.norm(output)
            output_batch.append(output)
        output_batch = torch.stack(output_batch, dim=0)                         
        return output_batch
        
    def forwardbatch(self, nodes_emb, adj_mat):
        '''
        : param nodes_emb: [B, n_nodes, D_emb]
        : param adj_mat: [B, n_core, n_nodes, n_nodes]
        
        : return output: [B, n_nodes, D_out]
        '''
        D_emb = nodes_emb.size(-1)
        batch_size, num_core, num_nodes, _ = adj_mat.size()
        hx_list = []
        adj_mat_resh = adj_mat.reshape(batch_size*num_core, num_nodes, num_nodes)
        nodes_emb_resh = nodes_emb[:, None, :, :].expand(-1, num_core, -1, -1).reshape(batch_size*num_core, num_nodes, D_emb)
        hx = torch.bmm(adj_mat_resh, nodes_emb_resh)                                  
        hx = hx.reshape(batch_size, num_core, num_nodes, D_emb)
                                      
                                                                              
        hx = hx.permute(0, 2, 1, 3).reshape(batch_size*num_nodes, num_core, D_emb)                                 
                                                                                                   
        self.rnn.flatten_parameters()
        output, _ = self.rnn(hx)
        output = output.sum(dim=1)                         
                                                                           
        output = self.norm(output).reshape(batch_size, num_nodes, -1)
        
        return output
