               
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


                                       
class CoreDiffusion(nn.Module):
    input_dim: int
    output_dim: int
    layer_num: int
    bias: bool
    rnn_type: str

    def __init__(self, input_dim, output_dim, core_num=1, bias=True, rnn_type='GRU'):
        super(CoreDiffusion, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.core_num = core_num
        self.rnn_type = rnn_type

        self.linear = nn.Linear(input_dim, output_dim)
                                                                     
        assert self.rnn_type in ['LSTM', 'GRU']
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=input_dim, hidden_size=output_dim, num_layers=1, bias=bias, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=input_dim, hidden_size=output_dim, num_layers=1, bias=bias, batch_first=True)
        self.norm = nn.LayerNorm(output_dim)
                                 

                                 
                                                      
                                             

    def forward(self, x, adj_list):
        '''
        INput:
            x: [B, n_nodes, embed_dim]
            adj_list: [B, max_core, n_nodes, n_nodes]
        '''
        batch_size = len(adj_list)
                       
        output_batch = []
        for batch in range(batch_size):
            hx_list = []
            for i, adj in enumerate(adj_list[batch]):
                if i == 0:
                    res = torch.mm(adj, x[batch])
                else:
                    res = hx_list[-1] + torch.mm(adj, x[batch])
                                       
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

class CoreDiffusionBatch(CoreDiffusion):
    def __init__(self, input_dim, output_dim, core_num=1, bias=True, rnn_type='GRU'):
        super().__init__(input_dim, output_dim, core_num, bias, rnn_type)
    
    def forward(self, x, adj_list):
        '''
        Input:
            x: [B, n_nodes, embed_dim]
            adj_list: [B, max_core, n_nodes, n_nodes]
        Output:
            output: [B, num_nodes, D_output]
        '''
        batch_size = len(adj_list)
        
        _, num_nodes, D_emb = x.size()
        max_core = adj_list.size(1)
        adj_list_resh = adj_list.reshape(batch_size*max_core, num_nodes, num_nodes)
        x_resh = x[:, None, :, :].expand(-1, max_core, -1, -1).reshape(batch_size*max_core, num_nodes, D_emb)
        hx = torch.bmm(adj_list_resh, x_resh)                                  
        hx = hx.reshape(batch_size, max_core, num_nodes, D_emb)
        hx = hx.permute(0, 2, 1, 3).reshape(batch_size*num_nodes, max_core, D_emb)                                 
                                                                                                   
        self.rnn.flatten_parameters()
        output, _ = self.rnn(hx)
        output = output.sum(dim=1)                         
                                                                           
        output = self.norm(output).reshape(batch_size, num_nodes, -1)
        
        
        return output
    
    
                                   
class MLP(nn.Module):
    input_dim: int
    hidden_dim: int
    output_dim: int
    layer_num: int
    bias: bool
    activate_type: str

    def __init__(self, input_dim, hidden_dim, output_dim, layer_num, bias=True, activate_type='N'):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer_num = layer_num
        self.bias = bias
        self.activate_type = activate_type
        assert self.activate_type in ['L', 'N']
        assert self.layer_num > 0

        if layer_num == 1:
            self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        else:
            self.linears = torch.nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim, bias=bias))
            for layer in range(layer_num - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            self.linears.append(nn.Linear(hidden_dim, output_dim, bias=bias))

    def forward(self, x):
        if self.layer_num == 1:                
            x = self.linear(x)
            if self.activate_type == 'N':
                x = F.selu(x)
            return x
        h = x       
        for layer in range(self.layer_num):
            h = self.linears[layer](h)
            if self.activate_type == 'N':
                h = F.selu(h)
        return h
