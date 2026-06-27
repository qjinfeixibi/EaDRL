import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def weight_init(m):
    if isinstance(m, nn.Linear):
        size = m.weight.size()
        fan_out = size[0]                 
        fan_in = size[1]                    
        variance = 0.001                                  
        m.weight.data.normal_(0.0, variance)
        try:
            m.bias.data.normal_(0.0, 0.0001)
        except:
            pass

class mlp_layer(nn.Module):
    def __init__(self, input_size, output_size, activation='tanh', drouput_prob=0.0):
        super(mlp_layer, self).__init__()

        self.affine = nn.Linear(input_size, output_size)
                                                
        weight_init(self.affine)

        if activation.lower() == 'tanh':
            self.activation = torch.tanh
        elif activation.lower() == 'relu':
            self.activation = F.relu()

    def forward(self, x):
        x = self.activation(self.affine(x))
                               
        return x

class Attention_batch(nn.Module):
    def __init__(self, hidden_size, query_size, use_softmax=False):
        super().__init__()

        self.use_softmax = use_softmax
        self.W_query = nn.Linear(query_size, hidden_size, bias=True)
        self.W_ref = nn.Linear(hidden_size, hidden_size, bias=False)
                                                                        
        V = torch.normal(torch.zeros(hidden_size), 0.0001)
        self.V = nn.Parameter(V)
        weight_init(V)
        weight_init(self.W_query)
        weight_init(self.W_ref)

    def forward(self, query, ref):
        """
        Args:
            query: [B, hidden_size]
            ref:   [B, seq_len, hidden_size]
        Output:
            logits: [B, seq_len]
        """

        seq_len  = ref.size(-2)
        query = self.W_query(query)                   

        _ref = self.W_ref(ref)                             
                                       
        
        query_batch = query[:, None, :].expand(-1, _ref.size(1), -1)                               
        
        m = torch.tanh(query_batch + _ref)                            

                                                                                 
        logits = torch.matmul(m, self.V).squeeze(-1)                  
        if self.use_softmax:
            logits = torch.softmax(logits, dim=-1)                
        else:
            logits = logits

        return logits

class query_att_batch(nn.Module):
    def __init__(self, hidden_size, use_softmax=False, as_sum=True):
        super().__init__()

        self.attention = Attention_batch(hidden_size, hidden_size, use_softmax=True)


    def forward(self, query, ref):
        """
        Args:
            query: [B, E]
            ref:   [B, n_node, E]
        Output:
            ret: [B, 1, E]
        """

        softmax_res = self.attention(query, ref)                 
        ret = torch.bmm(softmax_res[:, None, :], ref)               
        return ret