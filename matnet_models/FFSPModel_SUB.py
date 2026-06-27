
"""
The MIT License

Copyright (c) 2021 MatNet

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
                                                  
    
        added = input1 + input2
                                            

        transposed = added.transpose(1, 2)
                                            

        normalized = self.norm(transposed)

        back_trans = normalized.transpose(1, 2)
                                            

        return back_trans

class AddAndInstanceNormalization_Edge(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.norm = nn.InstanceNorm1d(1, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
                                                  
    
        added = input1 + input2
                                            

        transposed = added.transpose(1, 2)
                                            

        normalized = self.norm(transposed)

        back_trans = normalized.transpose(1, 2)
                                            

        return back_trans

class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
                                                  

        return self.W2(F.relu(self.W1(input1)))


class FeedForward_Edge(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(1, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, 1)

    def forward(self, input1):
                                                  

        return self.W2(F.relu(self.W1(input1)))

class MixedScore_MultiHeadAttention(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        head_num = model_params['head_num']
        ms_hidden_dim = model_params['ms_hidden_dim']
        mix1_init = model_params['ms_layer1_init']
        mix2_init = model_params['ms_layer2_init']

        mix1_weight = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample((head_num, 2, ms_hidden_dim))
        mix1_bias = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample((head_num, ms_hidden_dim))
        self.mix1_weight = nn.Parameter(mix1_weight)
                                     
        self.mix1_bias = nn.Parameter(mix1_bias)
                                  

        mix2_weight = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample((head_num, ms_hidden_dim, 1))
        mix2_bias = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample((head_num, 1))
        self.mix2_weight = nn.Parameter(mix2_weight)
                                     
        self.mix2_bias = nn.Parameter(mix2_bias)
                          

    def forward(self, q, k, v, cost_mat):
                                                      
                                                        
                                                   

        batch_size = q.size(0)
        row_cnt = q.size(2)
        col_cnt = k.size(2)

        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        sqrt_qkv_dim = self.model_params['sqrt_qkv_dim']

        dot_product = torch.matmul(q, k.transpose(2, 3))
                                                    

        dot_product_score = dot_product / sqrt_qkv_dim
                                                    

        cost_mat_score = cost_mat[:, None, :, :].expand(batch_size, head_num, row_cnt, col_cnt)
                                                    

        two_scores = torch.stack((dot_product_score, cost_mat_score), dim=4)
                                                       

        two_scores_transposed = two_scores.transpose(1,2)
                                                       

        ms1 = torch.matmul(two_scores_transposed, self.mix1_weight)
                                                                   

        ms1 = ms1 + self.mix1_bias[None, None, :, None, :]
                                                                   

        ms1_activated = F.relu(ms1)

        ms2 = torch.matmul(ms1_activated, self.mix2_weight)
                                                       

        ms2 = ms2 + self.mix2_bias[None, None, :, None, :]
                                                       

        mixed_scores = ms2.transpose(1,2)
                                                       

        mixed_scores = mixed_scores.squeeze(4)
                                                    

        weights = nn.Softmax(dim=3)(mixed_scores)
                                                    

        out = torch.matmul(weights, v)
                                                    

        out_transposed = out.transpose(1, 2)
                                                    

        out_concat = out_transposed.reshape(batch_size, row_cnt, head_num * qkv_dim)
                                                   

        return out_concat

class MixedScore_MultiHeadAttention_WithEdge(MixedScore_MultiHeadAttention):
    def __init__(self, **model_params):
        super().__init__(**model_params)
    
    def forward(self, q, k, v, cost_mat):
                                                      
                                                        
                                                   

        batch_size = q.size(0)
        row_cnt = q.size(2)
        col_cnt = k.size(2)

        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        sqrt_qkv_dim = self.model_params['sqrt_qkv_dim']

        dot_product = torch.matmul(q, k.transpose(2, 3))
                                                          

        dot_product_score = dot_product / sqrt_qkv_dim
                                                       

        cost_mat_score = cost_mat[:, None, :, :].expand(batch_size, head_num, row_cnt, col_cnt)
                                                        

        two_scores = torch.stack((dot_product_score, cost_mat_score), dim=4)
                                                          

        two_scores_transposed = two_scores.transpose(1,2)
                                                           

        ms1 = torch.matmul(two_scores_transposed, self.mix1_weight)
                                                                           

        ms1 = ms1 + self.mix1_bias[None, None, :, None, :]
                                                                        

        ms1_activated = F.relu(ms1)       

        ms2 = torch.matmul(ms1_activated, self.mix2_weight)
                                                            

        ms2 = ms2 + self.mix2_bias[None, None, :, None, :]
                                                       

        mixed_scores = ms2.transpose(1,2)
                                                       

        mixed_scores = mixed_scores.squeeze(4)
                                                                          

        weights = nn.Softmax(dim=3)(mixed_scores)
                                                                                          

        out = torch.matmul(weights, v)
                                                    

        out_transposed = out.transpose(1, 2)
                                                    

        out_concat = out_transposed.reshape(batch_size, row_cnt, head_num * qkv_dim)
                                                   

        return out_concat, mixed_scores


class MultiHeadAttention(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        head_num = model_params['head_num']
        ms_hidden_dim = model_params['ms_hidden_dim']
        mix1_init = model_params['ms_layer1_init']
        mix2_init = model_params['ms_layer2_init']


    def forward(self, q, k, v):
                                                      
                                                        
                                                   

        batch_size = q.size(0)
        row_cnt = q.size(2)
        col_cnt = k.size(2)

        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        sqrt_qkv_dim = self.model_params['sqrt_qkv_dim']

        dot_product = torch.matmul(q, k.transpose(2, 3))
                                                    

        dot_product_score = dot_product / sqrt_qkv_dim
                                                    

        weights = nn.Softmax(dim=3)(dot_product_score)
                                                    

        out = torch.matmul(weights, v)
                                                    

        out_transposed = out.transpose(1, 2)
                                                    

        out_concat = out_transposed.reshape(batch_size, row_cnt, head_num * qkv_dim)
                                                   

        return out_concat


class Heterogeneous_MultiHeadAttention(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        head_num = model_params['head_num']
        ms_hidden_dim = model_params['ms_hidden_dim']
        mix1_init = model_params['ms_layer1_init']
        mix2_init = model_params['ms_layer2_init']

        mix1_weight = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample((head_num, 2, ms_hidden_dim))
        mix1_bias = torch.torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample((head_num, ms_hidden_dim))
        self.mix1_weight = nn.Parameter(mix1_weight)
                                     
        self.mix1_bias = nn.Parameter(mix1_bias)
                                  

        mix2_weight = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample((head_num, ms_hidden_dim, 1))
        mix2_bias = torch.torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample((head_num, 1))
        self.mix2_weight = nn.Parameter(mix2_weight)
                                     
        self.mix2_bias = nn.Parameter(mix2_bias)
                          
    
    def forward(self, q, k_row, v_row, k_col, v_col, cost_mat):

        row_col_atten = self._get_row_col_atten(q, k_col, v_col, cost_mat)                           
        self_row_atten = self._get_self_row_atten(q, k_row, v_row)                                   
        return row_col_atten + self_row_atten

    def _get_row_col_atten(self, q, k, v, cost_mat):
        '''
        Input:
            q: [B, H, row_cnt, qkv_dim]
            k: [B, H, col_cnt, qkv_dim]
            cost_mat: [B, row_cnt, col_cnt]
        Output:
            out_concat: [B, row_cnt, H*qkv_dim]
        '''
        

        batch_size = q.size(0)
        row_cnt = q.size(2)
        col_cnt = k.size(2)

        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        sqrt_qkv_dim = self.model_params['sqrt_qkv_dim']

        dot_product = torch.matmul(q, k.transpose(2, 3))
                                                    

        dot_product_score = dot_product / sqrt_qkv_dim
                                                    

        cost_mat_score = cost_mat[:, None, :, :].expand(batch_size, head_num, row_cnt, col_cnt)
                                                    

        two_scores = torch.stack((dot_product_score, cost_mat_score), dim=4)
                                                       

        two_scores_transposed = two_scores.transpose(1,2)
                                                       
        
        ms1 = torch.matmul(two_scores_transposed, self.mix1_weight)
                                                                   

        ms1 = ms1 + self.mix1_bias[None, None, :, None, :]
                                                                   

        ms1_activated = F.relu(ms1)

        ms2 = torch.matmul(ms1_activated, self.mix2_weight)
                                                       

        ms2 = ms2 + self.mix2_bias[None, None, :, None, :]
                                                       

        mixed_scores = ms2.transpose(1,2)
                                                       

        mixed_scores = mixed_scores.squeeze(4)
                                                    

        weights = nn.Softmax(dim=3)(mixed_scores)
                                                    
        
        out = torch.matmul(weights, v)
                                                    

        out_transposed = out.transpose(1, 2)
                                                    

        out_concat = out_transposed.reshape(batch_size, row_cnt, head_num * qkv_dim)
                                                   

        return out_concat
    
    def _get_self_row_atten(self, q, k, v):
        batch_size = q.size(0)
        row_cnt = q.size(2)
        col_cnt = k.size(2)

        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        sqrt_qkv_dim = self.model_params['sqrt_qkv_dim']

        dot_product = torch.matmul(q, k.transpose(2, 3))
                                                    

        dot_product_score = dot_product / sqrt_qkv_dim
                                                    

        weights = nn.Softmax(dim=3)(dot_product_score)
                                                    

        out = torch.matmul(weights, v)
                                                    

        out_transposed = out.transpose(1, 2)
                                                    

        out_concat = out_transposed.reshape(batch_size, row_cnt, head_num * qkv_dim)
                                                   

        return out_concat