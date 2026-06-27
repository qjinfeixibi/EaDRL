import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from matnet_models.FFSPModel_SUB import AddAndInstanceNormalization, FeedForward, MixedScore_MultiHeadAttention


                                        
         
                                        
class TFJSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        encoder_layer_num = model_params['encoder_layer_num']
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, row_emb, col_emb, cost_mat):
                                                    
                                                    
                                                   

        for layer in self.layers:
            row_emb, col_emb = layer(row_emb, col_emb, cost_mat)

        return row_emb, col_emb


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.row_encoding_block = EncodingBlock(**model_params)
        self.col_encoding_block = EncodingBlock(**model_params)

    def forward(self, row_emb, col_emb, cost_mat):
                                                    
                                                    
                                                   
        row_emb_out = self.row_encoding_block(row_emb, col_emb, cost_mat)                   
        col_emb_out = self.col_encoding_block(col_emb, row_emb, cost_mat.transpose(1, 2))                  

        return row_emb_out, col_emb_out


class EncodingBlock(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.mixed_score_MHA = MixedScore_MultiHeadAttention(**model_params)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

    def forward(self, row_emb, col_emb, cost_mat):
                                                                                
                                                   
                                                   
                                                   
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(row_emb), head_num=head_num)
                                                      
        k = reshape_by_heads(self.Wk(col_emb), head_num=head_num)
        v = reshape_by_heads(self.Wv(col_emb), head_num=head_num)
                                                       

        out_concat = self.mixed_score_MHA(q, k, v, cost_mat)
                                                   

        multi_head_out = self.multi_head_combine(out_concat)
                                            

        out1 = self.add_n_normalization_1(row_emb, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3
                                            



                                        
         
                                        

class TFJSP_Decoder_hetero_attention(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.encoded_NO_JOB = nn.Parameter(torch.rand(1, 1, embedding_dim))

        self.Wq_1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_3 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        
        self.Wk_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.multi_head_combine_2 = nn.Linear(head_num * qkv_dim, embedding_dim)
        
        self.proj_ope = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.k = None                                       
        self.v = None                                         
        self.single_head_key = None                                        
    
    
    
    def _select_ma(self, encoded_nodes, state):
        '''
        Input:
            encoded_nodes: [B, n_mas, D_emb]
        Output:
            encoded_current_ma: [B, 1, D_emb]
            select_ma: [B, 1]
        '''
        batch_size = state.ope_ma_adj_batch.size(0)
        num_opes = state.ope_ma_adj_batch.size(1)
        num_mas = state.ope_ma_adj_batch.size(2)
        embed_dim = encoded_nodes.size(-1)
        
                                         
        batch_idxes = state.batch_idxes
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)               
        ope_ma_adj = state.ope_ma_adj_batch[batch_idxes].bool()                     
        ma_eligible = ~state.mask_ma_procing_batch[batch_idxes]             
        job_eligible = ~(state.mask_job_procing_batch[batch_idxes] + state.mask_job_finish_batch[batch_idxes])                
                                                        
        mask_ope_step = torch.full(size=(len(batch_idxes), num_opes), dtype=torch.bool, fill_value=False)   
        for batch in batch_idxes:
            mask_ope_step[batch, ope_step_batch[batch]] = True               
                                                         
        mask_ope_by_job = torch.full(size=(batch_size, num_opes), dtype=torch.bool, fill_value=False)               
        for batch in batch_idxes:
            mask_ope_by_job[batch] = torch.repeat_interleave(job_eligible[batch], state.nums_ope_batch[batch])
                                                                                           
        ope_eligible = mask_ope_by_job & mask_ope_step               
        ope_eligible_add_ma = ope_eligible[:, :, None].expand(-1, -1, num_mas) & ma_eligible[:, None, :].expand(-1, num_opes, -1)                       
                                                          
        ma_elig_on_ope = ope_eligible_add_ma & ope_ma_adj                      
                                                                               
                                                                  
        ma_elig_on_ope = ma_elig_on_ope.transpose(1, 2).any(dim=2)               
        ma_elig_on_ope_float = torch.where(ma_elig_on_ope == True, 1.0, float('-inf'))
        
        select_ma = torch.softmax(ma_elig_on_ope_float, dim=1).multinomial(1)           
                                                            
        encoded_current_ma = encoded_nodes.gather(1, select_ma[..., None].expand(-1, -1 , embed_dim))                 
        
        return encoded_current_ma, select_ma
    
    def set_ope_kv(self, encoded_opes):
        '''
        Input:
            encoded_row: [B, n_opes, E]
        '''
        head_num = self.model_params['head_num']
        
        self.k_ope = reshape_by_heads(self.Wk(encoded_opes), head_num=head_num)                             
        self.v_ope = reshape_by_heads(self.Wv(encoded_opes), head_num=head_num)                             
        self.single_head_key_ope = encoded_opes.transpose(1, 2)                   
        
    def set_ma_kv(self, encoded_mas):
        '''
        Input:
            encoded_mas: [B, n_mas, E]
        '''
        head_num = self.model_params['head_num']
        self.k_ma = reshape_by_heads(self.Wk_2(encoded_mas), head_num=head_num)                            
        self.v_ma = reshape_by_heads(self.Wv_2(encoded_mas), head_num=head_num)                            
        self.single_head_key_ma = encoded_mas.transpose(1, 2)                
        
    
    def forward(self, embed_nodes, embed_context, state, mask, mask_ope_x_ma,
                training=True, eval_type='softmax', baseline=False):
        '''
        Input:
            embed_nodes: [B, n_opes + n_mas, E]
            state:
            mask: [B, n_jobs, n_mas]
            mask_ope_x_ma: [B, n_opes, n_mas]
            
        '''
        head_num = self.model_params['head_num']
        batch_size, num_opes, num_mas = state.ope_ma_adj_batch.size()
        num_jobs = state.mask_job_procing_batch.size(1)
        
            
        
                                
        embed_opes = embed_nodes[:, :num_opes, :]
        embed_mas = embed_nodes[:, num_opes:, :]
        
                                   
        ninf_mask_ope_x_ma = torch.where(mask_ope_x_ma == True, 0., -math.inf)                      
        mask_ma = torch.where(mask_ope_x_ma.sum(dim=1) > 0, True, False)                
        ninf_mask_ma = torch.where(mask_ma == True, 0., -math.inf)[:, None, :]                   
        
                                     
        q_ma = reshape_by_heads(self.Wq_2(embed_context), head_num=head_num)                       
        out_concat_ma = self._multi_head_attention_for_decoder(q_ma, self.k_ma, self.v_ma,
                                                            rank3_ninf_mask=ninf_mask_ma)                            
        mh_atten_out_ma = self.multi_head_combine_2(out_concat_ma)             
        
                                                                    
        score_ma = torch.matmul(mh_atten_out_ma, self.single_head_key_ma)                   

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled_ma = score_ma / sqrt_embedding_dim

        score_clipped_ma = logit_clipping * torch.tanh(score_scaled_ma)
        score_masked_ma = score_clipped_ma + ninf_mask_ma                   
        all_mas_prob = F.softmax(score_masked_ma, dim=2).squeeze(1)               
        
        batch_idxes = state.batch_idxes
        if training or eval_type == 'softmax':
            while True:                                                                      
                select_ma = all_mas_prob.reshape(batch_size, -1).multinomial(1).squeeze(dim=1).reshape(batch_size, 1)         
                ma_prob = all_mas_prob.gather(1, select_ma)         
                non_finish_batch = torch.full(size=(batch_size, 1), dtype=torch.bool, fill_value=False)
                non_finish_batch[batch_idxes] = True
                finish_batch = torch.where(non_finish_batch == True, False, True)
                ma_prob[finish_batch] = 1                                     
                if (ma_prob != 0).all():
                    break
        else:
            select_ma = all_mas_prob.argmax(dim=1).unsqueeze(1)            
            ma_prob = torch.zeros(size=(batch_size, 1))                      
        if baseline:
            select_ma = all_mas_prob.argmax(dim=1).unsqueeze(1)            
            ma_prob = torch.zeros(size=(batch_size, 1))                      
        embed_select_ma = embed_mas.gather(1, select_ma[:, :, None].expand(-1, -1, embed_mas.size(-1)))             
        
                                       
                                                       
        ninf_mask_ope_on_ma = ninf_mask_ope_x_ma.gather(2, select_ma[:, None, :].expand(-1, ninf_mask_ope_x_ma.size(1), -1)).transpose(1,2)                  
        
                                       
        q_ope = reshape_by_heads(self.Wq_3(embed_select_ma), head_num=head_num)                       
        out_concat = self._multi_head_attention_for_decoder(q_ope, self.k_ope, self.v_ope,
                                                            rank3_ninf_mask=ninf_mask_ope_on_ma)                            
        mh_atten_out = self.multi_head_combine(out_concat)             

                                                                    
        score_ope = torch.matmul(mh_atten_out, self.single_head_key_ope)                      

        score_scaled_ope = score_ope / sqrt_embedding_dim

        score_clipped_ope = logit_clipping * torch.tanh(score_scaled_ope)
        score_masked_ope = score_clipped_ope + ninf_mask_ope_on_ma                      
        all_opes_prob = F.softmax(score_masked_ope, dim=2).squeeze(1)                  
                                                                      
        
                                 
        if training or eval_type == 'softmax':
            while True:                                                                      
                select_ope = all_opes_prob.reshape(batch_size, -1).multinomial(1).squeeze(dim=1).reshape(batch_size, 1)         
                ope_prob = all_opes_prob.gather(1, select_ope)         
                non_finish_batch = torch.full(size=(batch_size, 1), dtype=torch.bool, fill_value=False)
                non_finish_batch[batch_idxes] = True
                finish_batch = torch.where(non_finish_batch == True, False, True)
                ope_prob[finish_batch] = 1                                     
                if (ope_prob != 0).all():
                    break
        else:
            select_ope = all_opes_prob.argmax(dim=1).unsqueeze(1)            
            ope_prob = torch.zeros(size=(batch_size, 1))                      
        if baseline:
            select_ope = all_opes_prob.argmax(dim=1).unsqueeze(1)            
            ope_prob = torch.zeros(size=(batch_size, 1))                      
        select_job = self.from_ope_to_job(select_ope.squeeze(1), state, num_jobs, num_opes).unsqueeze(1)            
        embed_select_ope = embed_opes.gather(1, select_ope[:, :, None].expand(-1, -1, embed_opes.size(-1)))
        
                             
        action = torch.cat([select_ope, select_ma, select_job], dim=1).transpose(1, 0)          
        prob = ope_prob * ma_prob
        
        return action, prob.log(), embed_select_ope + embed_select_ma
    
    
    
    def from_ope_to_job(self, select_ope, state, num_jobs, num_opes):
        '''
        Input:
            select_ope: [B,]: selected operation index
        Output:
            selecct_job: [B,]
        '''
        batch_size = state.ope_step_batch.size(0)
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch, state.end_ope_biases_batch, state.ope_step_batch)               
        empty_ope_batch = torch.ones(size=(batch_size, 1)) * num_opes
        ope_step_batch_plus1 = torch.cat([ope_step_batch, empty_ope_batch], dim=1)                  
        
        select_job = torch.where(ope_step_batch_plus1 == select_ope[:, None].expand(-1, num_jobs+1))[1]       
        return select_job
    
        
        
    def _multi_head_attention_for_decoder(self, q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
        '''
        Input:
            q: [B, H, 1, qkv_dim]
            k, v: [B, H, n_opes, qkv_dim]
            rank2_ninf_mask: [B, n_opes]
            rank3_ninf_mask: [B, n_mas, n_opes] or [B, 1, n_opes]
        '''
        batch_size = q.size(0)
        n = q.size(2)
        num_node = k.size(2)

        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        sqrt_qkv_dim = self.model_params['sqrt_qkv_dim']

        score = torch.matmul(q, k.transpose(2, 3))                                

        score_scaled = score / sqrt_qkv_dim

        if rank2_ninf_mask is not None:
            score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_size, head_num, n, num_node)
        if rank3_ninf_mask is not None:
            score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_size, head_num, n, num_node)

        weights = nn.Softmax(dim=3)(score_scaled)                                 

        out = torch.matmul(weights, v)                      

        out_transposed = out.transpose(1, 2)                        

        out_concat = out_transposed.reshape(batch_size, n, head_num * qkv_dim)                      

        return out_concat
    
class TFJSP_Decoder_hetero_attention_v2(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.encoded_NO_JOB = nn.Parameter(torch.rand(1, 1, embedding_dim))

        self.Wq_1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_3 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        
        self.Wk_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.multi_head_combine_2 = nn.Linear(head_num * qkv_dim, embedding_dim)
        
        self.proj_ope = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.k = None                                       
        self.v = None                                         
        self.single_head_key = None                                        


    def _select_ma(self, encoded_nodes, state):
        '''
        Input:
            encoded_nodes: [B, n_mas, D_emb]
        Output:
            encoded_current_ma: [B, 1, D_emb]
            select_ma: [B, 1]
        '''
        batch_size = state.ope_ma_adj_batch.size(0)
        num_opes = state.ope_ma_adj_batch.size(1)
        num_mas = state.ope_ma_adj_batch.size(2)
        embed_dim = encoded_nodes.size(-1)
        
                                         
        batch_idxes = state.batch_idxes
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)               
        ope_ma_adj = state.ope_ma_adj_batch[batch_idxes].bool()                     
        ma_eligible = ~state.mask_ma_procing_batch[batch_idxes]             
        job_eligible = ~(state.mask_job_procing_batch[batch_idxes] + state.mask_job_finish_batch[batch_idxes])                
                                                        
        mask_ope_step = torch.full(size=(len(batch_idxes), num_opes), dtype=torch.bool, fill_value=False)   
        for batch in batch_idxes:
            mask_ope_step[batch, ope_step_batch[batch]] = True               
                                                         
        mask_ope_by_job = torch.full(size=(batch_size, num_opes), dtype=torch.bool, fill_value=False)               
        for batch in batch_idxes:
            mask_ope_by_job[batch] = torch.repeat_interleave(job_eligible[batch], state.nums_ope_batch[batch])
                                                                                           
        ope_eligible = mask_ope_by_job & mask_ope_step               
        ope_eligible_add_ma = ope_eligible[:, :, None].expand(-1, -1, num_mas) & ma_eligible[:, None, :].expand(-1, num_opes, -1)                       
                                                          
        ma_elig_on_ope = ope_eligible_add_ma & ope_ma_adj                      
                                                                               
                                                                  
        ma_elig_on_ope = ma_elig_on_ope.transpose(1, 2).any(dim=2)               
        ma_elig_on_ope_float = torch.where(ma_elig_on_ope == True, 1.0, float('-inf'))
        
        select_ma = torch.softmax(ma_elig_on_ope_float, dim=1).multinomial(1)           
                                                            
        encoded_current_ma = encoded_nodes.gather(1, select_ma[..., None].expand(-1, -1 , embed_dim))                 
        
        return encoded_current_ma, select_ma
    
    def set_ope_kv(self, encoded_opes):
        '''
        Input:
            encoded_row: [B, n_opes, E]
        '''
        head_num = self.model_params['head_num']
        
        self.k_ope = reshape_by_heads(self.Wk(encoded_opes), head_num=head_num)                             
        self.v_ope = reshape_by_heads(self.Wv(encoded_opes), head_num=head_num)                             
        self.single_head_key_ope = encoded_opes.transpose(1, 2)                   
        
    def set_ma_kv(self, encoded_mas):
        '''
        Input:
            encoded_mas: [B, n_mas, E]
        '''
        head_num = self.model_params['head_num']
        self.k_ma = reshape_by_heads(self.Wk_2(encoded_mas), head_num=head_num)                            
        self.v_ma = reshape_by_heads(self.Wv_2(encoded_mas), head_num=head_num)                            
        self.single_head_key_ma = encoded_mas.transpose(1, 2)                
        
    
    def forward(self, embed_nodes, embed_context, state, mask, mask_ope_x_ma,
                training=False, eval_type='softmax', baseline=False):
        '''
        Input:
            embed_nodes: [B, n_opes + n_mas, E]
            state:
            mask: [B, n_jobs, n_mas]
            mask_ope_x_ma: [B, n_opes, n_mas]
            
        '''
        head_num = self.model_params['head_num']
        batch_size, num_opes, num_mas = state.ope_ma_adj_batch.size()
        num_jobs = state.mask_job_procing_batch.size(1)
        
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)               
        
                                
        embed_jobs = embed_nodes[:, :num_jobs, :]
        embed_mas = embed_nodes[:, num_jobs:, :]
        
                                   
        ninf_mask = torch.where(mask == True, 0., -math.inf)                      
        mask_ma = torch.where(mask.sum(dim=1) > 0, True, False)                
        ninf_mask_ma = torch.where(mask_ma == True, 0., -math.inf)[:, None, :]                   
        
                                     
        q_ma = reshape_by_heads(self.Wq_2(embed_context), head_num=head_num)                       
        out_concat_ma = self._multi_head_attention_for_decoder(q_ma, self.k_ma, self.v_ma,
                                                            rank3_ninf_mask=ninf_mask_ma)                            
        mh_atten_out_ma = self.multi_head_combine_2(out_concat_ma)             
        
                                                                    
        score_ma = torch.matmul(mh_atten_out_ma, self.single_head_key_ma)                   

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled_ma = score_ma / sqrt_embedding_dim

        score_clipped_ma = logit_clipping * torch.tanh(score_scaled_ma)
        score_masked_ma = score_clipped_ma + ninf_mask_ma                   
        all_mas_prob = F.softmax(score_masked_ma, dim=2).squeeze(1)               
        
        batch_idxes = state.batch_idxes
                                                                                     
        if training or eval_type == 'softmax':
            while True:                                                                      
                select_ma = all_mas_prob.reshape(batch_size, -1).multinomial(1).squeeze(dim=1).reshape(batch_size, 1)         
                ma_prob = all_mas_prob.gather(1, select_ma)         
                non_finish_batch = torch.full(size=(batch_size, 1), dtype=torch.bool, fill_value=False)
                non_finish_batch[batch_idxes] = True
                finish_batch = torch.where(non_finish_batch == True, False, True)
                ma_prob[finish_batch] = 1                                     
                if (ma_prob != 0).all():
                    break
        else:
            select_ma = all_mas_prob.argmax(dim=1).unsqueeze(1)            
            ma_prob = torch.zeros(size=(batch_size, 1))                      
        if baseline:
            select_ma = all_mas_prob.argmax(dim=1).unsqueeze(1)            
            ma_prob = torch.zeros(size=(batch_size, 1))                      
        embed_select_ma = embed_mas.gather(1, select_ma[:, :, None].expand(-1, -1, embed_mas.size(-1)))             
        
                                 
                                                 
        ninf_mask_job_on_ma = ninf_mask.gather(2, select_ma[:, None, :].expand(-1, ninf_mask.size(1), -1)).transpose(1,2)                  
        
                                       
        q_ope = reshape_by_heads(self.Wq_3(embed_select_ma), head_num=head_num)                       
        out_concat = self._multi_head_attention_for_decoder(q_ope, self.k_ope, self.v_ope,
                                                            rank3_ninf_mask=ninf_mask_job_on_ma)                            
        mh_atten_out = self.multi_head_combine(out_concat)             

                                                                    
        score_ope = torch.matmul(mh_atten_out, self.single_head_key_ope)                      

        score_scaled_ope = score_ope / sqrt_embedding_dim

        score_clipped_ope = logit_clipping * torch.tanh(score_scaled_ope)
        score_masked_ope = score_clipped_ope + ninf_mask_job_on_ma                      
        all_jobs_prob = F.softmax(score_masked_ope, dim=2).squeeze(1)                  
        
                                 
        if training or eval_type == 'softmax':
            while True:                                                                      
                select_job = all_jobs_prob.reshape(batch_size, -1).multinomial(1).squeeze(dim=1).reshape(batch_size, 1)         
                job_prob = all_jobs_prob.gather(1, select_job)         
                non_finish_batch = torch.full(size=(batch_size, 1), dtype=torch.bool, fill_value=False)
                non_finish_batch[batch_idxes] = True
                finish_batch = torch.where(non_finish_batch == True, False, True)
                job_prob[finish_batch] = 1                                     
                if (job_prob != 0).all():
                    break
        else:
            select_job = all_jobs_prob.argmax(dim=1).unsqueeze(1)            
            job_prob = torch.zeros(size=(batch_size, 1))                      
        if baseline:
            select_job = all_jobs_prob.argmax(dim=1).unsqueeze(1)            
            job_prob = torch.zeros(size=(batch_size, 1))                      
                                                                                                                      
        select_ope = ope_step_batch.gather(1, select_job)           
        embed_select_job = embed_jobs.gather(1, select_job[:, :, None].expand(-1, -1, embed_jobs.size(-1)))
        
                             
        action = torch.cat([select_ope, select_ma, select_job], dim=1).transpose(1, 0)          
        prob = job_prob * ma_prob
        
        return action, prob.log(), embed_select_job + embed_select_ma
    
    
    
    def from_ope_to_job(self, select_ope, state, num_jobs, num_opes):
        '''
        Input:
            select_ope: [B,]: selected operation index
        Output:
            selecct_job: [B,]
        '''
        batch_size = state.ope_step_batch.size(0)
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch, state.end_ope_biases_batch, state.ope_step_batch)               
        empty_ope_batch = torch.ones(size=(batch_size, 1)) * num_opes
        ope_step_batch_plus1 = torch.cat([ope_step_batch, empty_ope_batch], dim=1)                  
        
        select_job = torch.where(ope_step_batch_plus1 == select_ope[:, None].expand(-1, num_jobs+1))[1]       
        return select_job
    
        
        
    def _multi_head_attention_for_decoder(self, q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
        '''
        Input:
            q: [B, H, 1, qkv_dim]
            k, v: [B, H, n_opes, qkv_dim]
            rank2_ninf_mask: [B, n_opes]
            rank3_ninf_mask: [B, n_mas, n_opes] or [B, 1, n_opes]
        '''
        batch_size = q.size(0)
        n = q.size(2)
        num_node = k.size(2)

        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        sqrt_qkv_dim = self.model_params['sqrt_qkv_dim']

        score = torch.matmul(q, k.transpose(2, 3))                                

        score_scaled = score / sqrt_qkv_dim

        if rank2_ninf_mask is not None:
            score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_size, head_num, n, num_node)
        if rank3_ninf_mask is not None:
            score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_size, head_num, n, num_node)

        weights = nn.Softmax(dim=3)(score_scaled)                                 

        out = torch.matmul(weights, v)                      

        out_transposed = out.transpose(1, 2)                        

        out_concat = out_transposed.reshape(batch_size, n, head_num * qkv_dim)                      

        return out_concat
    
        
class FFSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.encoded_NO_JOB = nn.Parameter(torch.rand(1, 1, embedding_dim))

        self.Wq_1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_3 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None                                       
        self.v = None                                         
        self.single_head_key = None                                        

    def set_kv(self, encoded_jobs):
                                                     
        batch_size = encoded_jobs.size(0)
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']

        encoded_no_job = self.encoded_NO_JOB.expand(size=(batch_size, 1, embedding_dim))
        encoded_jobs_plus_1 = torch.cat((encoded_jobs, encoded_no_job), dim=1)
                                              

                                                                                    
                                                                                    
        
        self.k = reshape_by_heads(self.Wk(encoded_jobs), head_num=head_num)                          
        self.v = reshape_by_heads(self.Wv(encoded_jobs), head_num=head_num)
                                                  
                                                                    
        self.single_head_key = encoded_jobs.transpose(1, 2)                     
                                          

    def forward(self, encoded_machine, ninf_mask):
                                                         
                                                   
        
        head_num = self.model_params['head_num']

                               
                                                               
        q = reshape_by_heads(self.Wq_3(encoded_machine), head_num=head_num)
                                                 
        out_concat = self._multi_head_attention_for_decoder(q, self.k, self.v,
                                                            rank3_ninf_mask=ninf_mask)                       
                                                
        mh_atten_out = self.multi_head_combine(out_concat)                 
                                         

                                                             
                                                               
        score = torch.matmul(mh_atten_out, self.single_head_key)                    
                                         

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
                                         

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

                                                
        probs = F.softmax(score_masked, dim=2)
                                         

        return probs

    def _multi_head_attention_for_decoder(self, q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
        '''
        Input:
            q: [B, H, n (selected job: 1), qkv_dim]
            k, v: [B, H, n_opes, qkv_dim]
            rank2_ninf_mask: [B, n_opes]
        '''
                                                                                      
                                                          
                                                   
                                                      
        batch_size = q.size(0)
        n = q.size(2)
        job_cnt_plus_1 = k.size(2)

        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        sqrt_qkv_dim = self.model_params['sqrt_qkv_dim']

        score = torch.matmul(q, k.transpose(2, 3))                     
                                                

        score_scaled = score / sqrt_qkv_dim

        if rank2_ninf_mask is not None:
            score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_size, head_num, n, job_cnt_plus_1)
        if rank3_ninf_mask is not None:
            score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_size, head_num, n, job_cnt_plus_1)

        weights = nn.Softmax(dim=3)(score_scaled)
                                                

        out = torch.matmul(weights, v)
                                              

        out_transposed = out.transpose(1, 2)
                                              

        out_concat = out_transposed.reshape(batch_size, n, head_num * qkv_dim)
                                             

        return out_concat



class FFSP_Decoder_update(FFSP_Decoder):
    def __init__(self, **model_params):
        super().__init__(**model_params)

class FJSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.encoded_NO_JOB = nn.Parameter(torch.rand(1, 1, embedding_dim))

        self.Wq_1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_3 = nn.Linear(2 * embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(2 * embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(2 * embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        
        self.single_head_combine = nn.Linear(embedding_dim, 1)

        self.k = None                                       
        self.v = None                                         
        self.single_head_key = None                                        

    def set_kv(self, encoded_jobs_mas):
        '''
        Input:
            encoded_jobs_mas: [B, n_jobs, n_mas, 2 * D_emb]
        '''
                                                     
        batch_size = encoded_jobs_mas.size(0)
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']

        k = reshape_by_heads(self.Wk(encoded_jobs_mas), head_num=head_num)                                 
        v = reshape_by_heads(self.Wv(encoded_jobs_mas), head_num=head_num)
                                                  
                                                                                  
        
                                                                                               
        return k, v

    def forward(self, encoded_job, encoded_machine, ninf_mask):
        '''
        Input:
            encoded_job: [B, n_jobs, D_emb]
            encoded_machine: [B, n_mas, D_emb]
            ninf_mask: [B, n_opes, n_mas]
        Output:
            probs: [B, n_opes * n_mas]
        '''
        head_num = self.model_params['head_num']
        batch_size = encoded_job.size(0)
        num_jobs = encoded_job.size(1)
        num_mas = encoded_machine.size(1)
                                                        
        encoded_jobs_pool = encoded_job[:, :, None, :].expand(-1, -1, num_mas, -1)                                
        encoded_mas_pool = encoded_machine[:, None, :, :].expand(-1, num_jobs, -1, -1)
        encoded_jobs_mas = torch.cat([encoded_jobs_pool, encoded_mas_pool], dim=-1)                                  
        
                        
        k, v = self.set_kv(encoded_jobs_mas)                                    

                                     
                                                               
        q = reshape_by_heads(self.Wq_3(encoded_jobs_mas), head_num=head_num)                                    
        out_concat = self._multi_head_attention_for_decoder(q, k, v,
                                                            rank3_ninf_mask=ninf_mask)                                    
                                                

        mh_atten_out = self.multi_head_combine(out_concat)                              
                                         
                                                
        
                                                             
                                                               
                                                                                      
                                         
        
        score = self.single_head_combine(mh_atten_out).squeeze(-1)                       

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
                                         

        score_clipped = logit_clipping * torch.tanh(score_scaled)
        
        score_masked = score_clipped + ninf_mask.reshape(batch_size, -1)                         

        probs = F.softmax(score_masked, dim=1)
                                         

        return probs

    def _multi_head_attention_for_decoder(self, q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
        '''
        Input:
            q: [B, H, n_jobs, n_mas, qkv_dim]
            k, v: [B, H, n_jobs, n_mas, qkv_dim]
            rank3_ninf_mask: [B, n_jobs, n_mas]
        '''
                                                                                      
                                                          
                                                   
                                                      
        batch_size = q.size(0)
        n_jobs = q.size(2)
        n_mas = q.size(3)
                       
                                    

        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        sqrt_qkv_dim = self.model_params['sqrt_qkv_dim']
        
        q_resh = q.reshape(batch_size, head_num, n_jobs * n_mas, qkv_dim)
        k_resh = k.reshape(batch_size, head_num, n_jobs * n_mas, qkv_dim)
        v_resh = v.reshape(batch_size, head_num, n_jobs * n_mas, qkv_dim)
        rank3_ninf_mask_resh = rank3_ninf_mask.reshape(batch_size, n_jobs * n_mas)

        score = torch.matmul(q_resh, k_resh.transpose(2, 3))                                          
                                                

        score_scaled = score / sqrt_qkv_dim

                                                                                                                                           
        
        weights = nn.Softmax(dim=3)(score_scaled)                                           
                                                        
        
        out = torch.matmul(weights, v_resh)                                  
                                                                                       

        out_transposed = out.transpose(1, 2)                                     

        out_concat = out_transposed.reshape(batch_size, n_jobs * n_mas, head_num * qkv_dim)                                   

        return out_concat

class MLPCritic(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLPCritic, self).__init__()

        self.linear_or_not = True                           
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
                          
            self.linear = nn.Linear(input_dim, output_dim)
        else:
                               
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            '''
            self.batch_norms = torch.nn.ModuleList()
            '''

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            '''
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
            '''

    def forward(self, x):
        if self.linear_or_not:
                             
            return self.linear(x)
        else:
                    
            h = x
            for layer in range(self.num_layers - 1):
                '''
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
                '''
                h = torch.tanh(self.linears[layer](h))
                                                    
            return self.linears[self.num_layers - 1](h)

                                        
                  
                                        


def reshape_by_heads(qkv, head_num):
                                                                                 
                               
    batch_s = qkv.size(0)
    qkv_len = len(qkv.size())
    
    if qkv_len == 3:
        n = qkv.size(1)

        q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
                                              

        q_transposed = q_reshaped.transpose(1, 2)
                                              

        return q_transposed
    elif qkv_len == 4:
        n = qkv.size(1)
        m = qkv.size(2)
        q_reshaped = qkv.reshape(batch_s, n, m, head_num, -1)                                   
        q_transposed =q_reshaped.permute(0, 3, 1, 2, 4)                                 
        
        return q_transposed
    else:
        raise ValueError("qkv_len error!")
