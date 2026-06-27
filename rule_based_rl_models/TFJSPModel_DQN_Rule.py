from copy import deepcopy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.distributions import Categorical

from rule_based_rl_models.TFJSPModel_DQN_Rule_sub import MLPActor, MLPCritic
from utils.memory import ReplayBuffer
from env.common_func import get_mask_ope_ma, norm_disc_rewards


SPT = 0
LPT = 1
FIFO = 2
LUM_SPT = 3
LUM_LPT = 4


    

class TFJSPModel_DQN_Rule(nn.Module):
    def __init__(self,
                 model_paras,
                 train_paras):
        super().__init__()
        
        self.ma_feat_dim = model_paras["in_size_ma"]                                                         
        self.ope_feat_dim = model_paras["in_size_ope"]                                                           
        self.veh_feat_dim = model_paras["in_size_veh"]                                                           
        
        self.embedding_dim = model_paras["embedding_dim"]                                 
        
        self.actor_dim = model_paras["actor_in_dim"]                            
        self.critic_dim = model_paras["critic_in_dim"]                             
        self.n_latent_actor = model_paras["n_latent_actor"]                                  
        self.n_latent_critic = model_paras["n_latent_critic"]                                   
        self.n_hidden_actor = model_paras["n_hidden_actor"]                             
        self.n_hidden_critic = model_paras["n_hidden_critic"]                              
                                   
                                                    
        self.action_dim = model_paras["action_dim"]  
        
        self.device = model_paras["device"]
        
                                  
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        
                                  
        self.init_embed_opes = nn.Linear(self.ope_feat_dim, self.embedding_dim)
        self.init_embed_mas = nn.Linear(self.ma_feat_dim, self.embedding_dim)
        self.init_embed_vehs = nn.Linear(self.veh_feat_dim, self.embedding_dim)
        self.init_embed_proc = nn.Linear(1, 1)
        self.init_embed_trans = nn.Linear(1, 1)
        
                            
        self.actor = MLPActor(self.n_hidden_actor, self.embedding_dim, self.n_latent_actor, self.action_dim)
        self.critic = MLPCritic(self.n_hidden_critic, self.embedding_dim, self.n_latent_critic, 1)
        
        self.MseLoss = nn.MSELoss()
        
        
        
    def init(self, state):
        self.batch_size, self.num_opes, self.num_mas = state.ope_ma_adj_batch.size()
        self.num_jobs = state.mask_job_finish_batch.size(1)
        self.num_vehs = state.mask_veh_procing_batch.size(1)
        
    def act(self, state, memory=None):
        action, act_idx, act_logprob = self.forward(state)
        
        if memory is not None:
            memory.add_action_info(action.transpose(1,0).detach().cpu().numpy(), 
                                    act_idx.detach().cpu().numpy(), 
                                    act_logprob.detach().cpu().numpy())
        
        return action, act_logprob
    
    def forward(self, state):
        '''
        Output:
            action: [4, B]
            log_p: [B, 1]
        '''
                                   
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
                                                            
                               
        embed_state, _, _ = self._get_norm_raw_feat(state)
        
        
                                                            
        if np.random.rand() <= self.epsilon:
            rule_idx = torch.randint(0, self.action_dim-1, size=(self.batch_size, 1)).long()
            rule_prob = torch.ones(size=(self.batch_size, 1)).div(self.action_dim)
        else:
            q_values = self.actor(embed_state)                     
            rule_probs = F.softmax(q_values, dim=1)
            rule_idx = rule_probs.argmax(dim=1).unsqueeze(1).long()         
            rule_prob = rule_probs.gather(1, rule_idx)            
                                                         
                                       
        
                                   
        select_ope = torch.zeros(size=(self.batch_size, 1), dtype=torch.long)
        select_job = torch.zeros(size=(self.batch_size, 1), dtype=torch.long)
        select_ma = torch.zeros(size=(self.batch_size, 1), dtype=torch.long)

        spt_batch = torch.where(rule_idx == SPT)
        if spt_batch[0].nelement() > 0:
            select_ope[spt_batch], select_ma[spt_batch], select_job[spt_batch] =\
                self._select_OMPair_on_ProcTime(spt_batch, state, proc_crit='short')
        
        lpt_batch = torch.where(rule_idx == LPT)
        if lpt_batch[0].nelement() > 0:
            select_ope[lpt_batch], select_ma[lpt_batch], select_job[lpt_batch] =\
            self._select_OMPair_on_ProcTime(lpt_batch, state, proc_crit='long')
        
        fifo_batch = torch.where(rule_idx == FIFO)
        if fifo_batch[0].nelement() > 0:
            select_job[fifo_batch], select_ope[fifo_batch] = self._select_fifo_job(fifo_batch, state)
            select_ma[fifo_batch] = self._select_fifo_ma(fifo_batch, state, select_job)

        lum_spt_batch = torch.where(rule_idx == LUM_SPT)
        if lum_spt_batch[0].nelement() > 0:
            select_ma[lum_spt_batch] = self._select_ma_on_util(lum_spt_batch, state, util_crit='low')
            select_ope[lum_spt_batch], select_job[lum_spt_batch] = self._select_oper_given_ma(lum_spt_batch, state, select_ma, proc_crit='short')

        lum_lpt_batch = torch.where(rule_idx == LUM_LPT)
        if lum_lpt_batch[0].nelement() > 0:
            select_ma[lum_lpt_batch] = deepcopy(self._select_ma_on_util(lum_lpt_batch, state, util_crit='low'))
            select_ope[lum_lpt_batch], select_job[lum_lpt_batch] = self._select_oper_given_ma(lum_lpt_batch, state, select_ma, proc_crit='long')
        
                                          
        veh_dict = self._select_nearest_veh(state, select_ma, select_job)
        select_veh = veh_dict['veh_id'].long()         
        
                                          
        action = torch.cat([select_ope, select_ma, select_job, select_veh], dim=1).transpose(1, 0)          
        
        return action, rule_idx, rule_prob.log()
    
    
    def _select_ma_on_util(self, batch, state, util_crit='low'):
        '''
        Input:
            batch: tensor of batch_idx (ex. tensor([0, 1, 2, ...]))
            state: ,
            util_crit: 
                'low': lowest utilization machine
                'high': highest utilization machine
        Output:
            select_ma: [len(batch), 1]
        '''
        mask, _ = get_mask_ope_ma(state)

        mask_ma = torch.where(mask.sum(dim=1) == 0, False, True)                
        
        util = deepcopy(state.feat_mas_batch[:, 2, :])                
        if util_crit == 'low':
            util[~mask_ma] = 1000
            select_ma = util.min(dim=1, keepdim=True)[1]            
        elif util_crit == 'high':
            util[~mask_ma] = 0
            select_ma = util.max(dim=1, keepdim=True)[1]            
        else:
            raise Exception('util_cirt error!')
        return select_ma[batch]
        
    def _select_oper_given_ma(self, batch, state, select_ma, proc_crit='short'):
        '''
        Input:
            state:
            select_ma: [B, 1]
            proc_crit:
                'short': shortest processing time operation given the machine
                'long': longest processing time operation given the machine
        Output:
            select_ope: [len(batch), 1]
            select_job: [len(batch), 1]
        '''
        batch_size, num_opes, num_mas = state.ope_ma_adj_batch.size()
        
                                  
        mask, mask_ope_ma = get_mask_ope_ma(state)
        
                                                                 
        proc_time = deepcopy(state.proc_times_batch)                     
        if proc_crit == 'short':
            proc_time[torch.where(mask_ope_ma == False)] = 1000
            proc_time = proc_time.gather(2, select_ma[:, None, :].expand(-1, num_opes, -1)).squeeze(2)               
            select_ope = proc_time.argmin(dim=1, keepdim=True)          
        elif proc_crit == 'long':
            proc_time[torch.where(mask_ope_ma == False)] = 0
            proc_time = proc_time.gather(2, select_ma[:, None, :].expand(-1, num_opes, -1)).squeeze(2)               
            select_ope = proc_time.argmax(dim=1, keepdim=True)          
        else:
            raise Exception('_select_oper_given_ma() error!')
        
        select_job = self.from_ope_to_job(select_ope.squeeze(1), state).unsqueeze(1).long()            
        return select_ope[batch], select_job[batch]
    
    def _select_nearest_veh(self, state, select_ma, select_job):
        '''
        Input:
            state:
            select_ope: [B, 1]
            select_ma: [B, 1]
            select_job: [B, 1]
        Output:
            results = {
                'veh_id': [B, 1]
                'trans_time': [B, 1]
            }
            
        '''
        trans_times_batch = state.trans_times_batch                      
        veh_loc_batch = state.veh_loc_batch              
        prev_ope_locs_batch = state.prev_ope_locs_batch              
        batch_size = trans_times_batch.size(0)
        
        elig_vehs = ~state.mask_veh_procing_batch                
        
        prev_ope_locs = prev_ope_locs_batch.gather(1, select_job)           
        
        
                                                                   
        results = {
            'veh_id': torch.zeros(size=(batch_size, 1)),
            'trans_time': torch.zeros(size=(batch_size, 1)),
        }
        for b in range(batch_size):
            elig_veh_ids = torch.where(elig_vehs[b, :] == True)

            veh_locs = veh_loc_batch[b, elig_veh_ids[0]]                   
            tmp_prev_ope_locs = prev_ope_locs[b].expand(veh_locs.size(0))
            tmp_select_ma = select_ma[b].expand(veh_locs.size(0))
            
            empty_trans = trans_times_batch[b, veh_locs, tmp_prev_ope_locs]
            travel_trans = trans_times_batch[b, tmp_prev_ope_locs, tmp_select_ma]
            trans_time = empty_trans + travel_trans
            min_value, min_idx = trans_time.min(dim=0, keepdim=True)
            results['veh_id'][b] = elig_veh_ids[0][min_idx]
            results['trans_time'][b] = min_value
                                     
        return results
    
    def _select_fifo_ma(self, batch, state, select_job):
        '''
        among multiple eligible machines, select one randomly
        
        Input:
            state:
            select_job: [batch, 1]
        Output:
            select_ma: [batch, 1]
        '''
        batch_size, num_opes, num_mas = state.ope_ma_adj_batch.size()
        num_jobs = state.mask_job_procing_batch.size(1)
        batch_idxes = state.batch_idxes
        
        mask, _ = get_mask_ope_ma(state)                        
        
        avail_mask_mas = mask.gather(1, select_job[:, :, None].expand(-1, -1, num_mas)).squeeze(1)              
        infeas_batch = torch.where(avail_mask_mas.sum(dim=1) == 0)[0]                   
        avail_mask_mas[infeas_batch, :] = True                          
        avail_mas = torch.where(avail_mask_mas == True, 0., -math.inf)              
        avail_ma_probs = F.softmax(avail_mas, dim=1)                
        
        while True:                                                                      
            select_ma = avail_ma_probs.reshape(batch_size, -1).multinomial(1).squeeze(dim=1).reshape(batch_size, 1)         
            ma_prob = avail_ma_probs.gather(1, select_ma)         
            if (ma_prob != 0).all():
                break
            else:
                raise Exception("select ma_prob with 0!")
        return select_ma[batch]
    
    
    def _select_fifo_job(self, batch, state):
        '''
        fifo: eligible jobs are processed as fast as possible
        
        when there are multiple eligible jobs, we select one randomly
        
        Input:
            state:
        Output:
            select_job: [batch, 1]
            select_ope: [batch, 1]
        '''
        batch_size = state.ope_ma_adj_batch.size(0)
        batch_idxes = state.batch_idxes
        
                                                                              
        mask, _ = get_mask_ope_ma(state)                        
        avail_jobs = torch.where(mask.sum(dim=2) > 0, 0., -math.inf)               
        avail_job_probs = F.softmax(avail_jobs, dim=1)               
        
        while True:                                                                      
            select_job = avail_job_probs.reshape(batch_size, -1).multinomial(1).squeeze(dim=1).reshape(batch_size, 1)         
            job_prob = avail_job_probs.gather(1, select_job)         
            non_finish_batch = torch.full(size=(batch_size, 1), dtype=torch.bool, fill_value=False)
            non_finish_batch[batch_idxes] = True
            finish_batch = torch.where(non_finish_batch == True, False, True)
            job_prob[finish_batch] = 1                                     
            if (job_prob != 0).all():
                break
                                                    
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)               
        select_ope = ope_step_batch.gather(1, select_job)           
        
        return select_job[batch], select_ope[batch]
    
    
    def _select_OMPair_on_ProcTime(self, batch, state, proc_crit='short'):
        '''
        Input:
            prec_cirt: 
                'short': shortest processing time
                'long': longest processing time
        Ouput:
            ope: [batch, 1]
            ma: [batch, 1]
            job: [batch, 1]
        '''
        batch_size, num_opes, num_mas = state.ope_ma_adj_batch.size()
        
                                  
        mask, mask_ope_ma = get_mask_ope_ma(state)
        
                                                  
        proc_time = deepcopy(state.proc_times_batch)                     
        if proc_crit == 'short':
            proc_time[torch.where(mask_ope_ma == False)] = 1000
            proc_time_resh = proc_time.reshape(batch_size, -1)                      
            OM_idx = proc_time_resh.argmin(dim=1, keepdim=True)          
        elif proc_crit == 'long':
            proc_time[torch.where(mask_ope_ma == False)] = 0
            proc_time_resh = proc_time.reshape(batch_size, -1)                      
            OM_idx = proc_time_resh.argmax(dim=1, keepdim=True)          
        else:
            raise Exception('implement this!')
                                                                
        num_mas_torch = torch.ones(size=(batch_size, 1)) * num_mas
        ma = torch.remainder(OM_idx, num_mas_torch).long()          
        ope = torch.div(OM_idx, num_mas_torch).floor().long()           
        job = self.from_ope_to_job(ope.squeeze(1), state).unsqueeze(1).long()            
        
                                                                     
        return ope[batch], ma[batch], job[batch]
    
    def evaluate(self, raw_opes, raw_mas, raw_vehs, proc_time, trans_time, ope_step_batch, 
                 old_action_idxes,
                ):
        '''
        Input:
            raw_opes: [B, n_opes, feat_ope_dim]
            raw_mas:
            proc_time:
            ope_step_batch: [B, n_jobs]
            old_action_idxes: [B, ]
        Output:
            act_logprob: [B, ]
            state_values: [B, ]
            dist_entropy: [B, ]
        '''
        raw_jobs = raw_opes.gather(1, ope_step_batch[:, :, None].expand(-1, -1, raw_opes.size(2)))               
        
                                 
        features = self.get_normalized(raw_jobs, raw_mas, raw_vehs, proc_time, trans_time,\
            flag_sample=True, flag_train=True)
        norm_jobs = (deepcopy(features[0]))
        norm_mas = (deepcopy(features[1]))
        norm_vehs = (deepcopy(features[2]))
        norm_proc_time = (deepcopy(features[3]))
        norm_trans_time = (deepcopy(features[4]))
                          
        embed_feat_ope = self.init_embed_opes(norm_jobs)                    
        embed_feat_ma = self.init_embed_mas(norm_mas)                  
        embed_feat_veh = self.init_embed_vehs(norm_vehs)                    
                                  
        embed_states = torch.cat([embed_feat_ope, embed_feat_ma, embed_feat_veh], dim=1)                                     
        embed_states = embed_states.mean(dim=1)         
                               
        act_score = self.actor(embed_states)                     
        act_probs = F.softmax(act_score, dim=1)
        state_values = self.critic(embed_states)
        dist = Categorical(act_probs)
        act_logprob = dist.log_prob(old_action_idxes)
        dist_entropy = dist.entropy()                                                                                           
        
        
        return act_logprob, state_values.squeeze().double(), dist_entropy
    
    def from_ope_to_job(self, select_ope, state, batch=None):
        '''
        Input:
            select_ope: [B,]: selected operation index
        Output:
            selecct_job: [B,]
        '''
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch, state.end_ope_biases_batch, state.ope_step_batch)               
        select_job = torch.where(ope_step_batch == select_ope[:, None].expand(-1, self.num_jobs))[1]       
        
        return select_job
    
    def _get_norm_raw_feat(self, state):
        '''
        Output:
            embed_states: [B, E]
            norm_proc_time: [B, n_opes, n_mas]
            norm_trans_time: [B, n_mas, n_mas]
        '''
        batch_size, num_opes, num_mas = state.ope_ma_adj_batch.size()
        num_jobs = state.mask_job_finish_batch.size(1)
        num_vehs = state.mask_veh_procing_batch.size(1)
                                       
        batch_idxes = state.batch_idxes
                           
        ope_ma_adj_batch = deepcopy(state.ope_ma_adj_batch)
        ope_ma_adj_batch = torch.where(ope_ma_adj_batch == 1, True, False)
        raw_opes = state.feat_opes_batch.transpose(1, 2)[batch_idxes]                              
        raw_mas = state.feat_mas_batch.transpose(1, 2)[batch_idxes]                          
        raw_vehs = state.feat_vehs_batch.transpose(1, 2)[batch_idxes]                              
        proc_time = state.proc_times_batch[batch_idxes]                     
        trans_time = state.trans_times_batch[batch_idxes]                      
                                              
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)               
        raw_jobs = raw_opes.gather(1, ope_step_batch[:, :, None].expand(-1, -1, raw_opes.size(2)))               
        
        
                           
        nums_opes = state.nums_opes_batch[batch_idxes]
        features = self.get_normalized(raw_jobs, raw_mas, raw_vehs, proc_time, trans_time,\
            flag_sample=True, flag_train=True)
        norm_jobs = (deepcopy(features[0]))
        norm_mas = (deepcopy(features[1]))
        norm_vehs = (deepcopy(features[2]))
        norm_proc_time = (deepcopy(features[3]))
        norm_trans_time = (deepcopy(features[4]))
        
                          
        embed_feat_ope = self.init_embed_opes(norm_jobs)                    
        embed_feat_ma = self.init_embed_mas(norm_mas)                  
        embed_feat_veh = self.init_embed_vehs(norm_vehs)                    
        
                                  
        embed_states = torch.cat([embed_feat_ope, embed_feat_ma, embed_feat_veh], dim=1)                                     
        embed_states = embed_states.mean(dim=1)         
        
        return embed_states, norm_proc_time, norm_trans_time
            

    def feature_normalize(self, data):
        return (data - torch.mean(data)) / ((data.std() + 1e-5))

    def get_normalized(self, raw_opes, raw_mas, raw_vehs, proc_time, trans_time, flag_sample=False, flag_train=False):
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

    def update(self, memory, optimizer, minibatch_size,
               gamma, K_epochs, eps_clip, A_coeff, vf_coeff, entropy_coeff):
        
                                   
        old_ope_ma_adj, old_ope_pre_adj, old_ope_sub_adj,\
            old_raw_opes, old_raw_mas, old_raw_vehs,\
            old_proc_time, old_trans_time,\
            old_ope_step_batch, old_eligible,\
            old_rewards, old_is_terminals,\
            old_logprobs, old_action_indexes = memory.all_sample()
        
        old_ope_ma_adj = old_ope_ma_adj.transpose(1,0).flatten(0, 1)                            
        old_ope_pre_adj = old_ope_pre_adj.transpose(1,0).flatten(0, 1)                             
        old_ope_sub_adj = old_ope_sub_adj.transpose(1,0).flatten(0, 1)                             
        
        old_raw_opes = old_raw_opes.transpose(1,0).flatten(0, 1).transpose(1,2)                                       
        old_raw_mas = old_raw_mas.transpose(1,0).flatten(0, 1).transpose(1,2)                                       
        old_raw_vehs = old_raw_vehs.transpose(1,0).flatten(0, 1).transpose(1,2)                                         
        
        old_proc_time = old_proc_time.transpose(1,0).flatten(0, 1)                                  
        old_trans_time = old_trans_time.transpose(1,0).flatten(0, 1)                                 
        old_ope_step_batch = old_ope_step_batch.transpose(1,0).flatten(0, 1)                     
        old_eligible = old_eligible.transpose(1,0).flatten(0, 1)                                  

        old_rewards = old_rewards.transpose(1,0)                                    
        old_is_terminals = old_is_terminals.transpose(1,0)                          
        
        old_logprobs = old_logprobs.transpose(1,0).flatten(0, 1).squeeze()                       
        old_action_indexes = old_action_indexes.transpose(1,0).flatten(0, 1).squeeze()               

                                              
        rewards_batch = norm_disc_rewards(old_rewards, old_is_terminals, gamma, self.device)             
        
                                            
        loss_epochs = 0
        full_batch_size = old_ope_ma_adj.size(0)
        num_complete_minibatches = math.floor(full_batch_size / minibatch_size)
        for _ in range(K_epochs):
            for i in range(num_complete_minibatches+1):
                if i < num_complete_minibatches:
                    start_idx = i * minibatch_size
                    end_idx = (i + 1) * minibatch_size
                else:
                    start_idx = i * minibatch_size
                    end_idx = full_batch_size
        
                logprobs, state_values, dist_entropy =\
                    self.evaluate(
                        old_raw_opes[start_idx: end_idx, :, :], 
                        old_raw_mas[start_idx: end_idx, :, :], 
                        old_raw_vehs[start_idx: end_idx, :, :], 
                        old_proc_time[start_idx: end_idx, :, :], 
                        old_trans_time[start_idx: end_idx, :, :], 
                        old_ope_step_batch[start_idx: end_idx, :], 
                        old_action_indexes[start_idx: end_idx]
                        )                                         
                
                ratios = torch.exp(logprobs - old_logprobs[i*minibatch_size:(i+1)*minibatch_size].detach())
                advantages = rewards_batch[i*minibatch_size:(i+1)*minibatch_size] - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
                loss = - A_coeff * torch.min(surr1, surr2)\
                    + vf_coeff * self.MseLoss(state_values, rewards_batch[i*minibatch_size:(i+1)*minibatch_size])\
                    - entropy_coeff * dist_entropy
                loss_epochs += loss.mean().detach()

                optimizer.zero_grad()
                loss.mean().backward()
                optimizer.step()
        

        return loss_epochs.item() / K_epochs

