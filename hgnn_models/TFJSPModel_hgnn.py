
from copy import deepcopy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.distributions import Categorical

from hgnn_models.TFJSPModel_hgnn_sub import GATedge, MLPsim, MLPs
from rule_based_rl_models.TFJSPModel_DQN_Rule_sub import MLPActor, MLPCritic
from env.common_func import get_normalized, norm_disc_rewards, select_vehicle_v2

class TFJSPModel_hgnn(nn.Module):
    def __init__(self,
                 env_paras,
                 model_paras,
                 ):
        super().__init__()
        
                                         
        self.in_size_ma = env_paras["ma_feat_dim"]                                                         
        self.in_size_ope = env_paras["ope_feat_dim"]                                                           

        self.device = model_paras["device"]
        self.out_size_ma = model_paras["out_size_ma"]                                               
        self.out_size_ope = model_paras["out_size_ope"]                                                 
        self.hidden_size_ope = model_paras["hidden_size_ope"]                                 
        self.actor_dim = model_paras["actor_in_dim"]                            
        self.critic_dim = model_paras["critic_in_dim"]                             
        self.n_latent_actor = model_paras["n_latent_actor"]                                  
        self.n_latent_critic = model_paras["n_latent_critic"]                                   
        self.n_hidden_actor = model_paras["n_hidden_actor"]                             
        self.n_hidden_critic = model_paras["n_hidden_critic"]                              
        self.action_dim = model_paras["action_dim"]                             
        
        self.num_heads = model_paras["num_heads"]
        self.dropout = model_paras["dropout"]
        
                                
        self.get_machines = nn.ModuleList()
        self.get_machines.append(GATedge((self.in_size_ope, self.in_size_ma), self.out_size_ma, self.num_heads[0],
                                    self.dropout, self.dropout, activation=F.elu))
        for i in range(1,len(self.num_heads)):
            self.get_machines.append(GATedge((self.out_size_ope, self.out_size_ma), self.out_size_ma, self.num_heads[i],
                                    self.dropout, self.dropout, activation=F.elu))

                                  
        self.get_operations = nn.ModuleList()
        self.get_operations.append(MLPs([self.out_size_ma, self.in_size_ope, self.in_size_ope, self.in_size_ope],
                                        self.hidden_size_ope, self.out_size_ope, self.num_heads[0], self.dropout))
        for i in range(len(self.num_heads)-1):
            self.get_operations.append(MLPs([self.out_size_ma, self.out_size_ope, self.out_size_ope, self.out_size_ope],
                                            self.hidden_size_ope, self.out_size_ope, self.num_heads[i], self.dropout))

        self.actor = MLPActor(self.n_hidden_actor, self.actor_dim, self.n_latent_actor, self.action_dim).to(self.device)
        self.critic = MLPCritic(self.n_hidden_critic, self.critic_dim, self.n_latent_critic, 1).to(self.device)
        
        self.MseLoss = nn.MSELoss()
        
    
    def init(self, state, dataset=None, loader=None):
        pass
    
    def act(self, state, memory=None):
        action, act_idx, act_prob = self.forward(state)
        if memory is not None:
            memory.add_action_info(action.transpose(1,0).detach().cpu().numpy(), 
                                    act_idx.detach().cpu().numpy(), 
                                    act_prob.detach().cpu().numpy())
        
        return action, act_prob
    
    def forward(self, state, flag_sample=True, flag_train=True):
                                                                                                                 
        action_probs, ope_step_batch, _ = self.get_action_prob(state, flag_sample, flag_train=flag_train)                         

                                               
        if flag_sample:
            dist = Categorical(action_probs)
            action_indexes = dist.sample()
                                                                      
        else:
            action_indexes = action_probs.argmax(dim=1)
        select_act_prob = dist.log_prob(action_indexes).exp().unsqueeze(-1)           

                                                                                  
        mas = (action_indexes / state.mask_job_finish_batch.size(1)).long()           
        jobs = (action_indexes % state.mask_job_finish_batch.size(1)).long()          
        opes = ope_step_batch[state.batch_idxes, jobs]
        
                                
        veh_dict = select_vehicle_v2(state, mas.unsqueeze(1), jobs.unsqueeze(1))
        vehs = veh_dict['veh_id'].long()         

        return torch.stack((opes, mas, jobs, vehs), dim=1).t(), action_indexes, select_act_prob

    def get_action_prob(self, state, memories=None, flag_sample=False, flag_train=False):
        '''
        Get the probability of selecting each action in decision-making
        '''
                               
        batch_idxes = state.batch_idxes
        
                              
        raw_opes = state.feat_opes_batch.transpose(1, 2)[batch_idxes]                              
        raw_mas = state.feat_mas_batch.transpose(1, 2)[batch_idxes]                          
        raw_vehs = state.feat_vehs_batch.transpose(1, 2)[batch_idxes]                              
        proc_time = state.proc_times_batch[batch_idxes]                     
        trans_time = state.trans_times_batch[batch_idxes]                      
                                              
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)               
        raw_jobs = raw_opes.gather(1, ope_step_batch[:, :, None].expand(-1, -1, raw_opes.size(2)))               
        
                           
        nums_opes = state.nums_opes_batch[batch_idxes]
        features = get_normalized(raw_opes, raw_mas, raw_vehs, proc_time, trans_time,\
            flag_sample=True, flag_train=True)
        norm_opes = (deepcopy(features[0]))
        norm_mas = (deepcopy(features[1]))
        norm_vehs = (deepcopy(features[2]))
        norm_proc_time = (deepcopy(features[3]))
        norm_trans_time = (deepcopy(features[4]))
        
        feat_tuple = (features[0], features[1], features[3])
                                  
        for i in range(len(self.num_heads)):
                                                 
                                                             
            h_mas = self.get_machines[i](state.ope_ma_adj_batch, state.batch_idxes, feat_tuple)                     
            feat_tuple = (feat_tuple[0], h_mas, feat_tuple[2])
                                                    
                                                                    
            h_opes = self.get_operations[i](state.ope_ma_adj_batch, state.ope_pre_adj_batch, state.ope_sub_adj_batch,
                                            state.batch_idxes, feat_tuple)                           
            feat_tuple = (h_opes, feat_tuple[1], feat_tuple[2])

                              
        h_mas_pooled = h_mas.mean(dim=-2)                                          
                                                                                                            
        if not flag_sample and not flag_train:
            h_opes_pooled = []
            for i in range(len(batch_idxes)):
                h_opes_pooled.append(torch.mean(h_opes[i, :nums_opes[i], :], dim=-2))
            h_opes_pooled = torch.stack(h_opes_pooled)                                
        else:
            h_opes_pooled = h_opes.mean(dim=-2)                                           

                                                                                                 
        jobs_gather = ope_step_batch[..., :, None].expand(-1, -1, h_opes.size(-1))[batch_idxes]                                  
        h_jobs = h_opes.gather(1, jobs_gather)                               
        
        
                                                              
                                                      
        eligible_proc = state.ope_ma_adj_batch[batch_idxes].gather(1,
                          ope_step_batch[..., :, None].expand(-1, -1, state.ope_ma_adj_batch.size(-1))[batch_idxes])
        h_jobs_padding = h_jobs.unsqueeze(-2).expand(-1, -1, state.proc_times_batch.size(-1), -1)                                        
                                                                   
                                                         
        h_mas_padding = h_mas.unsqueeze(-3).expand_as(h_jobs_padding)                                            
                                                                                                       
        h_mas_pooled_padding = h_mas_pooled[:, None, None, :].expand_as(h_jobs_padding)                                          
        h_opes_pooled_padding = h_opes_pooled[:, None, None, :].expand_as(h_jobs_padding)
                                                           
                                                      
        ma_eligible = ~state.mask_ma_procing_batch[batch_idxes].unsqueeze(1).expand_as(h_jobs_padding[..., 0])
                                                   
        job_eligible = ~(state.mask_job_procing_batch[batch_idxes] +
                         state.mask_job_finish_batch[batch_idxes])[:, :, None].expand_as(h_jobs_padding[..., 0])
        eligible = job_eligible & ma_eligible & (eligible_proc == 1)
        if (~(eligible)).all():
            print("No eligible O-M pair!")
            return
                            
                                                                                    
        h_actions = torch.cat((h_jobs_padding, h_mas_padding, h_opes_pooled_padding, h_mas_pooled_padding),
                              dim=-1).transpose(1, 2)
        h_pooled = torch.cat((h_opes_pooled, h_mas_pooled), dim=-1)              
        mask = eligible.transpose(1, 2).flatten(1)              
                                                                                           
        scores = self.actor(h_actions).flatten(1)               
        scores[~mask] = float('-inf')
        action_probs = F.softmax(scores, dim=1)


        return action_probs, ope_step_batch, h_pooled
    
    def evaluate(self, ope_ma_adj, ope_pre_adj, ope_sub_adj, raw_opes, raw_mas, proc_time,
                 ope_step_batch, eligible, action_idx, flag_sample=False):
        '''
        Input:
            ope_step_batch: [B, n_jobs]
        '''
        batch_idxes = torch.arange(0, ope_ma_adj.size(-3)).long()
        features = (raw_opes, raw_mas, proc_time)

                                  
        for i in range(len(self.num_heads)):
            h_mas = self.get_machines[i](ope_ma_adj, batch_idxes, features)
            features = (features[0], h_mas, features[2])
            h_opes = self.get_operations[i](ope_ma_adj, ope_pre_adj, ope_sub_adj, batch_idxes, features)
            features = (h_opes, features[1], features[2])

                              
        h_mas_pooled = h_mas.mean(dim=-2)
        h_opes_pooled = h_opes.mean(dim=-2)

                                                                                                  
        h_jobs = h_opes.gather(1, ope_step_batch[:, :, None].expand(-1, -1, h_opes.size(2)))                               
        h_jobs_padding = h_jobs.unsqueeze(-2).expand(-1, -1, proc_time.size(-1), -1)
        h_mas_padding = h_mas.unsqueeze(-3).expand_as(h_jobs_padding)
        h_mas_pooled_padding = h_mas_pooled[:, None, None, :].expand_as(h_jobs_padding)
        h_opes_pooled_padding = h_opes_pooled[:, None, None, :].expand_as(h_jobs_padding)

        h_actions = torch.cat((h_jobs_padding, h_mas_padding, h_opes_pooled_padding, h_mas_pooled_padding),
                              dim=-1).transpose(1, 2)
        h_pooled = torch.cat((h_opes_pooled, h_mas_pooled), dim=-1)
        scores = self.actor(h_actions).flatten(1)                        
        mask = eligible.transpose(1, 2).flatten(1)                       

        scores[~mask] = float('-inf')
        action_probs = F.softmax(scores, dim=1)
        state_values = self.critic(h_pooled)
        dist = Categorical(action_probs.squeeze())
        action_logprobs = dist.log_prob(action_idx)          
        dist_entropys = dist.entropy()
        
        return action_logprobs, state_values.squeeze().double(), dist_entropys

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

                                              
        rewards_batch, _ = norm_disc_rewards(old_rewards, old_is_terminals, gamma, self.device)           
        rewards_batch = rewards_batch.reshape(-1).double()            
        
                                            
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
                        old_ope_ma_adj[start_idx: end_idx, :, :],
                        old_ope_pre_adj[start_idx: end_idx, :, :],
                        old_ope_sub_adj[start_idx: end_idx, :, :],
                        old_raw_opes[start_idx: end_idx, :, :], 
                        old_raw_mas[start_idx: end_idx, :, :], 
                        old_proc_time[start_idx: end_idx, :, :], 
                        old_ope_step_batch[start_idx: end_idx, :], 
                        old_eligible[start_idx: end_idx, :], 
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