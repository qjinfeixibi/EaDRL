from json import encoder
import torch
import numpy as np
import os
from copy import deepcopy
from collections import deque
import random
import time
import pandas as pd
import math
import torch.nn.functional as F
from env.case_generator_v2 import CaseGenerator
from env.tfjsp_env import TFJSPEnv

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from logging import getLogger
from utils.utils import *

from DHJS_models.TFJSPModel_dhjs import TFJSPModel_DHJS
from DHJS_models.meta_learner import subgraph_meta_learner
from SAHJS_models.temporal_graph_dataset import set_GraphData
from torch_geometric.loader.dataloader import DataLoader
from sat_models.sat.data import GraphDataset
from DHJS_models.TFJSPTester import validate_multi_models,generate_vali_env,validate_one_model

from env.tfjsp_env_batch_grid import GridTFJSPEnv
import dynamicAMR.mapf.cactus.algorithms as algorithms
from dynamicAMR.mapf.cactus.constants import *
from dynamicAMR.mapf.cactus.env.env_generator import generate_batch_gridworld

params = {}
params[EPISODES_PER_EPOCH] = 1
params[ENV_OBSERVATION_SIZE] = 7
params[ENV_NR_AGENTS] = 6
params[HIDDEN_LAYER_DIM] = 64
params[ENV_GAMMA] = 1.0
params[RENDER_MODE] = False
params[ENV_MAKESPAN_MODE] = False
params[GRAD_NORM_CLIP] = 10
params[VDN_MODE] = False
params[REWARD_SHARING] = False
params[MIXING_HIDDEN_SIZE] = 128
params[ENV_TIME_LIMIT] = 204800
params[ENV_STEPTIME] = 1
params[ALGORITHM_NAME] = ALGORITHM_PPO_QMIX
params["observation_dim"] = [5, 7, 7]

def clip_grad_norms(param_groups, max_norm=math.inf):

    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf, 
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


class TFJSPTrainer_DHJS():
    def __init__(self,
                env_paras,
                model_paras,
                train_paras,
                optimizer_paras,
                test_paras,
                change_paras,
                model_version=1,
                ):
        self.env_paras = env_paras
        self.model_paras = model_paras
        self.train_paras = train_paras
        self.optimizer_paras = optimizer_paras
        self.test_paras = test_paras
        self.change_paras = change_paras
        
        self.accum_steps = 16 
        self.grad_accum_counter = 0 
        
        self.device = self.model_paras["device"]
                                
        self.env_valid_paras = deepcopy(env_paras)
        self.env_valid_paras["batch_size"] = env_paras["valid_batch_size"]
        
        self.num_jobs = env_paras["num_jobs"]
        self.num_opes = env_paras["num_opes"]
        self.num_mas = env_paras["num_mas"]
        self.num_vehs = env_paras["num_vehs"]
        model_paras['num_opes'] = env_paras["num_opes"]
        model_paras['num_mas'] = env_paras["num_mas"]
        model_paras['num_vehs'] = env_paras["num_vehs"]
        
        self.opes_per_job_min = int(self.num_mas * 0.8)
        self.opes_per_job_max = int(self.num_mas * 1.2)        
        self.proctime_per_ope_max = env_paras["proctime_per_ope_max"]
        self.transtime_btw_ma_max = env_paras["transtime_btw_ma_max"]
        self.dynamic = env_paras['dynamic']
        
        
                                           
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()
        
        self.start_epoch = 1
        
                                             
        encoder_version = 0
        decoder_version = 1
        if model_version == 1:
            encoder_version = 1
            decoder_version = 1
        elif model_version == 2:
            encoder_version = 1
            decoder_version = 2
        elif model_version == 3:
            encoder_version = 2
            decoder_version = 1
        elif model_version == 4:
            encoder_version = 3
            decoder_version = 1
        elif model_version == 5:
            encoder_version = 4
            decoder_version = 1
        elif model_version == 6:
            encoder_version = 5
            decoder_version = 1
        elif model_version == 7:
            encoder_version = 6
            decoder_version = 1
        elif model_version == 8:
            encoder_version = 1
            decoder_version = 3
        elif model_version == 9:
            encoder_version = 7
            decoder_version = 1
        elif model_version == 10:
            encoder_version = 8
            decoder_version = 1
        elif model_version == 11:
            encoder_version = 9
            decoder_version = 1
        elif model_version == 12:
            encoder_version = 10
            decoder_version = 1
        elif model_version == 13:
            encoder_version = 11
            decoder_version = 1
        elif model_version == 14:
            encoder_version = 12
            decoder_version = 1
               
                                                               

        self.encoder_version = encoder_version
                                               
        self.model = TFJSPModel_DHJS(
            embedding_dim_=model_paras["embedding_dim"],
            hidden_dim_=model_paras["hidden_dim"],
            problem=None,
            ope_feat_dim=model_paras["in_size_ope"],
            ma_feat_dim=model_paras["in_size_ma"],
            veh_feat_dim=model_paras["in_size_veh"],
            mask_inner=True,
            mask_logits=True,
            encoder_version=encoder_version,
            decoder_version=decoder_version,
            meta_rl=train_paras['meta_rl'] if train_paras['meta_rl']['enable'] else None,
            **model_paras
        ).to(model_paras["device"])
        self.base_model = deepcopy(self.model)
        

        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_paras['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_paras['scheduler'])        
        
                                                  
        self.prev_model_para = None
        
    
    def run(self):
        worst_epi_len = 200
        env, train_dataset, train_loader, controller= self._generate_gridenv(
            self.num_jobs, self.num_opes, self.num_mas, self.num_vehs, self.device, 
            self.opes_per_job_min, self.opes_per_job_max,
            self.dynamic,
            self.proctime_per_ope_max, self.transtime_btw_ma_max,
            job_centric=self.model_paras['job_centric'],       
            new_job_flag=self.env_paras['new_job']        
        )
        self.logger.info('----- Initial environment: job_{}, opes_{} mas_{}, veh_{} -----'\
            .format(env.num_jobs, env.num_opes, env.num_mas, env.num_vehs))
        self.logger.info(f'num_opes_list:{env.num_opes_list}')
        
                                          
        change_env_idx = 1
        sc = []
        st = []
        for epoch in range(self.start_epoch, self.train_paras['epochs']+1):
            self.logger.info('=================================================================')
            
            if  self.change_paras['enable'] == True:
                if (epoch % self.change_paras['change_interval'] == 0) and (epoch != 1):
                    self._change_env_paras(self.env_paras, self.change_paras, change_env_idx)
                    change_env_idx += 1
                    self.init(self.env_paras)
                    self.logger.info('----- change environment with job_{}, opes_{}, mas_{}, veh_{} -----'.format(env.num_jobs, self.num_opes, self.num_mas, self.num_vehs))
                    env, train_dataset, train_loader = self._generate_env(
                        self.num_jobs, self.num_opes, self.num_mas, self.num_vehs, self.device, 
                        self.opes_per_job_min, self.opes_per_job_max,
                        self.dynamic,
                        self.proctime_per_ope_max, self.transtime_btw_ma_max,
                        job_centric=self.model_paras['job_centric'],
                        new_job_flag=self.env_paras['new_job']
                    )
                        
            else:
                if epoch % self.train_paras["parallel_iter"] == 0 and epoch > 0:
                    env, train_dataset, train_loader, controller = self._generate_gridenv(
                        self.num_jobs, self.num_opes, self.num_mas, self.num_vehs, self.device, 
                        self.opes_per_job_min, self.opes_per_job_max,
                        self.dynamic,
                        self.proctime_per_ope_max, self.transtime_btw_ma_max,
                        job_centric=self.model_paras['job_centric'],
                        new_job_flag=self.env_paras['new_job']
                    )
                    self.logger.info('------ new environment generation:  job_{}, opes_{} mas_{}, veh_{} -----'\
                        .format(env.num_jobs, env.num_opes, env.num_mas, env.num_vehs))
                                                                            
            
                           
            train_score, train_loss = self._train_ope_epoch(
                epoch, env, controller,
                train_dataset, train_loader,
            )
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)
            
                                      
            all_done = (epoch == self.train_paras['epochs'])
            model_save_interval = self.train_paras['logging']['model_save_interval']
            img_save_interval = self.train_paras['logging']['img_save_interval']
            
            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))
            
            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(image_prefix, self.train_paras['logging']['log_image_params_1'],
                                               self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.train_paras['logging']['log_image_params_2'],
                                               self.result_log, labels=['train_loss'])

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)



    
    def _train_ope_epoch(self, epoch, env, controller,
                        train_dataset=None, train_loader=None):
        train_num_episode = self.train_paras["train_episodes"]
        episode = 0
        loop_cnt = 0
        train_cnt = 0

        score_list = []
        loss_list = []

        while episode < train_num_episode:
                 
            remaining = train_num_episode - episode
            
            
                             
            if self.train_paras['meta_rl']['enable']:
                if self.train_paras['meta_rl']['use_subgraphs']:
                                                         
                    graph_list, minibatch_size_list = subgraph_meta_learner(
                        self.env_paras['meta_rl']['minibatch'], self.num_opes, self.num_mas, self.num_vehs, 
                        self.proctime_per_ope_max, self.transtime_btw_ma_max,
                        self.env_paras, self.device, self.dynamic
                        )
                                                                         
                else:
                    minibatch_size = self.train_paras['meta_rl']['minibatch']
                    num_graph = self.train_paras['meta_rl']['num_graph']
                    minibatch_size_list = [minibatch_size] * num_graph
                                              
                    graph_list = self._generate_graph_list(
                        num_graph, self.num_opes, self.num_mas, self.num_vehs, self.device, self.dynamic,
                        self.proctime_per_ope_max, self.transtime_btw_ma_max
                    )
                
                                    
                score,loss = self._meta_train(minibatch_size_list, graph_list)
                batch_size = sum(minibatch_size_list)
            else:
                batch_size = self.env_paras['batch_size']
                                                 
                          
                score, loss = self._train_grid_batch(
                    batch_size, env, controller, train_dataset, train_loader)
            
            episode += batch_size
            train_cnt += 1

            score_list.append(score)
            loss_list.append(loss)

                                                          
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score, loss))
        avg_score = sum(score_list) / len(score_list)
        avg_loss = sum(loss_list) / len(loss_list)
                                  
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 avg_score, avg_loss))
        return avg_score, avg_loss
    
    def _meta_train(self, minibatch_size_list, graph_list):
        assert self.encoder_version != 3 or self.encoder_version != 5
        num_graph = len(graph_list)
                                 
        self.model.train()
        state_list = [env.reset() for env in graph_list]
        
        base_env_list = [deepcopy(env) for env in graph_list]
        self.base_model.load_state_dict(self.model.state_dict())
        
                             
        graphs_loss = 0
        graphs_score = 0
        
        for idx, env in enumerate(graph_list):
            done = False
            dones = env.done_batch
            epi_log_p = torch.zeros(size=(minibatch_size_list[idx], 0))
            all_rewards = torch.zeros(size=(minibatch_size_list[idx],))
            state = state_list[idx]
            while not done:
                action, log_p = self.model.act(state)                            
                state, rewards, dones = env.step(action)
                done = dones.all()
                epi_log_p = torch.cat([epi_log_p, log_p], dim=1)                
                all_rewards += rewards

                                  
            baseline_value = self._baseline(base_env_list[idx], self.base_model)
            advantage = rewards - baseline_value
                                                  
            epi_log_p = epi_log_p.sum(dim=1)            
            loss = -advantage * epi_log_p
            loss_mean = loss.mean()
            graphs_loss += loss_mean
            
                           
            score = env.get_makespan().mean()
            graphs_score += score
        self.optimizer.zero_grad()
        graphs_loss.backward()
        self.optimizer.step()
        
        return graphs_score.item(), graphs_loss.item()
    def _train_grid_batch(self, batch_size, env, controller, dataset=None, loader=None):
                                 
        self.model.train()
        state = env.reset()
        self.model.init(state)                    
        old_log_probs, rewards, values = [], [], []
        rollout_states, rollout_actions, rollout_prev_embeds = [], [], []

                             
        done = False
        dones = env.done_batch
        logits_list = []
        loop = 0
        while not done:
            rollout_states.append(deepcopy(state))
            rollout_prev_embeds.append(self.model.prev_embed.detach().clone())
            action, log_p, value, logits = self.model.act(state, return_state_vec=True)
            state, reward, dones = env.step(action, controller, step_reward="version2")

            rollout_actions.append(action.detach().clone())
            old_log_probs.append(log_p.squeeze(-1).detach())
            rewards.append(reward)
            values.append(value.squeeze(-1).detach())
            logits_list.append(logits)

            done = dones.all()
            loop += 1

                                                      
        with torch.no_grad():
            if env.done or dones.all():
                next_value = torch.zeros_like(values[-1])       
            else:
                                             
                _, _, next_value, _ = self.model.act(state, return_state_vec=True)
                next_value = next_value.squeeze(-1)       

        values = torch.stack(values)                                      
        values = torch.cat([values, next_value[None, :]], dim=0)            

                             
        entropy_terms = []
        for step_logits in logits_list:
            dist_ope = torch.distributions.Categorical(logits=step_logits['ope_logits'])
            dist_ma  = torch.distributions.Categorical(logits=step_logits['ma_logits'])
            dist_veh = torch.distributions.Categorical(logits=step_logits['veh_logits'])
            step_entropy = dist_ope.entropy() + dist_ma.entropy() + dist_veh.entropy()       
            entropy_terms.append(step_entropy)
        entropy_terms = torch.stack(entropy_terms)          
        entropy_bonus = entropy_terms.mean()
        entropy_coeff = 0.02

                                
        rewards = torch.stack(rewards)          
        advantages_raw = compute_gae(rewards, values, gamma=1, lam=0.95)          
        returns = advantages_raw + values[:-1]          
        advantages = (advantages_raw - advantages_raw.mean()) / (advantages_raw.std() + 1e-8)

        old_log_probs = torch.stack(old_log_probs)          

        k_epochs = self.train_paras.get('K_epochs', 1)
        eps_clip = self.train_paras.get('eps_clip', 0.2)
        vf_coeff = self.train_paras.get('vf_coeff', 0.5)
        entropy_coeff = self.train_paras.get('entropy_coeff', 0.01)
        max_grad_norm = self.train_paras.get('max_grad_norm', None)

        actor_loss = None
        critic_loss = None
        total_loss = None
        for _ in range(k_epochs):
            current_log_probs = []
            current_values = []
            current_entropy_terms = []
            for step_state, step_prev_embed, step_action in zip(
                rollout_states, rollout_prev_embeds, rollout_actions
            ):
                log_p_new, value_new, step_logits = self.model.evaluate_actions(
                    step_state, step_action, step_prev_embed
                )
                current_log_probs.append(log_p_new.squeeze(-1))
                current_values.append(value_new.squeeze(-1))
                dist_ope = torch.distributions.Categorical(logits=step_logits['ope_logits'])
                dist_ma = torch.distributions.Categorical(logits=step_logits['ma_logits'])
                dist_veh = torch.distributions.Categorical(logits=step_logits['veh_logits'])
                current_entropy_terms.append(
                    dist_ope.entropy() + dist_ma.entropy() + dist_veh.entropy()
                )

            current_log_probs = torch.stack(current_log_probs)
            current_values = torch.stack(current_values)
            current_entropy = torch.stack(current_entropy_terms).mean()

            ratios = torch.exp(current_log_probs - old_log_probs)
            surr1 = ratios * advantages.detach()
            surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages.detach()
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(current_values, returns.detach())
            total_loss = actor_loss + vf_coeff * critic_loss - entropy_coeff * current_entropy

            self.optimizer.zero_grad()
            total_loss.backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.optimizer.step()

        print(f"[Train Batch] Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}")
        print("reward:", rewards.mean().item(), "makespan:", env.get_makespan().mean().item())
        score = env.get_makespan().mean()
        return score.item(), total_loss.item()
    
    def _baseline(self, env, controller, model):
        model.eval()
        state = env.reset()
        model.init(state)
        
        done = False
        dones = env.done_batch
        while not done:
            with torch.no_grad():
                action, _ = model.act(state, baseline=True)                    
            state, rewards, dones = env.step(action, controller)
            done = dones.all()
        return rewards
    
    def _change_env_paras(self, env_paras, change_paras, change_env_idx):
        candidate_idx = change_env_idx % change_paras['num_candidate']
        
                                                                
        env_paras['num_jobs'] = change_paras['env_paras']['num_jobs'][candidate_idx]
        env_paras['num_mas'] = change_paras['env_paras']['num_mas'][candidate_idx]
        env_paras['num_vehs'] = change_paras['env_paras']['num_vehs'][candidate_idx]
        
        
    
    def _generate_graph_list(
        self, num_graph, num_opes, num_mas, num_vehs, device, dynamic=None,
        proctime_per_ope_max=20, transtime_btw_ma_max=10, version=None
    ):
        assert num_mas >= 2 and num_vehs >= 2 and num_opes >= 2
        graph_list = []
        for i in range(num_graph):
                                  
            tmp_num_opes = random.randint(2, num_opes)
            tmp_num_mas = random.randint(2, num_mas)
            tmp_num_vehs = random.randint(2, num_vehs)

            env = self._generate_env(
                tmp_num_opes, tmp_num_mas, tmp_num_vehs, device, 
                dynamic,
                proctime_per_ope_max, transtime_btw_ma_max, version
            )
            graph_list.append(env)
        return graph_list
    
    
    def _generate_env(self, 
            num_jobs, num_opes, num_mas, num_vehs, device, 
            opes_per_job_min, opes_per_job_max,
            dynamic=None,
            proctime_per_ope_max=20, transtime_btw_ma_max=10, version=None,
            job_centric=False,
            new_job_flag=False,
        ):
                               
                                                                                       
                                                         
                                    
           
        case = CaseGenerator(
            num_jobs, num_opes, num_mas, num_vehs, device,
            opes_per_job_min, opes_per_job_max,
            proctime_per_ope_max, transtime_btw_ma_max,
            dynamic, job_centric
        )
        new_job_dict = None
        if new_job_flag:
            new_job_dict = {
                'new_job_idx': torch.full(size=(self.env_paras['batch_size'], num_jobs), fill_value=False),
                'release': torch.full(size=(self.env_paras['batch_size'], num_jobs), fill_value=0)
            }
            n_newJobs = self.env_paras['num_newJobs']
            newJob_idxes = [random.randint(0, num_jobs-1) for _ in range(n_newJobs)]
            new_job_dict['new_job_idx'][:, newJob_idxes] = True
            print(f"now_new_job_dict['new_job_idx']:{new_job_dict['new_job_idx']}")
        env = TFJSPEnv(case=case, env_paras=self.env_paras, new_job_dict=new_job_dict)

        graph_dataset = []
        edge_index_bias_list = []
        train_dataset = None
        train_loader = None
        for batch in range(env.batch_size):
            data = set_GraphData(                                                              
                env.num_opes, env.num_mas, env.num_vehs, env.nums_ope_batch[batch],
                env.ope_ma_adj_batch[batch], env.proc_times_batch[batch], env.trans_times_batch[batch],
                node_feat_dim=8, edge_feat_dim=1
            )
            graph_dataset.append(data)
        train_dataset = GraphDataset(graph_dataset, degree=True, 
            k_hop=1, se='khopgnn',
            use_subgraph_edge_attr=True
        )
        train_loader = DataLoader(train_dataset, batch_size=env.batch_size,
            shuffle=False,
            generator=torch.Generator(device=self.device)
        )
        
        return env, train_dataset, train_loader

    def _generate_gridenv(self, 
            num_jobs, num_opes, num_mas, num_vehs, device, 
            opes_per_job_min, opes_per_job_max,
            dynamic=None,
            proctime_per_ope_max=20, transtime_btw_ma_max=10, version=None,
            job_centric=False,
            new_job_flag=False,
        ):
                               
                                                                                       
                                                         
                                    
           
        case = CaseGenerator(
            num_jobs, num_opes, num_mas, num_vehs, device,
            opes_per_job_min, opes_per_job_max,
            proctime_per_ope_max, transtime_btw_ma_max,
            dynamic, job_centric
        )
        new_job_dict = None
        if new_job_flag:
            new_job_dict = {
                'new_job_idx': torch.full(size=(self.env_paras['batch_size'], num_jobs), fill_value=False),
                'release': torch.full(size=(self.env_paras['batch_size'], num_jobs), fill_value=0)
            }
            n_newJobs = self.env_paras['num_newJobs']
            newJob_idxes = [random.randint(0, num_jobs-1) for _ in range(n_newJobs)]
            new_job_dict['new_job_idx'][:, newJob_idxes] = True
            print(f"now_new_job_dict['new_job_idx']:{new_job_dict['new_job_idx']}")
        
        batch_size = self.env_paras['batch_size']
        params[BATCH_SIZE] = batch_size
        grid_env, new_params = generate_batch_gridworld(6, 6, 11, 0.0, params)
            
        controller = algorithms.make(new_params)
        controller.load_model_weights("dynamicAMR/mapf/example_models/mappo")
        env = GridTFJSPEnv(case=case, env_paras=self.env_paras, grid_envs=grid_env, new_job_dict=new_job_dict)
              
                                                   
                                      
                                                                      
                                                                
                                  
                                                                
                                                                
                                                    
                                      
                                               
                                                                              
                                          
                                                                     
                           
                                                
                                                                                                                

        graph_dataset = []
        edge_index_bias_list = []
        train_dataset = None
        train_loader = None
        for batch in range(env.batch_size):
            data = set_GraphData(                                                              
                env.num_opes, env.num_mas, env.num_vehs, env.nums_ope_batch[batch],
                env.ope_ma_adj_batch[batch], env.proc_times_batch[batch], env.trans_times_batch[batch],
                node_feat_dim=8, edge_feat_dim=1
            )
            graph_dataset.append(data)
        train_dataset = GraphDataset(graph_dataset, degree=True, 
            k_hop=1, se='khopgnn',
            use_subgraph_edge_attr=True
        )
        train_loader = DataLoader(train_dataset, batch_size=env.batch_size,
            shuffle=False,
            generator=torch.Generator(device=self.device)
        )
        
        return env, train_dataset, train_loader, controller

def generate_vali_gridenv(
    nm,nv,size, test_env_paras, logger, device, 
    opes_per_job_min, opes_per_job_max,
    proctime_per_ope_max, transtime_btw_ma_max, 
    job_centric=True,
    new_job_flag=True,
    test_len=5):
    num_jobs = test_env_paras['num_jobs']
    num_opes = test_env_paras['num_opes']
    num_mas = test_env_paras['num_mas']
    num_vehs = test_env_paras['num_vehs']
    dynamic = test_env_paras['dynamic']
    batch_size = test_env_paras["batch_size"]
    case_list = []
    env_list = []
    
    test_dataset_list = []
    test_loader_list = []
    for _ in range(test_len):
        case = CaseGenerator(
            num_jobs, num_opes, num_mas, num_vehs, device,
            opes_per_job_min, opes_per_job_max,
            proctime_per_ope_max, transtime_btw_ma_max,
            dynamic, job_centric
        )
        new_job_dict = None
        new_job_flag = False
        if new_job_flag:
            new_job_dict = {
                'new_job_idx': torch.full(size=(test_env_paras['batch_size'], num_jobs), fill_value=False),
                'release': torch.full(size=(test_env_paras['batch_size'], num_jobs), fill_value=0)
            }
                                                                  
            n_newJobs = num_jobs - 10
                           
            newJob_idxes = random.sample(range(num_jobs), n_newJobs)
            new_job_dict['new_job_idx'][:, newJob_idxes] = True
            print(f"new_job_dict['new_job_idx']:{new_job_dict['new_job_idx']}")
        
        batch_size = test_env_paras['batch_size']
        params[BATCH_SIZE] = batch_size
        grid_env, new_params = generate_batch_gridworld(nm, nv, int(size), 0.0, params)      
            
        controller = algorithms.make(new_params)
        controller.load_model_weights("dynamicAMR/mapf/example_models/cac_marl")
        env = GridTFJSPEnv(case=case, env_paras=test_env_paras, grid_envs=grid_env, new_job_dict=new_job_dict)
        case_list.append(case)
        env_list.append(env)
        logger.info('vali_env info: num_jobs:{}, num_opes:{} num_mas:{}, num_veh:{}, batch_size:{}, map_size{}'\
            .format(env.num_jobs, env.num_opes, env.num_mas, env.num_vehs, test_env_paras['batch_size'],size))
        
                                             
                            
                                         
                                   
                                                                                     
                                                                                                         
                                                  
               
                                        

                                                                  
                                    
                                         
           
                                                                       
                            
                                                      
           
                                                
                                              


                                                                         
    return env_list,controller
                              
                                                
def compute_gae(rewards, values, gamma=1, lam=0.95):
    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    gae = 0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    return advantages
