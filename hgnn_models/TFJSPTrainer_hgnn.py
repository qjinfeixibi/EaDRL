
import torch
import numpy as np
import os
from copy import deepcopy
from collections import deque
import random
import time
import pandas as pd
import math

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from DHJS_models.TFJSPTrainer_dhjs import TFJSPTrainer_DHJS
from hgnn_models.TFJSPModel_hgnn import TFJSPModel_hgnn
from utils.memory import ReplayBuffer


class TFJSPTrainer_hgnn(TFJSPTrainer_DHJS):
    def __init__(self,
                 env_paras,
                model_paras,
                train_paras,
                optimizer_paras,
                test_paras,
                change_paras,
                 ):
        super().__init__(
            env_paras,
            model_paras,
            train_paras,
            optimizer_paras,
            test_paras,
            change_paras,
        )
        
        self.lr = train_paras["lr"]                 
        self.betas = train_paras["betas"]                          
        self.gamma = train_paras["gamma"]                   
        self.eps_clip = train_paras["eps_clip"]                      
        self.K_epochs = train_paras["K_epochs"]                              
        self.A_coeff = train_paras["A_coeff"]                               
        self.vf_coeff = train_paras["vf_coeff"]                              
        self.entropy_coeff = train_paras["entropy_coeff"]                                
        self.num_envs = env_paras["batch_size"]                                
        self.device = model_paras["device"]                  
        
        model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
        model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]
        model_paras["action_dim"] = 1
        
        self.model = TFJSPModel_hgnn(env_paras, model_paras).to(self.device)
        self.model_old = deepcopy(self.model)
        self.model_old.load_state_dict(self.model.state_dict())
                                                                                                  
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_paras['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_paras['scheduler'])
        
                               
        self.memory = ReplayBuffer(self.device, max_size=int(1e5))
        
    def _train_one_batch(self, batch_size, env, train_dataset=None, train_loader=None):
                                 
        state = env.reset()
        
        prob_list = torch.zeros(size=(batch_size, 0))
        
                             
        done = False
        dones = env.done_batch
        while not done:
            self.memory.add_state_info(deepcopy(state))
            with torch.no_grad():
                action, prob = self.model_old.act(state, memory=self.memory)                   
            
            state, rewards, dones = env.step(action)
            
            done = dones.all()
            prob_list = torch.cat((prob_list, prob), dim=1)
            
            self.memory.add_reward_info(rewards.detach().cpu().numpy(), dones.detach().cpu().numpy(), batch_size)
        
                          
        loss = self.model.update(self.memory, self.optimizer, self.train_paras['minibatch_size'],
                          self.gamma, self.K_epochs, self.eps_clip, 
                          self.A_coeff, self.vf_coeff, self.entropy_coeff)
                                          
        self.model_old.load_state_dict(self.model.state_dict())
        self.memory.clear()
        
                       
        score = env.get_makespan().mean()
        
        return score.item(), loss
