
import random
import time
import torch
import numpy as np
from copy import deepcopy

class CaseGenerator:                             
    '''
    FJSP instance generator
    '''
    def __init__(self, num_jobs, num_opes, num_mas, num_vehs, device,
                opes_per_job_min, opes_per_job_max,
                proctime_per_ope_mas=20, transtime_btw_ma_max=10,
                dynamic=None, job_centric=True,
                data_source='case', case_config=None
    ):
        '''
        :param data_source: 'case' or 'benchmark'
        '''
        self.num_jobs = num_jobs
        self.num_opes = num_opes
        self.num_mas = num_mas
        self.num_vehs = num_vehs
        self.device = device
        self.data_source = data_source
        
        if data_source == 'case':
            if not job_centric:
                                                              
                if dynamic is None:
                    self.num_jobs = self.num_opes//self.num_mas
                    self.num_opes_list = self._get_num_opes_list(self.num_opes, self.num_jobs, None)
                                                                        
                else:
                    self.num_opes_list = self._get_num_opes_list(self.num_opes, None, dynamic)
                    self.num_jobs = len(self.num_opes_list)
            else:     
                self.num_opes_list = [random.randint(opes_per_job_min, opes_per_job_max) for _ in range(self.num_jobs)]                                           
                self.num_opes = sum(self.num_opes_list)
        elif data_source == 'benchmark':
            self.num_jobs = case_config['num_jobs']
            self.num_mas = case_config['num_mas']
            self.num_vehs = case_config['num_vehs']
            self.num_opes_list = case_config['num_opes_list']
            self.num_opes = case_config['num_opes']
            self.ope_ma_adj = case_config['ope_ma_adj']
            self.proc_time_mat = case_config['proc_time']
            self.trans_time_mat = case_config['trans_time']
        else:
            raise Exception('data_source error')
                 
        self.mas_per_ope_min = 1                                                                
        self.mas_per_ope_max = self.num_mas
                     
        self.opes_per_job_min = opes_per_job_min                                              
        self.opes_per_job_max = opes_per_job_max
                
        self.proctime_per_ope_min = proctime_per_ope_mas-15                                   
                                                                          
        self.proctime_per_ope_max = proctime_per_ope_mas
        self.proctime_dev = 0.2
                
        self.transtime_btw_ma_min = 1                                
        self.transtime_btw_ma_max = transtime_btw_ma_max
        self.transtime_dev = 0.2
        
    def _get_num_opes_list(self, num_opes, num_jobs, dynamic=None):     
        '''
        Output:
            num_opes_list: [number of operations per job, ...], len = num_jobs
        '''
        num_opes_list = []
        if dynamic is None:
            rest_opes = num_opes%num_jobs
            for i in range(num_jobs):
                if i < rest_opes:
                    num_opes_list.append(num_opes//num_jobs + 1)
                else:
                    num_opes_list.append(num_opes//num_jobs)
        else:
            min_ope_per_job = dynamic['min_ope_per_job']
                                                          
            max_ope_per_job = max(num_opes//3, min_ope_per_job+1)
            total_opes = 0
            while True:
                tmp_num = np.random.randint(min_ope_per_job, max_ope_per_job)
                if total_opes + tmp_num > num_opes:
                    tmp_num = num_opes - total_opes
                    if tmp_num > 0:
                        num_opes_list.append(tmp_num)
                    break
                else:
                    num_opes_list.append(tmp_num)
                    total_opes += tmp_num
        return num_opes_list
    
    def get_case_for_transport(self, idx=0):
        '''
        Generate FJSP instance
        :param idx: The instance number
        
        Output:
           0 matrix_proc_time: tensor: [num_opes, num_mas]
           1 matrix_ope_ma_adj: tensor: [num_opes, num_mas]
           2 matrix_pre_ope_adj: tensor: [num_opes, num_opes]
           3 matrix_suc_ope_adj: tensor: [num_opes, num_opes]
           4 matrix_cal_cumul: tensor: [num_opes, num_opes]
           5 nums_ope: tensor: [num_jobs,]:
           10 matrix_ma_veh_adj: tensor: [num_mas, num_vehs]
            
        '''
        
                                                 
                                                           
        assert self.num_jobs <= self.num_opes
        assert len(self.num_opes_list) == self.num_jobs
        assert self.num_opes == sum(self.num_opes_list)
        
                                                                    
        self.num_cpt_mas_list = [random.randint(self.mas_per_ope_min, self.mas_per_ope_max) for _ in range(self.num_opes)]                                                           
        self.num_cpt_mas = sum(self.num_cpt_mas_list)              
        
                                 
        self.num_ope_biases = [sum(self.num_opes_list[0:i]) for i in range(self.num_jobs)]
                         
        self.end_ope_biases = [val + self.num_opes_list[i] - 1 for i, val in enumerate(self.num_ope_biases)]
        self.num_ma_biases = [sum(self.num_cpt_mas_list[0:i]) for i in range(self.num_opes)]
        
        
                                                                                                     
        self.ope_ma = []
        for val in self.num_cpt_mas_list:
            sample_mas = sorted(random.sample(range(1, self.num_mas+1), val))                
            sample_mas_idx = [val - 1 for val in sample_mas]          
            self.ope_ma.append(sample_mas_idx)             
        
                                                             
        matrix_ope_ma_adj = torch.zeros(size=(self.num_opes, self.num_mas))
        for ope_idx in range(self.num_opes):
            matrix_ope_ma_adj[ope_idx, self.ope_ma[ope_idx]] = 1          
        
                                                                    
        self.proc_time = []
                
        self.proc_times_mean = [random.randint(self.proctime_per_ope_min, self.proctime_per_ope_max) for _ in range(self.num_opes)]                                            
        
        for i in range(len(self.num_cpt_mas_list)):                                            
            low_bound = max(self.proctime_per_ope_min,round(self.proc_times_mean[i]*(1-self.proctime_dev)))
            high_bound = min(self.proctime_per_ope_max,round(self.proc_times_mean[i]*(1+self.proctime_dev)))
            proc_time_ope = [random.randint(low_bound, high_bound) for _ in range(self.num_cpt_mas_list[i])]                                                                      
                                                              
            self.proc_time.append(proc_time_ope)                    
        
        
                                              
        matrix_proc_time = torch.zeros(size=(self.num_opes, self.num_mas))
        for ope_idx in range(self.num_opes):
            for n, ma_idx in enumerate(self.ope_ma[ope_idx]):
                matrix_proc_time[ope_idx, ma_idx] = self.proc_time[ope_idx][n]
        
                                                                        
        self.trans_time = []
        self.trans_time_mean = [random.randint(self.transtime_btw_ma_min, self.transtime_btw_ma_max) for _ in range(self.num_mas)]
        for i in range(self.num_mas):
            low_bound = max(self.transtime_btw_ma_min, round(self.trans_time_mean[i]*(1-self.transtime_dev)))
            high_bound = min(self.transtime_btw_ma_max, round(self.trans_time_mean[i]*(1+self.transtime_dev)))
            trans_time_ma = [random.randint(low_bound, high_bound) for _ in range(self.num_mas)]
            self.trans_time.append(trans_time_ma)
        
                                                            
        matrix_trans_time = torch.zeros(size=(self.num_mas, self.num_mas), dtype=torch.float)
        for from_ma in range(self.num_mas):
            for to_ma in range(self.num_mas):
                if from_ma == to_ma:
                    matrix_trans_time[from_ma, to_ma] = 0
                elif from_ma < to_ma:                     
                    matrix_trans_time[from_ma, to_ma] = self.trans_time[from_ma][to_ma]
                    matrix_trans_time[to_ma, from_ma] = self.trans_time[from_ma][to_ma]
                                                     
                                                         
        
                                                       
        matrix_ma_veh_adj = torch.ones(size=(self.num_mas, self.num_vehs))
        
        
                                                     
        matrix_pre_ope_adj_np = np.eye(self.num_opes, k=1, dtype=np.bool)
        matrix_pre_ope_adj_np[self.end_ope_biases, :] = False
        matrix_pre_ope_adj = torch.from_numpy(matrix_pre_ope_adj_np).to(self.device)          

                                                   
        matrix_suc_ope_adj_np = np.eye(self.num_opes, k=-1, dtype=np.bool)
        matrix_suc_ope_adj_np[self.num_ope_biases, :] = False
        matrix_suc_ope_adj = torch.from_numpy(matrix_suc_ope_adj_np).to(self.device)
        
                                                                                               
                       
        matrix_cal_cumul = torch.zeros(size=(self.num_opes, self.num_opes)).float()       
        job_idx = 0              
        cunt_ope = 0               
        for col in range(self.num_opes):           
            if col not in self.num_ope_biases:       
                vector = torch.zeros(size=(self.num_opes,)) 
                vector[self.num_ope_biases[job_idx] + cunt_ope - 1] = 1
                matrix_cal_cumul[:, col] = matrix_cal_cumul[:, col-1] + vector              
            
            if cunt_ope == self.num_opes_list[job_idx]-1:        
                job_idx += 1        
                cunt_ope = 0
            else:            
                cunt_ope += 1

                                                                        
        opes_appertain = torch.zeros(size=(self.num_opes,), dtype=torch.long)
        for job_idx in range(self.num_jobs):
            opes_appertain[self.num_ope_biases[job_idx] : self.num_ope_biases[job_idx] + self.num_opes_list[job_idx]] = job_idx
        
        
             
        num_opes_list = torch.tensor(self.num_opes_list, dtype=torch.long)
        num_ope_biases = torch.tensor(self.num_ope_biases, dtype=torch.long)
        end_ope_biases = torch.tensor(self.end_ope_biases, dtype=torch.long)
        
                                                
        if self.data_source == 'benchmark':
            matrix_ope_ma_adj = self.ope_ma_adj
            matrix_proc_time = self.proc_time_mat
            matrix_trans_time = self.trans_time_mat
        return (matrix_proc_time, matrix_ope_ma_adj, matrix_pre_ope_adj, matrix_suc_ope_adj, matrix_cal_cumul,\
            num_opes_list, num_ope_biases, end_ope_biases, opes_appertain, matrix_trans_time, matrix_ma_veh_adj)
