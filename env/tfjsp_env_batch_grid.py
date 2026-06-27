import sys
import gym
import torch
from copy import deepcopy
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from dataclasses import dataclass
from env.load_data import load_tfjs
from utils.utils_fjspt import read_json, write_json

from env.tfjsp_env import *
import time
from collections import defaultdict


class GridTFJSPEnv(TFJSPEnv):
    def __init__(self, case, env_paras, grid_envs, data_source='case', new_job_dict=None):
        """
        :param data_source: 'case' or 'benchmark'
        :param new_job_dict:
            {
                'new_job_idx': [B, n_jobs], bool, True if the new job index
                'release': [B, n_jobs], scalar value if the new job index, 0 otherwise
            }
        """
                                 
        self.show_mode = env_paras["show_mode"]
        self.num_jobs = case.num_jobs
        self.num_opes_list = case.num_opes_list
        self.num_mas = case.num_mas
        self.num_vehs = case.num_vehs
        self.proctime_per_ope_max = case.proctime_per_ope_max
        self.transtime_btw_ma_max = case.transtime_btw_ma_max

        if env_paras["meta_rl"] is not None:
            self.batch_size = env_paras["meta_rl"]['minibatch']
        else:
            self.batch_size = env_paras["batch_size"]

        self.paras = env_paras
        self.device = env_paras["device"]
        self.new_job_dict = new_job_dict
        self.grid_envs = grid_envs

                                                      
        self._build_case_data(case)

                                        
        self.dyn_ope_ma_adj_batch = deepcopy(self.ope_ma_adj_batch)
        self.dyn_ope_veh_adj_batch = deepcopy(self.ope_veh_adj_batch)
        self.ope_status = torch.full(
            size=(self.batch_size, self.num_opes),
            dtype=torch.bool,
            fill_value=False,
            device=self.device
        )

                              
        obs = self.grid_envs.reset()
        self.obs_batch = obs
        self.trans_times_batch = self.grid_envs.init_trans_time.to(self.device)

                                    
        self._build_initial_features()

                                  
        self._init_new_jobs()

                                          
        self.sched_ope_dim = 6
        self.sched_mas_dim = 5
        self.sched_veh_dim = 4

                                  
        self._init_runtime_buffers()

                                                 
        self._recompute_min_estimate()

                               
        self.state = EnvState(
            batch_idxes=self.batch_idxes,
            feat_opes_batch=self.feat_opes_batch,
            feat_mas_batch=self.feat_mas_batch,
            feat_vehs_batch=self.feat_vehs_batch,
            proc_times_batch=self.proc_times_batch,
            trans_times_batch=self.trans_times_batch,
            ope_ma_adj_batch=self.ope_ma_adj_batch,
            ope_veh_adj_batch=self.ope_veh_adj_batch,
            ope_pre_adj_batch=self.ope_pre_adj_batch,
            ope_sub_adj_batch=self.ope_sub_adj_batch,
            ma_veh_adj_batch=self.ma_veh_adj_batch,
            prev_ope_locs_batch=self.prev_ope_locs_batch,
            allo_ma_batch=self.allo_ma_batch,
            veh_loc_batch=self.veh_loc_batch,
            mask_job_procing_batch=self.mask_job_procing_batch,
            mask_job_finish_batch=self.mask_job_finish_batch,
            mask_ma_procing_batch=self.mask_ma_procing_batch,
            mask_veh_procing_batch=self.mask_veh_procing_batch,
            opes_appertain_batch=self.opes_appertain_batch,
            ope_step_batch=self.ope_step_batch,
            end_ope_biases_batch=self.end_ope_biases_batch,
            time_batch=self.time,
            nums_ope_batch=self.nums_ope_batch,
            nums_opes_batch=self.nums_opes,
            ope_status=self.ope_status,
            ope_adj_batch=self.ope_adj_batch,
            dyn_ope_ma_adj_batch=self.dyn_ope_ma_adj_batch,
            dyn_ope_veh_adj_batch=self.dyn_ope_veh_adj_batch
        )

        self.loop = 0
        self.k = 0

                                                 
        self._save_initial_state()
                                                  
        self.ll_infer_time_total = 0.0                                 
        self.ll_infer_count = 0                                  
        self.ll_infer_time_list = []                                

        self.ll_active_env_total = 0                                            
        self.ll_avg_time_per_active_env_list = []                              
                                                               
                            
                                                               
    def _build_case_data(self, case):
        case_results = []
        self.num_opes = 0

        for _ in range(self.batch_size):
            case_results.append(case.get_case_for_transport())
            self.num_opes = max(self.num_opes, case.num_opes)

        num_data = len(case_results[0])
        tensors = [[] for _ in range(num_data)]

        for i in range(self.batch_size):
            load_data = load_tfjs(case_results[i], self.num_mas, self.num_opes)
            for j in range(num_data):
                tensors[j].append(load_data[j])

        self.proc_times_batch = torch.stack(tensors[0], dim=0).to(self.device)                           
        self.ope_ma_adj_batch = torch.stack(tensors[1], dim=0).long().to(self.device)                    
        self.ope_veh_adj_batch = torch.ones(
            size=(self.batch_size, self.num_opes, self.num_vehs),
            dtype=torch.long,
            device=self.device
        )                                                                                                 
        self.ope_pre_adj_batch = torch.stack(tensors[2], dim=0).to(self.device)                          
        self.ope_sub_adj_batch = torch.stack(tensors[3], dim=0).to(self.device)                          
        self.cal_cumul_adj_batch = torch.stack(tensors[4], dim=0).float().to(self.device)                
        self.nums_ope_batch = torch.stack(tensors[5], dim=0).to(self.device)                          
        self.num_ope_biases_batch = torch.stack(tensors[6], dim=0).long().to(self.device)             
        self.end_ope_biases_batch = torch.stack(tensors[7], dim=0).long().to(self.device)             
        self.opes_appertain_batch = torch.stack(tensors[8], dim=0).long().to(self.device)             
        self.trans_times_batch = torch.stack(tensors[9], dim=0).to(self.device)                          
        self.ma_veh_adj_batch = torch.stack(tensors[10], dim=0).long().to(self.device)                   

        self.nums_opes = torch.sum(self.nums_ope_batch, dim=1)       

    def _build_initial_features(self):
        feat_opes_batch = torch.zeros(
            size=(self.batch_size, self.paras["ope_feat_dim"], self.num_opes),
            device=self.device
        )
        feat_mas_batch = torch.zeros(
            size=(self.batch_size, self.paras["ma_feat_dim"], self.num_mas),
            device=self.device
        )
        feat_vehs_batch = torch.zeros(
            size=(self.batch_size, self.paras["veh_feat_dim"], self.num_vehs),
            device=self.device
        )

                                        
        feat_opes_batch[:, N_NEIGH_MA, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=2)
        feat_opes_batch[:, PROC_TIME, :] = torch.sum(self.proc_times_batch, dim=2).div(
            feat_opes_batch[:, N_NEIGH_MA, :] + 1e-9
        )
        feat_opes_batch[:, N_UNSCHED_OPE, :] = convert_feat_job_2_ope(
            self.nums_ope_batch, self.opes_appertain_batch
        )
        feat_opes_batch[:, ESTI_START, :] = torch.bmm(
            feat_opes_batch[:, PROC_TIME, :].unsqueeze(1),
            self.cal_cumul_adj_batch
        ).squeeze(1)
        end_time_batch = (
            feat_opes_batch[:, ESTI_START, :] + feat_opes_batch[:, PROC_TIME, :]
        ).gather(1, self.end_ope_biases_batch)
        feat_opes_batch[:, JOB_COMP_TIME, :] = convert_feat_job_2_ope(
            end_time_batch, self.opes_appertain_batch
        )
        feat_opes_batch[:, ALLO_MA, :] = 0.0
        feat_opes_batch[:, N_NEIGH_VEH, :] = torch.count_nonzero(self.ope_veh_adj_batch, dim=2)

                                       
        non_zero_proc_times_batch = torch.where(
            self.proc_times_batch == 0,
            torch.tensor(1000.0, device=self.device),
            self.proc_times_batch.float()
        )
        self.min_proc_times_batch = torch.min(non_zero_proc_times_batch, dim=2)[0]
        self.min_esti_start = torch.bmm(
            self.min_proc_times_batch.unsqueeze(1),
            self.cal_cumul_adj_batch
        ).squeeze(1)
        self.min_esti_end = self.min_esti_start + self.min_proc_times_batch

                                      
        feat_mas_batch[:, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=1)

                                      
        feat_vehs_batch[:, 0, :] = torch.count_nonzero(self.ope_veh_adj_batch, dim=1)

        self.feat_opes_batch = feat_opes_batch
        self.feat_mas_batch = feat_mas_batch
        self.feat_vehs_batch = feat_vehs_batch

    def _init_new_jobs(self):
        if self.new_job_dict:
            n_newJobs = self.new_job_dict['new_job_idx'].count_nonzero(dim=1)[0].item()
            tmp_end_time_batch = deepcopy(
                (self.feat_opes_batch[:, ESTI_START, :] + self.feat_opes_batch[:, PROC_TIME, :]).gather(
                    1, self.end_ope_biases_batch
                )
            )
            tmp_end_time_batch[self.new_job_dict['new_job_idx']] = 0
            max_job_comp_times = tmp_end_time_batch.max(dim=1)[0].tolist()
            rand_release_time = self._sample_poisson_release_times(n_newJobs, max_job_comp_times)
            self.new_job_dict['release'][self.new_job_dict['new_job_idx']] = rand_release_time.reshape(-1)
            self.new_job_dict['release'] = self.new_job_dict['release'].to(self.device)
            self.new_job_dict['new_job_idx'] = self.new_job_dict['new_job_idx'].to(self.device)

    def _init_runtime_buffers(self):
        self.batch_idxes = torch.arange(self.batch_size, device=self.device)
        self.time = torch.zeros(size=(self.batch_size,), device=self.device)
        self.N = torch.zeros(size=(self.batch_size,), dtype=torch.int, device=self.device)

        self.ope_step_batch = deepcopy(self.num_ope_biases_batch)
        self.veh_loc_batch = torch.zeros(
            size=(self.batch_size, self.num_vehs), dtype=torch.long, device=self.device
        )
        self.prev_ope_locs_batch = torch.zeros(
            size=(self.batch_size, self.num_jobs), dtype=torch.long, device=self.device
        )
        self.allo_ma_batch = torch.zeros(
            size=(self.batch_size, self.num_opes), dtype=torch.long, device=self.device
        )

        aver_trans_time = self.trans_times_batch.flatten(1).float().mean()
        self.ope_trans_time_batch = torch.ones(
            size=(self.batch_size, self.num_opes), device=self.device
        ) * aver_trans_time

        self.ope_adj_batch = deepcopy(self.ope_pre_adj_batch)

        self.mask_job_procing_batch = torch.full(
            size=(self.batch_size, self.num_jobs),
            dtype=torch.bool,
            fill_value=False,
            device=self.device
        )
        self.mask_job_finish_batch = torch.full(
            size=(self.batch_size, self.num_jobs),
            dtype=torch.bool,
            fill_value=False,
            device=self.device
        )
        self.mask_ma_procing_batch = torch.full(
            size=(self.batch_size, self.num_mas),
            dtype=torch.bool,
            fill_value=False,
            device=self.device
        )
        self.mask_veh_procing_batch = torch.full(
            size=(self.batch_size, self.num_vehs),
            dtype=torch.bool,
            fill_value=False,
            device=self.device
        )

        if self.new_job_dict:
            self.mask_job_procing_batch[self.new_job_dict['new_job_idx']] = True

        self._init_sched_buffers()

        self.makespan_batch = torch.max(self.feat_opes_batch[:, JOB_COMP_TIME, :], dim=1)[0]
        self.done_batch = self.mask_job_finish_batch.all(dim=1)
        self.done = self.done_batch.all()

        self.esti_trans_time = torch.zeros_like(self.time)

        self.task_buffer = torch.empty((0, 9), dtype=torch.float32, device=self.device)
        self.finish_list = torch.empty((0, 9), dtype=torch.float32, device=self.device)
        self.end_list = torch.empty((0, 9), dtype=torch.float32, device=self.device)

    def _init_sched_buffers(self):
        self.sched_opes_batch = torch.zeros(
            size=(self.batch_size, self.num_opes, self.sched_ope_dim),
            device=self.device
        )
        self.sched_opes_batch[:, :, 2] = self.feat_opes_batch[:, ESTI_START, :]
        self.sched_opes_batch[:, :, 3] = (
            self.feat_opes_batch[:, ESTI_START, :] + self.feat_opes_batch[:, PROC_TIME, :]
        )

        self.sched_mas_batch = torch.zeros(
            size=(self.batch_size, self.num_mas, self.sched_mas_dim),
            device=self.device
        )
        self.sched_mas_batch[:, :, 0] = 1.0
        self.sched_mas_batch[:, :, 4] = -1.0

        self.sched_vehs_batch = torch.zeros(
            size=(self.batch_size, self.num_vehs, self.sched_veh_dim),
            device=self.device
        )
        self.sched_vehs_batch[:, :, 0] = 1.0

    def _recompute_min_estimate(self):
        non_zero_proc_times_batch = torch.where(
            self.proc_times_batch == 0,
            torch.tensor(1000.0, device=self.device),
            self.proc_times_batch.float()
        )
        self.min_proc_times_batch = torch.min(non_zero_proc_times_batch, dim=2)[0]
        self.min_esti_start = torch.bmm(
            self.min_proc_times_batch.unsqueeze(1),
            self.cal_cumul_adj_batch
        ).squeeze(1)
        self.min_esti_end = self.min_esti_start + self.min_proc_times_batch
        min_esti_end_of_job = self.min_esti_end.gather(1, self.end_ope_biases_batch)
        self.prev_esti_makespan = torch.max(min_esti_end_of_job, dim=1)[0]

    def _save_initial_state(self):
        self.old_proc_times_batch = deepcopy(self.proc_times_batch)
        self.old_trans_times_batch = deepcopy(self.trans_times_batch)
        self.old_ope_ma_adj_batch = deepcopy(self.ope_ma_adj_batch)
        self.old_ope_veh_adj_batch = deepcopy(self.ope_veh_adj_batch)
        self.old_cal_cumul_adj_batch = deepcopy(self.cal_cumul_adj_batch)
        self.old_feat_opes_batch = deepcopy(self.feat_opes_batch)
        self.old_feat_mas_batch = deepcopy(self.feat_mas_batch)
        self.old_feat_vehs_batch = deepcopy(self.feat_vehs_batch)
        self.old_state = deepcopy(self.state)
        self.old_ope_status = deepcopy(self.ope_status)
        self.old_ope_adj_batch = deepcopy(self.ope_adj_batch)
        self.old_dyn_ope_ma_adj_batch = deepcopy(self.dyn_ope_ma_adj_batch)
        self.old_dyn_ope_veh_adj_batch = deepcopy(self.dyn_ope_veh_adj_batch)
        self.old_obs_batch = deepcopy(self.obs_batch)

                                                               
           
                                                               
    def reset(self):
        """
        Reset the environment to its initial state
        """
        self.proc_times_batch = deepcopy(self.old_proc_times_batch)
        self.trans_times_batch = deepcopy(self.old_trans_times_batch)
        self.ope_ma_adj_batch = deepcopy(self.old_ope_ma_adj_batch)
        self.ope_veh_adj_batch = deepcopy(self.old_ope_veh_adj_batch)
        self.cal_cumul_adj_batch = deepcopy(self.old_cal_cumul_adj_batch)
        self.feat_opes_batch = deepcopy(self.old_feat_opes_batch)
        self.feat_mas_batch = deepcopy(self.old_feat_mas_batch)
        self.feat_vehs_batch = deepcopy(self.old_feat_vehs_batch)
        self.state = deepcopy(self.old_state)
        self.ope_status = deepcopy(self.old_ope_status)
        self.ope_adj_batch = deepcopy(self.old_ope_adj_batch)
        self.dyn_ope_ma_adj_batch = deepcopy(self.old_dyn_ope_ma_adj_batch)
        self.dyn_ope_veh_adj_batch = deepcopy(self.old_dyn_ope_veh_adj_batch)
        self.obs_batch = self.grid_envs.reset()

        self._init_runtime_buffers()
        self._recompute_min_estimate()
        self._refresh_state()
                                                        
        self.ll_infer_time_total = 0.0
        self.ll_infer_count = 0
        self.ll_infer_time_list = []

        self.ll_active_env_total = 0
        self.ll_avg_time_per_active_env_list = []
        return self.state

                                                               
               
                                                               
    def step(self, action, controller, step_reward=None):
        """
        Input:
            action: [ope, ma, job, veh]: [4, B]
        """
        ope = action[0, :]
        ma = action[1, :]
        job = action[2, :]
        veh = action[3, :]

        proc_times, trans_times = self._apply_dispatch_action(ope, ma, job, veh)
        self._compute_reward(ope, proc_times, step_reward)

        flag_trans_2_next_time = self.if_no_eligible()

        if (flag_trans_2_next_time > 0).all(dim=0):
            self._refresh_state()
            return self.state, self.reward_batch, self.done_batch

        while ((flag_trans_2_next_time == 0) & (~self.done_batch)).any(dim=0):
            self._dispatch_tasks_to_grid()
            self._handle_transport_start_events()
            self._handle_process_end_events()

            self._clear_task_buffer()

            flag_trans_2_next_time = self.if_no_eligible()
            need_step = (flag_trans_2_next_time == 0) & (~self.done_batch)

            self._advance_lower_level(need_step, controller)

            if self.new_job_dict is not None:
                self._check_newJobInsert(self.grid_envs.time_step.float())

        self._refresh_state()
        return self.state, self.reward_batch, self.done_batch

                                                               
                    
                                                               
    def _apply_dispatch_action(self, ope, ma, job, veh):
        self.N += 1
        self.ope_status[self.batch_idxes, ope] = True
        self.ope_adj_batch[self.batch_idxes, ope, :] = False

        self._fix_selected_arcs(ope, ma, veh)

        self.ope_step_batch[self.batch_idxes, job] += 1
        self.dyn_ope_ma_adj_batch, self.dyn_ope_veh_adj_batch = self.get_dyn_adj_mat()

        proc_times = self.proc_times_batch[self.batch_idxes, ope, ma]
        trans_times = self._compute_transport_time(ma, job, veh)

        self.esti_trans_time = deepcopy(trans_times)
        self.ope_trans_time_batch[self.batch_idxes, ope] = trans_times

        self._append_task_buffer(ope, ma, job, veh, proc_times)
        self._update_features_after_dispatch(ope, ma, job, veh, proc_times, trans_times)
        self._update_schedule_after_dispatch(ope, ma, job, veh, proc_times, trans_times)
        self._update_masks_after_dispatch(job, ma, veh)

        self.done_batch = self.mask_job_finish_batch.all(dim=1)
        self.done = self.done_batch.all()

        return proc_times, trans_times

    def _fix_selected_arcs(self, ope, ma, veh):
        remain_ope_ma_adj = torch.zeros(
            size=(self.batch_size, self.num_mas),
            dtype=torch.long,
            device=self.device
        )
        remain_ope_ma_adj[self.batch_idxes, ma] = 1
        self.ope_ma_adj_batch[self.batch_idxes, ope] = remain_ope_ma_adj[self.batch_idxes, :]
        self.proc_times_batch *= self.ope_ma_adj_batch

        remain_ope_veh_adj = torch.zeros(
            size=(self.batch_size, self.num_vehs),
            dtype=torch.long,
            device=self.device
        )
        remain_ope_veh_adj[self.batch_idxes, veh] = 1
        self.ope_veh_adj_batch[self.batch_idxes, ope] = remain_ope_veh_adj[self.batch_idxes, :]

        remain_ma_veh_adj = torch.zeros(
            size=(self.batch_size, self.num_vehs),
            dtype=torch.long,
            device=self.device
        )
        remain_ma_veh_adj[self.batch_idxes, veh] = 1
        self.ma_veh_adj_batch[self.batch_idxes, ma] = remain_ma_veh_adj[self.batch_idxes, :]

    def _compute_transport_time(self, ma, job, veh):
        prev_ope_loc = self.prev_ope_locs_batch[self.batch_idxes, job].long()
        empty_trans_time = self.trans_times_batch[
            self.batch_idxes,
            self.veh_loc_batch[self.batch_idxes, veh],
            prev_ope_loc
        ]
        travel_trans_time = self.trans_times_batch[self.batch_idxes, prev_ope_loc, ma]
        trans_times = empty_trans_time + travel_trans_time
        return trans_times

    def _append_task_buffer(self, ope, ma, job, veh, proc_times):
        loc_veh_tensor = self.veh_loc_batch[self.batch_idxes, veh]
        start_time_tensor = self.time[self.batch_idxes]
        pro_time_tensor = proc_times
        op_tensor = ope
        job_id_tensor = job
        pre_ma_tensor = self.prev_ope_locs_batch[self.batch_idxes, job]
        target_ma_tensor = ma
        veh_id_tensor = veh
        batch_idx_tensor = self.batch_idxes.float()

        task_tensor = torch.stack([
            batch_idx_tensor,
            loc_veh_tensor.float(),
            start_time_tensor.float(),
            op_tensor.float(),
            job_id_tensor.float(),
            pro_time_tensor.float(),
            pre_ma_tensor.float(),
            target_ma_tensor.float(),
            veh_id_tensor.float()
        ], dim=1)

        self.task_buffer = torch.cat([self.task_buffer, task_tensor], dim=0)

    def _update_features_after_dispatch(self, ope, ma, job, veh, proc_times, trans_times):
                                         
        self.feat_opes_batch[self.batch_idxes, :3, ope] = torch.stack(
            (
                torch.ones(self.batch_idxes.size(0), dtype=torch.float, device=self.device),
                torch.ones(self.batch_idxes.size(0), dtype=torch.float, device=self.device),
                proc_times.float()
            ),
            dim=1
        )

                                    
        self.feat_opes_batch[self.batch_idxes, N_NEIGH_MA, :] = torch.count_nonzero(
            self.dyn_ope_ma_adj_batch[self.batch_idxes, :, :], dim=2
        ).float()
        self.feat_opes_batch[self.batch_idxes, N_NEIGH_VEH, :] = torch.count_nonzero(
            self.dyn_ope_veh_adj_batch[self.batch_idxes, :, :], dim=2
        ).float()

                                 
        last_ope = torch.where(
            ope - 1 < self.num_ope_biases_batch[self.batch_idxes, job],
            torch.tensor(self.num_opes - 1, device=self.device),
            ope - 1
        )
        self.cal_cumul_adj_batch[self.batch_idxes, last_ope, :] = 0

                                     
        start_ope = self.num_ope_biases_batch[self.batch_idxes, job]
        end_ope = self.end_ope_biases_batch[self.batch_idxes, job]
        for b, s, e, m in zip(
            self.batch_idxes.tolist(),
            start_ope.tolist(),
            end_ope.tolist(),
            ma.tolist()
        ):
            self.feat_opes_batch[b, N_UNSCHED_OPE, s:e + 1] -= 1
            self.feat_opes_batch[b, ALLO_MA, s:e + 1] = float(m)

                                          
        self.feat_opes_batch[self.batch_idxes, ESTI_START, ope] = self.time[self.batch_idxes]

        is_scheduled = self.feat_opes_batch[self.batch_idxes, STATUS, :]
        mean_proc_time = self.feat_opes_batch[self.batch_idxes, PROC_TIME, :]
        start_times = self.feat_opes_batch[self.batch_idxes, ESTI_START, :] * is_scheduled

        mean_trans_time = self.ope_trans_time_batch[self.batch_idxes, :]
        un_scheduled = 1 - is_scheduled

        estimate_times = torch.bmm(
            (start_times + mean_trans_time + mean_proc_time).unsqueeze(1),
            self.cal_cumul_adj_batch[self.batch_idxes, :, :]
        ).squeeze(1) * un_scheduled

        self.feat_opes_batch[self.batch_idxes, ESTI_START, :] = start_times + estimate_times

        end_times_batch = (
            self.feat_opes_batch[self.batch_idxes, ESTI_START, :]
            + mean_trans_time
            + self.feat_opes_batch[self.batch_idxes, PROC_TIME, :]
        ).gather(1, self.end_ope_biases_batch[self.batch_idxes, :])

        self.feat_opes_batch[self.batch_idxes, JOB_COMP_TIME, :] = convert_feat_job_2_ope(
            end_times_batch,
            self.opes_appertain_batch[self.batch_idxes, :]
        ).squeeze()

                             
        self.feat_mas_batch[self.batch_idxes, 0, :] = torch.count_nonzero(
            self.ope_ma_adj_batch[self.batch_idxes, :, :], dim=1
        ).float()

        utilize = self.sched_mas_batch[self.batch_idxes, :, 2]
        cur_time = self.time[self.batch_idxes, None].expand_as(utilize)
        utilize = torch.minimum(utilize, cur_time)
        utilize = utilize.div(self.time[self.batch_idxes, None] + 1e-9)
        self.feat_mas_batch[self.batch_idxes, 2, :] = utilize

                             
        self.feat_vehs_batch[self.batch_idxes, 0, :] = torch.count_nonzero(
            self.ope_veh_adj_batch[self.batch_idxes, :, :], dim=1
        ).float()
        self.feat_vehs_batch[self.batch_idxes, 3, veh] = ma.float()

        self.feat_mas_batch[self.batch_idxes, 1, ma] = self.time[self.batch_idxes] + trans_times + proc_times
        self.feat_vehs_batch[self.batch_idxes, 1, veh] = self.time[self.batch_idxes] + trans_times

        utilize_v = self.sched_vehs_batch[self.batch_idxes, :, 2]
        cur_time_v = self.time[self.batch_idxes, None].expand_as(utilize_v)
        utilize_v = torch.minimum(utilize_v, cur_time_v)
        utilize_v = utilize_v.div(self.time[self.batch_idxes, None] + 1e-9)
        self.feat_vehs_batch[self.batch_idxes, 2, :] = utilize_v
        self.feat_vehs_batch[self.batch_idxes, 4, veh] = trans_times

        self.allo_ma_batch[self.batch_idxes, ope] = ma

    def _update_schedule_after_dispatch(self, ope, ma, job, veh, proc_times, trans_times):
        mean_proc_time = self.feat_opes_batch[self.batch_idxes, PROC_TIME, :]
        mean_trans_time = self.ope_trans_time_batch[self.batch_idxes, :]

        self.sched_opes_batch[self.batch_idxes, ope, :2] = torch.stack(
            (torch.ones(self.batch_idxes.size(0), device=self.device), ma.float()),
            dim=1
        )
        self.sched_opes_batch[self.batch_idxes, ope, 4] = veh.float()

        self.sched_opes_batch[self.batch_idxes, :, 2] = self.feat_opes_batch[self.batch_idxes, ESTI_START, :]
        self.sched_opes_batch[self.batch_idxes, :, 3] = (
            self.feat_opes_batch[self.batch_idxes, ESTI_START, :]
            + mean_trans_time
            + mean_proc_time
        )
        self.sched_opes_batch[self.batch_idxes, ope, 5] = (
            self.feat_opes_batch[self.batch_idxes, ESTI_START, ope] + trans_times
        )

        self.sched_mas_batch[self.batch_idxes, ma, 0] = 0
        self.sched_mas_batch[self.batch_idxes, ma, 1] = self.time[self.batch_idxes] + trans_times + proc_times
        self.sched_mas_batch[self.batch_idxes, ma, 2] += proc_times
        self.sched_mas_batch[self.batch_idxes, ma, 3] = job.float()
        self.sched_mas_batch[self.batch_idxes, ma, 4] = veh.float()

        self.sched_vehs_batch[self.batch_idxes, veh, 0] = 0
        self.sched_vehs_batch[self.batch_idxes, veh, 1] = self.time[self.batch_idxes] + trans_times
        self.sched_vehs_batch[self.batch_idxes, veh, 2] += trans_times
        self.sched_vehs_batch[self.batch_idxes, veh, 3] = ma.float()

    def _update_masks_after_dispatch(self, job, ma, veh):
        self.mask_job_finish_batch = torch.where(
            self.ope_step_batch == self.end_ope_biases_batch + 1,
            True,
            self.mask_job_finish_batch
        )
        self.mask_veh_procing_batch[self.batch_idxes, veh] = True
        self.mask_job_procing_batch[self.batch_idxes, job] = True
        self.mask_ma_procing_batch[self.batch_idxes, ma] = True

                                                               
            
                                                               
    def _compute_reward(self, ope, proc_times, step_reward):
        if step_reward is not None:
            if step_reward == 'version1':
                max_value = torch.max(
                    self.feat_opes_batch[self.batch_idxes, JOB_COMP_TIME, :], dim=1
                )[0]
                self.reward_batch = self.makespan_batch - max_value
                self.makespan_batch = max_value
            elif step_reward == 'version2':
                self.reward_batch = self.get_step_reward(ope, proc_times)
            else:
                raise Exception('step_reward error!')
        else:
            if self.done:
                self.reward_batch = -self.get_makespan()
            else:
                self.reward_batch = torch.zeros(
                    size=(self.batch_size,),
                    dtype=torch.float,
                    device=self.device
                )

                                                               
                        
                                                               
    def _dispatch_tasks_to_grid(self):
        if self.task_buffer.numel() > 0:
            finish_batch = self.grid_envs.accept_task(self.task_buffer)
            if finish_batch is not None and finish_batch.numel() > 0:
                self.finish_list = torch.cat([self.finish_list, finish_batch.to(self.device)], dim=0)

    def _handle_transport_start_events(self):
        if len(self.finish_list) == 0:
            return

        batch_idx_tensor = self.finish_list[:, 0].long()
        end_time = self.finish_list[:, 1]
        pro_start = self.finish_list[:, 2]
        trans_time = self.finish_list[:, 3]
        pro_time = self.finish_list[:, 4]
        op = self.finish_list[:, 5].long()
        job_id = self.finish_list[:, 6].long()
        target_ma = self.finish_list[:, 7].long()
        veh_id = self.finish_list[:, 8].long()
        B = batch_idx_tensor

        time_step_batch = self.grid_envs.time_step[B]
        mask_pro_start = (pro_start == time_step_batch)

        if not mask_pro_start.any():
            return

        idx = mask_pro_start.nonzero(as_tuple=True)[0]
        b = B[idx]
        o = op[idx]
        v = veh_id[idx]
        m = target_ma[idx]
        j = job_id[idx]
        t = trans_time[idx]
        p = pro_time[idx]

        is_scheduled = self.feat_opes_batch[b, STATUS, :]
        mean_proc_time = self.feat_opes_batch[b, PROC_TIME, :]
        start_times = self.feat_opes_batch[b, ESTI_START, :] * is_scheduled

        self.ope_trans_time_batch[b, o] = t
        mean_trans_time = self.ope_trans_time_batch[b]
        un_scheduled = 1 - is_scheduled

        estimate_times = torch.bmm(
            (start_times + mean_trans_time + mean_proc_time).unsqueeze(1),
            self.cal_cumul_adj_batch[b]
        ).squeeze(1) * un_scheduled

        self.feat_opes_batch[b, ESTI_START, :] = start_times + estimate_times

        end_times_batch = (
            self.feat_opes_batch[b, ESTI_START, :]
            + mean_trans_time
            + self.feat_opes_batch[b, PROC_TIME, :]
        ).gather(1, self.end_ope_biases_batch[b])

        job_comp_ep = convert_feat_job_2_ope(end_times_batch, self.opes_appertain_batch[b])
        self.feat_opes_batch[b, JOB_COMP_TIME, :] = job_comp_ep.squeeze()

        self.sched_opes_batch[b, :, 2] = self.feat_opes_batch[b, ESTI_START, :]
        self.sched_opes_batch[b, :, 3] = (
            self.feat_opes_batch[b, ESTI_START, :] + mean_trans_time + mean_proc_time
        )
        self.sched_opes_batch[b, o, 5] = self.feat_opes_batch[b, ESTI_START, o] + t

        self.sched_mas_batch[b, m, 1] = self.time[b] + t + p
        self.feat_mas_batch[b, 1, m] = self.sched_mas_batch[b, m, 1]

        self.feat_vehs_batch[b, 1, v] = self.time[b] + t
        self.sched_vehs_batch[b, v, 1] = self.time[b] + t
        self.sched_vehs_batch[b, v, 2] += t - self.esti_trans_time[b]

        cur_time = self.time[b].reshape(-1, 1).expand_as(self.sched_vehs_batch[b, :, 2])
        utilize = torch.minimum(self.sched_vehs_batch[b, :, 2], cur_time)
        utilize = utilize.div(self.time[b, None] + 1e-9)
        self.feat_vehs_batch[b, 2, :] = utilize
        self.feat_vehs_batch[b, 4, v] = t

        self.prev_ope_locs_batch[b, j] = m
        self.veh_loc_batch[b, v] = m
        self.mask_veh_procing_batch[b, v] = False

        self.end_list = torch.cat([self.end_list, self.finish_list[idx]], dim=0)

        N = self.finish_list.shape[0]
        all_idx = torch.arange(N, device=self.finish_list.device)
        keep_mask = ~torch.isin(all_idx, idx)
        self.finish_list = self.finish_list[keep_mask]

    def _handle_process_end_events(self):
        if len(self.end_list) == 0:
            return

        batch_idx_tensor = self.end_list[:, 0].long()
        end_time = self.end_list[:, 1]
        op = self.end_list[:, 5].long()
        job_id = self.end_list[:, 6].long()
        target_ma = self.end_list[:, 7].long()
        B = batch_idx_tensor

        time_step_batch = self.grid_envs.time_step[B]
        mask_end_time = (end_time == time_step_batch)

        if not mask_end_time.any():
            return

        idx = mask_end_time.nonzero(as_tuple=True)[0]
        b = B[idx]
        m = target_ma[idx]
        j = job_id[idx]
        batches = B[mask_end_time]

        self.time[batches] = self.grid_envs.time_step[batches].float()

        if self.new_job_dict is not None:
            self._check_newJobInsert(self.grid_envs.time_step.float())

        self.sched_mas_batch[b, m, 0] = 1
        self.sched_mas_batch[b, m, 4] = -1

        utilize = self.sched_mas_batch[b, :, 2]
        cur_time = self.time[b].reshape(-1, 1).expand_as(self.sched_mas_batch[b, :, 2])
        utilize = torch.minimum(utilize, cur_time)
        utilize = utilize.div(self.time[b, None] + 1e-9)
        self.feat_mas_batch[b, 2, :] = utilize

        self.mask_job_procing_batch[b, j] = False
        self.mask_ma_procing_batch[b, m] = False
        self.mask_job_finish_batch[b] = torch.where(
            self.ope_step_batch[b] == self.end_ope_biases_batch[b] + 1,
            True,
            self.mask_job_finish_batch[b]
        )

        N = self.end_list.shape[0]
        all_idx = torch.arange(N, device=self.end_list.device)
        keep_mask = ~torch.isin(all_idx, idx)
        self.end_list = self.end_list[keep_mask]

        self.done_batch = self.mask_job_finish_batch.all(dim=1)
        self.done = self.done_batch.all()

    def _clear_task_buffer(self):
        self.task_buffer = self.task_buffer[:0]

    def _advance_lower_level(self, need_step, controller):
        joint_action = self._build_full_joint_action(need_step, controller)
        self.obs_batch, self.trans_times_batch, task_finish = self.grid_envs.step(joint_action, need_step)
        self.trans_times_batch = self.trans_times_batch.to(self.device)

        if task_finish is not None and task_finish.numel() > 0:
            self.finish_list = torch.cat([self.finish_list, task_finish.to(self.device)], dim=0)

                                                               
                   
                                                               
    def _refresh_state(self):
        self.state.update(
            self.batch_idxes,
            self.feat_opes_batch, self.feat_mas_batch, self.feat_vehs_batch,
            self.proc_times_batch, self.trans_times_batch,
            self.ope_ma_adj_batch, self.ma_veh_adj_batch, self.ope_veh_adj_batch,
            self.prev_ope_locs_batch, self.veh_loc_batch, self.allo_ma_batch,
            self.mask_job_procing_batch, self.mask_job_finish_batch,
            self.mask_ma_procing_batch, self.mask_veh_procing_batch,
            self.ope_step_batch, self.time,
            self.ope_status, self.ope_adj_batch,
            self.dyn_ope_ma_adj_batch, self.dyn_ope_veh_adj_batch
        )

                                                               
                                      
                                                               
  

    def _build_full_joint_action(self, need_step, controller):
        """
        need_step: BoolTensor [B]，表示哪些环境这次需要推进
        返回: LongTensor [B, A]，未推进的行置零
        """
        if isinstance(need_step, np.ndarray):
            need_step = torch.from_numpy(need_step).to(self.device)

        idx = torch.nonzero(need_step, as_tuple=False).squeeze(1)
        A = getattr(controller, "nr_agents", self.num_vehs)
        ja_full = torch.zeros((self.batch_size, A), dtype=torch.long, device=self.device)

        if idx.numel() == 0:
            return ja_full

        sub_obs = [self.obs_batch[b] for b in idx.tolist()]
        active_env_count = int(idx.numel())

        with torch.no_grad():
                                             
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            if hasattr(controller, "batch_policy"):
                ja_sub = controller.batch_policy(sub_obs)
                if not torch.is_tensor(ja_sub):
                    ja_sub = torch.stack(ja_sub, dim=0)
                ja_sub = ja_sub.to(dtype=torch.long, device=self.device)
            else:
                ja_list = []
                for o in sub_obs:
                    act = controller.joint_policy(o)
                    if not torch.is_tensor(act):
                        act = torch.as_tensor(act, dtype=torch.long, device=self.device)
                    else:
                        act = act.to(dtype=torch.long, device=self.device)
                    if act.dim() != 1:
                        act = act.view(-1)
                    ja_list.append(act)
                ja_sub = torch.stack(ja_list, dim=0)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()

        infer_dt = t1 - t0

                          
        self.ll_infer_time_total += infer_dt
        self.ll_infer_count += 1
        self.ll_infer_time_list.append(infer_dt)

        self.ll_active_env_total += active_env_count
        self.ll_avg_time_per_active_env_list.append(infer_dt / max(active_env_count, 1))

        ja_full[idx] = ja_sub
        return ja_full
    
    def get_lower_level_infer_stats(self):
        avg_ll_infer_time = (
        self.ll_infer_time_total / self.ll_infer_count
        if self.ll_infer_count > 0 else 0.0
    )

        avg_ll_infer_time_per_active_env = (
            float(np.mean(self.ll_avg_time_per_active_env_list))
            if len(self.ll_avg_time_per_active_env_list) > 0 else 0.0
        )

        return {
            "ll_infer_time_total_s": float(self.ll_infer_time_total),
            "ll_infer_count": int(self.ll_infer_count),
            "avg_ll_infer_time_s": float(avg_ll_infer_time),
            "avg_ll_infer_time_ms": float(avg_ll_infer_time * 1000.0),
            "avg_ll_infer_time_per_active_env_s": float(avg_ll_infer_time_per_active_env),
            "avg_ll_infer_time_per_active_env_ms": float(avg_ll_infer_time_per_active_env * 1000.0),
            "ll_active_env_total": int(self.ll_active_env_total),
            "ll_infer_time_list_s": list(self.ll_infer_time_list),
            "ll_avg_time_per_active_env_list_s": list(self.ll_avg_time_per_active_env_list),
        }
