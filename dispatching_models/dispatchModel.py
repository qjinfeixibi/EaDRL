import torch
from copy import deepcopy
import math
import torch.nn.functional as F

INF = 1e9


class dispatchModel():
    def __init__(self, rule='spt', **model_paras):
        self.rule = rule
        self.model_paras = model_paras

    def init(self, state, dataset=None, loader=None):
        self.batch_size = state.ope_ma_adj_batch.size(0)
        self.num_opes = state.ope_ma_adj_batch.size(1)
        self.num_mas = state.ope_ma_adj_batch.size(2)
        self.num_jobs = state.mask_job_finish_batch.size(1)
        self.num_vehs = state.mask_veh_procing_batch.size(1)

    def act(self, state):



        if self.rule == 'spt':
            select_ope, select_ma, select_job = self._select_OMPair_on_ProcTime(state, proc_crit='short')
            veh_dict = self._select_nearest_veh(state, select_ope, select_ma, select_job)
            select_veh = veh_dict['veh_id'].long()

        elif self.rule == 'lpt':
            select_ope, select_ma, select_job = self._select_OMPair_on_ProcTime(state, proc_crit='long')
            veh_dict = self._select_nearest_veh(state, select_ope, select_ma, select_job)
            select_veh = veh_dict['veh_id'].long()

        elif self.rule == 'fifo':
            select_job, select_ope = self._select_fifo_job(state)
            select_ma = self._select_fifo_ma(state, select_job)
            veh_dict = self._select_nearest_veh(state, select_ope, select_ma, select_job)
            select_veh = veh_dict['veh_id'].long()

        elif self.rule == 'lum_spt':
            select_ma = self._select_ma_on_util(state, util_crit='low')
            select_ope, select_job = self._select_oper_given_ma(state, select_ma, proc_crit='short')
            veh_dict = self._select_nearest_veh(state, select_ope, select_ma, select_job)
            select_veh = veh_dict['veh_id'].long()

        elif self.rule == 'lum_lpt':
            select_ma = self._select_ma_on_util(state, util_crit='low')
            select_ope, select_job = self._select_oper_given_ma(state, select_ma, proc_crit='long')
            veh_dict = self._select_nearest_veh(state, select_ope, select_ma, select_job)
            select_veh = veh_dict['veh_id'].long()




        elif self.rule == 'est_eet_eet':

            select_job, select_ope = self._select_job_on_EST_with_transport(state)

            select_ma = self._select_ma_on_EET_with_transport(state, select_job, select_ope)

            veh_dict = self._select_veh_on_EET_given_job_ma(state, select_job, select_ma)
            select_veh = veh_dict['veh_id'].long()

        else:
            raise Exception('dispatch rule error!')


        action = torch.cat([select_ope, select_ma, select_job, select_veh], dim=1).transpose(1, 0)
        return action, 0




    def _select_job_on_EST_with_transport(self, state):
        """
        For each job j, consider its current operation ope_j.
        For each feasible machine m and idle vehicle v:
            base_m = max(time, ma_avail[m])
            start  = max(base_m, veh_avail[v])
            trans  = T(veh_loc[v] -> job_loc[j]) + T(job_loc[j] -> m)
            arrive = start + trans   (earliest processing start time on m using v)
        job_EST = min_{m,v} arrive
        Choose job with min job_EST.

        Returns:
            select_job: [B,1]
            select_ope: [B,1]
        """
        device = state.ope_ma_adj_batch.device
        B, _, n_mas = state.ope_ma_adj_batch.size()
        n_jobs = self.num_jobs
        n_vehs = self.num_vehs

        batch_idxes = state.batch_idxes
        active = torch.zeros(B, dtype=torch.bool, device=device)
        active[batch_idxes] = True


        time = state.time_batch.to(device)


        ma_avail = state.feat_mas_batch[:, 1, :].to(device)
        veh_avail = state.feat_vehs_batch[:, 1, :].to(device)


        ope_step = torch.where(
            state.ope_step_batch > state.end_ope_biases_batch,
            state.end_ope_biases_batch,
            state.ope_step_batch
        ).to(device)


        job_ok = ~(state.mask_job_procing_batch | state.mask_job_finish_batch).to(device)
        ma_ok = (~state.mask_ma_procing_batch).to(device)
        veh_ok = (~state.mask_veh_procing_batch).to(device)
        has_veh = veh_ok.any(dim=1, keepdim=True)


        ope_ma = state.ope_ma_adj_batch.gather(
            1, ope_step[..., None].expand(-1, -1, n_mas)
        ).bool().to(device)

        jm_ok = job_ok[..., None] & ma_ok[:, None, :] & (ope_ma == 1)
        job_has_ma = jm_ok.any(dim=2)


        prev_loc = state.prev_ope_locs_batch.to(device)
        veh_loc = state.veh_loc_batch.to(device)
        trans = state.trans_times_batch.to(device)


        veh_loc_exp = veh_loc[:, None, :].expand(-1, n_jobs, -1)
        prev_loc_exp = prev_loc[:, :, None].expand(-1, -1, n_vehs)
        empty_t = trans[torch.arange(B, device=device)[:, None, None], veh_loc_exp, prev_loc_exp]


        prev_loc_m = prev_loc[:, :, None].expand(-1, -1, n_mas)
        mas_idx = torch.arange(n_mas, device=device)[None, None, :].expand(B, n_jobs, n_mas)
        travel_t = trans[torch.arange(B, device=device)[:, None, None], prev_loc_m, mas_idx]


        base_m = torch.maximum(time[:, None, None], ma_avail[:, None, :]).expand(-1, n_jobs, -1)


        base_m4 = base_m[:, :, :, None].expand(-1, -1, -1, n_vehs)
        veh_av4 = veh_avail[:, None, None, :].expand(-1, n_jobs, n_mas, -1)
        start = torch.maximum(base_m4, veh_av4)


        empty4 = empty_t[:, :, None, :].expand(-1, -1, n_mas, -1)
        travel4 = travel_t[:, :, :, None].expand(-1, -1, -1, n_vehs)
        trans_bjmv = empty4 + travel4

        arrive = start + trans_bjmv


        jm_ok4 = jm_ok[:, :, :, None].expand(-1, -1, -1, n_vehs)
        veh_ok4 = veh_ok[:, None, None, :].expand(-1, n_jobs, n_mas, -1)
        active4 = active[:, None, None, None].expand_as(jm_ok4)
        feas = jm_ok4 & veh_ok4 & active4

        arrive_masked = arrive.clone()
        arrive_masked[~feas] = INF


        job_est = arrive_masked.min(dim=3).values.min(dim=2).values
        job_est[~job_has_ma] = INF
        job_est[~has_veh.expand_as(job_est)] = INF
        job_est[~active[:, None].expand_as(job_est)] = INF

        select_job = job_est.argmin(dim=1, keepdim=True)
        select_ope = ope_step.gather(1, select_job)


        select_job[~active] = 0
        select_ope[~active] = 0
        return select_job.long(), select_ope.long()




    def _select_ma_on_EET_with_transport(self, state, select_job, select_ope):
        """
        Choose machine m minimizing:
            EET(m) = min_v [ max(max(time, ma_avail[m]), veh_avail[v]) + trans(v,job,m) ] + proc(ope,m)
        """
        device = state.ope_ma_adj_batch.device
        B, _, n_mas = state.ope_ma_adj_batch.size()
        n_vehs = self.num_vehs

        batch_idxes = state.batch_idxes
        active = torch.zeros(B, dtype=torch.bool, device=device)
        active[batch_idxes] = True

        time = state.time_batch.to(device)
        ma_avail = state.feat_mas_batch[:, 1, :].to(device)
        veh_avail = state.feat_vehs_batch[:, 1, :].to(device)
        ma_ok = (~state.mask_ma_procing_batch).to(device)
        veh_ok = (~state.mask_veh_procing_batch).to(device)

        trans = state.trans_times_batch.to(device)
        veh_loc = state.veh_loc_batch.to(device)
        prev_loc = state.prev_ope_locs_batch.to(device)


        job_loc = prev_loc.gather(1, select_job).squeeze(1)


        b_idx = torch.arange(B, device=device)[:, None]
        empty_t = trans[b_idx, veh_loc, job_loc[:, None].expand(-1, n_vehs)]


        mas_idx = torch.arange(n_mas, device=device)[None, :].expand(B, n_mas)
        travel_t = trans[torch.arange(B, device=device)[:, None], job_loc[:, None].expand(-1, n_mas), mas_idx]


        trans_vm = empty_t[:, :, None] + travel_t[:, None, :]


        proc = state.proc_times_batch.gather(
            1, select_ope[:, :, None].expand(-1, -1, n_mas)
        ).squeeze(1).to(device)


        ope_ma = state.ope_ma_adj_batch.gather(
            1, select_ope[:, :, None].expand(-1, -1, n_mas)
        ).squeeze(1).bool().to(device)

        mask_m = active[:, None] & ma_ok & ope_ma & (proc > 0)

        base_m = torch.maximum(time[:, None], ma_avail)
        base_m_ = base_m[:, None, :].expand(-1, n_vehs, -1)
        veh_av_ = veh_avail[:, :, None].expand(-1, -1, n_mas)
        start = torch.maximum(base_m_, veh_av_)
        arrive = start + trans_vm


        veh_ok_ = veh_ok[:, :, None].expand(-1, -1, n_mas)
        arrive[~veh_ok_] = INF

        best_arrive = arrive.min(dim=1).values
        eet = best_arrive + proc

        eet[~mask_m] = INF
        select_ma = eet.argmin(dim=1, keepdim=True)
        select_ma[~active] = 0
        return select_ma.long()




    def _select_veh_on_EET_given_job_ma(self, state, select_job, select_ma):
        """
        Choose vehicle v minimizing:
            arrive(v) = max(max(time, ma_avail[m]), veh_avail[v]) + trans(v,job,m)
        """
        device = state.ope_ma_adj_batch.device
        B = state.ope_ma_adj_batch.size(0)
        n_vehs = self.num_vehs

        batch_idxes = state.batch_idxes
        active = torch.zeros(B, dtype=torch.bool, device=device)
        active[batch_idxes] = True

        time = state.time_batch.to(device)
        ma_avail = state.feat_mas_batch[:, 1, :].to(device)
        veh_avail = state.feat_vehs_batch[:, 1, :].to(device)
        veh_ok = (~state.mask_veh_procing_batch).to(device)

        trans = state.trans_times_batch.to(device)
        veh_loc = state.veh_loc_batch.to(device)
        prev_loc = state.prev_ope_locs_batch.to(device)

        job_loc = prev_loc.gather(1, select_job).squeeze(1)
        ma = select_ma.squeeze(1)


        b_idx = torch.arange(B, device=device)[:, None]
        empty_t = trans[b_idx, veh_loc, job_loc[:, None].expand(-1, n_vehs)]
        travel_t = trans[torch.arange(B, device=device), job_loc, ma]
        trans_v = empty_t + travel_t[:, None]

        base_m = torch.maximum(time, ma_avail.gather(1, select_ma).squeeze(1))
        start = torch.maximum(base_m[:, None], veh_avail)
        arrive = start + trans_v

        arrive[~veh_ok] = INF
        arrive[~active[:, None].expand_as(arrive)] = INF

        veh_id = arrive.argmin(dim=1, keepdim=True)
        trans_end = arrive.gather(1, veh_id)

        veh_id[~active] = 0
        trans_end[~active] = 0
        return {'veh_id': veh_id.long(), 'trans_end': trans_end}




    def _select_ma_on_util(self, state, util_crit='low'):
        mask, _ = self._get_mask_ope_ma(state)
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
        return select_ma

    def _select_oper_given_ma(self, state, select_ma, proc_crit='short'):
        batch_size, num_opes, num_mas = state.ope_ma_adj_batch.size()
        mask, mask_ope_ma = self._get_mask_ope_ma(state)

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
            raise Exception("proc_crit error")

        select_job = self.from_ope_to_job(select_ope.squeeze(1), state).unsqueeze(1).long()
        return select_ope, select_job

    def _select_fifo_ma(self, state, select_job):
        batch_size, num_opes, num_mas = state.ope_ma_adj_batch.size()
        batch_idxes = state.batch_idxes

        mask, _ = self._get_mask_ope_ma(state)
        avail_mask_mas = mask.gather(1, select_job[:, :, None].expand(-1, -1, num_mas)).squeeze(1)
        avail_mas = torch.where(avail_mask_mas == True, 0., -math.inf)
        avail_ma_probs = F.softmax(avail_mas, dim=1)

        while True:
            select_ma = avail_ma_probs.reshape(batch_size, -1).multinomial(1).squeeze(dim=1).reshape(batch_size, 1)
            ma_prob = avail_ma_probs.gather(1, select_ma)
            non_finish_batch = torch.full(size=(batch_size, 1), dtype=torch.bool, fill_value=False)
            non_finish_batch[batch_idxes] = True
            finish_batch = torch.where(non_finish_batch == True, False, True)
            ma_prob[finish_batch] = 1
            if (ma_prob != 0).all():
                break
        return select_ma

    def _select_fifo_job(self, state):
        batch_size = state.ope_ma_adj_batch.size(0)
        batch_idxes = state.batch_idxes

        mask, _ = self._get_mask_ope_ma(state)
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
        return select_job, select_ope

    def _select_nearest_veh(self, state, select_ope, select_ma, select_job):
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

    def _select_OMPair_on_ProcTime(self, state, proc_crit='short'):
        batch_size, num_opes, num_mas = state.ope_ma_adj_batch.size()
        mask, mask_ope_ma = self._get_mask_ope_ma(state)

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
        return ope, ma, job

    def _get_mask_ope_ma(self, state):
        batch_idxes = state.batch_idxes
        num_opes = state.ope_ma_adj_batch.size(1)
        num_mas = state.ope_ma_adj_batch.size(2)
        num_jobs = state.mask_job_procing_batch.size(1)

        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)
        opes_appertain_batch = state.opes_appertain_batch
        mask_ma = ~state.mask_ma_procing_batch[batch_idxes]

        eligible_proc = state.ope_ma_adj_batch[batch_idxes].gather(
            1, ope_step_batch[..., None].expand(-1, -1, state.ope_ma_adj_batch.size(-1))[batch_idxes]
        )
        dummy_shape = torch.zeros(size=(len(batch_idxes), self.num_jobs, self.num_mas))
        ma_eligible = ~state.mask_ma_procing_batch[batch_idxes].unsqueeze(1).expand_as(dummy_shape)
        job_eligible = ~(state.mask_job_procing_batch[batch_idxes] +
                         state.mask_job_finish_batch[batch_idxes])[:, :, None].expand_as(dummy_shape)
        eligible = job_eligible & ma_eligible & (eligible_proc == 1)

        if (~(eligible)).all():
            print("No eligible J-M pair!")
            return
        mask = eligible

        mask_ope_step = torch.full(size=(self.batch_size, num_opes), dtype=torch.bool, fill_value=False)
        tmp_batch_idxes = batch_idxes.unsqueeze(-1).repeat(1, num_jobs)
        mask_ope_step[tmp_batch_idxes, ope_step_batch] = True

        mask_job = torch.where(mask.sum(dim=-1) > torch.zeros(size=(self.batch_size, self.num_jobs)),
                               True, False)
        mask_ope_by_job = mask_job.gather(1, opes_appertain_batch)
        mask_ope = mask_ope_by_job & mask_ope_step

        mask_ope_padd = mask_ope[:, :, None].expand(-1, -1, num_mas)
        mask_ma_padd = mask_ma[:, None, :].expand(-1, num_opes, -1)
        ope_ma_adj = state.ope_ma_adj_batch[batch_idxes]
        mask_ope_ma = mask_ope_padd & mask_ma_padd & (ope_ma_adj == 1)

        return mask, mask_ope_ma

    def from_ope_to_job(self, select_ope, state):
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)
        select_job = torch.where(ope_step_batch == select_ope[:, None].expand(-1, self.num_jobs))[1]
        return select_job










        





    






    


























        


        














        










        













        


        










        


        




        








        




        
        










    



        

        








        




        









        




        

    












            





        

        

        
        











            








    












        


        



















        










        





        













        







        









        

        





        



        











