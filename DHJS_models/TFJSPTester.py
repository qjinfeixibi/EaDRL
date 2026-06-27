
import random
import torch
import time

from env.case_generator_v2 import CaseGenerator
from env.tfjsp_env import TFJSPEnv
from SAHJS_models.temporal_graph_dataset import set_GraphData
from sat_models.sat.data import GraphDataset
from torch_geometric.loader.dataloader import DataLoader
from dispatching_models.dispatchModel import dispatchModel

def validate(env, model, logger=None, test_dataset=None, test_loader=None):
    if logger is not None:
        logger.info('========== validating ==========')
                         
    if not isinstance(model, dispatchModel):
        model.eval()
    state = env.reset()
    model.init(state, dataset=test_dataset, loader=test_loader)
    
    done = False
    dones = env.done_batch
    start_time = time.time()
    while not done:
        with torch.no_grad():
            action, _ = model.act(state)                    
        state, rewards, dones = env.step(action)
        done = dones.all()
    spand_time = time.time() - start_time

    score = -rewards.mean() 
    batch_scores = -rewards       
    gantt_result = env.validate_gantt()[0]
    if not gantt_result:
        if logger is not None:
            logger.info("Scheduling Error!!!")
        else:
            print("Scheduling Error!!!")
                                                           
    if logger is not None:
        logger.info('score (the lower the better): {}'.format(score))
    return score.item(), spand_time, batch_scores

def validate_multi_models(
    env_list, models, model_names, logger, result_folder, result_log, test_len=100,
    test_dataset_list=None, test_loader_list=None,
):
    
    for i, model in enumerate(models):
        avg_score = 0
        avg_spand_time = 0
        
        for test_idx in range(test_len):
            if test_dataset_list is not None:
                score, spand_time, rewards = validate(
                    env_list[test_idx], model, 
                    test_dataset=test_dataset_list[test_idx],
                    test_loader=test_loader_list[test_idx]
                )
            else:
                score, spand_time, batch_scores = validate(env_list[test_idx], model)
            avg_score += score
            avg_spand_time += spand_time
        avg_score /= test_len
        avg_spand_time /= test_len
        
        logger.info('{} Score: {:0.2f} | SpandTime: {:0.2f} '.format(model_names[i], avg_score, avg_spand_time))
        result_log.append('{}_score'.format(model_names[i]), avg_score)
        result_log.append('{}_SpandTime'.format(model_names[i]), avg_spand_time)
        result_log.append('{}_batch_scores'.format(model_names[i]), batch_scores.cpu().tolist())
        
    
    print(f'result_log.get_raw_data():{result_log.get_raw_data()}')
    
                                            
    result_dict = {
        'result_log': result_log.get_raw_data()
    }
    torch.save(result_dict, '{}/test_results.pt'.format(result_folder))
        
def validate_one_model(
    env_list, model, model_name, logger, result_folder, result_log, test_len=100,
    test_dataset_list=None, test_loader_list=None,
):
    
    avg_score = 0
    avg_spand_time = 0
    all_batch_scores = []                    

                    
    for test_idx in range(test_len):
                         
        if test_dataset_list is not None:
            score, spand_time, rewards = validate(
                env_list[test_idx], model, 
                test_dataset=test_dataset_list[test_idx],
                test_loader=test_loader_list[test_idx]
            )
        else:
            score, spand_time, batch_scores = validate(env_list[test_idx], model)
            all_batch_scores.extend(batch_scores.cpu().tolist())          
        
                 
        avg_score += score
        avg_spand_time += spand_time

                   
    avg_score /= test_len
    avg_spand_time /= test_len

                     
    logger.info(f'{model_name} | Avg Score: {avg_score:.2f} | Avg Time: {avg_spand_time:.2f}s')
    result_log.append(f'{model_name}_score', avg_score)
    result_log.append(f'{model_name}_time', avg_spand_time)
    result_log.append(f'{model_name}_batch_scores', all_batch_scores)               

                    
    result_dict = {
        'result_log': result_log.get_raw_data()
    }
    torch.save(result_dict, f'{result_folder}/test_results.pt')   
    return result_log

def generate_vali_env(
    test_env_paras, logger, device, 
    opes_per_job_min, opes_per_job_max,
    proctime_per_ope_max, transtime_btw_ma_max, 
    job_centric=True,
    new_job_flag=False,
    test_len=100):
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
    for i in range(test_len):
        case = CaseGenerator(
            num_jobs, num_opes, num_mas, num_vehs, device,
            opes_per_job_min, opes_per_job_max,
            proctime_per_ope_max, transtime_btw_ma_max,
            dynamic, job_centric
        )
        new_job_dict = None
        if new_job_flag:
            new_job_dict = {
                'new_job_idx': torch.full(size=(test_env_paras['batch_size'], num_jobs), fill_value=False),
                'release': torch.full(size=(test_env_paras['batch_size'], num_jobs), fill_value=0)
            }
            n_newJobs = test_env_paras['num_newJobs']
            newJob_idxes = [random.randint(0, num_jobs-1) for _ in range(n_newJobs)]
            new_job_dict['new_job_idx'][:, newJob_idxes] = True
            print(f"new_job_dict['new_job_idx']:{new_job_dict['new_job_idx']}")
        env = TFJSPEnv(case=case, env_paras=test_env_paras, new_job_dict=new_job_dict)
        case_list.append(case)
        env_list.append(env)
        logger.info('vali_env info: num_jobs:{}, num_opes:{} num_mas:{}, num_veh:{}, batch_size:{}'\
            .format(env.num_jobs, env.num_opes, env.num_mas, env.num_vehs, test_env_paras['batch_size']))
        
                                             
                            
                                         
                                   
                                                                                     
                                                                                                         
                                                  
               
                                        

                                                                  
                                    
                                         
           
                                                                       
                            
                                                      
           
                                                
                                              


    
    return env_list
                                                


                                                 
         
            
                       
                        
                         
           
         
                                                                               
                                                                       
                                                           
    
                  
                            
import os, re, torch
from glob import glob

def _resolve_ckpt_path(path_dir_or_file: str) -> str:

    if os.path.isdir(path_dir_or_file):
              
        files = []
        for ext in ('*.pt', '*.pth', '*.ckpt'):
            files += glob(os.path.join(path_dir_or_file, ext))
        if not files:
            raise FileNotFoundError(f"目录 {path_dir_or_file} 下没有 *.pt/*.pth/*.ckpt 可加载")

        def priority(f):
            name = os.path.basename(f).lower()
            if 'best' in name: return (0, -os.path.getmtime(f))
            if 'final' in name or 'last' in name: return (1, -os.path.getmtime(f))
            return (2, -os.path.getmtime(f))
        files.sort(key=priority)
        chosen = files[0]
        print(f"[restore_model] 传入目录，自动选择权重：{chosen}")
        return chosen
    return path_dir_or_file

def restore_model(model, device, path=None, strict=False, **_):
                    
    path = _resolve_ckpt_path(path)

                      
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)                         

                                                          
                                                                                                      
                                           

           
    missing_unexp = model.load_state_dict(state, strict=strict)
    print("[restore_model] load_state_dict 完成:", missing_unexp)
    return model
