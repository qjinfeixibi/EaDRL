import torch
import numpy as np

                       

def load_tfjs(case_results, num_mas, num_opes):
    '''
    Input:
        case_results:
            matrix_proc_time: tensor: [num_opes, num_mas]
            matrix_ope_ma_adj: tensor: [num_opes, num_mas]
            matrix_pre_ope_adj: tensor: [num_opes, num_opes]
            matrix_suc_ope_adj: tensor: [num_opes, num_opes]
            matrix_cal_cumul: tensor: [num_opes, num_opes]
            nums_ope: tensor: [num_jobs,]: each element is the number of operation on a job
            num_ope_biases: tensor: [num_jobs,]
            end_ope_biases: tensor: [num_jobs,]
            opes_appertain: tensor: [num_opes,]
            matrix_trans_time: tensor: [num_mas, num_mas]
            matrix_ma_veh_adj: tensor: [num_mas, num_mas]
            
    '''
    num_opes_short = case_results[0].size(0)                 

    matrix_proc_time = torch.zeros(size=(num_opes, num_mas))         
    matrix_proc_time[:num_opes_short, :] = case_results[0]
    
    matrix_ope_ma_adj = torch.zeros(size=(num_opes, num_mas))   
    matrix_ope_ma_adj[:num_opes_short, :] = case_results[1]
    
    matrix_pre_ope_adj = torch.zeros(size=(num_opes, num_opes), dtype=torch.bool)
    matrix_pre_ope_adj[:num_opes_short, :num_opes_short] = case_results[2]

    matrix_suc_ope_adj = torch.zeros(size=(num_opes, num_opes), dtype=torch.bool)
    matrix_suc_ope_adj[:num_opes_short, :num_opes_short] = case_results[3]
    
    matrix_cal_cumul = torch.zeros(size=(num_opes, num_opes)).float()
    matrix_cal_cumul[:num_opes_short, :num_opes_short] = case_results[4]
    
    opes_appertain = torch.zeros(size=(num_opes, ), dtype=torch.long)
    opes_appertain[:num_opes_short] = case_results[8]
    
    return matrix_proc_time, matrix_ope_ma_adj, matrix_pre_ope_adj, matrix_suc_ope_adj, matrix_cal_cumul,\
        case_results[5], case_results[6], case_results[7], opes_appertain, case_results[9], case_results[10]
    
    

def load_fjs(lines, num_mas, num_opes):
    '''
    Load the local FJSP instance.
    '''
    flag = 0
    matrix_proc_time = torch.zeros(size=(num_opes, num_mas))
    matrix_pre_proc = torch.full(size=(num_opes, num_opes), dtype=torch.bool, fill_value=False)
    matrix_cal_cumul = torch.zeros(size=(num_opes, num_opes)).int()
    nums_ope = []                                                   
    opes_appertain = np.array([])
    num_ope_biases = []                                             
                             
    for line in lines:
                    
        if flag == 0:
            flag += 1
                   
        elif line is "\n":
            break
               
        else:
            num_ope_bias = int(sum(nums_ope))                                             
            num_ope_biases.append(num_ope_bias)
                                                                                
            num_ope = edge_detec(line, num_ope_bias, matrix_proc_time, matrix_pre_proc, matrix_cal_cumul)
            nums_ope.append(num_ope)
                                   
                                           
                                                       
                                                                     
            opes_appertain = np.concatenate((opes_appertain, np.ones(num_ope)*(flag-1)))
            flag += 1
    matrix_ope_ma_adj = torch.where(matrix_proc_time > 0, 1, 0)
                                                                             
    opes_appertain = np.concatenate((opes_appertain, np.zeros(num_opes-opes_appertain.size)))
    return matrix_proc_time, matrix_ope_ma_adj, matrix_pre_proc, matrix_pre_proc.t(),\
           torch.tensor(opes_appertain).int(), torch.tensor(num_ope_biases).int(),\
           torch.tensor(nums_ope).int(), matrix_cal_cumul

def nums_detec(lines):
    '''
    Count the number of jobs, machines and operations
    '''
    num_opes = 0
                             
    for i in range(1, len(lines)):
        num_opes += int(lines[i].strip().split()[0]) if lines[i]!="\n" else 0
    line_split = lines[0].strip().split()
    num_jobs = int(line_split[0])
    num_mas = int(line_split[1])
    return num_jobs, num_mas, num_opes

def edge_detec(line, num_ope_bias, matrix_proc_time, matrix_pre_proc, matrix_cal_cumul):
    '''
    Detect information of a job
    '''
    line_split = line.split()
                                       
    flag = 0
    flag_time = 0
    flag_new_ope = 1
    idx_ope = -1
    num_ope = 0                                              
    num_option = np.array([])                                                                           
    mac = 0
    for i in line_split:
        x = int(i)
                                                                         
        if flag == 0:
            num_ope = x
            flag += 1
                                
        elif flag == flag_new_ope:
            idx_ope += 1
            flag_new_ope += x * 2 + 1
            num_option = np.append(num_option, x)
            if idx_ope != num_ope-1:
                matrix_pre_proc[idx_ope+num_ope_bias][idx_ope+num_ope_bias+1] = True
            if idx_ope != 0:
                vector = torch.zeros(matrix_cal_cumul.size(0))
                vector[idx_ope+num_ope_bias-1] = 1
                matrix_cal_cumul[:, idx_ope+num_ope_bias] = matrix_cal_cumul[:, idx_ope+num_ope_bias-1]+vector
            flag += 1
                                 
        elif flag_time == 0:
            mac = x-1
            flag += 1
            flag_time = 1
                   
        else:
            matrix_proc_time[idx_ope+num_ope_bias][mac] = x
            flag += 1
            flag_time = 0
    
    return num_ope