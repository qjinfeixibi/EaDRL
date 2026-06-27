import logging
import json

import numpy as np
import torch

from options import get_options
from utils.utils import *
from logging import getLogger
from train import setup_seed

from DHJS_models.TFJSPTester import restore_model, validate_multi_models
from dispatching_models.dispatchModel import dispatchModel
from GTrans_models.TFJSPModel_gtrans import TFJSPModel_GTrans
from dynamicAMR.mapf.cactus.constants import *
from DHJS_models.TFJSPTrainer_dhjs import generate_vali_gridenv

process_start_time = datetime.now(pytz.timezone("Asia/Seoul"))

def _print_config():
    logger = logging.getLogger('root')
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

def main(opts, num_jobs=None, num_mas=None, num_vehs=None, size=None):

    with open("./configs/config.json", 'r') as load_f:
        load_dict = json.load(load_f)
    env_paras = load_dict["env_paras"]
    model_paras = load_dict["model_paras"]
    train_paras = load_dict["train_paras"]
    optimizer_paras = load_dict["optimizer_paras"]
    logger_paras = load_dict["logger_paras"]
    test_paras = load_dict["test_paras"]

    params = {}
    params[EPISODES_PER_EPOCH] = 1
    params[ENV_OBSERVATION_SIZE] = 7
    params[ENV_NR_MACHINE] = num_mas
    params[ENV_NR_AGENTS] = num_vehs
    params[HIDDEN_LAYER_DIM] = 64
    params[ENV_GAMMA] = 1.0
    params[RENDER_MODE] = False
    params[ENV_MAKESPAN_MODE] = False
    params[GRAD_NORM_CLIP] = 10
    params[VDN_MODE] = False
    params[REWARD_SHARING] = False
    params[MIXING_HIDDEN_SIZE] = 128
    params[ENV_TIME_LIMIT] = 15000
    params[ENV_STEPTIME] = 1
    params[ALGORITHM_NAME] = ALGORITHM_PPO_QMIX
    params["observation_dim"] = [5, 7, 7]
    params[ENV_SIZE] = size

    if num_jobs is not None and num_mas is not None and num_vehs is not None:
        test_paras['num_jobs'] = num_jobs
        test_paras['num_mas'] = num_mas
        test_paras['num_vehs'] = num_vehs
        opts.log_file_desc = f'test_{num_jobs}_{num_mas}_{num_vehs}_{size}'

    model_paras['sqrt_embedding_dim'] = model_paras['embedding_dim']**(1/2)
    model_paras['sqrt_qkv_dim'] = model_paras['qkv_dim']**(1/2)
    model_paras['ms_layer1_init'] = (1/2)**(1/2)
    model_paras['ms_layer2_init'] = (1/16)**(1/2)

    logger_paras['log_file']['desc'] = opts.log_file_desc
    logger_paras['log_file']['filepath'] = './result/' + 'time_cpl/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}'       

    setup_seed(seed=2)
    create_logger(**logger_paras)
    _print_config()

    device = torch.device("cuda:"+str(opts.cuda) if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    print("PyTorch device: ", device)
    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

    env_paras["device"] = device
    test_paras["device"] = device
    model_paras["device"] = device

    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]
    test_paras["batch_size"] = env_paras["batch_size"]
    model_paras["batch_size"] = env_paras["batch_size"]
    model_paras["proctime_per_ope_max"] = env_paras["proctime_per_ope_max"]
    model_paras["transtime_btw_ma_max"] = env_paras["transtime_btw_ma_max"]

    model_paras["checkpoint_encoder"] = opts.checkpoint_encoder
    model_paras["algorithm"] = opts.algorithm
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]
    model_paras['num_opes'] = test_paras["num_opes"]
    model_paras['num_mas'] = test_paras["num_mas"]
    model_paras['num_vehs'] = test_paras["num_vehs"]

    test_paras["dynamic"]['max_ope_per_job'] = test_paras['num_opes'] // test_paras['num_mas']
    if opts.static:
        test_paras['dynamic'] = None

    logger = getLogger(name='tester')
    result_folder = get_result_folder()
    result_log = LogData()

    operation_count_reference_machines = 15
    opes_per_job_min = int(operation_count_reference_machines * 0.8)
    opes_per_job_max = int(operation_count_reference_machines * 1.2)
    vali_env, controller = generate_vali_gridenv(
        num_mas, num_vehs, int(size), test_paras, logger, device,
        opes_per_job_min, opes_per_job_max,
        env_paras["proctime_per_ope_max"], env_paras["transtime_btw_ma_max"],
        job_centric=model_paras['job_centric'],
        new_job_flag=True,
        test_len=test_paras['num_test'],
    )
    model_paras['encoder_layer_num'] = 2

    gt_drl = TFJSPModel_GTrans(
        embedding_dim_=model_paras["embedding_dim"],
        hidden_dim_=model_paras["hidden_dim"],
        problem=None,
        ope_feat_dim=model_paras["in_size_ope"],
        ma_feat_dim=model_paras["in_size_ma"],
        veh_feat_dim=model_paras["in_size_veh"],
        mask_inner=True,
        mask_logits=True,
        encoder_version=2,
        decoder_version=5,
        meta_rl=train_paras['meta_rl'] if train_paras['meta_rl']['enable'] else None,
        **model_paras
    ).to(device)

    models = [
        gt_drl,
                    
    ]

    model_names = []
    model_loads = []
    for key, val in test_paras['models'].items():
        model_names.append(val['name'])
        model_loads.append(val)

    for i, model in enumerate(models):
        print(f'model_loads[{i}]:{model_loads[i]["name"]}')
        restore_model(model, device, **model_loads[i])

    if opts.test_dispatch:
        dispatch_fifo = dispatchModel(rule='fifo', **model_paras)
        models.append(dispatch_fifo)
        model_names.append('dispatch_fifo')

    validate_multi_models(
        vali_env, models, controller, model_names, logger,
        result_folder, result_log, test_len=test_paras['num_test'],
        test_dataset_list=None,
        test_loader_list=None
    )

    result_dict = {'result_log': result_log.get_raw_data()}
    torch.save(result_dict, f'{result_folder}/test_results.pt')


if __name__ == '__main__':
    args = get_options()

    if not hasattr(args, "multi_test") or (hasattr(args, "multi_test") and args.multi_test is False):
        args.multi_test = True

    num_jobs, num_mas, num_vehs, map_size = [], [], [], []

    num_jobs += [ 60]
    num_mas  += [40]
    num_vehs += [ 15]
    map_size += [ 20]

    for n_jobs, n_mas, n_vehs, size in zip(num_jobs, num_mas, num_vehs, map_size):
        main(args, n_jobs, n_mas, n_vehs, size)
