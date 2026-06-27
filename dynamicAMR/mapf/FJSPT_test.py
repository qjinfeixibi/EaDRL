import torch
import numpy as np
from cactus.env.factory_gridworld import FactoryGridWorld 
from cactus.algorithms import make as make_controller
from cactus.constants import *

def test_factory_task_by_accept(
    model_dir, nr_agents=6, nr_machines=6, map_size=11, density=0, steptime=1.0, max_time_steps=256
):
    params = {}
    params[ENV_NR_AGENTS] = nr_agents
    params[ENV_OBSERVATION_SIZE] = 7
    params[ENV_TIME_LIMIT] = max_time_steps
    params[ENV_GAMMA] = 1.0
    params[HIDDEN_LAYER_DIM] = 64
    params[MIXING_HIDDEN_SIZE] = 64
    params[TORCH_DEVICE] = torch.device("cpu")
    params[ALGORITHM_NAME] = ALGORITHM_PPO_QMIX
    params[CRITIC_NAME] = CRITIC_QMIX
    params[EPISODES_PER_EPOCH] = 1
    params[ENV_NR_MACHINE] = nr_machines
    params[ENV_STEPTIME] = steptime

    obstacle_map = np.zeros((map_size, map_size))
    params[ENV_OBSTACLES] = obstacle_map

    env = FactoryGridWorld(params)
    params[ENV_NR_ACTIONS] = env.nr_actions
    params[ENV_OBSERVATION_DIM] = [5, params[ENV_OBSERVATION_SIZE], params[ENV_OBSERVATION_SIZE]]

    controller = make_controller(params)
    controller.load_model_weights(model_dir)

    observations = env.reset()

    agent_target_counts = {a: 0 for a in range(nr_agents)}
    agent_target_times = {a: [] for a in range(nr_agents)}

    for agent_id in range(nr_agents):
        task = generate_single_task(env, agent_id)
        env.accept_task(task)

    done = False
    while not done:
        joint_action = controller.joint_policy(observations)
        result = env.step(joint_action)

        observations, _, time_step, task_finish = result
        processed_indices = [] 
        for idx,record in enumerate(task_finish):
            end_time, pro_start, trans_time, pro_time, op, job_id, target_ma, veh_id = record
            elapsed = end_time - pro_start
            agent_target_times[veh_id].append(elapsed)
            agent_target_counts[veh_id] += 1

            new_task = generate_single_task(env, veh_id)
            env.accept_task(new_task)
            processed_indices.append(idx)
        for idx in reversed(processed_indices):
            del env.task_finish[idx]

        if time_step >= max_time_steps:
            done = True

            
    for agent_id, times in agent_target_times.items():
        print(f"\nAgent {agent_id} 完成任务 {len(times)} 个，各任务耗时：")
        for idx, t in enumerate(times):
            print(f"  第 {idx+1} 个任务：{t} 步")

def generate_single_task(env, agent_id):

    nr_machines = env.nr_machine
    while True:
        pre_ma = np.random.randint(0, nr_machines)
        target_ma = np.random.randint(0, nr_machines)
        if pre_ma != target_ma:
            break
    loc_veh = 0 
    start_time = env.time_step
    op = 0  
    job_id = agent_id
    pro_time = 5  
    veh_id = agent_id  
    return [(loc_veh, start_time, op, job_id, pro_time, pre_ma, target_ma, veh_id)]

if __name__ == "__main__":
    model_path = "dynamicAMR/mapf/example_models/cactus"  
    test_factory_task_by_accept(
        model_path, nr_agents=6, nr_machines=6, map_size=11, density=0, steptime=1.0, max_time_steps=256
    )
