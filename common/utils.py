# Utils files to have the helper functions #

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from collections import deque
from tqdm import tqdm

# Find atanh #
def atanh(x, eps=1e-6):
    x = x.clamp(-1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

# Soft Update #
def SoftUpdate(target_network, source_network, tau=0.005):
    for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(
            (1 - tau) * target_param.data + tau * source_param.data
        )

# Load expert trajectories into replay buffer #
def load_demo_trajectories(replay_buffer, demo_file, demo_env, demo_ocp, n_step=1, gamma=0.999):
    demo_file = demo_file
    with open(demo_file, "rb") as file:
        TRAJ = pickle.load(file)

    count_land, count_crash = 0, 0 
    test_env = demo_env
    buff = deque(maxlen=n_step)

    # Loading the trajectories #
    for key in tqdm(TRAJ.keys(), desc="Loading Trajectories", unit="trj"):
        tr_ = TRAJ[key]
        result = tr_["solution_vector"]

        x_int, z_int, u_int, w_int, omg_int, lmbi_int, tht0_int, thtP_int = demo_ocp.getInterpolators(result)
        tf = result[-1]

        success = False
        for eps in (0.0, -0.025, 0.025, -0.05, 0.05):
            x_int, z_int, u_int, w_int, omg_int, lmbi_int, tht0_int, thtP_int = demo_ocp.getInterpolators(result)
            init = (x_int(-1), z_int(-1)+eps, u_int(-1), w_int(-1), omg_int(-1), tht0_int(-1), thtP_int(-1), lmbi_int(-1))
            action_int = (tht0_int, thtP_int)

            test_env.dt=0.02
            test_env.rk4_dt=0.01
            sim_traj = test_env._similate_trajectory(init, action_int, tf)

            if sim_traj[-1][3] and sim_traj[-1][-1]["constraints"][0][2]:
                success = True
                break

        if success:
            count_land += 1
            for i in range(len(sim_traj)):
                experience = sim_traj[i]
                state, action, reward, termination, truncation, next_state, info = experience
                replay_buffer.add(0, state, action, reward, termination, truncation, next_state)
                # Offline Buffer add
                experience = sim_traj[-i-1]
                state, action, reward, termination, truncation, next_state, info = experience
                if i == 0:
                    value_fn = 0
                    n_step_value_fn = 0
                else:
                    value_fn = buff[-1][0] + gamma*buff[-1][1]
                    n_step_value_fn = 0
                    for j in range(len(buff)):
                        n_step_value_fn += buff[-j-1][0]*gamma**(j)
                    n_step_value_fn += buff[0][1]*gamma**(len(buff))
                buff.append((reward, value_fn))
                replay_buffer.add(0, state, action, reward, termination, truncation, next_state)

            replay_buffer.clear_n_step_buffer(0)
        else:
            count_crash += 1
            # tqdm.write(f"Failed : {key}")

    print(f"Loaded all successful trajectories... No of Sucess : {count_land}; No of Crash : {count_crash}")
    return replay_buffer