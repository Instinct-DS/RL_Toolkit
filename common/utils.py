# Utils files to have the helper functions #

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from collections import deque
from tqdm import tqdm
from scipy.interpolate import BarycentricInterpolator, CubicSpline

# Find atanh #
def atanh(x, eps=1e-3):
    x = x.clamp(-1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

# Soft Update #
def SoftUpdate(target_network, source_network, tau=0.005):
    for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(
            (1 - tau) * target_param.data + tau * source_param.data
        )

# Nodes Used in OCP #
def lglnodes(N):
    N1 = N + 1
    x = np.cos(np.pi * np.arange(N1) / N)
    P = np.zeros((N1, N1))
    xold = 2 * np.ones_like(x)
    while np.max(np.abs(x - xold)) > np.finfo(float).eps:
        xold = x.copy()
        P[:, 0] = 1
        P[:, 1] = x
        for k in range(2, N1):
            P[:, k] = ((2 * k - 1) * x * P[:, k - 1] - (k - 1) * P[:, k - 2]) / k
        x = xold - (x * P[:, N] - P[:, N - 1]) / (N1 * P[:, N])
    w = 2 / (N * N1 * P[:, N]**2)
    return np.flip(x), np.flip(w)    

# Get Interpolator #
def getInterpolators(result, nds):
    n = len(nds)
    Z = result
    states = np.reshape(Z[:6*n], shape=(n,6), order='F')
    ctrl = np.reshape(Z[6*n:8*n], shape=(n,2), order="F")
    tf = Z[-1]

    x_i = states[:, 0]
    z_i = states[:, 1]
    u_i = states[:, 2]
    w_i = states[:, 3]
    omg_i = states[:, 4]
    lmbi = states[:, 5]
    tht0 = ctrl[:, 0]
    thtP = ctrl[:, 1]

    x_int = CubicSpline(nds, x_i)
    z_int = CubicSpline(nds, z_i)
    u_int = CubicSpline(nds, u_i)
    w_int = CubicSpline(nds, w_i)
    omg_int = CubicSpline(nds, omg_i)
    lmbi_int = CubicSpline(nds, lmbi)
    tht0_int = CubicSpline(nds, tht0)
    thtP_int = CubicSpline(nds, thtP)

    return x_int, z_int, u_int, w_int, omg_int, lmbi_int, tht0_int, thtP_int


# Load expert trajectories into replay buffer #
def load_demo_trajectories(replay_buffer, demo_file, demo_env, nds, gamma=0.999):
    demo_file = demo_file
    with open(demo_file, "rb") as file:
        TRAJ = pickle.load(file)

    count_land, count_crash = 0, 0 
    test_env = demo_env

    # Loading the trajectories #
    for key in tqdm(TRAJ.keys(), desc="Loading Trajectories", unit="trj"):
        tr_ = TRAJ[key]
        result = tr_["solution_vector"]

        x_int, z_int, u_int, w_int, omg_int, lmbi_int, tht0_int, thtP_int = getInterpolators(result, nds)
        tf = result[-1]

        success = False
        for eps in (0.0, -0.025, 0.025, -0.05, 0.05):
            x_int, z_int, u_int, w_int, omg_int, lmbi_int, tht0_int, thtP_int = getInterpolators(result, nds)
            init = (x_int(-1), z_int(-1)+eps, u_int(-1), w_int(-1), omg_int(-1), tht0_int(-1), thtP_int(-1), lmbi_int(-1))
            action_int = (tht0_int, thtP_int)
            sim_traj = test_env._similate_trajectory(init, action_int, tf)

            if sim_traj[-1][3] and sim_traj[-1][-1]["constraints"][0][2]:
                success = True
                break

        if success:
            count_land += 1
            for i in range(len(sim_traj)):
                experience = sim_traj[i]
                state, action, reward, termination, truncation, next_state, info = experience
                for buffer_i in replay_buffer:
                    buffer_i.add(0, state, action, reward, termination, truncation, next_state)

            for buffer_i in replay_buffer:
                buffer_i.clear_n_step_buffer(0)
        else:
            count_crash += 1
            # tqdm.write(f"Failed : {key}")

    print(f"Loaded all successful trajectories... No of Sucess : {count_land}; No of Crash : {count_crash}")
    return replay_buffer