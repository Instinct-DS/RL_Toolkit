# AWAC : Advantage Weighted Actor Critic #

# Notes #
# 01) Uses AWAC style losses for policy (without entropy term) but uses SAC/TD3 style critic losses as per AWAC paper
# 02) Behavioural Cloning style of loss is not added as per AWAC paper (setting bc_weight=0)

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
from time import time
from tqdm import tqdm
import torch.optim as optim
from common.replaybuffer import NStepReplayBuffer
from common.logger import MLFlowLogger
from common.utils import atanh, SoftUpdate, load_demo_trajectories
from networks.network import Q_network
from networks.policy import TanhGaussianPolicy
from collections import deque, namedtuple

# AWAC Agent #
class AWAC:
    def __init__(
            self,
            state_dim,
            action_dim,
            tau=0.005,
            gamma=0.999,
            lr=3e-4,
            batch_size=256,
            buffer_size=1_000_000,
            n_step=3,
            train_freq=1,
            gradient_steps=1,
            target_update_interval=1,
            seed=None,
            stats_window_size=100,
            n_envs =4,
            alpha = 0.0, 
            bc_weight = 0.0,
            awac_lambda = 3.0,
            max_weight = 2.0,
            policy_update_period = 5,
            q_update_period = 1,
            use_automatic_entropy_tuning = False,
            logger_name="mlflow",
            experiment_name = "",
            run_name = "",
            device="cuda" if torch.cuda.is_available() else "cpu"   
    ):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = tau
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.n_step = n_step
        self.train_freq = train_freq
        self.buffer_size = buffer_size
        self.gradient_steps = gradient_steps
        self.target_update_interval = target_update_interval
        self.seed = seed
        self.stats_window_size = stats_window_size
        self.n_envs = n_envs
        self.logger_name = logger_name
        self.device = device
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.bc_weight = bc_weight
        self.policy_update_period = policy_update_period
        self.q_update_period = q_update_period
        self.awac_lambda = awac_lambda
        self.max_weight = max_weight

        # Logging variables
        self._n_train_steps_total = 0
        self.logger_name = logger_name
        self.stats_window_size = stats_window_size
        self.ep_rewards = deque(maxlen=stats_window_size) 
        self.ep_lengths = deque(maxlen=stats_window_size) 
        self.ep_step_count = 0
        self.start_time = time()

        # Set random seeds
        if seed is not None:
            # torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        # Actor
        self.actor = TanhGaussianPolicy(state_dim, action_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Critics
        self.critic1 = Q_network(state_dim, action_dim).to(self.device)
        self.critic2 = Q_network(state_dim, action_dim).to(self.device)
        self.critic1_optimizer = optim.SGD(self.critic1.parameters(), lr=lr, momentum=0.9)
        self.critic2_optimizer = optim.SGD(self.critic2.parameters(), lr=lr, momentum=0.9)

        # Target critics
        self.target_critic1 = Q_network(state_dim, action_dim).to(self.device)
        self.target_critic2 = Q_network(state_dim, action_dim).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Temperature parameter for entropy tuning
        self.alpha_init=alpha
        target_entropy = -float(action_dim)  # heuristic from SB3
        self.target_entropy = target_entropy
        if use_automatic_entropy_tuning:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        else:
            self.log_alpha = torch.log(torch.tensor(alpha)) if alpha != 0 else -torch.inf

        self.replay_buffer = NStepReplayBuffer(state_dim, action_dim, buffer_size, batch_size, n_step, gamma, n_envs)

        # Logger Init #
        if logger_name == "mlflow":
            self.logger = MLFlowLogger(uri="http://127.0.0.1:5000", experiment_name=experiment_name, run_name=run_name)

        # Log hyperparams #
        self.hparams = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "tau": self.tau,
            "gamma": self.gamma,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "buffer_size": self.buffer_size,
            "n_step": self.n_step,
            "train_freq": self.train_freq,
            "gradient_steps": self.gradient_steps,
            "target_update_interval": self.target_update_interval,
            "seed": seed,
            "stats_window_size": self.stats_window_size,
            "n_envs": n_envs,
            "logger_name": self.logger_name,
            "device": self.device,
        }

    def _run_bc_batch(self, replay_buffer):
        batch_m, terminal_flag_m = replay_buffer.sample()
        states_m, actions_b_m, rewards_b_m, next_states_m, terminations_m, truncations_m = batch_m

        pred_actions, log_prob = self.actor.sample(states_m)

        mse_loss = F.mse_loss(pred_actions, actions_b_m).mean()
        policy_logpp = -self.actor.log_probs(states_m, actions_b_m).mean()
        policy_loss = policy_logpp

        return policy_loss, mse_loss
    
    def load_trajectories(self, demo_file, demo_env, demo_ocp):
        self.replay_buffer = load_demo_trajectories(self.replay_buffer, demo_file, demo_env, demo_ocp, self.n_step, self.gamma)
        return self.replay_buffer
    
    def _train(self):
        batch_m, terminal_flag_m = self.replay_buffer.sample()
        states_m, actions_b_m, rewards_b_m, next_states_m, terminations_m, truncations_m = batch_m

        # Convert to tensors
        states_m = torch.FloatTensor(states_m).to(self.device)
        actions_b_m = torch.FloatTensor(actions_b_m).to(self.device)
        rewards_b_m = torch.FloatTensor(rewards_b_m).to(self.device)
        next_states_m = torch.FloatTensor(next_states_m).to(self.device)
        terminations_m= torch.FloatTensor(terminations_m).to(self.device)
        truncations_m = torch.FloatTensor(truncations_m).to(self.device)
        terminal_flag_m = torch.FloatTensor(terminal_flag_m).to(self.device)

        actions_cur, log_probs = self.actor.sample(states_m)

        # ALPHA UPDATE #
        if self.use_automatic_entropy_tuning:
            ALPHA_LOSS = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            ALPHA_LOSS.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp().detach() # Check this #
        else:
            ALPHA_LOSS = torch.tensor(0.0, device=self.device)
            alpha = self.alpha_init

        # VALUE FUNCTION UPDATE #
        q1_pred = self.critic1(states_m, actions_b_m)
        q2_pred = self.critic2(states_m, actions_b_m)
        with torch.no_grad():
            next_actions, next_log_pi = self.actor.sample(next_states_m)
            target_q_values = torch.min(
                self.target_critic1(next_states_m, next_actions),
                self.target_critic2(next_states_m, next_actions)
            ) - alpha*next_log_pi
            
            q_target = rewards_b_m + (1 - terminations_m) * (self.gamma**terminal_flag_m) * target_q_values

        QF1_LOSS = F.mse_loss(q1_pred, q_target)
        QF2_LOSS = F.mse_loss(q2_pred, q_target)

        # POLICY UPDATE #
        with torch.no_grad():
            actions_pi, _ = self.actor.sample_det(states_m)
            v1_pi = self.critic1(states_m, actions_pi)
            v2_pi = self.critic2(states_m, actions_pi)
            v_pi = torch.min(v1_pi, v2_pi)

            q_adv = torch.min(q1_pred, q2_pred)
            score = q_adv - v_pi

            exp_adv = torch.clamp(score / self.awac_lambda, min=-self.max_weight, max=self.max_weight)
            weights = torch.exp(exp_adv)
            # Normalize weights
            # adv_mean = torch.mean(score)
            # adv_std = torch.std(score) + 1e-5
            # normalized_score = (score - adv_mean) / adv_std
            # weights = torch.exp(normalized_score / self.awac_lambda)


        policy_logpp = self.actor.log_probs(states_m, actions_b_m)
        POLICY_LOSS = -alpha * log_probs.mean()
        POLICY_LOSS = POLICY_LOSS + (-policy_logpp *weights).mean()
        if self.bc_weight > 0:
            train_policy_loss, train_mse_loss = self._run_bc_batch(self.replay_buffer)
            POLICY_LOSS = POLICY_LOSS + self.bc_weight*train_policy_loss

        # UPDATE NETWORKS #
        if self._n_train_steps_total % self.q_update_period == 0:
            self.critic1_optimizer.zero_grad()
            QF1_LOSS.backward()
            self.critic1_optimizer.step()

            self.critic2_optimizer.zero_grad()
            QF2_LOSS.backward()
            self.critic2_optimizer.step()

        if self._n_train_steps_total % self.policy_update_period == 0:
            self.actor_optimizer.zero_grad()
            POLICY_LOSS.backward()
            self.actor_optimizer.step()

        if self._n_train_steps_total % self.target_update_interval == 0:
            # SOFT UPDATES #
            SoftUpdate(self.target_critic1, self.critic1, self.tau)
            SoftUpdate(self.target_critic2, self.critic2, self.tau)

        self._n_train_steps_total += 1

        return POLICY_LOSS.item(), QF1_LOSS.item(), QF2_LOSS.item(), ALPHA_LOSS.item(), alpha  
    
    def train_loop_awac(
            self,
            env,
            train_offline_steps=10_000, 
            train_online_steps=100_000,
            progress_bar = True,
            log_interval = 10,
            verbose = 1,
    ):
        self.train_offline_steps = train_offline_steps
        self.train_online_steps = train_online_steps
        self.log_interval = log_interval
        log_interval_offline = self.log_interval*10
        self.verbose = verbose
        self.progress_bar = progress_bar

        progress_bar_1 = tqdm(total=self.train_offline_steps, desc="Offline Training Steps")
        actor_offline_loss = deque(maxlen=log_interval_offline)
        critic1_offline_loss = deque(maxlen=log_interval_offline)
        critic2_offline_loss = deque(maxlen=log_interval_offline)
        alpha_offline_loss = deque(maxlen=log_interval_offline)
        alpha_offline = deque(maxlen=log_interval_offline)

        # Logger Init #
        self.logger.start()
        self.logger._log_params(self.hparams)

        while self._n_train_steps_total <= self.train_offline_steps:
            policy_loss, critic1_loss, critic2_loss, alpha_loss, alpha = self._train()
            actor_offline_loss.append(policy_loss)
            critic1_offline_loss.append(critic1_loss)
            critic2_offline_loss.append(critic2_loss)
            alpha_offline_loss.append(alpha_loss)
            alpha_offline.append(alpha)
            
            if self._n_train_steps_total % log_interval_offline == 0:
                self.logger.log_metric("actor_loss", np.mean(actor_offline_loss), step=self._n_train_steps_total)
                self.logger.log_metric("critic1_loss", np.mean(critic1_offline_loss), step=self._n_train_steps_total)
                self.logger.log_metric("critic2_loss", np.mean(critic2_offline_loss), step=self._n_train_steps_total)
                self.logger.log_metric("alpha_loss", np.mean(alpha_offline_loss), step=self._n_train_steps_total)
                self.logger.log_metric("alpha", np.mean(alpha_offline), step=self._n_train_steps_total)

            progress_bar_1.update(1)
        progress_bar_1.close()

        # Reset for online training #
        self._n_train_steps_total = 0
        progress_bar_2 = tqdm(total=self.train_online_steps, desc="Online Training Steps")
        obs, _ = env.reset()
        episode_start = np.zeros(env.num_envs, dtype=bool)
        episode_rewards = np.zeros(shape=(env.num_envs,))
        episode_lengths = np.zeros(shape=(env.num_envs,))
        self.logger_count = 1
        self.timesteps_ran = 0
        while self.timesteps_ran <= self.train_online_steps:
            with torch.no_grad():
                obs_cuda = torch.FloatTensor(obs).to(self.device)
                actions = self.actor.sample(obs_cuda)[0].cpu().numpy()
            # Environment step
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            dones = np.logical_or(terminations, truncations)

            # Add transitions to replay buffer (flatten batch items)
            for i in range(env.num_envs):
                if not episode_start[i]:
                    self.replay_buffer.add(
                        i,
                        obs[i],
                        actions[i],
                        rewards[i],
                        terminations[i],
                        truncations[i],
                        next_obs[i]
                    )
                else:
                    self.ep_step_count += 1
                    self.logger_count += 1
                    self.replay_buffer.clear_n_step_buffer(i)
                    self.ep_lengths.append(episode_lengths[i])
                    self.ep_rewards.append(episode_rewards[i])
                    episode_rewards[i], episode_lengths[i] = 0, -1


            # Track episode rewards and lengths
            episode_rewards += rewards
            episode_lengths += 1
            self.timesteps_ran += env.num_envs
            obs = next_obs
            episode_start = dones
            
            if self._n_train_steps_total % self.train_freq == 0:
                for _ in range(self.gradient_steps):
                    policy_loss, critic1_loss, critic2_loss, alpha_loss, alpha = self._train()
                    actor_offline_loss.append(policy_loss)
                    critic1_offline_loss.append(critic1_loss)
                    critic2_offline_loss.append(critic2_loss)
                    alpha_offline_loss.append(alpha_loss)
                    alpha_offline.append(alpha)

            if self.logger_count % self.log_interval == 0:
                # Compute mean metrics over interval
                mean_ep_length = np.mean(self.ep_lengths) if len(self.ep_lengths) > 0 else 0
                mean_ep_reward = np.mean(self.ep_rewards) if len(self.ep_rewards) > 0 else 0
                elapsed_time = time() - self.start_time
                fps = self.timesteps_ran / elapsed_time if elapsed_time > 0 else 0

                # Log metrics to MLflow
                self.logger.log_metric("mean_episode_length", mean_ep_length, step=self.timesteps_ran)
                self.logger.log_metric("mean_episode_reward", mean_ep_reward, step=self.timesteps_ran)
                self.logger.log_metric("frames_per_second", fps, step=self.timesteps_ran)
                self.logger.log_metric("actor_loss", np.mean(actor_offline_loss), step=self.timesteps_ran)
                self.logger.log_metric("critic1_loss", np.mean(critic1_offline_loss), step=self.timesteps_ran)
                self.logger.log_metric("critic2_loss", np.mean(critic2_offline_loss), step=self.timesteps_ran)
                self.logger.log_metric("entropy_coefficient", np.mean(alpha_offline), step=self.timesteps_ran)
                self.logger.log_metric("alpha_loss", np.mean(alpha_offline_loss), step=self.timesteps_ran)


                if verbose != 0:
                    tqdm.write("\n" + "="*90)
                    tqdm.write(f" Step: {self.timesteps_ran:<8d} | MeanEpLen: {mean_ep_length:.2f} | MeanEpRew: {mean_ep_reward:.2f} | FPS: {fps:.0f}")
                    tqdm.write(f" Actor Loss: {np.mean(actor_offline_loss):.4f} | Critic Loss: {min(np.mean(critic2_offline_loss), np.mean(critic1_offline_loss)):.4f} | Alpha: {np.mean(alpha_offline):.4f}")
                    tqdm.write("="*90)
                
                self.logger_count = 1

            progress_bar_2.update(env.num_envs)
        progress_bar_2.close()

