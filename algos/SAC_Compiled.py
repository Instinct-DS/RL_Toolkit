# Implementation of SAC algoirthm #

# Importing the necessary libraries
import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
from time import time
from tqdm import tqdm
import copy
import torch.optim as optim
from common.replaybuffer import NStepReplayBuffer
from common.logger import MLFlowLogger
from common.utils import atanh, SoftUpdate, load_demo_trajectories, load_demo_trajectories_parallel
from networks.network import Q_network
from networks.policy import DeterministicPolicy, TanhGaussianPolicy
from collections import deque, namedtuple
import mlflow
from tensordict.nn import CudaGraphModule

# Replay buffer with n-step support
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'termination', 'truncation', 'next_state'])

# SAC Agent #
class SACAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        tau=0.005,
        gamma=0.99,
        alpha=0.2,
        lr=3e-4,
        batch_size=256,
        buffer_size=1_000_000,
        n_steps=3,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=1,
        target_entropy=None,
        seed=None,
        stats_window_size=10,
        n_envs=4,
        logger_name="mlflow",
        device="cuda" if th.cuda.is_available() else "cpu",
        policy_kwargs = dict(),
        critic_kwargs = dict(),
        experiment_name = "",
        run_name = "",
        compiled = True
    ):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = tau
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.n_steps = n_steps
        self.train_freq = train_freq
        self.gradient_steps= gradient_steps
        self.target_update_interval = target_update_interval
        self.target_entropy = target_entropy
        self.seed = seed
        self.stats_window_size = stats_window_size
        self.n_envs = n_envs
        self.logger_name = logger_name
        self.device = device
        self.policy_kwargs = policy_kwargs
        self.critic_kwargs = critic_kwargs


        # Logging Variables
        self._count_total_gradients_taken = 0
        self._ep_rewards = deque(maxlen=stats_window_size)
        self._ep_lengths = deque(maxlen=stats_window_size)
        self._start_time = time()

        # Set Random seeds
        if seed is not None:
            th.manual_seed(seed)
            th.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

        # Actor
        self.actor = TanhGaussianPolicy(state_dim=state_dim, action_dim=action_dim, **policy_kwargs).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Critics
        self.critic1 = Q_network(state_dim, action_dim, **critic_kwargs).to(self.device)
        self.critic2 = Q_network(state_dim, action_dim, **critic_kwargs).to(self.device)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        # Target critics
        self.target_critic1 = Q_network(state_dim, action_dim, **critic_kwargs).to(self.device)
        self.target_critic2 = Q_network(state_dim, action_dim, **critic_kwargs).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Temperature parameter for entropy tuning
        if target_entropy is None:
            target_entropy = -float(action_dim)  # heuristic from SB3
        self.target_entropy = target_entropy
        self.log_alpha = th.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        # Replay Buffer
        self.replay_buffer = NStepReplayBuffer(state_dim, action_dim, buffer_size, batch_size, n_steps, gamma, n_envs)

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
            "n_step": self.n_steps,
            "train_freq": self.train_freq,
            "gradient_steps": self.gradient_steps,
            "target_update_interval": self.target_update_interval,
            "target_entropy": self.target_entropy,
            "seed": seed,
            "stats_window_size": self.stats_window_size,
            "n_envs": n_envs,
            "logger_name": self.logger_name,
            "device": self.device,
            "policy_kwargs" : policy_kwargs,
            "critic_kwargs" : critic_kwargs
        }

        # Enabling Compile Mode
        if compiled:
            self.enable_torch_compile()

    def log_hyperparameters(self):
        """
        Log SAC hyperparameters to MLflow for the current run.
        Call this only inside an active mlflow.start_run() context.
        """
        if self.logger_name == "mlflow":
            self.logger.log_params(self.hparams)

    def enable_torch_compile(self, mode="default"):
        self.update_critics = th.compile(self.update_critics, mode=mode)
        self.update_actor_and_alpha = th.compile(self.update_actor_and_alpha, mode=mode)
        print(f"torch.compile enabled (mode={mode})")

        # self.update_critics = CudaGraphModule(self.update_critics)
        # self.update_actor_and_alpha = CudaGraphModule(self.update_actor_and_alpha)
        # print(f"torch.compile enabled with cuda-graphs (mode={mode})")

    def update_critics(
        self,
        states,
        actions,
        rewards,
        next_states,
        terminations,
        terminal_flag,
    ):
        with th.no_grad():
            next_actions, next_logp = self.actor.sample(next_states)
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = th.min(target_q1, target_q2)
            entropy_term = th.exp(self.log_alpha.detach()) * next_logp
            targets = rewards + (1 - terminations) * (self.gamma ** terminal_flag) * (target_q - entropy_term)

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        loss = 0.5 * (
            F.mse_loss(q1, targets) + F.mse_loss(q2, targets)
        )

        self.critic1_optimizer.zero_grad(set_to_none=True)
        self.critic2_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        return loss.detach()
    
    def update_actor_and_alpha(self, states):
        actions, logp = self.actor.sample(states)
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        q = th.min(q1, q2)

        alpha = th.exp(self.log_alpha.detach())
        actor_loss = (alpha * logp - q).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return actor_loss.detach(), alpha_loss.detach()



    def soft_update(self, target_network, source_network):
        for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(
                (1 - self.tau) * target_param.data + self.tau * source_param.data
            )

    def select_action(self, state, deterministic=True):
        with th.no_grad():
            state = th.FloatTensor(state).to(self.device)
            mu, std = self.actor(state)

            if deterministic:
                action = th.tanh(mu)
                return action.cpu().numpy()
            else:
                action, _ = self.actor.sample(state)
                return action.cpu().numpy()
            
    def train(
            self,
            env,
            total_training_steps=1_000_000,
            learning_starts=10_000,
            progress_bar=True,
            verbose=1,
            log_interval=10.
    ):
        
        self.total_training_steps = total_training_steps
        self.learning_starts = learning_starts
        self.log_interval = log_interval
        self._total_timesteps_ran = 0

        if self.gradient_steps == -1:
            self.gradient_steps = env.n_envs

        if progress_bar:
            progress_bar = tqdm(total=self.total_training_steps, desc="Training Steps")

        # Logger Init #
        if hasattr(self, "logger"):
            self.logger.start()
            self.logger.log_params(self.hparams)

        # Start Collecting Rollout
        obs, _ = env.reset()
        _episode_start = np.zeros(env.num_envs, dtype=bool)
        _episode_rewards = np.zeros(shape=(env.num_envs,))
        _episode_lengths = np.zeros(shape=(env.num_envs,))
        self.logger_count = 1

        while self._total_timesteps_ran <= self.total_training_steps:
            with th.no_grad():
                obs_cuda = th.FloatTensor(obs).to(self.device)
                actions, _ = self.actor.sample(obs_cuda)
                actions = actions.cpu().numpy()
            # Environment step
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            dones = np.logical_or(terminations, truncations)

            # Add transitions to replay buffer (flatten batch items)
            for i in range(env.num_envs):
                if not _episode_start[i]:
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
                    self.logger_count += 1
                    self.replay_buffer.clear_n_step_buffer(i)
                    self._ep_lengths.append(_episode_lengths[i])
                    self._ep_rewards.append(_episode_rewards[i])
                    _episode_rewards[i], _episode_lengths[i] = 0, -1

            # Track episode rewards and lengths
            _episode_rewards += rewards
            _episode_lengths += 1
            self._total_timesteps_ran += env.num_envs
            obs = next_obs
            _episode_start = dones

            # Update step - learn only after certain timesteps, periodically
            if self._total_timesteps_ran >= self.learning_starts and self._total_timesteps_ran % self.train_freq == 0:
                for _ in range(self.gradient_steps):
                    self._count_total_gradients_taken += 1
                    batch, terminal_flag = self.replay_buffer.sample()
                    states, actions_b, rewards_b, next_states, terminations, truncations = batch

                    # Convert to tensors
                    states = th.FloatTensor(states).to(self.device)
                    actions_b = th.FloatTensor(actions_b).to(self.device)
                    rewards_b = th.FloatTensor(rewards_b).to(self.device)
                    next_states = th.FloatTensor(next_states).to(self.device)
                    # done = torch.FloatTensor(done).to(self.device)
                    terminations = th.FloatTensor(terminations).to(self.device)
                    truncations = th.FloatTensor(truncations).to(self.device)
                    terminal_flag = th.FloatTensor(terminal_flag).to(self.device)

                    critic_loss = self.update_critics(
                        states,
                        actions_b,
                        rewards_b,
                        next_states,
                        terminations,
                        terminal_flag,
                    )

                    actor_loss, alpha_loss = self.update_actor_and_alpha(states)

                    # Soft update target networks
                    if self._count_total_gradients_taken % self.target_update_interval == 0:
                        self.soft_update(self.target_critic1, self.critic1)
                        self.soft_update(self.target_critic2, self.critic2)
            
            if self._total_timesteps_ran >= self.learning_starts and self.logger_count % self.log_interval == 0:
                # Compute mean metrics over interval
                mean_ep_length = np.mean(self._ep_lengths) if len(self._ep_lengths) > 0 else 0
                mean_ep_reward = np.mean(self._ep_rewards) if len(self._ep_rewards) > 0 else 0
                elapsed_time = time() - self._start_time
                fps = self._total_timesteps_ran / elapsed_time if elapsed_time > 0 else 0

                # Log metrics to MLflow
                self.logger.log_metric("mean_episode_length", mean_ep_length, step=self._total_timesteps_ran)
                self.logger.log_metric("mean_episode_reward", mean_ep_reward, step=self._total_timesteps_ran)
                self.logger.log_metric("frames_per_second", fps, step=self._total_timesteps_ran)
                self.logger.log_metric("actor_loss", actor_loss.item(), step=self._total_timesteps_ran)
                self.logger.log_metric("critic_loss", critic_loss.item(), step=self._total_timesteps_ran)
                self.logger.log_metric("ent_coeff_loss", alpha_loss.item(), step=self._total_timesteps_ran)
                self.logger.log_metric("entropy_coefficient", th.exp(self.log_alpha).item(), step=self._total_timesteps_ran)

                if verbose != 0:
                    tqdm.write("\n" + "="*90)
                    tqdm.write(f" Step: {self._total_timesteps_ran:<8d} | MeanEpLen: {mean_ep_length:.2f} | MeanEpRew: {mean_ep_reward:.2f} | FPS: {fps:.0f}")
                    tqdm.write(f" Actor Loss: {actor_loss.item():.4f} | Critic Loss: {critic_loss.item():.4f} | Alpha: {th.exp(self.log_alpha).item():.4f}")
                    tqdm.write("="*90)
                
                self.logger_count = 1

            if progress_bar:
                progress_bar.update(env.num_envs)

        return self.actor
