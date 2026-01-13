# Replay buffer to store transitions and support sampling #

import numpy as np
from collections import deque, namedtuple
import random

# Replay buffer with n-step support
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'termination', 'truncation', 'next_state'])

class NStepReplayBuffer:
    def __init__(self, state_dim, action_dim, buffer_size=1_000_000, batch_size=256, n_step=3, gamma=0.99, n_envs=4):
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.batch_size = batch_size
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffers = [deque(maxlen=n_step) for i in range(self.n_envs)]
        self.ptr, self.size = 0, 0

        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.terminations = np.zeros((buffer_size,), dtype=np.float32)
        self.truncations = np.zeros((buffer_size,), dtype=np.float32)
        self.terminal_flag = np.ones((buffer_size,), dtype=np.float32) * self.n_step

    def add(self, env_idx, state, action, reward, termination, truncation, next_state):
        n_step_buf = self.n_step_buffers[env_idx]
        n_step_buf.append(Transition(state, action, reward, termination, truncation, next_state))

        if len(n_step_buf) < self.n_step:
            return

        # Compute n-step return
        reward_n, next_state_n, termination_n, truncation_n, terminal_flag = self._get_n_step_info(n_step_buf)
        state_n = n_step_buf[0].state
        action_n = n_step_buf[0].action

        # Store to main buffer
        self.states[self.ptr] = state_n
        self.actions[self.ptr] = action_n
        self.rewards[self.ptr] = reward_n
        self.next_states[self.ptr] = next_state_n
        # self.dones[self.ptr] = done_n
        self.terminations[self.ptr] = termination_n
        self.truncations[self.ptr] = truncation_n
        self.terminal_flag[self.ptr] = terminal_flag

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def flush_buffer(self, env_idx):
        n_step_buf = self.n_step_buffers[env_idx]
        while len(n_step_buf) > 0:
            reward_n, next_state_n, termination_n, truncation_n, terminal_flag = self._get_n_step_info(n_step_buf)
            state_n = n_step_buf[0].state
            action_n = n_step_buf[0].action
            
            self.states[self.ptr] = state_n
            self.actions[self.ptr] = action_n
            self.rewards[self.ptr] = reward_n
            self.next_states[self.ptr] = next_state_n
            self.terminations[self.ptr] = termination_n
            self.truncations[self.ptr] = truncation_n
            self.terminal_flag[self.ptr] = terminal_flag
            
            self.ptr = (self.ptr + 1) % self.buffer_size
            self.size = min(self.size + 1, self.buffer_size)
            n_step_buf.popleft()

    def _get_n_step_info(self, n_step_buf):
        reward, next_state, termination, truncation = 0.0, n_step_buf[-1].next_state, 0.0, 0.0
        for i, t in enumerate(n_step_buf):
            reward += (self.gamma ** i) * t.reward
            if t.termination:
                next_state = t.next_state
                termination = 1.0  # marks terminal transition
                break
            elif t.truncation:
                next_state = t.next_state
                truncation = 1.0  # marks terminal transition
                break
        return reward, next_state, termination, truncation, i+1

    def sample(self, size = None):
        if size is None:
            size = self.size
        # idx = np.random.randint(0, size, size=self.batch_size)
        idx = np.random.permutation(size)[:self.batch_size]
        batch = (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.terminations[idx],
            self.truncations[idx]
        )
        return batch, self.terminal_flag[idx]
    
    def clear_n_step_buffer(self, env_idx):
        self.flush_buffer(env_idx)
        self.n_step_buffers[env_idx].clear()

    def current_size(self):
        return self.size