from collections import deque
from dataclasses import dataclass

import numpy as np

import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow_probability as tfp
from dm_env import StepType


class RunningStats:
    """ Inspired by baselines
        https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py
    """

    def __init__(self, shape):

        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = 0 + 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = self._update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

    def normalize(self, x, eps=1e-8):
        normalized = (x - self.mean) / np.sqrt(self.var + eps)
        return normalized

    @staticmethod
    def _update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):

        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        return new_mean, new_var, new_count


@dataclass
class Experience:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


class SimpleReplayBuffer:
    """
    Simple Replay Buffer, assuming MDP-like state treatment
    """
    def __init__(self, maxlen=10**6):
        self.max_len = maxlen
        self.buffer = deque(maxlen=maxlen)
        self.count = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, exp):
        self.buffer.append(exp)
        self.count += 1

    def get_minibatch(self, batch_size):
        n = len(self.buffer)

        indices = np.random.choice(
            np.arange(n), replace=False, size=batch_size)

        selected_experiences = [self.buffer[idx] for idx in indices]

        states = np.vstack(
            [exp.state for exp in selected_experiences]
        ).astype(np.float32)

        actions = np.vstack(
            [exp.action for exp in selected_experiences]).astype(np.float32)

        rewards = np.vstack(
            [exp.reward for exp in selected_experiences]).reshape(-1, 1)

        next_states = np.vstack(
            [exp.next_state for exp in selected_experiences]).astype(np.float32)

        dones = np.vstack(
            [exp.done for exp in selected_experiences]).reshape(-1, 1)

        return states, actions, rewards, next_states, dones


class SingleTrajectoryReplayBuffer:
    """
    A Replay Buffer, which all experience is stored as a single long trajectory
    The buffer is first-in first-out
    """
    def __init__(self, maxlen=10**6):
        self.max_len = maxlen
        self.buffer = deque(maxlen=maxlen)
        self.count = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, exp):
        self.buffer.append(exp)
        self.count += 1

    def get_minibatch(self, batch_size):
        """
        MDP-based minibatch sampling
        :param batch_size:
        :return:
        """
        n = len(self.buffer)

        indices = np.random.choice(
            np.arange(n), replace=False, size=batch_size)

        selected_experiences = [self.buffer[idx] for idx in indices]

        states = np.vstack(
            [exp.state for exp in selected_experiences]
        ).astype(np.float32)

        actions = np.vstack(
            [exp.action for exp in selected_experiences]).astype(np.float32)

        rewards = np.vstack(
            [exp.reward for exp in selected_experiences]).reshape(-1, 1)

        next_states = np.vstack(
            [exp.next_state for exp in selected_experiences]).astype(np.float32)

        dones = np.vstack(
            [exp.done for exp in selected_experiences]).reshape(-1, 1)

        return states, actions, rewards, next_states, dones

    def get_trajectory(self, trajectory_length):
        """
        Sampling of a trajectory with a specific length
        :param trajectory_length:
        :return:
        """
        n = len(self.buffer)

        start_index = np.random.choice(np.arange(n - trajectory_length), replace=False)
        selected_experiences = [self.buffer[idx] for idx in range(start_index, start_index+trajectory_length)]

        states = np.vstack(
            [exp.state for exp in selected_experiences]
        ).astype(np.float32)

        actions = np.vstack(
            [exp.action for exp in selected_experiences]).astype(np.float32)

        rewards = np.vstack(
            [exp.reward for exp in selected_experiences]).reshape(-1, 1)

        next_states = np.vstack(
            [exp.next_state for exp in selected_experiences]).astype(np.float32)

        dones = np.vstack(
            [exp.done for exp in selected_experiences]).reshape(-1, 1)

        return states, actions, rewards, next_states, dones


class TrajectoryReplayBuffer:
    """
    Trajectory-based Replay Buffer
    """
    def __init__(self, maxlen=10**6):
        self.max_len = maxlen
        self.buffer = deque(maxlen=maxlen)
        self.count = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, trajectory):
        self.buffer.append(trajectory)
        self.count += 1

    def get_minibatch(self, batch_size):
        n = len(self.buffer)

        indices = np.random.choice(
            np.arange(n), replace=False, size=batch_size)

        trajectories = [self.buffer[idx] for idx in indices]
        return trajectories


@tf.function
def squashed_gaussian_logprob(mu, std, action, eps=1e-8):
    """
    Normalized Gaussian action log probability: squashed_action = tanh(action), action ~ Gaussian(mu, std**2)
    :param mu:
    :param std:
    :param action:
    :param eps:
    :return:
    """
    logprob = -0.5 * np.log(2 * np.pi)
    logprob += - tf.math.log(std + eps)
    logprob += - 0.5 * tf.square((action - mu) / (std + eps))
    logprob = tf.reduce_sum(logprob, axis=1, keepdims=True)
    logprob_squashed = logprob - tf.reduce_sum(
        tf.math.log(1 - tf.tanh(action) ** 2 + eps), axis=1, keepdims=True)
    return logprob_squashed


class DMC2GymWrapper:
    def __init__(self, env_dmc, max_step=1000):
        self.env_dmc = env_dmc
        self.max_step = max_step
        self._step = 0

    def reset(self):
        self._step = 0
        time_step = self.env_dmc.reset()
        obs, r, done, info = self.get_data_from_time_step(time_step)
        return obs

    def step(self, action):
        time_step = self.env_dmc.step(action)
        self._step += 1
        obs, r, done, info = self.get_data_from_time_step(time_step)
        return obs, r, done, info

    def get_data_from_time_step(self, time_step):
        obs = np.concatenate(list(time_step.observation.values()))
        r = time_step.reward
        done = True if (self._step >= self.max_step or time_step.step_type is StepType.LAST) else False
        info = {}
        return obs, r, done, info

    def sample_action(self):
        spec = self.env_dmc.action_spec()
        action = np.random.uniform(spec.minimum, spec.maximum, spec.shape)
        return action

