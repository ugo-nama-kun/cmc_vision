from collections import deque
from dataclasses import dataclass
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

import tensorflow as tf

import wandb
from dm_env import StepType
from matplotlib import animation

from util.base import AgentBase


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

    def __init__(self, maxlen=10 ** 6):
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

    def __init__(self, maxlen=10 ** 6):
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
        selected_experiences = [self.buffer[idx] for idx in range(start_index, start_index + trajectory_length)]

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

    def __init__(self, maxlen=10 ** 6):
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


class ActionSpace:
    def __init__(self, env_dmc):
        self.env_dmc = env_dmc
        self.spec = self.env_dmc.action_spec()

    @property
    def high(self):
        return np.ones(self.spec.shape) * self.spec.maximum

    @property
    def low(self):
        return np.ones(self.spec.shape) * self.spec.minimum

    def sample(self):
        spec = self.env_dmc.action_spec()
        action = np.random.uniform(spec.minimum, spec.maximum, spec.shape)
        return action


class DMC2GymWrapper:
    def __init__(self, env_dmc, max_step=1000, is_vision=False, frame_stack=3, im_size=84):
        self.env_dmc = env_dmc
        self.max_step = max_step
        self._step = 0
        self.is_vision = is_vision
        self.frames = deque(maxlen=frame_stack)
        self.im_size = im_size

    def reset(self):
        self._step = 0
        self.frames.clear()
        time_step = self.env_dmc.reset()
        obs, r, done, info = self.get_data_from_time_step(time_step)
        return obs

    def step(self, action):
        time_step = self.env_dmc.step(action)
        self._step += 1
        obs, r, done, info = self.get_data_from_time_step(time_step)
        return obs, r, done, info

    def get_data_from_time_step(self, time_step):
        if self.is_vision:
            single_vision = self.env_dmc.physics.render(camera_id=0, height=self.im_size, width=self.im_size)
            if len(self.frames) == 0:
                for _ in range(self.frames.maxlen):
                    self.frames.append(single_vision)
            else:
                self.frames.append(single_vision)
            obs = np.dstack(self.frames)
        else:
            obs = np.concatenate(list(time_step.observation.values()))
        r = time_step.reward
        done = True if (self._step >= self.max_step or time_step.step_type is StepType.LAST) else False
        info = {}
        return obs, r, done, info

    @property
    def action_space(self):
        return ActionSpace(self.env_dmc)


def display_video(frames, framerate=30):
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])

    def update(frame):
        im.set_data(frame)
        return [im]

    interval = 1000 / framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)
    return anim


def evaluate_single_episode(test_env: DMC2GymWrapper, agent: AgentBase, record: bool):
    episode_reward = 0
    local_steps = 0

    agent_infos = []
    env_infos = []
    frames = []

    done = False
    obs = test_env.reset()
    while not done:
        action, agent_info = agent.sample_action(np.atleast_2d(obs), frozen=True)

        next_obs, reward, done, env_info = test_env.step(action)
        agent.update(obs, action, reward, next_obs, done, frozen=True)

        if record:
            camera1 = test_env.env_dmc.physics.render(camera_id=1, height=200, width=200)
            frames.append(camera1)

        obs = next_obs
        episode_reward += reward

        local_steps += 1
        agent_infos.append(agent_info)
        env_infos.append(env_info)

    video_result = None
    if record:
        anim = display_video(frames, framerate=1. / test_env.env_dmc.control_timestep())
        writervideo = animation.FFMpegWriter(fps=1. / test_env.env_dmc.control_timestep())

        time_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        anim.save(f'video/test-{time_stamp}.mp4', writer=writervideo)

        video_result = wandb.Video(f"video/test-{time_stamp}.mp4",
                                   fps=1. / test_env.env_dmc.control_timestep(),
                                   format="mp4")

    return episode_reward, local_steps, agent_infos, env_infos, video_result


def training(agent, env, test_env, n_steps, evaluate_every, n_test):
    state = env.reset()
    episode_reward = 0
    local_steps = 0
    for step in range(n_steps):

        action, _ = agent.sample_action(np.atleast_2d(state))

        next_state, reward, done, _ = env.step(action)

        agent.update(state, action, reward, next_state, done)

        episode_reward += reward
        local_steps += 1

        if done:
            episode_reward = 0
            local_steps = 0
            state = env.reset()
        else:
            state = next_state

        if step == 0 or (step + 1) % evaluate_every == 0:
            video_test = None
            test_episode_rewards = []
            for n in range(n_test):
                test_episode_reward, test_local_steps, agent_infos, env_infos, video_result = evaluate_single_episode(
                    test_env=test_env, agent=agent, record=n == 0)
                test_episode_rewards.append(test_episode_reward)

                if video_result:
                    video_test = video_result

            step_now = step if step == 0 else step + 1
            print(
                f" {step_now} steps average reward: {np.array(test_episode_rewards, dtype=np.float32).mean()}, std: {np.array(test_episode_rewards, dtype=np.float32).std()}")

            wandb.log(
                {"average_return": np.array(test_episode_rewards, dtype=np.float32).mean(),
                 "std_return": np.array(test_episode_rewards, dtype=np.float32).std(),
                 "video": video_test},
                step=step_now
            )

    return episode_reward, local_steps
