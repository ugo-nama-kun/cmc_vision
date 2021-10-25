from copy import deepcopy
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.base import AgentBase
from util.util import RunningStats, SimpleReplayBufferPixel, Experience



class Encoder(nn.Module):
    def __init__(self, lr):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_features=35 * 35 * 32, out_features=50),
            nn.LayerNorm(50),
            nn.Tanh()
        )

        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)

    def forward(self, x):
        x = self.model(x)
        return x


class GaussianPolicy(nn.Module):
    def __init__(self, action_space, action_bound, state_space, lr):
        super(GaussianPolicy, self).__init__()
        self.action_space = action_space
        self.action_bound = action_bound

        self.fc1 = nn.Linear(state_space, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu = nn.Linear(256, self.action_space)
        self.logstd = nn.Linear(256, self.action_space)

        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.mu(x))
        logstd = self.logstd(x)
        return mu, logstd

    def sample_action(self, states):
        states = self.preprocess(states)

        mu, logstd = self.forward(states)
        std = torch.exp(logstd)
        noise = torch.randn_like(mu)

        actions = mu + std * noise
        actions_squashed = torch.tanh(actions).detach().numpy()

        logprob = self._compute_logprob(mu, std, actions)
        # what is this process?
        logprob_squashed = logprob - torch.sum(torch.log(1 - torch.tanh(actions)**2), dim=1, keepdim=True)

        actions_squashed = self.postprocess(actions_squashed)
        return actions_squashed, logprob_squashed

    def preprocess(self, states):
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=torch.float32)
        return states

    def postprocess(self, action):
        return action * self.action_bound

    def _compute_logprob(self, means, stdevs, actions):
        logprob = -0.5 * np.log(2 * np.pi)
        logprob += - torch.log(stdevs)
        logprob += -0.5 * torch.square((actions - means) / stdevs)
        logprob = torch.sum(logprob, dim=1, keepdim=True)
        return logprob


class QNetwork(nn.Module):
    def __init__(self, state_space, action_space):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_space + action_space, 256)
        self.fc2 = nn.Linear(256, 256)
        self.q = nn.Linear(256, 1)

    def forward(self, state, action):
        state, action = self.preprocess(state, action)
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q = self.q(x)
        return q

    def preprocess(self, state, action):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)
        return state, action


class DualQNetwork(nn.Module):
    def __init__(self, state_space, action_space, lr):
        super(DualQNetwork, self).__init__()
        self.q1 = QNetwork(state_space, action_space)
        self.q2 = QNetwork(state_space, action_space)

        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)

    def forward(self, state, action):
        q1 = self.q1(state, action)
        q2 = self.q2(state, action)
        return q1, q2


class SoftActorCriticAgentRawPixel(AgentBase):
    def __init__(self, env, max_experiences=10 ** 6,
                 min_experiences=512, update_period=4, gamma=0.99, tau=0.005, batch_size=256, lr=1e-4):
        super().__init__(env)
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.update_period = update_period
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Initialize State-Action Scaler
        dummy_obs = env.reset()
        self.obs_normalizer = RunningStats(shape=dummy_obs.shape)
        self.scale_action = (env.action_space.high - env.action_space.low) / 2.
        self.scale_action = self.scale_action.astype(np.float32)

        self.action_space = len(self.scale_action)
        self.state_space = len(dummy_obs)

        self.replay_buffer = SimpleReplayBufferPixel(maxlen=self.max_experiences)

        # Models
        self._encoder = Encoder(lr=lr)
        self._policy = GaussianPolicy(action_space=self.action_space,
                                      action_bound=self.scale_action,
                                      state_space=self.state_space,
                                      lr=lr)
        self._dualqnet = DualQNetwork(self.state_space, self.action_space, lr)
        self._target_dualqnet = DualQNetwork(self.state_space, self.action_space, lr)
        self._target_dualqnet.load_state_dict(deepcopy(self._dualqnet.state_dict()))

        self.log_alpha = torch.tensor(0., requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam(params=[self.log_alpha], lr=3e-4)
        self.target_entropy = -0.5 * len(self.scale_action)

        self.global_steps = 0

    def _preprocess(self, observations: np.ndarray, frozen=False):

        if not frozen:
            self.obs_normalizer.update(observations)

        return self.obs_normalizer.normalize(observations)

    def sample_action(self, observation: np.ndarray, frozen=False):
        with torch.no_grad():
            observation = self._preprocess(observation, frozen=frozen)

            action, _ = self._policy.sample_action(np.atleast_2d(observation))
            action = action[0]

            info = {
                "alpha": torch.exp(self.log_alpha).detach().numpy()
            }

        return self._postprocess(action), info

    def sample_actions(self, observations: np.ndarray, frozen=False):
        with torch.no_grad():

            observations = self._preprocess(observations, frozen=frozen)

            actions, _ = self._policy.sample_action(np.atleast_2d(observations))

            info = {
                "alpha": torch.exp(self.log_alpha).detach().numpy()
            }

        return self._postprocess(actions), info

    def _postprocess(self, actions, frozen=False):
        return self.scale_action * actions

    def update(self, observation, action, reward, next_observation, done, frozen=False):
        # main update rule of the agent. if frozen, no update
        if not frozen:
            exp = Experience(observation, action, reward, next_observation, done)
            self.replay_buffer.push(exp)

            if (len(self.replay_buffer) >= self.min_experiences
                    and self.global_steps % self.update_period == 0):
                self.update_params()

            self.global_steps += 1

    def update_params(self):
        # update rules of the networks and alpha

        # get numpy arrays
        obss, actions, rewards, next_obss, dones = self.replay_buffer.get_minibatch(self.batch_size)

        with torch.no_grad():
            # Normalize observations
            obss = self.obs_normalizer.normalize(obss)
            next_obss = self.obs_normalizer.normalize(next_obss)

            obss = torch.tensor(obss, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.float32)
            next_obss = torch.tensor(next_obss, dtype=torch.float32)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.int8)

            # Get alpha
            alpha = torch.exp(self.log_alpha)

            # Get latents
            next_latents = self._encoder(next_obss)

            # Update Q function
            next_actions, next_logprobs = self._policy.sample_action(next_latents)

            target_q1, target_q2 = self._target_dualqnet(next_latents, next_actions)

            min_q = torch.min(target_q1, target_q2)

            target = rewards + (1 - dones) * self.gamma * (min_q - alpha * next_logprobs)

        # Value Loss Optimization
        latents = self._encoder(obss)
        q1, q2 = self._dualqnet(latents, actions)
        loss_q1 = torch.mean(torch.sum(torch.square(target - q1), -1))
        loss_q2 = torch.mean(torch.sum(torch.square(target - q2), -1))
        loss_q = 0.5 * (loss_q1 + loss_q2)

        self._dualqnet.optimizer.zero_grad()
        loss_q.backward()
        self._dualqnet.optimizer.step()

        # Update policy
        latents = latents.detach()
        selected_actions, logprobs = self._policy.sample_action(latents)

        q1, q2 = self._dualqnet(latents, selected_actions)
        q_min = torch.min(q1, q2)

        loss_samples = torch.sum(alpha * logprobs - q_min, -1)
        loss_policy = torch.mean(loss_samples)

        self._policy.optimizer.zero_grad()
        loss_policy.backward()
        self._policy.optimizer.step()

        # Adjust alpha
        entropy_diff = - logprobs.detach() - self.target_entropy
        alpha_loss = torch.mean(torch.sum(torch.exp(self.log_alpha) * entropy_diff.detach(), -1))
        # print(f"alpha loss before: {alpha_loss.detach()}")

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Soft target update
        target_weights_dict = self._target_dualqnet.state_dict()
        for k, w in target_weights_dict.items():
            w_new = (1 - self.tau) * w + self.tau * self._dualqnet.state_dict()[k]
            w = w_new.detach()

    def save_model(self):
        # different from original tf implementation
        torch.save({
            "time_stamp": datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
            "encoder": self._encoder.state_dict(),
            "dualqnet": self._dualqnet.state_dict(),
            "target_dualqnet": self._target_dualqnet.state_dict(),
            "policy": self._policy.state_dict(),
        }, f"checkpoints/model.pth")
