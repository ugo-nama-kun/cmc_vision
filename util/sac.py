import numpy as np

import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow_probability as tfp

import wandb


from util.base import AgentBase
from util.util import RunningStats, SimpleReplayBuffer, Experience


class GaussianPolicy(tf.keras.Model):

    def __init__(self, action_space, lr=3e-4, eps=1e-8):
        super(GaussianPolicy, self).__init__()
        self.eps = eps

        self.action_space = action_space

        self.dense1 = kl.Dense(256, activation="relu")
        self.dense2 = kl.Dense(256, activation="relu")
        self.mu = kl.Dense(self.action_space, activation="tanh")
        self.logstd = kl.Dense(self.action_space)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    @tf.function
    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)

        mu = self.mu(x)
        logstd = self.logstd(x)
        return mu, logstd

    @tf.function
    def sample_action(self, states):

        mu, logstd = self(states)

        std = tf.math.exp(logstd)
        normal_noise = tf.random.normal(shape=mu.shape, mean=0., stddev=1.)
        actions = mu + std * normal_noise

        logprob = self.compute_logprob(mu, std, actions)

        actions_squashed = tf.tanh(actions)
        logprob_squashed = logprob - tf.reduce_sum(
            tf.math.log(1 - tf.tanh(actions) ** 2 + self.eps), axis=1, keepdims=True)

        return actions_squashed, logprob_squashed

    @tf.function
    def compute_logprob(self, means, stdevs, actions):
        logprob = -0.5 * np.log(2 * np.pi)
        logprob += - tf.math.log(stdevs + self.eps)
        logprob += - 0.5 * tf.square((actions - means) / (stdevs + self.eps))
        logprob = tf.reduce_sum(logprob, axis=1, keepdims=True)
        return logprob


class DualQNetwork(tf.keras.Model):
    def __init__(self, lr=3e-4):
        super(DualQNetwork, self).__init__()
        self.dense_11 = kl.Dense(256, activation="relu")
        self.dense_12 = kl.Dense(256, activation="relu")
        self.q1 = kl.Dense(1)

        self.dense_21 = kl.Dense(256, activation="relu")
        self.dense_22 = kl.Dense(256, activation="relu")
        self.q2 = kl.Dense(1)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def call(self, states, actions):
        inputs = tf.concat([states, actions], 1)
        x1 = self.dense_11(inputs)
        x1 = self.dense_12(x1)
        q1 = self.q1(x1)

        x2 = self.dense_21(inputs)
        x2 = self.dense_22(x2)
        q2 = self.q2(x2)
        return q1, q2


class SoftActorCriticAgent(AgentBase):
    def __init__(self, env, model_policy=GaussianPolicy, model_dual_qnet=DualQNetwork, max_experiences=10 ** 6,
                 min_experiences=512, update_period=4, gamma=0.99, tau=0.005, batch_size=256):
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

        self.replay_buffer = SimpleReplayBuffer(maxlen=self.max_experiences)

        # Models
        self._policy = model_policy(action_space=len(self.scale_action))
        self._dualqnet = model_dual_qnet()
        self._target_dualqnet = model_dual_qnet()

        self.log_alpha = tf.Variable(0.)
        self.alpha_optimizer = tf.keras.optimizers.Adam(3e-4)
        self.target_entropy = -0.5 * len(self.scale_action)

        self.global_steps = 0

        self._init(env)

    def _init(self, env):
        dummy_state = env.reset()
        dummy_state = (dummy_state[np.newaxis, ...]).astype(np.float32)

        dummy_action = env.action_space.sample()
        dummy_action = (dummy_action[np.newaxis, ...]).astype(np.float32)

        # Initialize network shapes by a run
        self._policy(dummy_state)
        self._dualqnet(dummy_state, dummy_action)
        self._target_dualqnet(dummy_state, dummy_action)
        self._target_dualqnet.set_weights(self._dualqnet.get_weights())

    def _preprocess(self, observations, frozen=False):

        if not frozen:
            self.obs_normalizer.update(observations)

        return self.obs_normalizer.normalize(observations)

    def sample_action(self, observation, frozen=False):

        observation = self._preprocess(observation, frozen=frozen)

        action, _ = self._policy.sample_action(np.atleast_2d(observation))
        action = action.numpy()[0]

        info = {
            "alpha": tf.exp(self.log_alpha).numpy()
        }

        return self._postprocess(action), info

    def sample_actions(self, observations, frozen=False):

        observations = self._preprocess(observations, frozen=frozen)

        actions, _ = self._policy.sample_action(np.atleast_2d(observations))
        actions = actions.numpy()

        info = {
            "alpha": tf.exp(self.log_alpha).numpy()
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

        states, actions, rewards, next_states, dones = self.replay_buffer.get_minibatch(self.batch_size)

        # Normalize observations
        states = self.obs_normalizer.normalize(states)
        next_states = self.obs_normalizer.normalize(next_states)

        # Get alpha
        alpha = tf.math.exp(self.log_alpha)

        # Update Q function
        next_actions, next_logprobs = self._policy.sample_action(next_states)

        target_q1, target_q2 = self._target_dualqnet(next_states, next_actions)

        target = rewards + (1 - dones) * self.gamma * (
                tf.minimum(target_q1, target_q2) + -1 * alpha * next_logprobs
        )
        with tf.GradientTape() as tape:
            q1, q2 = self._dualqnet(states, actions)
            loss_1 = tf.reduce_mean(tf.square(target - q1))
            loss_2 = tf.reduce_mean(tf.square(target - q2))
            loss = 0.5 * loss_1 + 0.5 * loss_2

        variables = self._dualqnet.trainable_variables
        grads = tape.gradient(loss, variables)
        self._dualqnet.optimizer.apply_gradients(zip(grads, variables))

        # Update policy
        with tf.GradientTape() as tape:
            selected_actions, logprobs = self._policy.sample_action(states)
            q1, q2 = self._dualqnet(states, selected_actions)
            q_min = tf.minimum(q1, q2)
            loss = -1 * tf.reduce_mean(q_min + -1 * alpha * logprobs)

        variables = self._policy.trainable_variables
        grads = tape.gradient(loss, variables)
        self._policy.optimizer.apply_gradients(zip(grads, variables))

        # Adjust alpha
        entropy_diff = -1 * logprobs - self.target_entropy
        with tf.GradientTape() as tape:
            tape.watch(self.log_alpha)
            alpha_loss = tf.reduce_mean(tf.exp(self.log_alpha) * entropy_diff)

        grad = tape.gradient(alpha_loss, self.log_alpha)
        self.alpha_optimizer.apply_gradients([(grad, self.log_alpha)])

        # Soft target update
        self._target_dualqnet.set_weights(
            (1 - self.tau) * np.array(self._target_dualqnet.get_weights())
            + self.tau * np.array(self._dualqnet.get_weights())
        )


def evaluate_single_episode(test_env, agent: SoftActorCriticAgent):
    episode_reward = 0
    local_steps = 0

    agent_infos = []
    env_infos = []

    done = False
    obs = test_env.reset()
    while not done:
        action, agent_info = agent.sample_action(np.atleast_2d(obs), frozen=True)

        next_obs, reward, done, env_info = test_env.step(action)
        agent.update(obs, action, reward, next_obs, done, frozen=True)

        obs = next_obs
        episode_reward += reward

        local_steps += 1
        agent_infos.append(agent_info)
        env_infos.append(env_info)

    return episode_reward, local_steps, agent_infos, env_infos


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
            test_episode_rewards = []
            for _ in range(n_test):
                test_episode_reward, test_local_steps, agent_infos, env_infos = evaluate_single_episode(
                    test_env=test_env, agent=agent)
                test_episode_rewards.append(test_episode_reward)

            step_now = step if step == 0 else step + 1
            print(f" {step_now} steps average reward: {np.array(test_episode_rewards, dtype=np.float32).mean()}, std: {np.array(test_episode_rewards, dtype=np.float32).std()}")
            # log_metric("average_test_episode_reward", np.array(test_episode_rewards, dtype=np.float32).mean(), step=step_now)
            wandb.log(
                {"maen_return": np.array(test_episode_rewards, dtype=np.float32).mean(),
                 "std_return": np.array(test_episode_rewards, dtype=np.float32).std()},
                step=step_now
            )

    return episode_reward, local_steps

#
# if __name__ == '__main__':
#     config = {
#         "env": "Pendulum-v0",
#         "algo": "sac-fnn",
#         "max_experience": 10**5,
#         "min_experiences": 512,
#         "update_period": 4,
#         "gamma": 0.99,
#         "tau": 0.005,
#         "batch_size": 256,
#     }
#
#     env = gym.make(config["env"])
#     test_env = gym.make(config["env"])
#     agent = SoftActorCriticAgent(env=env,
#                                  max_experiences=config["max_experience"],
#                                  min_experiences=config["min_experiences"],
#                                  update_period=config["update_period"],
#                                  gamma=config["gamma"],
#                                  tau=config["tau"],
#                                  batch_size=config["batch_size"])
#
#     training(agent, env, test_env, n_steps=300 * 100, evaluate_every=1000, n_test=3)
#
#

