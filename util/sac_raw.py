import numpy as np

import tensorflow as tf
import tensorflow.keras.layers as kl
from keras import Sequential
from tensorflow.keras import activations


from util.base import AgentBase
from util.util import Experience, SimpleReplayBufferPixel


class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()

        self.model = Sequential()
        self.model.add(kl.Conv2D(32, (3, 3), strides=2, activation="relu"))
        self.model.add(kl.Conv2D(32, (3, 3), strides=1, activation="relu"))
        self.model.add(kl.Conv2D(32, (3, 3), strides=1, activation="relu"))
        self.model.add(kl.Conv2D(32, (3, 3), strides=1, activation="relu"))
        self.model.add(kl.Flatten())
        self.model.add(kl.Dense(50, activation=activations.linear))
        self.model.add(kl.Activation(activation=activations.tanh))

    @tf.function
    def call(self, x):
        x = self.model(x)
        return x


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


class SoftActorCriticAgentRawPixel(AgentBase):
    def __init__(self, env, max_experiences=10 ** 6,
                 min_experiences=512, update_period=4, gamma=0.99, tau=0.005, batch_size=256):
        super().__init__(env)
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.update_period = update_period
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Initialize State-Action Scaler
        self.scale_action = (env.action_space.high - env.action_space.low) / 2.

        self.replay_buffer = SimpleReplayBufferPixel(maxlen=self.max_experiences)

        # Models
        self._encoder = Encoder()
        self._policy = GaussianPolicy(action_space=len(self.scale_action))
        self._dualqnet = DualQNetwork()
        self._target_dualqnet = DualQNetwork()

        self.log_alpha = tf.Variable(0.)
        self.alpha_optimizer = tf.keras.optimizers.Adam(3e-4)
        self.target_entropy = -0.5 * len(self.scale_action)

        self.global_steps = 0

        self._init(env)

    def _init(self, env):
        dummy_obs = env.reset()
        dummy_obs = (dummy_obs[np.newaxis, ...]).astype(np.float32)

        dummy_action = env.action_space.sample()
        dummy_action = (dummy_action[np.newaxis, ...]).astype(np.float32)

        # Initialize network shapes by a run
        latent = self._encoder(dummy_obs)
        self._policy(latent)
        self._dualqnet(latent, dummy_action)
        self._target_dualqnet(latent, dummy_action)
        self._target_dualqnet.set_weights(self._dualqnet.get_weights())

    def _preprocess(self, observations, frozen=False):
        return observations / 255.

    def sample_action(self, observation, frozen=False):

        observation = self._preprocess(observation, frozen=frozen)

        if len(observation.shape) < 4:
            observation = np.expand_dims(observation, axis=0)

        assert len(observation.shape) == 4

        latent = self._encoder((observation))
        action, _ = self._policy.sample_action(latent)
        action = action.numpy()[0]

        info = {
            "alpha": tf.exp(self.log_alpha).numpy()
        }

        return self._postprocess(action), info

    def sample_actions(self, observations, frozen=False):

        observations = self._preprocess(observations, frozen=frozen)

        assert len(observations.shape) == 4

        latents = self._encoder(np.atleast_2d(observations))
        actions, _ = self._policy.sample_action(latents)
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

        obss, actions, rewards, next_obss, dones = self.replay_buffer.get_minibatch(self.batch_size)

        # Get alpha
        alpha = tf.math.exp(self.log_alpha)

        # Update Q function
        next_latents = self._encoder(next_obss)
        next_actions, next_logprobs = self._policy.sample_action(next_latents)

        target_q1, target_q2 = self._target_dualqnet(next_latents, next_actions)

        target = rewards + (1 - dones) * self.gamma * (
                tf.minimum(target_q1, target_q2) + -1 * alpha * next_logprobs
        )
        with tf.GradientTape() as tape:
            latents = self._encoder(obss)
            q1, q2 = self._dualqnet(latents, actions)
            loss_1 = tf.reduce_mean(tf.square(target - q1))
            loss_2 = tf.reduce_mean(tf.square(target - q2))
            loss = 0.5 * loss_1 + 0.5 * loss_2

        variables = self._dualqnet.trainable_variables
        grads = tape.gradient(loss, variables)
        self._dualqnet.optimizer.apply_gradients(zip(grads, variables))

        # Update policy
        latents = self._encoder(obss)  # redundant?
        with tf.GradientTape() as tape:
            selected_actions, logprobs = self._policy.sample_action(latents)
            q1, q2 = self._dualqnet(latents, selected_actions)
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
            (1 - self.tau) * np.array(self._target_dualqnet.get_weights(), dtype="object")
            + self.tau * np.array(self._dualqnet.get_weights(), dtype="object")
        )




