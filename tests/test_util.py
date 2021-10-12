
import numpy as np

from dm_control import suite

from util.util import DMC2GymWrapper, SimpleReplayBufferPixel, Experience


def test_wrapper():
    env_dmc = suite.load("cartpole", "balance")
    env = DMC2GymWrapper(env_dmc, max_step=1000, is_vision=False)

    obs = env.reset()

    assert len(obs) == 5


def test_wrapper_vision():
    env_dmc = suite.load("cartpole", "balance")
    env = DMC2GymWrapper(env_dmc, max_step=1000, is_vision=True, im_size=84, frame_stack=3)

    obs = env.reset()

    assert obs.shape == (84, 84, 3 * 3)
    assert 1.0 < np.max(obs) < 255.0
    assert 0.0 < np.min(obs)


def test_experience():
    exp = Experience(np.array([0]), np.array([0]), 0, np.array([0]), False)

    assert exp.state == np.array([0])
    assert exp.action == np.array([0])
    assert exp.reward == 0
    assert exp.next_state == np.array([0])
    assert exp.done is False


def test_pixel_buffer():
    rb = SimpleReplayBufferPixel()

    env_dmc = suite.load("cartpole", "balance")
    env = DMC2GymWrapper(env_dmc, max_step=1000, is_vision=True, im_size=84, frame_stack=3)
    obs = env.reset()

    exp = Experience(obs, np.array([0, 0]), 0, obs, False)

    for i in range(100):
        rb.push(exp)

    o_, a, r, next_o_, dones = rb.get_minibatch(50)

    assert o_.shape == (50, 84, 84, 9)
    assert a.shape == (50, 2)
    assert next_o_.shape == (50, 84, 84, 9)
    assert r.shape == (50, 1)
    assert dones.shape == (50, 1)
