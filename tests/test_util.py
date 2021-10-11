from dm_control import suite

from util.util import DMC2GymWrapper


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
