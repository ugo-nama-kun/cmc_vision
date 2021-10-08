from dm_control import suite

from util.sac import SoftActorCriticAgent
from util.util import DMC2GymWrapper


def make_dmc_gym_env():
    env_dmc = suite.load("cartpole", "balance")
    return DMC2GymWrapper(env_dmc)


config = {
    "env": "Pendulum-v0",
    "algo": "sac-fnn",
    "max_experience": 10 ** 5,
    "min_experiences": 512,
    "update_period": 4,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 256,
}

env = make_dmc_gym_env()
test_env = make_dmc_gym_env()

agent = SoftActorCriticAgent(env=env,
                             max_experiences=config["max_experience"],
                             min_experiences=config["min_experiences"],
                             update_period=config["update_period"],
                             gamma=config["gamma"],
                             tau=config["tau"],
                             batch_size=config["batch_size"])

#training(agent, env, test_env, n_steps=300 * 100, evaluate_every=1000, n_test=3)

print("done.")
