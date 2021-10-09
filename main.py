import wandb
from dm_control import suite

from util.sac import SoftActorCriticAgent, training
from util.util import DMC2GymWrapper

project_name = 'cmc_dmc_vision'
entity = 'ugo-nama-kun'
config_experiment = {
    "env": ["cartpole", "balance"],
    "algo": "sac-fnn",
    "max_experience": 10 ** 6,
    "min_experiences": 512,
    "update_period": 5,
    "gamma": 0.99,
    "tau": 0.01,
    "batch_size": 256,
}

wandb.init(project=project_name, entity=entity)


def make_dmc_gym_env():
    env_dmc = suite.load("cartpole", "balance")
    return DMC2GymWrapper(env_dmc, max_step=1000)


config = wandb.config
config.update(config)

env = make_dmc_gym_env()
test_env = make_dmc_gym_env()

agent = SoftActorCriticAgent(env=env,
                             max_experiences=config_experiment["max_experience"],
                             min_experiences=config_experiment["min_experiences"],
                             update_period=config_experiment["update_period"],
                             gamma=config_experiment["gamma"],
                             tau=config_experiment["tau"],
                             batch_size=config_experiment["batch_size"])

training(agent, env, test_env, n_steps=3 * 10**6, evaluate_every=1000, n_test=3)

print("done.")
