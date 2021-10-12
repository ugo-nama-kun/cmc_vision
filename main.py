from datetime import datetime

import wandb
import tensorflow as tf

from dm_control import suite

from util.sac import SoftActorCriticAgent
from util.sac_raw import SoftActorCriticAgentRawPixel
from util.util import DMC2GymWrapper, training

# Trial params
agent_type = "raw_pixel"
gpu_id = 0

# Experiment Params
n_steps = 10 ** 6
evaluate_every = 3000
n_test = 5

env_id = ["cartpole", "balance"]

config_experiment = {
    "env": env_id,
    "algo": "sac-fnn",
    "max_experience": 10 ** 5,
    "min_experiences": 1000,
    "update_period": 2,
    "gamma": 0.99,
    "tau": 0.01,
    "batch_size": 512,
}

running_name = None
time_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
is_vision = True
if agent_type == "lowdim":
    running_name = agent_type + "-" + time_stamp
    is_vision = False
elif agent_type == "raw_pixel":
    running_name = agent_type + "-" + time_stamp

if running_name is None:
    raise ValueError("agent type is invalid")

#########################################
# Main content
#########################################

available_gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(available_gpus))
if available_gpus:
    try:
        tf.config.experimental.set_visible_devices(available_gpus[gpu_id], "GPU")
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(available_gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e)

project_name = 'cmc_dmc_vision'
entity = 'ugo-nama-kun'

wandb.init(project=project_name,
           entity=entity,
           group=config_experiment["env"][0],
           name=running_name)
config = wandb.config
config.update(config)


def make_dmc_gym_env(is_vision):
    env_dmc = suite.load("cartpole", "balance")
    return DMC2GymWrapper(env_dmc, max_step=1000, is_vision=is_vision, im_size=84, frame_stack=3)


env = make_dmc_gym_env(is_vision)
test_env = make_dmc_gym_env(is_vision)

if agent_type == "lowdim":
    agent = SoftActorCriticAgent(env=env,
                                 max_experiences=config_experiment["max_experience"],
                                 min_experiences=config_experiment["min_experiences"],
                                 update_period=config_experiment["update_period"],
                                 gamma=config_experiment["gamma"],
                                 tau=config_experiment["tau"],
                                 batch_size=config_experiment["batch_size"])
elif agent_type == "raw_pixel":
    agent = SoftActorCriticAgentRawPixel(env=env,
                                         max_experiences=config_experiment["max_experience"],
                                         min_experiences=config_experiment["min_experiences"],
                                         update_period=config_experiment["update_period"],
                                         gamma=config_experiment["gamma"],
                                         tau=config_experiment["tau"],
                                         batch_size=config_experiment["batch_size"])
else:
    raise ValueError("invalid agent type")

training(agent, env, test_env, n_steps=n_steps, evaluate_every=evaluate_every, n_test=n_test)

print("done.")
