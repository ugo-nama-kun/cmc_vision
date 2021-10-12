from datetime import datetime
import argparse

import wandb
import tensorflow as tf

from dm_control import suite

from util.sac import SoftActorCriticAgent
from util.sac_raw import SoftActorCriticAgentRawPixel
from util.util import DMC2GymWrapper, training


parser = argparse.ArgumentParser(description="Pixel Agent RL")

parser.add_argument("agent_type", help="agent type: [lowdim, raw_pixel]", type=str)
parser.add_argument("--gpu", help="GPU ID if available", default=0, type=int)
parser.add_argument("--domain", help="environment (domain) name, like cartpole", default="cartpole", type=str)
parser.add_argument("--task", help="task option, like balance", default="balance", type=str)
""" domain-task names
acrobot swingup
acrobot swingup_sparse
ball_in_cup catch
cartpole balance
cartpole balance_sparse
cartpole swingup
cartpole swingup_sparse
cheetah run
finger spin
finger turn_easy
finger turn_hard
fish upright
fish swim
hopper stand
hopper hop
humanoid stand
humanoid walk
humanoid run
manipulator bring_ball
pendulum swingup
point_mass easy
reacher easy
reacher hard
swimmer swimmer6
swimmer swimmer15
walker stand
walker walk
walker run
"""

args = parser.parse_args()

# Trial params
agent_type = args.agent_type
gpu_id = 0

# Experiment Params
n_steps = 10 ** 6
evaluate_every = 3000
n_test = 5

env_id = [args.domain, args.task]

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


print(f"domain: {args.domain}, task: {args.task}, agent_type: {args.agent_type}")


def make_dmc_gym_env(is_vision):
    env_dmc = suite.load(args.domain, args.task)
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
