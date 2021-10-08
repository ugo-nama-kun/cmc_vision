import copy

import numpy as np

from dm_control import suite
import matplotlib.pyplot as plt

random_state = np.random.RandomState(42)

env = suite.load("cartpole", "balance")
spec = env.action_spec()

duration = 4  # Seconds
frames = []
ticks = []
rewards = []
observations = []

while env.physics.data.time < duration:
    action = random_state.uniform(spec.minimum, spec.maximum, spec.shape)
    time_step = env.step(action)

    camera0 = env.physics.render(camera_id=0, height=200, width=200)
    camera1 = env.physics.render(camera_id=1, height=200, width=200)
    frames.append(np.hstack((camera0, camera1)))
    rewards.append(time_step.reward)
    observations.append(copy.deepcopy(time_step.observation))
    ticks.append(env.physics.data.time)

    print(env.physics.data.time, time_step)

plt.subplot(2, 1, 1)
plt.imshow(camera0)
plt.subplot(2, 1, 2)
plt.imshow(camera1)
