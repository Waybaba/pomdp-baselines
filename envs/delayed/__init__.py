from gym.envs.registration import register
import gym

### Below are pybullect (roboschool) environments, using BLT for Bullet
import pybullet_envs

"""
The observation space can be divided into several parts:
np.concatenate(
[
    z - self.initial_z, # pos
    np.sin(angle_to_target), # pos
    np.cos(angle_to_target), # pos
    0.3 * vx, # vel
    0.3 * vy, # vel
    0.3 * vz, # vel
    r, # pos
    p # pos
], # above are 8 dims
[j], # even elements [0::2] position, scaled to -1..+1 between limits
    # odd elements  [1::2] angular speed, scaled to show -1..+1
[self.feet_contact], # depends on foot_list, belongs to pos
])
"""

for delay in range(1, 16):
    register(
        "HopperBLT-Delay_{}-v0".format(delay),
        entry_point="envs.delayed.wrappers:DelayedWrapper",
        kwargs=dict(
            env0=gym.make("HopperBulletEnv-v0"),
            delay_steps=delay,
        ),
        max_episode_steps=1000,
    )
