import gym
from gym import spaces
import numpy as np
from queue import Queue
from copy import deepcopy
from types import SimpleNamespace



class POMDPWrapper(gym.Wrapper):
    def __init__(self, env, partially_obs_dims: list):
        super().__init__(env)
        self.partially_obs_dims = partially_obs_dims
        # can equal to the fully-observed env
        assert 0 < len(self.partially_obs_dims) <= self.observation_space.shape[0]

        self.observation_space = spaces.Box(
            low=self.observation_space.low[self.partially_obs_dims],
            high=self.observation_space.high[self.partially_obs_dims],
            dtype=np.float32,
        )

        if self.env.action_space.__class__.__name__ == "Box":
            self.act_continuous = True
            # if continuous actions, make sure in [-1, 1]
            # NOTE: policy won't use action_space.low/high, just set [-1,1]
            # this is a bad practice...
        else:
            self.act_continuous = False

    def get_obs(self, state):
        return state[self.partially_obs_dims].copy()

    def reset(self):
        state = self.env.reset()  # no kwargs
        return self.get_obs(state)

    def step(self, action):
        if self.act_continuous:
            # recover the action
            action = np.clip(action, -1, 1)  # first clip into [-1, 1]
            lb = self.env.action_space.low
            ub = self.env.action_space.high
            action = lb + (action + 1.0) * 0.5 * (ub - lb)
            action = np.clip(action, lb, ub)

        state, reward, done, info = self.env.step(action)

        return self.get_obs(state), reward, done, info


class global_config:
    def __init__(self, delay_steps):
        self.actor_input = SimpleNamespace()
        self.critic_input = SimpleNamespace()
        self.actor_input.history_merge_method = "cat_mlp"
        self.actor_input.history_num = delay_steps


class DelayedWrapper(gym.Wrapper):
    """
    ! TODO remove global_config related;
    ! TODO return last act and obs together
    """
    def __init__(self, env0: gym.Env, delay_steps=2):
        super().__init__(env0)
        self.env = env0
        self.delay_steps = delay_steps
        self.global_cfg = global_config(delay_steps)
        # cat 
        self.observation_space = spaces.Box(
            low=np.concatenate([self.env.observation_space.low, self.env.action_space.low]),
            high=np.concatenate([self.env.observation_space.high, self.env.action_space.high]),
            dtype=np.float32,
        )
        # delayed observations
        self.delayed_obs = np.zeros([delay_steps + 1] + list(self.observation_space.shape), dtype=np.float32)
        self.delay_buf = Queue(maxsize=delay_steps+1)

        # history merge
        if self.global_cfg.actor_input.history_merge_method != "none" or \
            self.global_cfg.critic_input.history_merge_method != "none":
            self.history_num = self.global_cfg.actor_input.history_num # short flag
            self.act_buf = [np.zeros(self.env.action_space.shape) for _ in range(self.history_num)]
        else:
            self.history_num = 0
        # ! TODO obs space

    def reset(self):
        res = self.env.reset()
        if isinstance(res, tuple): # (obs, {}) # discard info {}
            res = res[0]
        self.prev_act = np.zeros(self.env.action_space.shape) # ! TODO conditional set
        if self.history_num > 0:
            self.act_buf = [np.zeros(self.env.action_space.shape) for _ in range(self.history_num)]
        res = np.concatenate([res, self.prev_act], axis=0)
        return res

    def preprocess_fn(self, res, action):
        """
        preprocess the observation before the agent decision
        """
        if len(res) == 4: 
            obs_next_nodelay, reward, done, info = res
        elif len(res) == 5:
            obs_next_nodelay, reward, done, truncated, info = res
        else:
            raise ValueError("Invalid return value from env.step()")
        # operate the delayed observation queue
        self.delay_buf.put(obs_next_nodelay)
        while not self.delay_buf.full(): self.delay_buf.put(obs_next_nodelay) # make it full
        obs_next_delayed = self.delay_buf.get()
        info["obs_next_nodelay"] = obs_next_nodelay
        info["obs_next_delayed"] = obs_next_delayed
        # history merge
        if self.history_num > 0:
            prev_act = self.act_buf[-1]
            info["historical_act"] = np.concatenate(self.act_buf, axis=0)
            self.act_buf.append(action)
            self.act_buf.pop(0)
            obs_next_delayed = np.concatenate([obs_next_delayed, prev_act], axis=0)
        elif self.history_num == 0:
            raise NotImplementedError("Not implemented yet")
            info["historical_act"] = False
        return (deepcopy(obs_next_delayed), deepcopy(reward), deepcopy(done), deepcopy(info))

    def step(self, action):
        """
        make a queue of delayed observations, the size of the queue is delay_steps
        for example, if delay_steps = 2, then the queue is [s_{t-2}, s_{t-1}, s_t]
        for each step, the queue will be updated as [s_{t-1}, s_t, s_{t+1}]
        """
        res = self.env.step(action)
        return self.preprocess_fn(res, action)


    def get_obs(self):
        return self.delayed_obs[0]

    def get_oracle_obs(self):
        return self.delayed_obs[-1]

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


if __name__ == "__main__":
    import envs

    env = gym.make("HopperBLT-F-v0")
    obs = env.reset()
    done = False
    step = 0
    while not done:
        next_obs, rew, done, info = env.step(env.action_space.sample())
        step += 1
        print(step, done, info)
