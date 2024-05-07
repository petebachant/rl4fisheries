import gymnasium as gym
import numpy as np

from rl4fisheries import AsmEnv

class AsmCRLike(AsmEnv):
    """ observe mean weight, decide on a CR-like policy for biomass. """
    def __init__(self, render_mode = 'rgb_array', config={}):
        super().__init__(render_mode=render_mode, config=config)
        assert config.get("observation_fn_id", "observe_2o") == "observe_2o", (
            "AsmCRLike only compatible with observe_2o observation function atm, sorry!"
        )
        self.action_space = gym.spaces.Box(
            np.array(3 * [-1], dtype=np.float32),
            np.array(3 * [1], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            np.array([-1], dtype=np.float32),
            np.array([1], dtype=np.float32),
            dtype=np.float32,
        )
        
    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return np.array([obs[1]]), info

    def step(self, action):
        self.update_vuls()
        self.update_ssb()
        #
        x1, x2, y2 = self.unnormalize_action(action)
        self.state, reward = self.harvest(x1, x2, y2)
        #
        self.update_vuls()
        self.update_ssb()
        #
        self.state = self.population_growth()
        self.timestep += 1
        terminated = bool(self.timestep >= self.n_year)
        # 
        observation = self.observe()
        #
        return observation, reward, terminated, False, {}

    def unnormalize_action(self, action):
        x1 = self.bound * (action[0] + 1) / 2
        x2 = self.bound * (action[1] + 1) / 2
        y2 = (action[2] + 1) / 2
        return np.float32([x1,x2,y2])

    def harvest(self, x1, x2, y2):
        if  self.surv_vul_b < x1:
            intensity = 0
        elif x1 <= self.surv_vul_b < x2:
            intensity = y2 * (self.surv_vul_b - x1) / (x2 - x1)
        else: # vul_b >= x2
            intensity = y2

        f_yield = self.harv_vul_b * intensity
        new_state = self.parameters["s"] * self.state * (1 - self.parameters["harvest_vul"] * intensity)

        return new_state, reward

        










            
    