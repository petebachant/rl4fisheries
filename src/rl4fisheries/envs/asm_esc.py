import gymnasium as gym
import numpy as np

from rl4fisheries import AsmEnv
from rl4fisheries.envs.asm_fns import observe_mwt

class AsmEnvEsc(AsmEnv):
    """ actions are escapement levels (instead of exploitation rates). """
    def __init__(self, render_mode = 'rgb_array', config={}):
        super().__init__(render_mode=render_mode, config=config)

    def reset(self, *, seed=None, options=None):
        return super().reset(seed=seed, options=options)

    def step(self, action):
        self.update_vuls()
        self.update_ssb()
        #
        escapement = self.escapement_units(action)
        current_pop = self.population_units()
        if current_pop <= 0:
            mortality = 0
        else:
            mortality = (current_pop - escapement) / current_pop
        self.state, reward = self.harvest(mortality)
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

    def escapement_units(self, action):
        return self.bound * (action + 1) / 2