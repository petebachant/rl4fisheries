from itertools import product
import json
import os
import numpy as np
import polars as pl
from tqdm import tqdm
from .unit_interface import unitInterface

from rl4fisheries.agents.common import isVecObs

class CautionaryRule:
    def __init__(self, env, x1=0, x2=1, y2=1, observed_var='biomass', **kwargs):
        self.policy_type = f"CautionaryRule_{observed_var}"
        self.env = env
        self.observed_var = observed_var
        
        self.x1 = x1
        self.x2 = x2
        self.y2 = y2

        self.x1_pm1 = self.convert_to_pm1(x1) 
        self.x2_pm1 = self.convert_to_pm1(x2)
        self.y_pm1 = self.convert_to_pm1(y)

        assert x1 <= x2, "CautionaryRule error: x1 <= x2" 

    def convert_to_pm1(self, X):
        if self.observed_var == 'biomass':
            return self.env.bound * (X+1)/2
        elif self.observed_var == 'mean_wt':
            MAX_WT = self.env.parameters["max_wt"]
            MIN_WT = self.env.parameters["min_wt"]
            return MIN_WT + ((X+1) / 2) * (MAX_WT - MIN_WT)
    
    def predict(self, observation, **kwargs):
        if isVecObs(observation, self.env):
            observation = observation[0]
        raw_prediction = np.clip(self.predict_raw(observation), -1, 1)
        return np.float32([raw_prediction]), {}
    
    def predict_raw(self, observation):
        if observation < self.x1_pm1:
            return -1
        elif self.x1_pm1 <= observation < self.x2_pm1:
            return (
                -1 
                + (
                    (self.y2_pm1 + 1) 
                    * (observation - self.x1_pm1) 
                    / (self.x2_pm1 - self.x1_pm1)
                ) # -1 + (y2 - - 1) * fraction
            ),
        else:
            return self.y2_pm1

    def predict_effort(self, state):
        return (self.predict(state) + 1) / 2
    
    def state_to_pop(self, state):
        return (state + 1 ) / 2
    
