import gymnasium as gym
import numpy as np
from numbers import Number
from typing import Optional
import warning

class unitInterface:
    def __init__(
        self, 
        env: Optional[gym.Env] = None, 
        bounds: Optional[np.ndarray, Number, list] = None,
    ):
        if env and bound:
            warning.warn("`env` and `bound` provided to unitInterface(). `bound` value will override the obs_bound value of the env.")

        if not (env or bound):
            raise Warning("unitInterface() initializer needs either an `env` or a `bound` argument.")
        
        self.env = env
        self.bound = bound

        self.ignore_env = False
        if bound:
            self.ignore_env = True
        
        tmp_bound = bound or env.obs_bound
        self.bound_used = self.preprocess(tmp_bound)


    def to_natural_units(self, vec: np.ndarray):
        """ [-1, 1] space to [0, bound] space. """
        assert isinstance(vec, np.ndarray), "unitInterface.to_natural_units() `vec` argument must be an np.ndarray!"
        return self.bound_used * (vec + 1 ) / 2

    def to_norm_units(self, vec: np.ndarray):
        """ [0, bound] to [-1, 1] space. """
        assert isinstance(vec, np.ndarray), "unitInterface.to_norm_units() `vec` argument must be an np.ndarray!"
        return vec / self.bound_used
        self.bound_used * (vec + 1 ) / 2

    def preprocess(self, bound):
        if isinstance(bound, Number):
            assert bound > 0, "unitInteface() `bound` cannot be zero!"
            return np.array([bound])

        if isinstance(bound, list):
            assert len(bound) > 0, "unitInteface() `bound` cannot be an empty list!"
            assert min(bound) > 0, "unitInterface() `bound` entries must be strictly positive."
            return np.array(bound)

        if isinstance(bound, np.ndarray):
            assert len(bound) > 0, "unitInteface() `bound` cannot be an empty array!"
            assert min(bound) > 0, "unitInterface() `bound` entries must be strictly positive."
            return bound

        else:
            if self.ignore_env:
                raise Warning("unitInterface() `bound` argument: `bound` not in any of the allowed formats (list, number, np.ndarray)")
            else:
                raise Warning("unitInterface() `env` argument: `env.obs_bound` not in any of the allowed formats (list, number, np.ndarray)")
            