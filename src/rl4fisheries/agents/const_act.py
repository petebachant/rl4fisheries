import numpy as np

class ConstantAction:
    def __init__(self, env, action):
        self.env = env
        self.action=action
        
    def predict(self, observation):
        return self.action, {}
    #
    def action_to_mortality(self, action):
        return (self.action + 1 ) / 2
    def action_to_escapement(self, action):
        return self.env.bound * (self.action + 1 ) / 2

def ConstAct(*args, **kwargs):
    from warnings import warn
    warn("Name change: "
         "ConstAct -> ConstantAction. "
         "This old name still works but will be eventually discontinued."
        )
    return ConstantAction(*args, **kwargs)