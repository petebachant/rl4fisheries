from rl4fisheries import AsmEnv

class AsmAngerEnv(AsmEnv):
    def __init__(self, asm_config={}, anger_config={}):
        super().__init__(config=asm_config)
        self.anger_lvl = 1
        self.anger_config = anger_config
        self.action_threshold = anger_config.get('action_threshold', 0.01)

    def reset(self, *, seed=None, options=None):
        self.anger_lvl = 1
        return super().reset(seed=seed, options=options)
        

    def step(self, action):
        #
        # update anger
        if action < self.action_threshold: 
            self.anger_lvl += 1
        else:
            self.anger_lvl = max(self.anger_lvl-1, 1)

        obs, rew, term, trunc, info = super().step(action)
        rew = rew / self.anger_lvl
        
        return obs, rew, term, trunc, info
        
        