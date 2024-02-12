import json
import os

class Msy:
    def __init__(self, mortality=0, threshold=0, **kwargs):
        self.mortality = mortality
        self.threshold = threshold
        self.policy_type = "msy_and_threshold"

    def predict(self, population):
        if population < self.threshold:
            return 0
        else:
            return self.mortality

    def save(self, path = None)->None:       
        path = path or os.path.join(f'{self.policy_type}.json')
        with open(path, 'w') as f:
            json.dump(
                self.__dict__, 
                f,
            )
    
    @classmethod
    def load(self, path):
        assert path[-5:] == '.json', f"load error: msy.load(path) can only load json files currently!"

        with open(path, "r") as f:
            data = json.load(f)
            assert isinstance(data, dict), f"Target file is of type {type(data)}, it should be dict."

        # self.mortality = data.get("mortality")

        return Msy(**data)
        
        