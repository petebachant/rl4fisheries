import json
import os

class ConstEsc:
    def __init__(self, escapement=0, **kwargs):
        self.escapement = escapement
        self.policy_type = "constant_escapement"

    def predict(self, population):
        stabilizer_delta = 1e-6
        if population <= self.escapement:
            return 0
        else:
            return (population - self.escapement) / (population + stabilizer_delta)

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
        
        