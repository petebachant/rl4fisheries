import json
import os

class CautionaryRule:
    def __init__(self, x1=0, x2=1, y2=1, **kwargs):
        self.x1 = x1
        self.x2 = x2
        self.y2 = y2
        self.policy_type = "CautionaryRule_piecewise_linear"

        assert x1 < x2, "CautionaryRule error: x1 < x2" 

    def predict(self, population):
        if population < self.x1:
            return 0
        elif self.x1 < population < self.x2:
            return self.y2 * (population - self.x1) / (self.x2 - self.x1)
        else:
            return self.y_2

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