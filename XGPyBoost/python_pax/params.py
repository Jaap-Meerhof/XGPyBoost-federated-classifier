from dataclasses import dataclass
from objectives import softprob
from sketchtype import Sketch_type
@dataclass
class Params:
    #__slots__ = ['n_trees', 'max_depth', 'eta', 'lam', 'alpha', 'gamma', 'min_child_weight', 'max_delta_step', 'eA', 'objective', 'n_bins', 'n_participants', 'sketch_type']
    n_trees: int
    max_depth: int = 6
    eta: float = 0.3
    lam: float = 1.0
    alpha: float = 0.0
    gamma: float = 0.0
    min_child_weight: float = 1.0
    max_delta_step : float = 0.0
    eA:float = 0.1
    objective:object = softprob
    n_bins:int = 255
    n_participants:int = 5 # DEL?
    num_class:int = 100 # DEL?
    sketch_type:Sketch_type = Sketch_type.NORMAL

    def prettyprint(self):
        print(f"n_trees={self.n_trees}")
        print(f"max_depth={self.max_depth}")
        print(f"eta={self.eta:.2f}")
        print(f"lam={self.lam:.2f}")
        print(f"alpha={self.alpha:.2f}")
        print(f"gamma={self.gamma:.2f}")
        print(f"min_child_weight={self.min_child_weight:.2f}")
        print(f"max_delta_step={self.max_delta_step:.2f}")
        print(f"n_bins={self.n_bins}")
        print(f"n_participants={self.n_participants}")
        print(f"num_class={self.num_class}")
    
    def prettytext(self) -> str:
        pretty = ""
        pretty += f"n_trees={self.n_trees}\n"
        pretty += f"max_depth={self.max_depth}\n"
        pretty += f"eta={self.eta:.2f}\n"
        pretty += f"lam={self.lam:.2f}\n"
        pretty += f"alpha={self.alpha:.2f}\n"
        pretty += f"gamma={self.gamma:.2f}\n"
        pretty += f"min_child_weight={self.min_child_weight:.2f}\n"
        pretty += f"max_delta_step={self.max_delta_step:.2f}\n"
        pretty += f"n_bins={self.n_bins}\n"
        pretty += f"n_participants={self.n_participants}\n"
        pretty += f"num_class={self.num_class}\n"
        return pretty
