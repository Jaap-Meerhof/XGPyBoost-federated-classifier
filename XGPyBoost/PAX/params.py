from dataclasses import dataclass

@dataclass
class Params:
    #__slots__ = ['n_trees', 'max_depth', 'eta', 'lam', 'alpha', 'gamma', 'min_child_weight', 'max_delta_step']
    n_trees: int
    max_depth: int = 6
    eta: float = 0.3
    lam: float = 1.0
    alpha: float = 0.0
    gamma: float = 0.0
    min_child_weight: float = 1.0
    max_delta_step : float = 0.0