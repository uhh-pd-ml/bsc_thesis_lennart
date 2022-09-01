from typing import Callable
import numpy as np

def interpol_linear(a: np.ndarray, b: np.ndarray, n_points: int) -> tuple[Callable[[int], np.array]]:
    """
    Per-particle linear interpolation
    """

    n_particles = max(len(a), len(b))
    a = np.concatenate([a, np.zeros((n_particles-len(a), 3))])
    b = np.concatenate([b, np.zeros((n_particles-len(b), 3))])
    distance = np.linalg.norm(a - b)
    
    def _interpolate(step: float):
        lamb = step/(n_points-1)
        return (1-lamb)*a + lamb*b
        
    return distance, _interpolate
