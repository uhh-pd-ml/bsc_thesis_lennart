from typing import Callable, Optional
import numpy as np

def align_events(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_particles = max(len(a), len(b))
    larger_event = a if len(a) > len(b) else b
    a_aligned = np.concatenate((a, np.zeros((n_particles-len(a), 3))))
    b_aligned = np.concatenate((b, np.zeros((n_particles-len(b), 3))))
    a_aligned[len(a):, 1:3] = larger_event[len(a):, 1:3]
    b_aligned[len(b):, 1:3] = larger_event[len(b):, 1:3]    
    return a_aligned, b_aligned

def interpol_linear(a: np.ndarray, b: np.ndarray, n_points: int, radius: Optional[float] = None) -> tuple[Callable[[int], np.array]]:
    """
    Per-particle linear interpolation
    """
    # Sort both events by pt
    a = a[np.argsort(a[:, 0])][::-1]
    b = b[np.argsort(b[:, 0])][::-1]

    a, b = align_events(a, b)

    distance = np.linalg.norm(a - b)
    
    def _interpolate(step: float):
        lamb = step/(n_points-1)
        if distance and radius is not None:
            lamb *= radius/distance
        lamb = min(lamb, 1)
        return (1-lamb)*a + lamb*b
        
    return distance, _interpolate
