from typing import Callable, Optional
from energyflow.emd import emd
import numpy as np

__all__ = ("interpol_emd", )


def make_exchange_particles(e0: np.ndarray, e1: np.ndarray, R: float = 1.2):
    distance, flow = emd(e0, e1, R=R, return_flow=True)
    
    etaphi_start = e0[:, np.newaxis, 1:3].repeat(len(e1), axis=1)
    etaphi_end = e1[np.newaxis, :, 1:3].repeat(len(e0), axis=0)

    exchange_start = np.dstack([
        flow[:len(e0), :len(e1), np.newaxis],
        etaphi_start
    ])
    exchange_end = np.dstack([
        flow[:len(e0), :len(e1), np.newaxis],
        etaphi_end
    ])
        
    exchange_start = exchange_start.reshape(-1, 3)
    exchange_end = exchange_end.reshape(-1, 3)
    
    if e0[:, 0].sum() > e1[:, 0].sum():
        exchange_start = np.concatenate([
            exchange_start,
            np.hstack([flow[:, -1, np.newaxis], e0[:, 1:3]])
        ])
        exchange_end = np.concatenate([
            exchange_end,
            np.hstack([0*flow[:, -1, np.newaxis], e0[:, 1:3]])
        ])
    elif e0[:, 0].sum() < e1[:, 0].sum():
        exchange_start = np.concatenate([
            exchange_start,
            np.hstack([0*flow[-1, :, np.newaxis], e1[:, 1:3]])
        ])
        exchange_end = np.concatenate([
            exchange_end,
            np.hstack([flow[-1, :, np.newaxis], e1[:, 1:3]])
        ])
    
    return distance, exchange_start, exchange_end

def interpol_emd(e0: np.ndarray, e1: np.ndarray, n_points: int, R: float = 1.2, emd_radius: Optional[float] = None) -> tuple[float, Callable[[int], np.array]]:
    """
    This implementation should work the same as in emd_energyflow.py
    but perform faster

    emd_radius:
        Set a fixed radius in which to interpolate
    """
    distance, exchange_start, exchange_end = make_exchange_particles(e0, e1, R=R)
    
    def _interpolate(step: float):
        lamb = step/(n_points-1)
        if distance and emd_radius is not None:
            lamb *= emd_radius/distance
        lamb = min(lamb, 1)
        event = (1-lamb)*exchange_start + lamb*exchange_end
        return event[event[:, 0] > 0]
        
    return distance, _interpolate
