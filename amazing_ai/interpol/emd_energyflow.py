from typing import Callable
from energyflow.emd import emd
import numpy as np

__all__ = ("emd_energyflow", )


def merge(G: np.ndarray, ev0: np.ndarray, ev1: np.ndarray, R: float=1, lamb: float=0.5) -> np.ndarray:
    # Taken from https://github.com/pkomiske/EnergyFlow
    # This is from examples/animation_example.py
    
    merged = []
    for i in range(len(ev0)):
        for j in range(len(ev1)):
            if G[i, j] > 0:
                merged.append([G[i,j], lamb*ev0[i,1] + (1-lamb)*ev1[j,1], 
                                       lamb*ev0[i,2] + (1-lamb)*ev1[j,2]])

    # detect which event has more pT
    if np.sum(ev0[:,0]) > np.sum(ev1[:,0]):
        for i in range(len(ev0)):
            if G[i,-1] > 0:
                merged.append([G[i,-1]*lamb, ev0[i,1], ev0[i,2]])
    elif np.sum(ev0[:,0]) < np.sum(ev1[:,0]):
        for j in range(len(ev1)):
            if G[-1,j] > 0:
                merged.append([G[-1,j]*(1-lamb), ev1[j,1], ev1[j,2]])            

    return np.asarray(merged)

def emd_energyflow(start: np.array, end: np.array, n_points: int, R: float = 1.2) -> tuple[float, Callable[[int], np.ndarray]]:
    distance, G = emd(start, end, R=R, return_flow=True)
    def interp(i: int) -> np.ndarray:
        return merge(G, start, end, R=R, lamb=1-i/(n_points-1))

    return distance, interp
