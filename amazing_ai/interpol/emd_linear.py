from typing import Callable
from scipy.interpolate import interp1d
from energyflow.emd import emd
import numpy as np
import warnings

__all__ = ("emd_linear", )


def make_exchange_particles(start: np.array, end: np.array, is_start: bool) -> tuple[float, np.array]:
    emdval, G = emd(start, end, return_flow=True)

    if is_start:
        # Angles from start
        angles = start[:, np.newaxis, 1:3].repeat(end.shape[0], axis=1)
    else:
        # Angles from end
        angles = end[np.newaxis, :, 1:3].repeat(start.shape[0], axis=0)

    # Combines the exchange particles' pt with their angles
    # For the start particles the angle should be inherited from their row,
    # for the end particles the angle should be inherited from their column
    exchange: np.ndarray = np.dstack([
        # emd may also generate a line/column for energy that gets thrown away
        G[:angles.shape[0], :angles.shape[1], np.newaxis],
        angles
    ])
    exchange = exchange.reshape(exchange.shape[0]*exchange.shape[1], 3)
    exchange = exchange[exchange[:, 0] > 0]
    return emdval, exchange

def emd_linear(start: np.array, end: np.array, n_points: int) -> tuple[float, Callable[[int], np.array]]:
    warnings.warn("emd_linear is deprecated", DeprecationWarning)
    _, exchange_start = make_exchange_particles(start, end, True)
    distance, _ = exchange_end = make_exchange_particles(start, end, False)
    return distance, interp1d((0, n_points-1), [exchange_start, exchange_end], axis=0)
