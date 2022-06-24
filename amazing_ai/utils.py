import warnings
import numpy as np
import h5py
from typing import Literal, Union

PT_I, ETA_I, PHI_I = 0, 1, 2

def split_datasets(
    data,
    train_size: Union[int, float] = 0.6,
    val_size: Union[int, float] = 0.1,
):
    data_len = len(data)

    train_end = train_size if type(train_size) == int else int(train_size * data_len)
    val_end = train_end + (
        val_size if type(train_size) == int else int(val_size * data_len)
    )

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, test_data, val_data


def get_jet(file: Union[h5py.File, str], i: int, jet: Literal[1, 2] = 1, normalize: bool = True) -> np.ndarray:
    warnings.warn("get_jet is deprecated", DeprecationWarning)
    # TODO cleanup
    if type(file) is str:
        from amazing_datasets import get_file
        file = get_file(file)
    jet = file[f"jet{jet}_PFCands"][i]

    if not normalize:
        return jet
    return normalize_jet(jet)

def normalize_jet(jet: np.ndarray) -> np.ndarray:
    if jet.ndim != 2:
        jet = jet.reshape(-1, 3)

    # jet[jet[:, PHI_I] < 0][:, PHI_I] += 2 * np.pi
    # jet[jet[:, PHI_I] < -np.pi][:, PHI_I] += 2 * np.pi
    # jet[jet[:, PHI_I] > np.pi][:, PHI_I] -= 2 * np.pi

    # Remove particles without momentum
    jet = jet[jet[:, 0] > 0]

    if np.average(np.abs(jet[:, PHI_I]) > 2.5):
        # Jet likely wraps in phi... make all phi positive
        jet[jet[:, PHI_I] < 0, PHI_I] += 2 * np.pi

    jet[:, PHI_I] = np.arcsin(np.sin(jet[:, PHI_I]))

    return jet

def cutoff_jet(jet: np.ndarray, cutoff_log: float = 1.) -> np.ndarray:
    return jet[jet[:, PT_I] > np.exp(cutoff_log)]

def rotate_jet(jet: np.ndarray) -> np.ndarray:
    # TODO: Decide on whether to do this functional or with side-effects
    if jet.shape[0] < 2:
        return jet

    coords = np.vstack([jet[:, ETA_I], jet[:, PHI_I]])
    cov = np.cov(coords, aweights=jet[:, PT_I])
    evals, evecs = np.linalg.eig(cov)

    e_max = np.argmax(evals)
    eta_v1, phi_v1 = evecs[:, e_max]  # Eigenvector with largest eigenvalue

    theta = np.arctan(eta_v1 / phi_v1)

    rotation_mat = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )

    transformed_mat = rotation_mat @ coords
    eta_transformed, phi_transformed = transformed_mat

    return np.column_stack([
        jet[:, PT_I],
        eta_transformed,
        phi_transformed
    ])

def center_jet(jet: np.ndarray) -> np.ndarray:
    # TODO: Decide on whether to do this functional or with side-effects
    eta_avg = np.average(jet[:, ETA_I], weights=jet[:,PT_I])
    phi_avg = np.average(jet[:, PHI_I], weights=jet[:,PT_I])
    jet[:, ETA_I] -= eta_avg
    jet[:, PHI_I] -= phi_avg
    return jet

def flip_jet(jet: np.ndarray) -> np.ndarray:
    # TODO: Decide on whether to do this functional or with side-effects
    argmax_pt = np.argmax(jet[:, PT_I])
    ptmax_eta, ptmax_phi = jet[argmax_pt, ETA_I], jet[argmax_pt, PHI_I]
    if ptmax_eta < 0:
        jet[:, ETA_I] *= -1.0
    if ptmax_phi < 0:
        jet[:, PHI_I] *= -1.0
    return jet
