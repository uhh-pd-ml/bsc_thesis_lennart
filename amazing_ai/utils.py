import warnings
import numpy as np
import h5py
from typing import Literal, Union, Tuple

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
    # TODO: Refactor to normalize_event
    if jet.ndim != 2:
        jet = jet.reshape(-1, 3)

    # Remove particles without momentum
    jet = jet[jet[:, 0] > 0]

    # jet[:, PHI_I] = (jet[:, PHI_I]) % np.pi - np.pi/2
    if np.average(np.abs(jet[:, PHI_I])) > 2.5:
        # Jet likely wraps in phi... make all phi positive
        jet[jet[:, PHI_I] < 0, PHI_I] += 2 * np.pi

    # jet[:, PHI_I] = np.mod(jet[:, PHI_I] + np.pi, 2*np.pi) - np.pi
    # jet[:, PHI_I] = (jet[:, PHI_I] + np.pi/2) % np.pi - np.pi/2
    # jet[:, PHI_I] = (jet[:, PHI_I]) % np.pi - np.pi/2

    # TODO: arcsin(sin(phi)) is probably pretty terrible
    # jet[:, PHI_I] = np.arcsin(np.sin(jet[:, PHI_I]))
    # Same as arcsin(sin(phi)) but 2x faster
    jet[:, PHI_I] = np.pi/2-np.abs(np.mod(jet[:, PHI_I]+np.pi/2, 2*np.pi) - np.pi)

    return jet

def cutoff_jet(jet: np.ndarray, cutoff_log: float = 1.) -> np.ndarray:
    # TODO: Refactor to cutoff_event
    return jet[jet[:, PT_I] > np.exp(cutoff_log)]

def rotate_jet(jet: np.ndarray, return_theta: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, bool]]:
    # TODO: Refactor to rotate_event
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

    rotated_event = np.column_stack([
        jet[:, PT_I],
        eta_transformed,
        phi_transformed
    ])

    if return_theta:
        return rotated_event, theta
    return rotated_event

def center_jet(jet: np.ndarray, return_center: bool = False):
    # TODO: Refactor to center_event
    eta_avg = np.average(jet[:, ETA_I], weights=jet[:,PT_I])
    phi_avg = np.average(jet[:, PHI_I], weights=jet[:,PT_I])
    jet[:, ETA_I] -= eta_avg
    jet[:, PHI_I] -= phi_avg
    if return_center:
        return jet, np.array((eta_avg, phi_avg))
    return jet

def flip_jet(jet: np.ndarray, return_flip: bool = False):
    # TODO: Refactor to flip_event
    argmax_pt = np.argmax(jet[:, PT_I])
    ptmax_eta, ptmax_phi = jet[argmax_pt, ETA_I], jet[argmax_pt, PHI_I]
    if ptmax_eta < 0:
        jet[:, ETA_I] *= -1.0
    if ptmax_phi < 0:
        jet[:, PHI_I] *= -1.0
    
    if return_flip:
        return jet, (ptmax_eta < 0, ptmax_phi < 0)
    return jet
