from typing import Optional
import numpy as np
import math
from amazing_ai.utils import rotate_jet, center_jet, flip_jet

PT_I, ETA_I, PHI_I = 0, 1, 2


def paint(target: np.ndarray, pos: tuple[int, int], brush: np.ndarray, norm: bool = False) -> np.ndarray:
    """
    Paints a brush to a 2D array
    """
    corner_0 = pos[0] - brush.shape[0] // 2
    corner_1 = pos[1] - brush.shape[1] // 2
    if corner_0 < 0:
        brush = brush[-corner_0:, :]
        corner_0 = 0
    if corner_1 < 0:
        brush = brush[:, -corner_1:]
        corner_1 = 0
    if corner_0+brush.shape[0] >= target.shape[0]:
        brush = brush[:target.shape[0]-(corner_0+brush.shape[0]), :]
    if corner_1+brush.shape[1] >= target.shape[1]:
        brush = brush[:, :target.shape[1]-(corner_1+brush.shape[1])]

    if norm and (brush_sum := brush.sum()):
        brush /= brush_sum
    target[corner_0:corner_0+brush.shape[0], corner_1:corner_1+brush.shape[1]] += brush
    return target


def pixelate(
    jet: np.ndarray,
    npix: int = 33,
    img_width: float = 1.2,
    rotate: bool = True,
    center: bool = True,
    flip: bool = False,
    norm: bool = True,
    blur: bool = False,
    sigma: float = 0.7,
    blur_size: int = 7
):
    # Augmented version of EnergyFlow package version to add rotation and flipping of jet in pre-processing
    # rotation code based on https://alyssaq.github.io/2015/computing-the-axes-or-orientation-of-a-blob/

    # the image is (img_width x img_width) in size
    pix_width = img_width / npix
    mid_pix = npix // 2
    
    jet_image = np.zeros((npix, npix), dtype=np.float16)

    # Prevent wrapping of phi. Moved to interpolation level
    # to keep the transformation continuous
    """"
    if np.average(np.abs(jet[:, PHI_I]) > 2.5):
        # Jet likely wraps in phi... make all phi positive
        jet[jet[:, PHI_I] < 0, PHI_I] += 2 * np.pi
    """

    if rotate:
        jet = rotate_jet(jet)

    if flip:
        jet = flip_jet(jet)

    # Normalize phi between [-pi, pi]
    jet[:, PHI_I] = np.mod(jet[:, PHI_I] + np.pi, 2*np.pi) - np.pi
    # jet[:, PHI_I][jet[:, PHI_I] < -np.pi] += 2 * np.pi
    # jet[:, PHI_I][jet[:, PHI_I] > np.pi] -= 2 * np.pi

    if center:
        jet = center_jet(jet)

    if not blur:
        # eta_indices = mid_pix + np.round(jet[:, ETA_I] / pix_width)
        # phi_indices = mid_pix + np.round(jet[:, PHI_I] / pix_width)
        # mask = (eta_indices >= 0) & (eta_indices < npix) & (phi_indices >= 0) & (phi_indices < npix)
        # eta_indices = eta_indices[mask].astype(int)
        # phi_indices = phi_indices[mask].astype(int)


        # construct grayscale image
        # TODO: Change to numpy.histogram2d
        # for phi_index, eta_index, pt in zip(phi_indices, eta_indices, jet[:, PT_I][mask]):
            # jet_image[phi_index, eta_index] += pt

        jet_image = np.histogram2d(
            jet[:, PHI_I],
            jet[:, ETA_I],
            weights=jet[:, PT_I],
            bins=npix,
            range=((-img_width/2, img_width/2), (-img_width/2, img_width/2))
        )[0].astype(np.float16)

    else:
        # transition to indices
        eta_indices = mid_pix + np.round(jet[:, ETA_I] / pix_width)
        phi_indices = mid_pix + np.round(jet[:, PHI_I] / pix_width)

        # delete elements outside of range
        mask = (eta_indices >= 0) & (eta_indices < npix) & (phi_indices >= 0) & (phi_indices < npix)
        for pt, eta, phi in jet[mask]:
            floored_eta_i = int(mid_pix + np.round(eta / pix_width))
            floored_phi_i = int(mid_pix + np.round(phi / pix_width))

            offset_eta = (eta % pix_width) / pix_width
            offset_phi = (phi % pix_width) / pix_width

            # Linear: https://www.desmos.com/calculator/kccwd1ql7t
            # Gaussian: https://www.desmos.com/calculator/bx49ydx5u7

            # (eta|phi)_dist in pixels
            eta_dist = np.arange(blur_size).astype(np.float64)-blur_size//2 - offset_eta
            phi_dist = np.arange(blur_size).astype(np.float64)-blur_size//2 - offset_phi
            pt_eta = np.exp(-(eta_dist/sigma)**2/2)  # / np.sqrt(2*np.pi*sigma)  will be normalized anyway
            pt_phi = np.exp(-(phi_dist/sigma)**2/2)  # / np.sqrt(2*np.pi*sigma)  will be normalized anyway
            pt_mask = np.tile(pt_eta, (blur_size, 1)) * np.tile(pt_phi, (blur_size, 1)).T
            pt_mask /= pt_mask.sum()

            paint(jet_image, (floored_phi_i, floored_eta_i), pt * pt_mask)

    # L1-normalize the pt channels of the jet image
    if norm:
        normfactor = np.sum(jet_image)
        if normfactor <= 0.0:
            raise FloatingPointError("Image had no particles!")
        else:
            jet_image /= normfactor

    return jet_image
