"""Elastic distortion adapted from the MIT-licensed Pointcept implementation.

Copyright (c) 2023 Pointcept. Authors: Xiaoyang Wu and Yujia Zhang.
Source: https://github.com/Pointcept/Pointcept/blob/main/pointcept/datasets/transform.py
The smoothing/interpolation equations are unchanged; the wrapper accepts a
coordinate tensor rather than a Pointcept dictionary. See NOTICE (Pointcept section).
"""

import random
import numpy as np
import scipy
import scipy.ndimage
import scipy.interpolate
import torch


class ElasticDistortion(object):
    def __init__(self, distortion_params=None):
        self.distortion_params = (
            [[0.2, 0.4], [0.8, 1.6]] if distortion_params is None else distortion_params
        )

    @staticmethod
    def elastic_distortion(coords, granularity, magnitude):
        blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
        blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
        blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
        coords_min = coords.min(0)
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)
        for _ in range(2):
            noise = scipy.ndimage.convolve(noise, blurx, mode="constant", cval=0)
            noise = scipy.ndimage.convolve(noise, blury, mode="constant", cval=0)
            noise = scipy.ndimage.convolve(noise, blurz, mode="constant", cval=0)
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(
                coords_min - granularity, coords_min + granularity * (noise_dim - 2), noise_dim
            )
        ]
        interp = scipy.interpolate.RegularGridInterpolator(
            ax, noise, bounds_error=False, fill_value=0
        )
        coords += interp(coords) * magnitude
        return coords

    def __call__(self, coord):
        coord = coord.numpy()
        if random.random() < 0.95:
            for granularity, magnitude in self.distortion_params:
                coord = self.elastic_distortion(coord, granularity, magnitude)
        return torch.from_numpy(coord)
