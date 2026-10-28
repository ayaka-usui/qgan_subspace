# Copyright 2025 GIQ, Universitat Autònoma de Barcelona
#
# Licensed under the MIT License. See the LICENSE file in the project root for the full license text.
# SPDX-License-Identifier: MIT
"""Optimizer file"""

import numpy as np

from config import CFG
from tools.data.data_reshapers import _flatten, _unflatten


class MomentumOptimizer:
    """Gradient descent with momentum, given an objective function to be minimized.

    The update formula is from:
    "On the importance of initialization and momentum in deep learning
    http://proceedings.mlr.press/v28/sutskever13.pdf"

    v_{t+1} = miu * v_{t} - eta * grad(theta_t)
    theta_{t+1} = theta_{t} + v_{t+1}

    Args:
        eta: learning rate > 0
        miu: momentum coefficient [0,1]
    """

    def __init__(self, eta: float = CFG.l_rate, miu: float = CFG.momentum_coeff):
        self.miu: float = miu
        self.eta: float = eta
        self.v = None

    def move_in_grad(self, theta: np.ndarray, grad_list: np.ndarray, min_or_max: str) -> np.ndarray:
        grad_list = _flatten(grad_list)
        theta_tmp = _flatten(theta)

        if min_or_max == "min":
            sign = -1.0
        elif min_or_max == "max":
            sign = 1.0
        else:
            raise ValueError("min_or_max must be either 'min' or 'max'")

        if self.v is None:
            self.v = [sign * self.eta * g for g in grad_list]
        else:
            self.v = [self.miu * v + sign * self.eta * g for v, g in zip(self.v, grad_list)]

        new_theta = [theta_i + v for theta_i, v in zip(theta_tmp, self.v)]
        new_theta = _unflatten(new_theta, theta)

        return new_theta
