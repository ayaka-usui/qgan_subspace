# Copyright 2025 GIQ, Universitat Autònoma de Barcelona
#
# Licensed under the MIT License. See the LICENSE file in the project root for the full license text.
# SPDX-License-Identifier: MIT
"""Numerical precision selected by CFG.precision ("float32" default, "float64")."""

import torch

from config import CFG


def rdtype() -> torch.dtype:
    """Real dtype for parameters (generator angles, discriminator coefficients)."""
    return torch.float64 if CFG.precision == "float64" else torch.float32


def cdtype() -> torch.dtype:
    """Complex dtype for states and operators."""
    return torch.complex128 if CFG.precision == "float64" else torch.complex64
