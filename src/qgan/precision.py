# Copyright 2025 GIQ, Universitat Autònoma de Barcelona
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Numerical precision selected by CFG.precision ("float32" default, "float64")."""

import torch

from config import CFG


def rdtype() -> torch.dtype:
    """Real dtype for parameters (generator angles, discriminator coefficients)."""
    return torch.float64 if CFG.precision == "float64" else torch.float32


def cdtype() -> torch.dtype:
    """Complex dtype for states and operators."""
    return torch.complex128 if CFG.precision == "float64" else torch.complex64
