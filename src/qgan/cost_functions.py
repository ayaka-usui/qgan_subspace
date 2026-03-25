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
"""Cost and Fidelity Functions — PyTorch version.

Reduced from the original cost_functions.py:
    - braket()                    -> REMOVED (replaced by torch.vdot in dis/gen compute_loss)
    - compute_fidelity()          -> kept, torch version
    - compute_fidelity_and_cost() -> kept, calls dis.compute_loss

Since the discriminator and generator now compute their own losses
internally via torch.trace and autograd, this module only needs to
provide evaluation metrics for logging during training.
"""

import torch
import numpy as np
from config import CFG

def compute_fidelity(final_target_state: torch.Tensor,
                     final_gen_state: torch.Tensor) -> float:
    """Calculate the fidelity between target and generated states.

    Fidelity = |<target|gen>|^2 (for pure states)

    Args:
        final_target_state: Target state, shape (d,) or (d, 1).
        final_gen_state: Generator state, shape (d,) or (d, 1).

    Returns:
        float: fidelity ∈ [0, 1].
    """
    t = final_target_state.reshape(-1)
    g = final_gen_state.reshape(-1)
    overlap = torch.vdot(t, g) #better to use .vdot than .dot(t.conj(), g)
    return float(torch.abs(overlap) ** 2)

def compute_fidelity_and_cost(dis, final_target_state: torch.Tensor,
                               final_gen_state: torch.Tensor) -> tuple[float, float]:
    """Calculate both fidelity and cost for logging.

    Args:
        dis: Discriminator (torch version).
        final_target_state: Target state.
        final_gen_state: Generator state.

    Returns:
        (fidelity, cost): both as floats.
    """
    fidelity = compute_fidelity(final_target_state, final_gen_state)
    with torch.no_grad():
        neg_cost = dis.compute_loss(final_target_state, final_gen_state)

    cost = float(-neg_cost)
    return fidelity, cost