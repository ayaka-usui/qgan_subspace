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
"""Cost and Fidelity Functions"""

import torch

from config import CFG


def compute_cost(dis, final_target_state: torch.Tensor, final_gen_state: torch.Tensor, config=CFG) -> float:
    """Calculate the cost function. Which is basically equivalent to the Wasserstein distance.

    Args:
        dis (Discriminator): the discriminator.
        final_target_state (np.ndarray): the target state to input into the Discriminator.
        final_gen_state (np.ndarray): the gen state to input into the Discriminator.
        config (Config): training configuration (defaults to CFG).
    Returns:
        float: the cost function.
    """
    A, B, psi, phi = dis.get_dis_matrices_rep()

    # Calculate the terms for the cost function
    final_gen_state = final_gen_state.flatten()
    final_target_state = final_target_state.flatten()
    A_final_gen_state = A @ final_gen_state
    B_final_gen_state = B @ final_gen_state

    term1 = torch.vdot(final_gen_state, A_final_gen_state) * torch.vdot(final_target_state, B @ final_target_state)

    term2 = torch.vdot(B_final_gen_state, final_target_state) * torch.vdot(final_target_state, A_final_gen_state)

    term3 = torch.vdot(A_final_gen_state, final_target_state) * torch.vdot(final_target_state, B_final_gen_state)

    term4 = torch.vdot(B_final_gen_state, final_gen_state) * torch.vdot(final_target_state, A @ final_target_state)

    psiterm = torch.trace(torch.outer(final_target_state, final_target_state.conj()) @ psi)
    phiterm = torch.trace(torch.outer(final_gen_state, final_gen_state.conj()) @ phi)

    regterm = (config.lamb / torch.e) * (config.cst1 * term1 - config.cst2 * (term2 + term3) + config.cst3 * term4)

    # The final loss must be a real-valued scalar tensor
    loss = (psiterm - phiterm - regterm).real
    return loss


def compute_fidelity(final_target_state: torch.Tensor, final_gen_state: torch.Tensor) -> float:
    """Calculate the fidelity between target state and gen state

    Args:
        final_target_state (np.ndarray): The final target state of the system.
        final_gen_state (np.ndarray): The final gen state of the system.

    Returns:
        float: the fidelity between the target state and the gen state.
    """
    braket_result = torch.vdot(final_target_state.flatten(), final_gen_state.flatten())
    # .item() extracts the scalar value from the tensor
    return torch.abs(braket_result).pow(2).item()


def compute_fidelity_and_cost(dis, final_target_state: torch.Tensor, final_gen_state: torch.Tensor) -> tuple[float, float]:
    """Calculate the fidelity and cost function

    Args:
        dis (Discriminator): the discriminator.
        final_target_state (np.ndarray): the target state.
        final_gen_state (np.ndarray): the gen state.

    Returns:
        tuple[float, float]: the fidelity and cost function.
    """
    fidelity = compute_fidelity(final_target_state, final_gen_state)
    cost = compute_cost(dis, final_target_state, final_gen_state)

    return fidelity, cost
