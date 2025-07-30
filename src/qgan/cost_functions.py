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

import numpy as np

from config import CFG

np.random.seed()


def braket(*args) -> float:
    """Calculate the braket (inner product) between two quantum states.

    Args:
        args: The arguments can be either two vectors (bra and ket), three (bra, operator, ket) or bigger (bra, operator^N, ket).

    Returns:
        float: The inner product of the two vectors.
    """
    bra, *ops, ket = args

    for op in ops:
        ket = np.matmul(op, ket)
    return np.matmul(bra.getH(), ket)


def compute_cost(dis, final_target_state: np.ndarray, final_gen_state: np.ndarray) -> float:
    """Calculate the cost function. Which is basically equivalent to the Wasserstein distance.

    Args:
        dis (Discriminator): the discriminator.
        final_target_state (np.ndarray): the target state to input into the Discriminator.
        final_gen_state (np.ndarray): the gen state to input into the Discriminator.

    Returns:
        float: the cost function.
    """
    A, B, psi, phi = dis.get_dis_matrices_rep()

    # Calculate the terms for the cost function
    psiterm = np.ndarray.item(braket(final_target_state, psi, final_target_state))
    phiterm = np.ndarray.item(braket(final_gen_state, phi, final_gen_state))

    # fmt: off
    term1 = braket(final_gen_state, A, final_gen_state) * braket(final_target_state, B, final_target_state)
    term2 = braket(final_target_state, A, final_gen_state) * braket(final_gen_state, B, final_target_state)
    term3 = braket(final_gen_state, A, final_target_state) * braket(final_target_state, B, final_gen_state)
    term4 = braket(final_target_state, A, final_target_state) * braket(final_gen_state, B, final_gen_state)

    regterm = np.ndarray.item(CFG.lamb / np.e * (CFG.cst1 * term1 - CFG.cst2 * (term2 + term3) + CFG.cst3 * term4))
    # fmt: on

    loss = np.real(psiterm - phiterm - regterm)

    return loss


def compute_fidelity(final_target_state: np.ndarray, final_gen_state: np.ndarray) -> float:
    """Calculate the fidelity between target state and gen state

    Args:
        final_target_state (np.ndarray): The final target state of the system.
        final_gen_state (np.ndarray): The final gen state of the system.

    Returns:
        float: the fidelity between the target state and the gen state.
    """
    braket_result = braket(final_target_state, final_gen_state)
    return np.abs(np.ndarray.item(braket_result)) ** 2
    # return np.abs(np.asscalar(np.matmul(target_state.getH(), total_final_state))) ** 2


def compute_fidelity_and_cost(dis, final_target_state: np.ndarray, final_gen_state: np.ndarray) -> tuple[float, float]:
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
