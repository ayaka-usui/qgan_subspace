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
from qgan.ancilla import get_final_fake_state_for_discriminator, get_final_real_state_for_discriminator

np.random.seed()


def get_final_comp_states_for_dis(gen, total_input_state: np.ndarray, total_real_state: np.ndarray) -> tuple:
    """Get the final fake state for the discriminator

    Args:
        gen (Generator): the generator.
        total_input_state (np.ndarray): the input state, which is the maximally entangled state.
        total_real_state (np.ndarray): the real state, which is the target state.

    Returns:
        tuple[np.ndarray]: the final fake and real states for the discriminator, and the fake norm
    """
    Untouched_x_G: np.ndarray = gen.get_Untouched_qubits_and_Gen()

    total_output_state: np.ndarray = np.matmul(Untouched_x_G, total_input_state)

    final_fake_state, fake_norm = get_final_fake_state_for_discriminator(total_output_state)
    final_real_state: np.ndarray = get_final_real_state_for_discriminator(total_real_state)
    return final_fake_state, final_real_state, fake_norm


def compute_cost(gen, dis, total_real_state: np.ndarray, total_input_state: np.ndarray) -> float:
    """Calculate the cost function

    Args:
        gen (Generator): the generator.
        dis (Discriminator): the discriminator.
        total_real_state (np.ndarray): the real state.
        total_input_state (np.ndarray): the input state.

    Returns:
        float: the cost function.
    """
    final_fake_state, final_real_state, norm = get_final_comp_states_for_dis(gen, total_input_state, total_real_state)
    A, B, psi, phi = dis.get_dis_matrices_rep()

    # fmt: off
    term1 = np.matmul(final_fake_state.getH(), np.matmul(A, final_fake_state))
    term2 = np.matmul(final_real_state.getH(), np.matmul(B, final_real_state))

    term3 = np.matmul(final_fake_state.getH(), np.matmul(B, final_real_state))
    term4 = np.matmul(final_real_state.getH(), np.matmul(A, final_fake_state))

    term5 = np.matmul(final_fake_state.getH(), np.matmul(A, final_real_state))
    term6 = np.matmul(final_real_state.getH(), np.matmul(B, final_fake_state))

    term7 = np.matmul(final_fake_state.getH(), np.matmul(B, final_fake_state))
    term8 = np.matmul(final_real_state.getH(), np.matmul(A, final_real_state))

    psiterm = np.trace(np.matmul(np.matmul(final_real_state, final_real_state.getH()), psi))
    phiterm = np.trace(np.matmul(np.matmul(final_fake_state, final_fake_state.getH()), phi))

    regterm = np.ndarray.item(CFG.lamb / np.e * (CFG.cst1 * term1 * term2 - CFG.cst2 * term3 * term4 - CFG.cst2 * term5 * term6 + CFG.cst3 * term7 * term8))
    # regterm = np.asscalar(CFG.lamb / np.e * (CFG.cst1 * term1 * term2 - CFG.cst2 * term3 * term4 - CFG.cst2 * term5 * term6 + CFG.cst3 * term7 * term8))
    # fmt: on

    # Penalization for ancilla qubit norm, if needed
    penalization = 0
    if CFG.extra_ancilla and CFG.ancilla_mode == "project" and CFG.ancilla_project_norm == "penalize":
        penalization = CFG.ancilla_norm_penalization * norm

    loss = np.real(psiterm - phiterm - regterm) + penalization

    return loss


def compute_fidelity(gen, total_real_state: np.ndarray, total_input_state: np.ndarray) -> float:
    """Calculate the fidelity between target state and fake state

    Args:
        gen (Generator): the generator.
        total_real_state (np.ndarray): the real state.
        total_input_state (np.ndarray): the input state.

    Returns:
        float: the fidelity between the target state and the fake state.
    """
    final_fake_state, final_real_state, _ = get_final_comp_states_for_dis(gen, total_input_state, total_real_state)

    return np.abs(np.ndarray.item(np.matmul(final_real_state.getH(), final_fake_state))) ** 2
    # return np.abs(np.asscalar(np.matmul(real_state.getH(), total_final_state))) ** 2
