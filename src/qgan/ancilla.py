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
"""Ancilla post-processing tools."""

import torch as np

from config import CFG

np.set_default_device(CFG.device)


def get_max_entangled_state_with_ancilla_if_needed(size: int) -> np.Tensor:
    """Get the maximally entangled state for the system size (With Ancilla if needed).

    Args:
        size (int): the size of the system.

    Returns:
        tuple[np.Tensor]: the maximally entangled states, plus ancilla if needed for generation and target.
    """
    # Generate the maximally entangled state for the system size
    state = np.zeros(2 ** (2 * size), dtype=np.complex64)
    dim_register = 2**size
    for i in range(dim_register):
        state[i * dim_register + i] = 1.0
    state /= np.sqrt(np.tensor(dim_register, dtype=np.complex64))

    # Add ancilla qubit at the end, if needed
    ancilla = np.zeros(2, dtype=np.complex64)
    ancilla[0] = 1.0
    initial_state_with_ancilla = np.kron(state, ancilla)

    # Different conditions for gen and target:
    initial_state_for_gen = initial_state_with_ancilla if CFG.extra_ancilla else state
    initial_state_for_target = initial_state_with_ancilla if CFG.extra_ancilla and CFG.ancilla_mode == "pass" else state

    return initial_state_for_gen.view(-1, 1), initial_state_for_target.view(-1, 1)


def project_ancilla_zero(state: np.Tensor, renormalize: bool = True) -> tuple[np.Tensor, float]:
    """Project the last qubit onto |0> and renormalize. Assumes state is a column vector.

    Args:
        state (np.Tensor): The quantum state vector to project.
        renormalize (bool): Whether to renormalize the projected state.

    Returns:
        np.Tensor: The projected state vector, normalized, with the ancilla qubit removed.
        float: The probability of the ancilla being in state |0>.
    """
    state = state.flatten()
    # Remove the ancilla qubit: keep only even indices:
    projected = state[::2]
    # Compute the norm of the projected state:
    norm = np.linalg.norm(projected)
    if norm == 0:
        return np.zeros((2 ** (CFG.system_size * 2), 1), dtype=np.complex64), 0.0
    # Renormalize if needed:
    if renormalize:
        if CFG.ancilla_project_norm == "re-norm":
            projected = projected / norm
        elif CFG.ancilla_project_norm != "pass":
            raise ValueError(f"Unknown ancilla_project_norm: {CFG.ancilla_project_norm}")
    return projected.view(-1, 1), float((norm**2).real)


# TODO: Think better what to do with this function... (how to use it)
def trace_out_ancilla(state: np.Tensor) -> np.Tensor:
    """Trace out the last qubit and return a sampled pure state from the reduced density matrix.

    Args:
        state (np.Tensor): The quantum state vector to trace out the ancilla.

    Returns:
        np.Tensor: The sampled pure state after tracing out the ancilla.
    """
    # state: (2**num_qubits, 1)
    state = state.flatten()
    # Reshape to (2**(n-1), 2) for last qubit
    state = state.view(-1, 2)
    # Compute reduced density matrix by tracing out last qubit
    rho_reduced = np.zeros((state.shape[0], state.shape[0]), dtype=np.complex64)
    for i in range(2):
        col = state[:, i].view(-1, 1)
        rho_reduced += col @ col.conj().T
    # Sample a pure state from the reduced density matrix
    eigvals, eigvecs = np.linalg.eigh(rho_reduced)
    eigvals = np.maximum(eigvals, 0)
    eigvals = eigvals / np.sum(eigvals)
    idx = np.multinomial(eigvals.real, 1).item()
    sampled_state = eigvecs[:, idx]
    return sampled_state.view(-1, 1)


def get_final_gen_state_for_discriminator(total_output_state: np.Tensor) -> np.Tensor:
    """Modifies the gen state to be passed to the discriminator, according to ancilla_mode.

    Args:
        total_output_state (np.Tensor): The output state from the generator.

    Returns:
        np.Tensor: The final state to be passed to the discriminator.
    """
    total_final_state = total_output_state
    if CFG.extra_ancilla:
        if CFG.ancilla_mode == "pass":
            return total_final_state
        if CFG.ancilla_mode == "project":
            projected, _ = project_ancilla_zero(total_final_state)
            return projected
        if CFG.ancilla_mode == "trace":
            return trace_out_ancilla(total_final_state)
        raise ValueError(f"Unknown ancilla_mode: {CFG.ancilla_mode}")
    return total_final_state
