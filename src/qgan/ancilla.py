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

import numpy as np

from config import CFG


def get_max_entangled_state_with_ancilla_if_needed(size: int) -> np.ndarray:
    """Get the maximally entangled state for the system size (With Ancilla if needed).

    Args:
        size (int): the size of the system.

    Returns:
        tuple[np.ndarray]: the maximally entangled states, plus ancilla if needed for generation and target.
    """
    # Generate the maximally entangled state for the system size
    state = np.zeros(2 ** (2 * size), dtype=complex)
    dim_register = 2**size
    for i in range(dim_register):
        state[i * dim_register + i] = 1.0
    state /= np.sqrt(dim_register)

    # Add ancilla qubit at the end, if needed
    initial_state_with_ancilla = np.kron(state, np.array([1, 0], dtype=complex))

    # Different conditions for gen and target:
    initial_state_for_gen = initial_state_with_ancilla if CFG.extra_ancilla else state
    initial_state_for_target = initial_state_with_ancilla if CFG.extra_ancilla and CFG.ancilla_mode == "pass" else state

    return np.asmatrix(initial_state_for_gen).T, np.asmatrix(initial_state_for_target).T


def project_ancilla_zero(state: np.ndarray, grad: bool, renormalize: bool = True) -> tuple[np.ndarray, float]:
    """Project the last qubit onto |0> and renormalize. Assumes state is a column vector.

    Args:
        state (np.ndarray): The quantum state vector to project.
        renormalize (bool): Whether to renormalize the projected state.

    Returns:
        np.ndarray: The projected state vector, normalized, with the ancilla qubit removed.
        float: The probability of the ancilla being in state |0>.
    """
    state = np.asarray(state).flatten()
    original_norm = np.linalg.norm(state)

    # FIXME: Remove this print after debugging
    print("Are we in a gradient? ", grad)
    print("Original norm, before projecting", original_norm**2)

    # Remove the ancilla qubit: keep only even indices:
    projected = state[::2]

    # Compute the norm of the projected state:
    norm = np.linalg.norm(projected)

    if norm == 0:  # Return the system part (without ancilla) as zeros
        return np.asmatrix(np.zeros((2 ** (CFG.system_size * 2), 1))), 0.0

    # Renormalize if needed:
    if renormalize:
        if CFG.ancilla_project_norm == "re-norm":
            projected = projected / norm * original_norm
        elif CFG.ancilla_project_norm != "pass":
            raise ValueError(f"Unknown ancilla_project_norm: {CFG.ancilla_project_norm}")

    # FIXME: Remove this print after debugging
    print("Probability of ancilla being in |0>: ", norm**2)

    return np.asmatrix(projected.reshape(-1, 1)), norm**2


# TODO: Think better what to do with this function... (how to use it)
def trace_out_ancilla(state: np.ndarray) -> np.ndarray:
    """Trace out the last qubit and return a sampled pure state from the reduced density matrix.

    Args:
        state (np.ndarray): The quantum state vector to trace out the ancilla.

    Returns:
        np.ndarray: The sampled pure state after tracing out the ancilla.
    """
    # state: (2**num_qubits, 1)
    state = np.asarray(state).flatten()
    # Reshape to (2**(n-1), 2) for last qubit
    state = state.reshape(-1, 2)
    # Compute reduced density matrix by tracing out last qubit
    rho_reduced = np.zeros((state.shape[0], state.shape[0]), dtype=complex)
    for i in range(2):
        rho_reduced += np.dot(state[:, i : i + 1], state[:, i : i + 1].conj().T)
    # Sample a pure state from the reduced density matrix
    eigvals, eigvecs = np.linalg.eigh(rho_reduced)
    eigvals = np.maximum(eigvals, 0)
    eigvals = eigvals / np.sum(eigvals)
    idx = np.random.choice(len(eigvals), p=eigvals)
    sampled_state = eigvecs[:, idx]
    return np.asmatrix(sampled_state.reshape(-1, 1))


def get_final_gen_state_for_discriminator(total_output_state: np.ndarray, grad: bool) -> np.ndarray:
    """Modifies the gen state to be passed to the discriminator, according to ancilla_mode.

    Args:
        total_output_state (np.ndarray): The output state from the generator.

    Returns:
        np.ndarray: The final state to be passed to the discriminator.
    """
    total_final_state = total_output_state
    if CFG.extra_ancilla:
        if CFG.ancilla_mode == "pass":
            # Pass ancilla to discriminator (current behavior)
            return total_final_state
        if CFG.ancilla_mode == "project":
            projected, _ = project_ancilla_zero(total_final_state, grad)
            return projected
        if CFG.ancilla_mode == "trace":
            return trace_out_ancilla(total_final_state)
        raise ValueError(f"Unknown ancilla_mode: {CFG.ancilla_mode}")
    return total_final_state
