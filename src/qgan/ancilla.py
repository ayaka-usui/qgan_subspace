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


# TODO: Add a test to cover that this projects correct subspace (last qubit is |0>).
def project_ancilla_zero(state: np.ndarray) -> tuple[np.ndarray, float]:
    """Project the last qubit onto |0> and renormalize. Assumes state is a column vector.

    Args:
        state (np.ndarray): The quantum state vector to project.

    Returns:
        np.ndarray: The projected state vector, normalized, with the ancilla qubit removed.
        float: The probability of the ancilla being in state |0>.
    """
    state = np.asarray(state).flatten()

    # Remove the ancilla qubit: keep only even indices:
    projected = state[::2]

    # Compute the norm of the projected state:
    norm = np.linalg.norm(projected)
    if norm == 0:  # Return the system part (without ancilla) as zeros
        return np.zeros((2 ** (CFG.system_size * 2), 1)), 0.0

    # Renormalize if needed:
    if CFG.ancilla_project_norm:
        projected = projected / norm
    elif CFG.ancilla_project_norm in ["pass", "penalize"]:
        pass
    else:
        raise ValueError(f"Unknown ancilla_project_norm: {CFG.ancilla_project_norm}")

    return projected.reshape(-1, 1), norm**2


# TODO: Think better what to do with this function... (how to use it)
# Right now, maybe its a cool way to codify if passing the "norm" or not.
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
    return sampled_state.reshape(-1, 1)


def get_final_fake_state_for_discriminator(total_output_state: np.ndarray) -> np.ndarray:
    """Return the fake state to be passed to the discriminator, according to ancilla_mode.

    Args:
        total_output_state (np.ndarray): The output state from the generator.

    Returns:
        np.ndarray: The final state to be passed to the discriminator.
    """
    norm = None
    total_final_state = total_output_state
    if CFG.extra_ancilla:
        if CFG.ancilla_mode == "pass":
            # Pass ancilla to discriminator (current behavior)
            return total_final_state
        if CFG.ancilla_mode == "project":
            projected, norm = project_ancilla_zero(total_final_state)
            return np.matrix(projected)
        if CFG.ancilla_mode == "trace":
            traced = np.matrix(trace_out_ancilla(total_final_state))
            return traced
        raise ValueError(f"Unknown ancilla_mode: {CFG.ancilla_mode}")
    return total_final_state, norm


def get_final_real_state_for_discriminator(total_output_state: np.ndarray) -> np.ndarray:
    """Return the real state to be passed to the discriminator, according to ancilla_mode.

    Args:
        total_output_state (np.ndarray): The output state from the target.

    Returns:
        np.ndarray: The final state to be passed to the discriminator.
    """
    total_final_state = total_output_state
    if CFG.extra_ancilla:
        n = CFG.system_size * 2 + 1  # total qubits (system + ancilla)
        if CFG.ancilla_mode == "pass":
            # Pass ancilla to discriminator (current behavior)
            return total_final_state
        if CFG.ancilla_mode in ["project", "trace"]:
            return total_final_state[: 2 ** (n - 1)]  # Return only the system part
        raise ValueError(f"Unknown ancilla_mode: {CFG.ancilla_mode}")
    return total_final_state
