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


def project_ancilla_zero(state: np.ndarray, renormalize: bool = True) -> tuple[np.ndarray, float]:
    """Project the last qubit onto |0> and renormalize. Assumes state is a column vector.

    Args:
        state (np.ndarray): The quantum state vector to project.
        renormalize (bool): Whether to renormalize the projected state.

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
    if renormalize:
        if CFG.ancilla_project_norm == "re-norm":
            projected = projected / norm
        elif CFG.ancilla_project_norm != "pass":
            raise ValueError(f"Unknown ancilla_project_norm: {CFG.ancilla_project_norm}")

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


def get_ancilla_reduced_density_matrix(total_output_state: np.ndarray) -> np.ndarray:
    """Return the reduced density matrix of the last qubit (ancilla).

    The input state is assumed to be a pure state vector for the full system,
    with the ancilla stored as the last qubit.

    Args:
        total_output_state (np.ndarray): Full output (pure and vector) state vector from the generator.

    Returns:
        np.ndarray: The 2x2 reduced density matrix of the ancilla.
    """
    state = np.asarray(total_output_state).flatten()
    if state.size % 2 != 0:
        raise ValueError("The total output state dimension must be even to trace out the ancilla qubit.")

    reshaped = state.reshape(-1, 2)
    rho = reshaped.conj().T @ reshaped
    tr = np.trace(rho)
    tr = np.real_if_close(tr, tol=1000)

    if not np.isclose(tr, 1.0, rtol=1e-2, atol=1e-2):
        raise ValueError(f"Reduced density matrix trace must be close to 1, but got {tr!r}.")

    rho = rho / tr
    # Enforce Hermiticity against numerical noise
    return (rho + rho.conj().T) / 2


def compute_ancilla_entanglement_entropy(total_output_state: np.ndarray) -> float:
    """Compute the von Neumann entanglement entropy of the ancilla qubit.

    The standard convention is used: S(rho) = -Tr(rho log_2(rho)).

    Args:
        total_output_state (np.ndarray): Full output (pure and vector) state vector from the generator.

    Returns:
        float: Entanglement entropy of the ancilla.

    Raises:
        ValueError: If eigenvalues fall outside the valid range [0, 1].
    """
    rho_ancilla = get_ancilla_reduced_density_matrix(total_output_state)
    eigvals = np.real(np.linalg.eigvalsh(rho_ancilla))

    # Check eigenvalues are in valid range [0, 1]
    if np.any(eigvals < -1e-10) or np.any(eigvals > 1.0 + 1e-10):
        raise ValueError(f"Eigenvalues of reduced density matrix must be in [0, 1], but got: {eigvals}")

    # Clamp to valid range to handle floating point errors near boundaries
    eigvals = np.clip(eigvals, 0.0, 1.0)
    non_zero = eigvals > 1e-12
    if not np.any(non_zero):
        return 0.0
    return float(-np.sum(eigvals[non_zero] * np.log2(eigvals[non_zero])))


def get_final_gen_state_for_discriminator(total_output_state: np.ndarray) -> np.ndarray:
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
            projected, _ = project_ancilla_zero(total_final_state)
            return projected
        if CFG.ancilla_mode == "trace":
            return trace_out_ancilla(total_final_state)
        raise ValueError(f"Unknown ancilla_mode: {CFG.ancilla_mode}")
    return total_final_state
