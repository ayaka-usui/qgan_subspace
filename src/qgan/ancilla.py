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
        np.matrix: the maximally entangled state, plus ancilla if needed.
    """
    # Generate the maximally entangled state for the system size
    state = np.zeros(2 ** (2 * size), dtype=complex)
    dim_register = 2**size
    for i in range(dim_register):
        state[i * dim_register + i] = 1.0
    state /= np.sqrt(dim_register)

    # Add ancilla qubit at the end, if needed
    initial_state_with_ancilla = np.kron(state, np.array([1, 0], dtype=complex))

    initial_state = initial_state_with_ancilla if CFG.extra_ancilla else state

    return np.asmatrix(initial_state).T


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
        raise ValueError(
            "The total output state dimension must be even to trace out the ancilla qubit."
        )

    reshaped = state.reshape(-1, 2)
    rho = reshaped.conj().T @ reshaped
    tr = np.trace(rho)
    tr = np.real_if_close(tr, tol=1000)

    if not np.isclose(tr, 1.0, rtol=1e-2, atol=1e-2):
        raise ValueError(
            f"Reduced density matrix trace must be close to 1, but got {tr!r}."
        )

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
        raise ValueError(
            f"Eigenvalues of reduced density matrix must be in [0, 1], but got: {eigvals}"
        )

    # Clamp to valid range to handle floating point errors near boundaries
    eigvals = np.clip(eigvals, 0.0, 1.0)
    non_zero = eigvals > 1e-12
    if not np.any(non_zero):
        return 0.0
    return float(-np.sum(eigvals[non_zero] * np.log2(eigvals[non_zero])))


def compute_bipartite_negativity(total_output_state: np.ndarray, global_i: int, global_j: int) -> float:
    """Compute the Negativity between two qubits in the given pure state.
    
    The Negativity is defined as (|| rho_ij^{T_i} ||_1 - 1) / 2.
    
    Args:
        total_output_state (np.ndarray): Full output (pure and vector) state vector.
        global_i (int): Global index of the first qubit.
        global_j (int): Global index of the second qubit.
        
    Returns:
        float: The Negativity between the two qubits.
    """
    state = np.asarray(total_output_state).flatten()
    num_qubits = int(np.log2(state.size))
    if 2**num_qubits != state.size:
        raise ValueError("State dimension is not a power of 2.")

    keep_indices = sorted([global_i, global_j])
    trace_indices = [idx for idx in range(num_qubits) if idx not in keep_indices]

    # Reshape state to tensor
    state_tensor = state.reshape([2] * num_qubits)

    # Partial trace
    rho = np.tensordot(state_tensor, state_tensor.conj(), axes=(trace_indices, trace_indices))

    # The axes of rho are now (keep_indices[0], keep_indices[1], keep_indices[0]', keep_indices[1]')
    # Partial transpose with respect to the first subsystem (axis 0)
    # Swap axis 0 and axis 2
    rho_pt = np.transpose(rho, (2, 1, 0, 3))

    # Reshape to 4x4
    rho_pt_mat = rho_pt.reshape(4, 4)

    # Compute trace norm
    eigvals = np.linalg.eigvalsh(rho_pt_mat)

    # Negativity = sum of absolute values of negative eigenvalues
    neg_eigvals = eigvals[eigvals < -1e-12]
    return 0.0 if len(neg_eigvals) == 0 else float(np.sum(np.abs(neg_eigvals)))
