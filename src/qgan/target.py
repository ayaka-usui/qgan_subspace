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
"""Target hamiltonian module — PennyLane rewrite.

Replaces the manual Kronecker-product term_* functions with PennyLane
Pauli operators, and uses qml.matrix(qml.exp(H)) for the unitary.
The original code built Hamiltonians manually with Kronecker products of
2×2 Pauli matrices, PennyLane replaces all of that with qml.PauliX(i) @ qml.PauliZ(j)
syntax, and qml.exp(H, coeff=-1j) for the matrix exponential.

Unchanged:
    get_target_unitary(target_type, size) 
    get_final_target_state(input_state)    

This module constructs the "answer" that the generator is trying to learn.
Given a Hamiltonian H, we compute the time-evolution operator U = e^{-iH}
and apply it to one half of a maximally entangled state (Choi representation).
"""

import sys

import numpy as np
import pennylane as qml

from config import CFG


# -- PAULI OPERATOR BUILDER ----------------------------------------
# Used by _pauli_word to translate strings like "XZX" into operator chains
# Same logic as in Generator.py
_PAULI_MAP = {
    "X": qml.PauliX,
    "Y": qml.PauliY,
    "Z": qml.PauliZ,
}


def _pauli_word(pauli_string: str, qubits: list[int]) -> qml.operation.Operator:
    """Build a PennyLane tensor-product Pauli operator from a string.

    This is the PennyLane replacement for all the original term_* functions.
    Instead of manually building Kronecker products of 2×2 matrices,
    we use PennyLane to handle the tensor structure via the @ operator.

    Meaning that:
        _pauli_word("XZX", [0, 1, 2])  ->  PauliX(0) @ PauliZ(1) @ PauliX(2)

    Moreover, PennyLane automatically handles the identity on qubits not mentioned.
    So PauliZ(0) @ PauliZ(1) on a 3-qubit system is implicitly Z \otimes Z \otimes I.

    Args:
        pauli_string: String of Pauli labels, e.g. "XZX", "ZZ", "X".
        qubits: List of qubit indices, same length as pauli_string.
    Returns:
        PennyLane operator representing the tensor product.
    """
    if len(pauli_string) != len(qubits):
        raise ValueError(
            f"Pauli string '{pauli_string}' has {len(pauli_string)} chars "
            f"but {len(qubits)} qubits were given."
        )

    # Build each single-qubit Pauli and chain them with @
    ops = [_PAULI_MAP[ch](q) for ch, q in zip(pauli_string, qubits)]
    result = ops[0]
    for op in ops[1:]:
        result = result @ op
    return result


# -- HAMILTONIAN BUILDER ------------------------
def _build_hamiltonian(size: int, terms: list[str],
                       strengths: list[float]) -> qml.Hamiltonian:
    """Build a PennyLane Hamiltonian from term strings and strengths.

    This is the PennyLane replacement for the original construct_target().
    The original looped over term types, built Kronecker products manually,
    and summed numpy matrices. Here we build a qml.Hamiltonian which is
    a symbolic sum of weighted Pauli operators.

    For each term, we iterate over all valid positions (nearest-neighbour).

    Supports: "I", "X", "Y", "Z" (1-body, all sites),
              "XX", "XZ", "ZZ" (2-body, nearest neighbours),
              "ZZZ", "XZX" (3-body, nearest neighbours),
              "ZZZZ", "XXXX" (4-body, nearest neighbours).

    Args:
        size: Number of qubits.
        terms: List of term labels.
        strengths: Coefficient for each term.

    Returns:
        qml.Hamiltonian
    """
    coeffs = []
    ops = []

    for strength, term in zip(strengths, terms):
        if term == "I":
            # Identity on full space: just a scalar shift to the energy.
            # H += strength · I  (doesn't change eigenstates, only shifts eigenvalues)
            coeffs.append(strength)
            ops.append(qml.Identity(0))

        elif len(term) == 1 and term in _PAULI_MAP:
            # 1-body term on every qubit: e.g. "X" -> \sum_i X_i 
            for i in range(size):
                coeffs.append(strength)
                ops.append(_PAULI_MAP[term](i))

        elif len(term) == 2:
            # 2-body nearest-neighbour: e.g. "ZZ" -> \sum_i Z_i·Z_{i+1} 
            for i in range(size - 1):
                coeffs.append(strength)
                ops.append(_pauli_word(term, [i, i + 1]))

        elif len(term) == 3:
            # 3-body nearest-neighbour: e.g. "XZX" -> \sum_i X_i·Z_{i+1}·X_{i+2} 
            for i in range(size - 2):
                coeffs.append(strength)
                ops.append(_pauli_word(term, [i, i + 1, i + 2]))

        elif len(term) == 4:
            # 4-body nearest-neighbour: e.g. "XXXX" -> \sum_i X_i·X_{i+1}·X_{i+2}·X_{i+3}
            # Used in surface code stabilizers
            for i in range(size - 3):
                coeffs.append(strength)
                ops.append(_pauli_word(term, [i, i + 1, i + 2, i + 3]))

        else:
            raise ValueError(f"Unknown or unsupported term '{term}'.")

    return qml.Hamiltonian(coeffs, ops)


def _hamiltonian_to_unitary(H: qml.Hamiltonian, size: int) -> np.ndarray:
    """Compute U = e^{-iH} from a PennyLane Hamiltonian.

    This replaces scipy.linalg.expm(-1j * H_matrix) from the original code.
    PennyLane's qml.exp computes the operator exponential symbolically,
    and qml.matrix converts it to a numpy array, so it can be used for 
    discriminator or cost function.

    The wire_order=range(size) ensures the matrix rows/columns follow
    the standard qubit ordering (qubit 0 = most significant bit).

    Uses qml.matrix(qml.exp(H, -1j)) to get the matrix representation
    of the time-evolution operator.

    Args:
        H: PennyLane Hamiltonian.
        size: Number of qubits (to set the wire order).

    Returns:
        np.ndarray of shape (2^size, 2^size).
    """
    return qml.matrix(qml.exp(H, coeff=-1j), wire_order=range(size))


# -- PREDEFINED HAMILTONIANS ------------------------
def _cluster_hamiltonian(size: int) -> qml.Hamiltonian:
    """Cluster Hamiltonian:  H = \sum_i X_iZ_{i+1}X_{i+2} + \sum_i Z_i
    """
    coeffs, ops = [], []
    # XZX terms: 3-body stabilizer-like interactions
    for i in range(size - 2):
        coeffs.append(1.0)
        ops.append(_pauli_word("XZX", [i, i + 1, i + 2]))
    # Z field on all qubits: local magnetic field
    for i in range(size):
        coeffs.append(1.0)
        ops.append(qml.PauliZ(i))
    return qml.Hamiltonian(coeffs, ops)


def _ising_hamiltonian(size: int) -> qml.Hamiltonian:
    """Transverse-field Ising:  H = -\sum_i Z_iZ_{i+1} - \sum_i X_i
    """
    coeffs, ops = [], []
    for i in range(size - 1):
        # ZZ coupling between neighbours
        coeffs.append(-1.0)
        ops.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))
        # X field on each qubit (except last, added separately)
        coeffs.append(-1.0)
        ops.append(qml.PauliX(i))
    # Last X term (the loop above misses the last qubit's X term)
    coeffs.append(-1.0)
    ops.append(qml.PauliX(size - 1))
    return qml.Hamiltonian(coeffs, ops)


def _rotated_surface_code_hamiltonian(size: int) -> qml.Hamiltonian:
    """Rotated surface code Hamiltonian (squared qubits only).

    The surface code is a topological error-correcting code.
    The Hamiltonian encodes the stabilizer generators:
      - X plaquettes (XXXX): detect Z errors
      - Z plaquettes (ZZZZ): detect X errors
      - Boundary terms (XX, ZZ): stabilizers at the edges

    Only supports 2×2 (4 qubits) and 3×3 (9 qubits) grids because
    the qubit connectivity is hardcoded for these specific layouts.

    For 4 qubits (2×2 grid):
        Q0 — Q1
        |    |
        Q2 — Q3

    For 3 qubits (3x3 grid):
        Q0 - Q1 - Q2
        |    |    |
        Q3 - Q4 - Q5
        |    |    |
        Q7 - Q8 - Q9
    """
    coeffs, ops = [], []

    if size == 4:
        # Single X plaquette covering all 4 qubits
        coeffs.append(-1.0)
        ops.append(_pauli_word("XXXX", [0, 1, 2, 3]))
        # Z boundary terms on edges
        coeffs.append(-1.0)
        ops.append(qml.PauliZ(0) @ qml.PauliZ(1))
        coeffs.append(-1.0)
        ops.append(qml.PauliZ(2) @ qml.PauliZ(3))

    elif size == 9:
        # X plaquettes (detect Z errors) — two interior plaquettes
        coeffs.append(-1.0)
        ops.append(_pauli_word("XXXX", [0, 1, 3, 4]))  # top-left plaquette
        coeffs.append(-1.0)
        ops.append(_pauli_word("XXXX", [4, 5, 7, 8]))  # bottom-right plaquette
        # X edges (boundary stabilizers)
        coeffs.append(-1.0)
        ops.append(qml.PauliX(2) @ qml.PauliX(5))  # right edge
        coeffs.append(-1.0)
        ops.append(qml.PauliX(3) @ qml.PauliX(6))  # left edge
        # Z plaquettes (detect X errors) — two interior plaquettes
        coeffs.append(-1.0)
        ops.append(_pauli_word("ZZZZ", [1, 2, 4, 5]))  # top-right plaquette
        coeffs.append(-1.0)
        ops.append(_pauli_word("ZZZZ", [3, 4, 6, 7]))  # bottom-left plaquette
        # Z edges (boundary stabilizers)
        coeffs.append(-1.0)
        ops.append(qml.PauliZ(0) @ qml.PauliZ(1))  # top edge
        coeffs.append(-1.0)
        ops.append(qml.PauliZ(7) @ qml.PauliZ(8))  # bottom edge
    else:
        sys.exit("Rotated surface code only supports size 4 or 9.")

    return qml.Hamiltonian(coeffs, ops)


# -- MAIN PUBLIC FUNCTIONS ------------------------------------------------
def get_target_unitary(target_type: str, size: int) -> np.ndarray:
    """Get the target unitary U = e^{-iH} based on the target type.

    The function dispatches to the appropriate Hamiltonian builder,
    then converts H -> U = e^{-iH} via matrix exponentiation.

    Args:
        target_type: One of "cluster_h", "rotated_surface_h", "ising_h", "custom_h".
        size: Number of qubits.

    Returns:
        np.ndarray of shape (2^size, 2^size).
    """
    # Build the Hamiltonian as a symbolic PennyLane operator
    if target_type == "cluster_h":
        H = _cluster_hamiltonian(size)
    elif target_type == "rotated_surface_h":
        H = _rotated_surface_code_hamiltonian(size)
    elif target_type == "ising_h":
        H = _ising_hamiltonian(size)
    elif target_type == "custom_h":
        # Custom Hamiltonian: terms and strengths from config
        H = _build_hamiltonian(size, CFG.custom_hamiltonian_terms,
                               CFG.custom_hamiltonian_strengths)
    else:
        raise ValueError(
            f"Unknown target type: {target_type}. "
            f"Expected 'cluster_h', 'rotated_surface_h', 'ising_h', or 'custom_h'."
        )

    # Convert Hamiltonian -> unitary: U = e^{-iH}
    return _hamiltonian_to_unitary(H, size)


def get_final_target_state(final_input_state: np.ndarray) -> np.ndarray:
    """Apply the target unitary to the Choi input state.

    Computes <I  \otimes U_target [\otimes I_ancilla]) |\Phi^+ >

    This uses the Choi-Jamiolkowski isomorphism. We apply I \otimes U to one
    half of a maximally entangled state. The resulting "Choi state" uniquely
    represents the channel and is what the discriminator compares against.

    If ancilla is present in "pass" mode, we extend with I_{ancilla} because
    the ancilla wire passes through to the discriminator unchanged.

    Args:
        final_input_state: The maximally entangled input state (column vector).

    Returns:
        np.ndarray: Target state as column vector, ready for the discriminator.
    """
    # Get the target unitary U = e^{-iH}
    target_unitary = get_target_unitary(CFG.target_hamiltonian, CFG.system_size)

    # I on Choi register \otimes U_target on system register
    # This creates a (2^{2N} × 2^{2N}) matrix
    identity_choi = np.eye(2 ** CFG.system_size)
    target_op = np.kron(identity_choi, target_unitary)

    # Extend with identity on ancilla wire if needed
    # Only when ancilla is present AND in "pass" mode (ancilla reaches discriminator)
    # In "project" or "trace" modes, the target doesn't need the ancilla dimension
    # because the ancilla is removed before reaching the discriminator
    if CFG.extra_ancilla and CFG.ancilla_mode == "pass":
        target_op = np.kron(target_op, np.eye(2))  # I_2 for one ancilla qubit

    # Apply the target operator to the input state: |target> = (I \otimes U)|\Phi^+ >
    return np.asmatrix(np.matmul(target_op, final_input_state))