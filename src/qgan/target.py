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
"""Target hamiltonian module"""

import sys

import torch as np

from config import CFG
from tools.qobjects.qgates import I, Identity, X, Y, Z

np.set_default_device(CFG.device)


##############################################################
# MAIN FUNCTIONS FOR TARGET HAMILTONIAN
##############################################################
def get_target_unitary(target_type: str, size: int) -> np.Tensor:
    """Get the target unitary based on the target type and size.

    Args:
        target_type (str): Type of target Hamiltonian, either cluster_h, rotated_surface_h, ising_h, or custom_h.
        size (int): Size of the system.

    Returns:
        np.Tensor: The target unitary.
    """
    if target_type == "cluster_h":
        return construct_clusterH(size)
    if target_type == "rotated_surface_h":
        return construct_RotatedSurfaceCode(size)
    if target_type == "ising_h":
        return construct_ising(size)
    if target_type == "custom_h":
        return construct_target(size, CFG.custom_hamiltonian_terms, CFG.custom_hamiltonian_strengths)
    raise ValueError(f"Unknown target type: {target_type}. Expected 'cluster_h', 'rotated_surface_h', or 'custom_h'.")


def get_final_target_state(final_input_state):
    """Initialize the target state. Applying the target unitary to the maximally entangled state.

    And returns it ready to be used in the discriminator.

    Args:
        final_input_state (np.Tensor): the input state, which is the maximally entangled state.

    Returns:
        np.Tensor: the target state after applying the target unitary.
    """
    target_unitary = get_target_unitary(CFG.target_hamiltonian, CFG.system_size)
    target_op = np.kron(Identity(CFG.system_size), target_unitary)
    if CFG.extra_ancilla and CFG.ancilla_mode == "pass":
        target_op = np.kron(target_op, Identity(1))
    return np.matmul(target_op, final_input_state)


##################################################################
# PREDEFINED TARGETS
##################################################################


def construct_target(size: int, terms: list[str], strengths: list[float]) -> np.Tensor:
    """Construct target Hamiltonian. Specify the terms to include as a list of strings.

    Args:
        size (int): the size of the system.
        terms (list[str]): which terms to include, e.g. ["I", "X", "Y", "Z", "XX", "XZ", "ZZ", "ZZZ", "ZZZZ", "XZX", "XXXX"]
        strenghs (list[float]): the strengths of the terms, in the same order as `terms`.

    Returns:
        np.Tensor: the target Hamiltonian.
    """
    H = np.zeros([2**size, 2**size], dtype=np.complex64)
    for term_i, term in enumerate(terms):
        if term == "I":
            H += strengths[term_i] * np.identity(2**size, dtype=np.complex64)
        elif term == "X":
            for i in range(size):
                H += strengths[term_i] * term_X(size, i)
        elif term == "Y":
            for i in range(size):
                H += strengths[term_i] * term_Y(size, i)
        elif term == "Z":
            for i in range(size):
                H += strengths[term_i] * term_Z(size, i)
        elif term == "XX":
            for i in range(size - 1):
                H += strengths[term_i] * term_XX(size, i, i + 1)
        elif term == "XZ":
            for i in range(size - 1):
                H += strengths[term_i] * term_XZ(size, i, i + 1)
        elif term == "ZZ":
            for i in range(size - 1):
                H += strengths[term_i] * term_ZZ(size, i, i + 1)
        elif term == "ZZZ":
            for i in range(size - 2):
                H += strengths[term_i] * term_ZZZ(size, i, i + 1, i + 2)
        elif term == "ZZZZ":
            for i in range(size - 3):
                H += strengths[term_i] * term_ZZZZ(size, i, i + 1, i + 2, i + 3)
        elif term == "XZX":
            for i in range(size - 2):
                H += strengths[term_i] * term_XZX(size, i, i + 1, i + 2)
        elif term == "XXXX":
            for i in range(size - 3):
                H += strengths[term_i] * term_XXXX(size, i, i + 1, i + 2, i + 3)
        # Add more terms as needed
        else:
            raise ValueError(f"Unknown term '{term}' in custom_hamiltonian_terms.")
    return np.matrix_exp(-1j * H.to("cpu")).to(CFG.device)


def construct_clusterH(size: int) -> np.Tensor:
    """Construct cluster Hamiltonian.

    Args:
        size (int): the size of the system.

    Returns:
        np.Tensor: the cluster Hamiltonian.
    """
    H = np.zeros([2**size, 2**size], dtype=np.complex64)
    for i in range(size - 2):
        H += term_XZX(size, i, i + 1, i + 2)
        H += term_Z(size, i)
    H += term_Z(size, size - 2)
    H += term_Z(size, size - 1)
    return np.matrix_exp(-1j * H.to("cpu")).to(CFG.device)


def construct_RotatedSurfaceCode(size: int) -> np.Tensor:
    """Construct rotated surface code Hamiltonian.

    Args:
        size (int): the size of the system.

    Returns:
        np.Tensor: the rotated surface code Hamiltonian.
    """
    H = np.zeros([2**size, 2**size], dtype=np.complex64)

    if size == 4:
        H += -term_XXXX(size, 0, 1, 2, 3)
        H += -term_ZZ(size, 0, 1)
        H += -term_ZZ(size, 2, 3)
    elif size == 9:
        H += -term_XXXX(size, 0, 1, 3, 4)
        H += -term_XXXX(size, 4, 5, 7, 8)
        H += -term_XX(size, 2, 5)
        H += -term_XX(size, 3, 6)
        H += -term_ZZZZ(size, 1, 2, 4, 5)
        H += -term_ZZZZ(size, 3, 4, 6, 7)
        H += -term_ZZ(size, 0, 1)
        H += -term_ZZ(size, 7, 8)
    else:
        sys.exit("system size is not 2*2 or 3*3 either")

    return np.matrix_exp(-1j * H.to("cpu")).to(CFG.device)


def construct_ising(size):
    """Construct Ising Hamiltonian."""
    H = np.zeros([2**size, 2**size], dtype=np.complex64)

    for i in range(size - 1):
        H += -term_ZZ(size, i, i + 1)
        H += -term_X(size, i)
    H += -term_X(size, size - 1)

    return np.matrix_exp(-1j * H.to("cpu")).to(CFG.device)


##############################################################
# HAMILTONIAN CUSTOM TERMS
##############################################################
def term_XXXX(size: int, qubit1: int, qubit2: int, qubit3: int, qubit4: int) -> np.Tensor:
    """Construct a term for the Hamiltonian with four X gates acting on specified qubits."""
    matrix = np.eye(1, dtype=np.complex64)
    for i in range(size):
        if (qubit1 == i) or (qubit2 == i) or (qubit3 == i) or (qubit4 == i):
            matrix = np.kron(matrix, X)
        else:
            matrix = np.kron(matrix, I)
    return matrix


def term_ZZZZ(size: int, qubit1: int, qubit2: int, qubit3: int, qubit4: int) -> np.Tensor:
    """Construct a term for the Hamiltonian with four Z gates acting on specified qubits"""
    matrix = np.eye(1, dtype=np.complex64)
    for i in range(size):
        if (qubit1 == i) or (qubit2 == i) or (qubit3 == i) or (qubit4 == i):
            matrix = np.kron(matrix, Z)
        else:
            matrix = np.kron(matrix, I)
    return matrix


def term_ZZZ(size: int, qubit1: int, qubit2: int, qubit3: int) -> np.Tensor:
    """Construct a term for the Hamiltonian with three Z gates acting on specified qubits."""
    matrix = np.eye(1, dtype=np.complex64)
    for i in range(size):
        if (qubit1 == i) or (qubit2 == i) or (qubit3 == i):
            matrix = np.kron(matrix, Z)
        else:
            matrix = np.kron(matrix, I)
    return matrix


def term_XZX(size: int, qubit1: int, qubit2: int, qubit3: int) -> np.Tensor:
    """Construct a term for the Hamiltonian with X and Z gates acting on specified qubits."""
    matrix = np.eye(1, dtype=np.complex64)
    for i in range(size):
        if (qubit1 == i) or (qubit3 == i):
            matrix = np.kron(matrix, X)
        elif qubit2 == i:
            matrix = np.kron(matrix, Z)
        else:
            matrix = np.kron(matrix, I)
    return matrix


def term_XX(size: int, qubit1: int, qubit2: int) -> np.Tensor:
    """Construct a term for the Hamiltonian with two X gates acting on specified qubits."""
    matrix = np.eye(1, dtype=np.complex64)
    for i in range(size):
        if (qubit1 == i) or (qubit2 == i):
            matrix = np.kron(matrix, X)
        else:
            matrix = np.kron(matrix, I)
    return matrix


def term_ZZ(size: int, qubit1: int, qubit2: int) -> np.Tensor:
    """Construct a term for the Hamiltonian with two Z gates acting on specified qubits."""
    matrix = np.eye(1, dtype=np.complex64)
    for i in range(size):
        if (qubit1 == i) or (qubit2 == i):
            matrix = np.kron(matrix, Z)
        else:
            matrix = np.kron(matrix, I)
    return matrix


def term_XZ(size, qubit1, qubit2):
    """Construct a term for the Hamiltonian with an X gate on one qubit and a Z gate on another."""
    matrix = np.eye(1, dtype=np.complex64)
    for i in range(size):
        if qubit1 == i:
            matrix = np.kron(matrix, X)
        elif qubit2 == i:
            matrix = np.kron(matrix, Z)
        else:
            matrix = np.kron(matrix, I)
    return matrix


def term_Z(size: int, qubit1: int) -> np.Tensor:
    """Construct a term for the Hamiltonian with a single Z gate acting on a specified qubit."""
    matrix = np.eye(1, dtype=np.complex64)
    for i in range(size):
        matrix = np.kron(matrix, Z) if qubit1 == i else np.kron(matrix, I)
    return matrix


def term_X(size, qubit1):
    """Construct a term for the Hamiltonian with a single X gate acting on a specified qubit."""
    matrix = np.eye(1, dtype=np.complex64)
    for i in range(size):
        matrix = np.kron(matrix, X) if qubit1 == i else np.kron(matrix, I)
    return matrix


def term_Y(size, qubit1):
    """Construct a term for the Hamiltonian with a single Y gate acting on a specified qubit."""
    matrix = np.eye(1, dtype=np.complex64)
    for i in range(size):
        matrix = np.kron(matrix, Y) if qubit1 == i else np.kron(matrix, I)
    return matrix
