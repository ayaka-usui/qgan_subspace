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
"""Target hamiltonian module using PyTorch."""

import sys
import torch
from config import CFG
from tools.qobjects.qgates import (
    I, Identity, X, Y, Z,
    _construct_operator, device, COMPLEX_TYPE
)

##############################################################
# HAMILTONIAN CUSTOM TERMS (Using the new PyTorch helper)
##############################################################

def term_X(size, *qubits):
    return _construct_operator(size, {q: X for q in qubits})

def term_Y(size, *qubits):
    return _construct_operator(size, {q: Y for q in qubits})

def term_Z(size, *qubits):
    return _construct_operator(size, {q: Z for q in qubits})

def term_XX(size, q1, q2):
    return _construct_operator(size, {q1: X, q2: X})

def term_XZ(size, q1, q2):
    return _construct_operator(size, {q1: X, q2: Z})

def term_ZZ(size, q1, q2):
    return _construct_operator(size, {q1: Z, q2: Z})

def term_ZZZ(size, q1, q2, q3):
    return _construct_operator(size, {q1: Z, q2: Z, q3: Z})

def term_ZZZZ(size, q1, q2, q3, q4):
    return _construct_operator(size, {q1: Z, q2: Z, q3: Z, q4: Z})

def term_XZX(size, q1, q2, q3):
    return _construct_operator(size, {q1: X, q2: Z, q3: X})

def term_XXXX(size, q1, q2, q3, q4):
    return _construct_operator(size, {q1: X, q2: X, q3: X, q4: X})


##################################################################
# PREDEFINED TARGETS
##################################################################

def construct_target(size: int, terms: list[str], strengths: list[float]) -> torch.Tensor:
    """Construct target Hamiltonian from a list of terms and strengths."""
    H = torch.zeros((2**size, 2**size), dtype=COMPLEX_TYPE, device=device)
    for i, term in enumerate(terms):
        s = strengths[i]
        if term == "I":
            H += s * torch.eye(2**size, dtype=COMPLEX_TYPE, device=device)
        elif term == "X":
            for i in range(size): H += s * term_X(size, i)
        elif term == "Y":
            for i in range(size): H += s * term_Y(size, i)
        elif term == "Z":
            for i in range(size): H += s * term_Z(size, i)
        elif term == "XX":
            for i in range(size-1): H += s * term_XX(size, i, i+1)
        elif term == "XZ":
            for i in range(size-1): H += s * term_XZ(size, i, i+1)
        elif term == "ZZ":
            for i in range(size-1): H += s * term_ZZ(size, i, i+1)
        elif term == "ZZZ":
            for i in range(size-2): H += s * term_ZZZ(size, i, i+1, i+2)
        elif term == "ZZZZ":
            for i in range(size-3): H += s * term_ZZZZ(size, i, i+1, i+2, i+3)
        elif term == "XZX":
            for i in range(size-2): H += s * term_XZX(size, i, i+1, i+2)
        elif term == "XXXX":
            for i in range(size-3): H += s * term_XXXX(size, i, i+1, i+2, i+3)
        else:
            raise ValueError(f"Unknown term '{term}' in custom_hamiltonian_terms.")
    return torch.matrix_exp(-1j * H)

def construct_clusterH(size: int) -> torch.Tensor:
    """Construct cluster Hamiltonian."""
    H = torch.zeros((2**size, 2**size), dtype=COMPLEX_TYPE, device=device)
    for i in range(size - 2):
        H += term_XZX(size, i, i + 1, i + 2)
        H += term_Z(size, i)
    H += term_Z(size, size - 2)
    H += term_Z(size, size - 1)
    return torch.matrix_exp(-1j * H)

def construct_RotatedSurfaceCode(size: int) -> torch.Tensor:
    """Construct rotated surface code Hamiltonian."""
    H = torch.zeros((2**size, 2**size), dtype=COMPLEX_TYPE, device=device)
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
    return torch.matrix_exp(-1j * H)

def construct_ising(size: int) -> torch.Tensor:
    """Construct Ising Hamiltonian."""
    H = torch.zeros((2**size, 2**size), dtype=COMPLEX_TYPE, device=device)
    for i in range(size - 1):
        H += -term_ZZ(size, i, i + 1)
        H += -term_X(size, i)
    H += -term_X(size, size - 1)
    return torch.matrix_exp(-1j * H)


##############################################################
# MAIN FUNCTIONS FOR TARGET HAMILTONIAN
##############################################################

def get_target_unitary(target_type: str, config = CFG) -> torch.Tensor:
    """Get the target unitary based on the target type and size."""
    if target_type == "cluster_h":
        return construct_clusterH(config.system_size)
    if target_type == "rotated_surface_h":
        return construct_RotatedSurfaceCode(config.system_size)
    if target_type == "ising_h":
        return construct_ising(config.system_size)
    if target_type == "custom_h":
        return construct_target(config.system_size, config.custom_hamiltonian_terms, config.custom_hamiltonian_strengths)
    raise ValueError(f"Unknown target type: {target_type}. Expected 'cluster_h', 'rotated_surface_h', or 'custom_h'.")

def get_final_target_state(final_input_state: torch.Tensor, config = CFG) -> torch.Tensor:
    """Initialize the target state by applying the target unitary."""
    target_unitary = get_target_unitary(config.target_hamiltonian, config)

    # Ensure the unitary is on the same device as the input state
    target_unitary = target_unitary.to(final_input_state.device)

    target_op = torch.kron(Identity(config.system_size), target_unitary)
    if config.extra_ancilla and config.ancilla_mode == "pass":
        target_op = torch.kron(target_op, Identity(1))
        
    return torch.matmul(target_op, final_input_state)