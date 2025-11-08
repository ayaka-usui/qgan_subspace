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
"""Including base components and definition of quantum gates using PyTorch."""

import torch
import torch.nn as nn

# --- Device Configuration ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA device selected.")
elif torch.backends.mps.is_available():
    device = torch.device("cpu")
    print("MPS device found, but CPU device selected since operator 'aten::linalg_matrix_exp' is not currently implemented for the MPS device..")
else:
    device = torch.device("cpu")
    print("CPU device selected.")

# Use torch.complex64 for complex numbers
COMPLEX_TYPE = torch.complex64

# --- Base Gates and Pauli Matrices as Tensors (on the selected device) ---
I = torch.eye(2, dtype=COMPLEX_TYPE, device=device)
X = torch.tensor([[0, 1], [1, 0]], dtype=COMPLEX_TYPE, device=device)
Y = torch.tensor([[0, -1j], [1j, 0]], dtype=COMPLEX_TYPE, device=device)
Z = torch.tensor([[1, 0], [0, -1]], dtype=COMPLEX_TYPE, device=device)
Hadamard = (torch.tensor([[1, 1], [1, -1]], dtype=COMPLEX_TYPE, device=device) / (2**0.5))

# --- Projectors ---
zero = torch.tensor([[1, 0], [0, 0]], dtype=COMPLEX_TYPE, device=device)
one = torch.tensor([[0, 0], [0, 1]], dtype=COMPLEX_TYPE, device=device)

# --- Two-Qubit Gates ---
CNOT_base = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=COMPLEX_TYPE, device=device)
SWAP = torch.tensor([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=COMPLEX_TYPE, device=device)
CZ = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]], dtype=COMPLEX_TYPE, device=device)


def Identity(n: int, device=None, dtype=torch.complex64) -> torch.Tensor:
    """Creates a 2^n x 2^n identity matrix efficiently on the given device and dtype."""
    dim = 2 ** n
    return torch.eye(dim, dtype=dtype, device=device)

def _construct_operator(size: int, op_map: dict[int, torch.Tensor]) -> torch.Tensor:
    """Helper to construct a multi-qubit operator from a map of ops and their qubit indices."""
    ops = [op_map.get(i, I) for i in range(size)]
    matrix = ops[0]
    for i in range(1, size):
        matrix = torch.kron(matrix, ops[i])
    return matrix

# --- Rotation Gate Implementations to match original file ---

def XX_Rotation(size: int, qubit1: int, qubit2: int, param: torch.Tensor) -> torch.Tensor:
    P = _construct_operator(size, {qubit1: X, qubit2: X})
    return torch.matrix_exp(-1j * param.to(P.device) * P)

def YY_Rotation(size: int, qubit1: int, qubit2: int, param: torch.Tensor) -> torch.Tensor:
    P = _construct_operator(size, {qubit1: Y, qubit2: Y})
    return torch.matrix_exp(-1j * param.to(P.device) * P)

def ZZ_Rotation(size: int, qubit1: int, qubit2: int, param: torch.Tensor) -> torch.Tensor:
    P = _construct_operator(size, {qubit1: Z, qubit2: Z})
    return torch.matrix_exp(0.5j * param.to(P.device) * P) # Note the different factor

def X_Rotation(size: int, qubit: int, param: torch.Tensor) -> torch.Tensor:
    P = _construct_operator(size, {qubit: X})
    return torch.matrix_exp(-0.5j * param.to(P.device) * P)

def Y_Rotation(size: int, qubit: int, param: torch.Tensor) -> torch.Tensor:
    P = _construct_operator(size, {qubit: Y})
    return torch.matrix_exp(-0.5j * param.to(P.device) * P)

def Z_Rotation(size: int, qubit: int, param: torch.Tensor) -> torch.Tensor:
    P = _construct_operator(size, {qubit: Z})
    return torch.matrix_exp(-0.5j * param.to(P.device) * P)

def Global_phase(size: int, param: torch.Tensor) -> torch.Tensor:
    phase = torch.exp(-1j * param.to(device)**2)
    return phase * torch.eye(2**size, dtype=COMPLEX_TYPE, device=device)

def mCNOT(size: int, control: int, target: int) -> torch.Tensor:
    """General multi-qubit CNOT gate, equivalent to expan_2qubit_gate."""
    if control == target or control >= size or target >= size:
        raise IndexError("Invalid control or target qubit index.")

    # Projector onto the |0> subspace of the control qubit
    P0 = _construct_operator(size, {control: zero})

    # Projector onto the |1> subspace of the control qubit, applying X to the target
    P1 = _construct_operator(size, {control: one, target: X})
    
    return P0 + P1

class QuantumGate(nn.Module):
    """A quantum gate as a PyTorch Module. The angle is a trainable parameter."""
    def __init__(self, name: str, qubit1: int, qubit2: int = None, angle: float = 0.0):
        super().__init__()
        self.name = name
        self.qubit1 = qubit1
        self.qubit2 = qubit2

        # Make the angle a trainable parameter if the gate is parameterized
        self.is_parameterized = name not in ["CNOT"]
        if self.is_parameterized:
            self.angle = nn.Parameter(torch.tensor(float(angle), dtype=torch.float32))
        else:
            self.angle = None

    def forward(self, size: int) -> torch.Tensor:
        """Returns the matrix representation of the gate."""
        if self.name == "XX":
            return XX_Rotation(size, self.qubit1, self.qubit2, self.angle)
        elif self.name == "YY":
            return YY_Rotation(size, self.qubit1, self.qubit2, self.angle)
        elif self.name == "ZZ":
            return ZZ_Rotation(size, self.qubit1, self.qubit2, self.angle)
        elif self.name == "X":
            return X_Rotation(size, self.qubit1, self.angle)
        elif self.name == "Y":
            return Y_Rotation(size, self.qubit1, self.angle)
        elif self.name == "Z":
            return Z_Rotation(size, self.qubit1, self.angle)
        elif self.name == "G":
            return Global_phase(size, self.angle)
        elif self.name == "CNOT":
            return mCNOT(size, self.qubit1, self.qubit2)
        else:
            raise ValueError(f"Gate '{self.name}' is not defined.")