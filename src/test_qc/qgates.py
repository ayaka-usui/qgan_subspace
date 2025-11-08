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
"""including base components and definition of quantum gates."""

import numpy as np
import scipy.linalg as linalg
from scipy.sparse import dok_matrix

I = np.eye(2)

# Pauli matrices
X = np.matrix([[0, 1], [1, 0]])  #: Pauli-X matrix
Y = np.matrix([[0, -1j], [1j, 0]])  #: Pauli-Y matrix
Z = np.matrix([[1, 0], [0, -1]])  #: Pauli-Z matrix
Hadamard = np.matrix([[1, 1], [1, -1]] / np.sqrt(2))  #: Hadamard gate

zero = np.matrix([[1, 0], [0, 0]])
one = np.matrix([[0, 0], [0, 1]])

# Two qubit gates
CNOT = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])  #: CNOT gate
SWAP = np.matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])  #: SWAP gate
CZ = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])  #: CZ gate

global param_table
param_table = {}


def Identity(size: int) -> np.ndarray:
    matrix = 1
    for i in range(1, size + 1):
        matrix = np.kron(matrix, I)
    return matrix


def CSWAP(size: int) -> np.ndarray:
    """Get control swap gate"""
    dim = 2 * size
    C_SWAP = dok_matrix((2 ** (dim + 1), 2 ** (dim + 1)))

    dim1 = 2**size
    SWAP = dok_matrix((dim1 * dim1, dim1 * dim1))

    for i in range(2**dim):
        C_SWAP[i, i] = 1

    for i in range(dim1):
        for j in range(dim1):
            SWAP[i * dim1 + j, j * dim1 + i] = 1
            SWAP[j * dim1 + i, i * dim1 + j] = 1
            C_SWAP[i * dim1 + j + 2**dim, j * dim1 + i + 2**dim] = 1
            C_SWAP[j * dim1 + i + 2**dim, i * dim1 + j + 2**dim] = 1
    # C_SWAP[SWAP.nonzero()] = SWAP[SWAP.nonzero()]
    return C_SWAP - np.zeros((2 ** (dim + 1), 2 ** (dim + 1))), SWAP


def CSWAP_T(size: int) -> np.ndarray:
    """Get control swap gate"""
    dim = 2 * size
    C_SWAP = dok_matrix((2 ** (dim + 1), 2 ** (dim + 1)))

    dim1 = 2**size
    SWAP = dok_matrix((dim1 * dim1, dim1 * dim1))
    # C_SWAP = np.zeros((2 ** (dim + 1), 2 ** (dim + 1)))
    # SWAP = np.zeros((dim * dim, dim * dim))

    for i in range(dim1):
        for j in range(dim1):
            SWAP[i * dim1 + j, j * dim1 + i] = 1
            SWAP[j * dim1 + i, i * dim1 + j] = 1

    C_SWAP[SWAP.nonzero()] = SWAP[SWAP.nonzero()]

    for i in range(2**dim, 2 ** (dim + 1)):
        C_SWAP[i, i] = 1

    return C_SWAP - np.zeros((2 ** (dim + 1), 2 ** (dim + 1))), SWAP


def mCNOT(size, control, target):
    gate = np.asarray(X)
    U = expan_2qubit_gate(gate, size, control, target)
    return U


def expan_2qubit_gate(gate, size, control, target):
    wires = np.asarray((control, target))

    if control > size - 1:
        raise IndexError("index is out of bound of wires")
    if target > size - 1:
        raise IndexError("index is out of bound of wires")
    if control - target == 0:
        raise IndexError("index should not be same")

    a = np.min(wires)
    b = np.max(wires)

    U_one = np.kron(Identity(control), np.kron(zero, Identity(size - control - 1)))
    if a == control:
        U_two = np.kron(
            Identity(control), np.kron(one, np.kron(Identity(b - a - 1), np.kron(gate, Identity(size - target - 1))))
        )
    else:
        U_two = np.kron(
            Identity(target), np.kron(gate, np.kron(Identity(a - b - 1), np.kron(one, Identity(size - control - 1))))
        )
    return U_one + U_two


def XX_Rotation(size, qubit1, qubit2, param, is_grad):
    matrix = 1
    for i in range(size):
        if (qubit1 == i) or (qubit2 == i):
            matrix = np.kron(matrix, X)
        else:
            matrix = np.kron(matrix, I)

    try:  # return matrix
        if is_grad:
            return -1j * np.matmul(matrix, linalg.expm(-1j * param * matrix))
        return linalg.expm(-1j * param * matrix)

    except Exception:
        print("param:\n:", param)


def YY_Rotation(size, qubit1, qubit2, param, is_grad):
    matrix = 1
    for i in range(size):
        if (qubit1 == i) or (qubit2 == i):
            matrix = np.kron(matrix, Y)
        else:
            matrix = np.kron(matrix, I)

    try:  # return matrix
        if is_grad:
            return -1j * np.matmul(matrix, linalg.expm(-1j * param * matrix))
        return linalg.expm(-1j * param * matrix)

    except Exception:
        print("param:\n:", param)


def ZZ_Rotation(size, qubit1, qubit2, param, is_grad):
    matrix = 1
    for i in range(size):
        if (qubit1 == i) or (qubit2 == i):
            matrix = np.kron(matrix, Z)
        else:
            matrix = np.kron(matrix, I)

    try:  # return matrix
        if is_grad:
            return 1j / 2 * np.matmul(matrix, linalg.expm(1j / 2 * param * matrix))
        return linalg.expm(1j / 2 * param * matrix)

    except Exception:
        print("param:\n:", param)


def X_Rotation(size, qubit, param, is_grad):
    matrix = 1
    for i in range(size):
        if qubit == i:
            if is_grad is False:
                try:
                    matrix = np.kron(matrix, linalg.expm(-1j / 2 * param * X))
                except Exception:
                    print("param:\n:", param)
            else:
                matrix = np.kron(matrix, -1j / 2 * X * linalg.expm(-1j / 2 * param * X))
        else:
            matrix = np.kron(matrix, I)

    return matrix


def Y_Rotation(size, qubit, param, is_grad):
    matrix = 1
    for i in range(size):
        if qubit == i:
            if is_grad is False:
                try:
                    matrix = np.kron(matrix, linalg.expm(-1j / 2 * param * Y))
                except Exception:
                    print("param:\n:", param)
            else:
                matrix = np.kron(matrix, -1j / 2 * Y * linalg.expm(-1j / 2 * param * Y))
        else:
            matrix = np.kron(matrix, I)

    return matrix


def Z_Rotation(size, qubit, param, is_grad):
    matrix = 1
    for i in range(size):
        if qubit == i:
            if is_grad is False:
                try:
                    matrix = np.kron(matrix, linalg.expm(-1j / 2 * param * Z))
                except Exception:
                    print("param:\n:", param)
            else:
                matrix = np.kron(matrix, -1j / 2 * Z * linalg.expm(-1j / 2 * param * Z))
        else:
            matrix = np.kron(matrix, I)

    return matrix


def Global_phase(size, param, is_grad):
    matrix = np.eye(2**size)
    eA = np.exp(-1j * param**2) * matrix
    return eA if is_grad is False else -1j * 2 * param * np.matmul(matrix, eA)


class QuantumGate:
    def __init__(self, name, qubit1=None, qubit2=None, **kwarg):
        self.name = name
        self.qubit1 = qubit1
        self.qubit2 = qubit2
        self.r = self.get_r()
        self.s = self.get_s()

        self.angle = kwarg.get("angle", None)

    def get_r(self):
        if self.name in ["X", "Y", "Z", "ZZ"]:
            return 1 / 2
        return 1 if self.name in ["XX", "YY"] else None

    def get_s(self):
        return np.pi / (4 * self.r) if self.r is not None else None

    def matrix_representation(self, size, is_grad):
        if self.angle is not None:
            try:
                param = float(self.angle)
            except:
                param = param_table[self.angle]

        if self.name == "XX":
            return XX_Rotation(size, self.qubit1, self.qubit2, param, is_grad)

        if self.name == "YY":
            return YY_Rotation(size, self.qubit1, self.qubit2, param, is_grad)

        if self.name == "ZZ":
            return ZZ_Rotation(size, self.qubit1, self.qubit2, param, is_grad)

        if self.name == "Z":
            return Z_Rotation(size, self.qubit1, param, is_grad)

        if self.name == "X":
            return X_Rotation(size, self.qubit1, param, is_grad)

        if self.name == "Y":
            return Y_Rotation(size, self.qubit1, param, is_grad)

        if self.name == "CNOT":
            return mCNOT(size, self.qubit1, self.qubit2)

        if self.name == "G":
            return Global_phase(size, param, is_grad)

        raise ValueError("Gate is not defined")

    def matrix_representation_shift_phase(self, size, is_grad, signal):
        if self.angle is not None:
            try:
                param = float(self.angle)
                if is_grad and self.name != "G":
                    if signal == "+":
                        param += self.s
                    else:
                        param -= self.s
                    is_grad = False
            except:
                param = param_table[self.angle]

        if self.name == "XX":
            return XX_Rotation(size, self.qubit1, self.qubit2, param, is_grad)

        if self.name == "YY":
            return YY_Rotation(size, self.qubit1, self.qubit2, param, is_grad)

        if self.name == "ZZ":
            return ZZ_Rotation(size, self.qubit1, self.qubit2, param, is_grad)

        if self.name == "Z":
            return Z_Rotation(size, self.qubit1, param, is_grad)

        if self.name == "X":
            return X_Rotation(size, self.qubit1, param, is_grad)

        if self.name == "Y":
            return Y_Rotation(size, self.qubit1, param, is_grad)

        if self.name == "G":
            return Global_phase(size, param, is_grad)

        if self.name == "CNOT":
            return mCNOT(size, self.qubit1, self.qubit2)

        raise ValueError("Gate is not defined")