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

import numpy
import torch as np

from config import CFG

np.set_default_device(CFG.device)

I = np.eye(2, dtype=np.complex64)

# Pauli matrices
X = np.tensor([[0, 1], [1, 0]], dtype=np.complex64)  #: Pauli-X matrix
Y = np.tensor([[0, -1j], [1j, 0]], dtype=np.complex64)  #: Pauli-Y matrix
Z = np.tensor([[1, 0], [0, -1]], dtype=np.complex64)  #: Pauli-Z matrix
Hadamard = np.tensor([[1, 1], [1, -1]], dtype=np.complex64) / np.tensor(
    numpy.sqrt(2), dtype=np.complex64
)  #: Hadamard gate

zero = np.tensor([[1, 0], [0, 0]], dtype=np.complex64)
one = np.tensor([[0, 0], [0, 1]], dtype=np.complex64)

# Two qubit gates
CNOT = np.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex64)  #: CNOT gate
SWAP = np.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex64)  #: SWAP gate
CZ = np.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=np.complex64)  #: CZ gate

global param_table
param_table = {}


def Identity(size: int):
    matrix = np.eye(1, dtype=np.complex64)
    for _ in range(size):
        matrix = np.kron(matrix, I)
    return matrix


def XX_Rotation(size, qubit1, qubit2, param, is_grad):
    matrix = np.eye(1, dtype=np.complex64)
    for i in range(size):
        if (qubit1 == i) or (qubit2 == i):
            matrix = np.kron(matrix, X)
        else:
            matrix = np.kron(matrix, I)
    param = np.tensor(param, dtype=np.complex64)
    cpu_matrix = matrix.cpu()
    cpu_param = param.cpu()
    if is_grad:
        exp_result = -1j * np.matmul(cpu_matrix, np.matrix_exp(-1j * cpu_param * cpu_matrix))
    else:
        exp_result = np.matrix_exp(-1j * cpu_param * cpu_matrix)
    return exp_result.to(CFG.device)


def YY_Rotation(size, qubit1, qubit2, param, is_grad):
    matrix = np.eye(1, dtype=np.complex64)
    for i in range(size):
        if (qubit1 == i) or (qubit2 == i):
            matrix = np.kron(matrix, Y)
        else:
            matrix = np.kron(matrix, I)
    param = np.tensor(param, dtype=np.complex64)
    cpu_matrix = matrix.cpu()
    cpu_param = param.cpu()
    if is_grad:
        exp_result = -1j * np.matmul(cpu_matrix, np.matrix_exp(-1j * cpu_param * cpu_matrix))
    else:
        exp_result = np.matrix_exp(-1j * cpu_param * cpu_matrix)
    return exp_result.to(CFG.device)


def ZZ_Rotation(size, qubit1, qubit2, param, is_grad):
    matrix = np.eye(1, dtype=np.complex64)
    for i in range(size):
        if (qubit1 == i) or (qubit2 == i):
            matrix = np.kron(matrix, Z)
        else:
            matrix = np.kron(matrix, I)
    param = np.tensor(param, dtype=np.complex64)
    cpu_matrix = matrix.cpu()
    cpu_param = param.cpu()
    if is_grad:
        exp_result = 1j / 2 * np.matmul(cpu_matrix, np.matrix_exp(1j / 2 * cpu_param * cpu_matrix))
    else:
        exp_result = np.matrix_exp(1j / 2 * cpu_param * cpu_matrix)
    return exp_result.to(CFG.device)


def X_Rotation(size, qubit, param, is_grad):
    matrix = np.eye(1, dtype=np.complex64)
    for i in range(size):
        if qubit == i:
            param = np.tensor(param, dtype=np.complex64)
            cpu_param = param.cpu()
            if not is_grad:
                kron_operand = np.matrix_exp(-1j / 2 * cpu_param * X.to("cpu"))
                matrix = np.kron(matrix, kron_operand.to(CFG.device))
            else:
                kron_operand = -1j / 2 * X.to("cpu") @ np.matrix_exp(-1j / 2 * cpu_param * X.to("cpu"))
                matrix = np.kron(matrix, kron_operand.to(CFG.device))
        else:
            matrix = np.kron(matrix, I)
    return matrix


def Y_Rotation(size, qubit, param, is_grad):
    matrix = np.eye(1, dtype=np.complex64)
    for i in range(size):
        if qubit == i:
            param = np.tensor(param, dtype=np.complex64)
            cpu_param = param.cpu()
            if not is_grad:
                kron_operand = np.matrix_exp(-1j / 2 * cpu_param * Y.to("cpu"))
                matrix = np.kron(matrix, kron_operand.to(CFG.device))
            else:
                kron_operand = -1j / 2 * Y.to("cpu") @ np.matrix_exp(-1j / 2 * cpu_param * Y.to("cpu"))
                matrix = np.kron(matrix, kron_operand.to(CFG.device))
        else:
            matrix = np.kron(matrix, I)
    return matrix


def Z_Rotation(size, qubit, param, is_grad):
    matrix = np.eye(1, dtype=np.complex64)
    for i in range(size):
        if qubit == i:
            param = np.tensor(param, dtype=np.complex64)
            cpu_param = param.cpu()
            if not is_grad:
                kron_operand = np.matrix_exp(-1j / 2 * cpu_param * Z.to("cpu"))
                matrix = np.kron(matrix, kron_operand.to(CFG.device))
            else:
                kron_operand = -1j / 2 * Z.to("cpu") @ np.matrix_exp(-1j / 2 * cpu_param * Z.to("cpu"))
                matrix = np.kron(matrix, kron_operand.to(CFG.device))
        else:
            matrix = np.kron(matrix, I)
    return matrix


def Global_phase(size, param, is_grad):
    matrix = np.eye(2**size, dtype=np.complex64)
    param = np.tensor(param, dtype=np.complex64)
    cpu_matrix = matrix.cpu()
    cpu_param = param.cpu()
    eA = np.exp(-1j * cpu_param**2) * cpu_matrix
    eA = eA
    return eA if not is_grad else -1j * 2 * param * np.matmul(matrix, eA)


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
        raise ValueError("Gate is not defined")
