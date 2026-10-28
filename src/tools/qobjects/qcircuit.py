# Copyright 2025 GIQ, Universitat Autònoma de Barcelona
#
# Licensed under the MIT License. See the LICENSE file in the project root for the full license text.
# SPDX-License-Identifier: MIT
"""definition of quantum circuit simulation."""

import os

import numpy as np

from tools.qobjects.qgates import Identity, QuantumGate


class QuantumCircuit:
    def __init__(self, size, name):
        self.size = size
        self.depth = 0
        self.gates = []
        self.name = name

    def check_circuit(self):
        for j, gate_j in enumerate(self.gates):
            if gate_j.qubit1 is not None and gate_j.qubit2 is not None:
                if gate_j.qubit1 > self.size - 1:
                    print(f"Error: #{j} gate:{gate_j.name} 1qubit is out of range")
                    os._exit(0)
                elif gate_j.qubit2 > self.size - 1:
                    print(f"Error: #{j} gate:{gate_j.name} 2qubit is out of range")
                    os._exit(0)

    def get_mat_rep(self):
        matrix = Identity(self.size)
        for gate in self.gates:
            g = gate.matrix_representation(self.size, False)
            matrix = np.matmul(g, matrix)
        return np.asmatrix(matrix)

    def get_grad_mat_rep(self, index, signal="none", type="matrix_multiplication") -> np.ndarray:
        """Matrix multipliction: explicit way to calculate the gradient using matrix multiplication.

        Shift_phase: generate two quantum circuit to calculate the gradient evaluating analytic gradients on quantum hardware:
        https://arxiv.org/pdf/1811.11184.pdf
        """
        if type == "shift_phase":
            matrix = Identity(self.size)
            for j, gate_j in enumerate(self.gates):
                if index == j:
                    g = gate_j.matrix_representation_shift_phase(self.size, True, signal)
                else:
                    g = gate_j.matrix_representation_shift_phase(self.size, False, signal)
                matrix = np.matmul(g, matrix)
            return np.asmatrix(matrix)

        if type == "matrix_multiplication":
            matrix = Identity(self.size)
            for j, gate_j in enumerate(self.gates):
                if index == j:
                    g = gate_j.matrix_representation(self.size, True)
                else:
                    g = gate_j.matrix_representation(self.size, False)
                matrix = np.matmul(g, matrix)
            return np.asmatrix(matrix)

        return None

    def get_grad_qc(self, indx, type="0"):
        qc_list = []
        for j, gate_j in enumerate(self.gates):
            tmp = QuantumGate(" ", qubit1=None, qubit2=None, angle=None)
            tmp.name = gate_j.name
            tmp.qubit1 = gate_j.qubit1
            tmp.qubit2 = gate_j.qubit2
            tmp.angle = gate_j.angle
            if j == indx:
                try:
                    if gate_j.name != "G" or gate_j.name != "CNOT":
                        if type == "+":
                            tmp.angle = gate_j.angle + gate_j.s
                        elif type == "-":
                            tmp.angle = gate_j.angle - gate_j.s
                except:
                    print("param value error")
                qc_list.append(tmp)
            else:
                qc_list.append(tmp)
        return qc_list

    def add_gate(self, quantum_gate):
        self.depth += 1
        self.gates.append(quantum_gate)
