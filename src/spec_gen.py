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
"""Circuit-resource report for the QGAN generator.

Reports two layers of cost for each config:

  [logical]   This is what qml.specs sees, where each
              Ising counts as ONE 2-qubit gate.

  [hardware]  the *native* cost on a chosen platform, obtained by mapping each
              logical gate to its native-gate cost via NATIVE_COST:

                superconducting (IBM/Google, CNOT-based):
                    IsingZZ -> 2 CNOT + 1 RZ
                    IsingXX -> 2 CNOT + 2 H
                    IsingYY -> 2 CNOT + 2 H + 2 S
                trapped-ion (quantinuum, RZZ native, all-to-all):
                    IsingZZ -> 1 RZZ  (native, no decomposition)
                    IsingXX -> 1 RZZ  + 2 H
                    IsingYY -> 1 RZZ  + 2 H + 2 S
"""

import torch
import pennylane as qml

from config import CFG
from qgan.precision import cdtype, rdtype
from qgan.generator import _build_qnode, _wire_layout, count_params


# -- PLATFORM-DEPENDENT NATIVE GATE COSTS ------------------------------------
NATIVE_COST = {
    "superconducting": {
        # gate        : (entangling CNOTs, single-qubit gates)
        "IsingZZ": (2, 1),   # 2 CNOT + 1 RZ
        "IsingXX": (2, 5),   # 2 CNOT + 1 RX-equiv + 4 H (basis change, both qubits)
        "IsingYY": (2, 9),   # 2 CNOT + RX-equiv + 4 H + 4 S (basis change, both qubits)
    },
    "trapped-ion": {
        # gate        : (native RZZ, single-qubit gates)
        "IsingZZ": (1, 0),   # native RZZ, nothing else
        "IsingXX": (1, 4),   # 1 RZZ + 4 H
        "IsingYY": (1, 8),   # 1 RZZ + 4 H + 4 S
    },
}

# single-qubit gate names that map 1:1 to a native single-qubit op
SINGLE_QUBIT_GATES = {"RX", "RY", "RZ", "Rot", "Hadamard", "PhaseShift",
                      "PauliX", "PauliY", "PauliZ", "S", "T"}

#   superconducting : a weight-2 Ising compiles to CNOT-RZ-CNOT -> 2 CNOTs in
#                     series -> 2q-depth contribution = 2.
#   trapped-ion      : native RZZ -> 1 entangling gate -> contribution = 1.
ENTANGLING_PER_2Q_GATE = {
    "superconducting": 2,
    "trapped-ion": 1,
}

# SEQUENTIAL LAYER depth each logical gate adds on the critical path, by
# platform. This counts time-ordered layers,
#   IsingZZ sc : CNOT-RZ-CNOT                       = 3 layers
#   IsingXX sc : H-CNOT-RZ-CNOT-H                    = 5 layers
#   IsingYY sc : (S^dag H)-CNOT-RZ-CNOT-(H S)        = 7 layers (S^dag,H separate)
#   IsingZZ i-t  : RZZ                                 = 1 layer
#   IsingXX i-t  : H-RZZ-H                             = 3 layers
#   IsingYY i-t  : (S^dag H)-RZZ-(H S)                 = 5 layers
LAYER_DEPTH = {
    "superconducting": {
        "IsingZZ": 3, "IsingXX": 5, "IsingYY": 7,
    },
    "trapped-ion": {
        "IsingZZ": 1, "IsingXX": 3, "IsingYY": 5,
    },
}


def hardware_counts(gate_types: dict, platform: str):
    """Translate logical gate_types (from qml.specs) into native hardware counts.

    Returns (n_entangling, n_single_qubit) for the given platform.
    Unknown gates are reported so nothing is silently dropped.
    """
    cost = NATIVE_COST[platform]
    n_ent = n_1q = 0
    unknown = {}
    for gname, count in gate_types.items():
        if gname in cost:
            e, s = cost[gname]
            n_ent += e * count
            n_1q += s * count
        elif gname in SINGLE_QUBIT_GATES:
            n_1q += count
        else:
            unknown[gname] = count
    return n_ent, n_1q, unknown


# -- 2-QUBIT DEPTH (logical) -------------------------------------------------
def two_qubit_depth(circuit, dummy_params, dummy_state):
    """Depth counting only 2-qubit gates (longest chain on any wire)."""
    tape = qml.workflow.construct_tape(circuit)(dummy_params, dummy_state)
    wire_depth = {}
    for op in tape.operations:
        if len(op.wires) == 2:
            w = list(op.wires)
            d = max(wire_depth.get(w[0], 0), wire_depth.get(w[1], 0)) + 1
            for wire in w:
                wire_depth[wire] = d
    return max(wire_depth.values(), default=0)


# -- 2-QUBIT DEPTH (hardware) -----------------------------------------
def two_qubit_depth_native(circuit, dummy_params, dummy_state, platform):
    """Native 2q-depth: same critical-path logic, but each logical 2-qubit gate
    contributes ENTANGLING_PER_2Q_GATE[platform] serial entangling gates on its
    wires (a weight-2 Ising is CNOT-RZ-CNOT = 2 CNOTs in series on superconducting,
    a single RZZ on trapped-ion). Single-qubit gates don't advance 2q-depth.
    """
    rung = ENTANGLING_PER_2Q_GATE[platform]
    tape = qml.workflow.construct_tape(circuit)(dummy_params, dummy_state)
    wire_depth = {}
    for op in tape.operations:
        if len(op.wires) == 2:
            w = list(op.wires)
            d = max(wire_depth.get(w[0], 0), wire_depth.get(w[1], 0)) + rung
            for wire in w:
                wire_depth[wire] = d
    return max(wire_depth.values(), default=0)


# -- TOTAL DEPTH (hardware) -----------------------------
def total_depth_native(circuit, dummy_params, dummy_state, platform):
    """Native TOTAL depth: critical path counting ALL native layers (1q + 2q).

    Each logical gate contributes its sequential LAYER_DEPTH on the wires it
    touches; a logical gate not in LAYER_DEPTH is treated as a single-qubit /
    single-layer op (depth 1).
    """
    layers = LAYER_DEPTH[platform]
    tape = qml.workflow.construct_tape(circuit)(dummy_params, dummy_state)
    wire_depth = {}
    for op in tape.operations:
        w = list(op.wires)
        add = layers.get(op.name, 1)  # unknown / plain 1q gate -> 1 layer
        base = max((wire_depth.get(wire, 0) for wire in w), default=0)
        d = base + add
        for wire in w:
            wire_depth[wire] = d
    return max(wire_depth.values(), default=0)


# -- MAIN REPORT -------------------------------------------------------------
def print_circuit_specs(config_override: dict = None,
                        platforms=("superconducting", "trapped-ion")):
    """Print logical + hardware gate counts for a given config, no training."""
    if config_override:
        original = {k: getattr(CFG, k) for k in config_override}
        for k, v in config_override.items():
            setattr(CFG, k, v)

    _, _, _, total_wires = _wire_layout()
    n_params = count_params(CFG.system_size, CFG.extra_ancilla)
    circuit = _build_qnode()

    dummy_params = torch.zeros(n_params, dtype=rdtype())
    dummy_state = torch.zeros(2 ** total_wires, dtype=cdtype())
    dummy_state[0] = 1.0

    raw = qml.specs(circuit)(dummy_params, dummy_state)
    res = raw["resources"]
    gate_types = dict(res.gate_types)

    # ---- logical (ansatz-level) ----
    print("  [logical / ansatz-level]")
    print(f"    depth         : {res.depth}")
    print(f"    2q depth      : {two_qubit_depth(circuit, dummy_params, dummy_state)}")
    print(f"    total gates   : {res.num_gates}")
    print(f"    1q gates      : {res.gate_sizes.get(1, 0)}")
    print(f"    2q gates      : {res.gate_sizes.get(2, 0)}")
    print(f"    gate types    : {gate_types}")

    # ---- hardware (per platform) ----
    for plat in platforms:
        n_ent, n_1q, unknown = hardware_counts(gate_types, plat)
        ent_name = "CNOT" if plat == "superconducting" else "RZZ"
        hw_2q_depth = two_qubit_depth_native(circuit, dummy_params,
                                             dummy_state, plat)
        hw_total_depth = total_depth_native(circuit, dummy_params,
                                            dummy_state, plat)
        print(f"  [hardware / {plat}]")
        print(f"    entangling ({ent_name}) : {n_ent}")
        print(f"    2q depth ({ent_name})   : {hw_2q_depth}")
        print(f"    total depth (1q+2q): {hw_total_depth}")
        print(f"    single-qubit gates : {n_1q}")
        print(f"    total native gates : {n_ent + n_1q}")
        if unknown:
            print(f"    WARNING unmapped gates (counted nowhere): {unknown}")

    if config_override:
        for k, v in original.items():
            setattr(CFG, k, v)


# -- RUN FOR EACH CONFIG -----------------------------------------------------
if __name__ == "__main__":
    for i, cfg in enumerate(CFG.reps_new_config):
        print(f"\nConfig {i}: {cfg}")
        print_circuit_specs(cfg)