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
"""Generator module — PyTorch + PennyLane rewrite.

Replaces all manual gradient computation (_grad_theta, _param_shift_grad_state,
_apply_momentum_step) with PyTorch autograd.

For the Choi representation, we create a maximally entangled.
For Haar batching, we generate a complex vector with Gaussian distributed
coefficients and transform it into a quantum state using qml.StatePrep (both cases).
"""

import os
from typing import Optional

import numpy as np
import pennylane as qml
import torch

from config import CFG
from qgan.precision import cdtype, rdtype
from qgan.ancilla import (
    as_gen_state,
    get_final_gen_state_torch,
    get_max_entangled_state_with_ancilla_if_needed,
)
from qgan.cost_functions import _calc_wasserstein
from qgan.dis_spectral import spectral_cost
from tools.data_managers import print_and_log

# -- GATE TERM DEFINITIONS ------------------------------------------------
_1Q_GATES = {
    "X": qml.RX,
    "Y": qml.RY,
    "Z": qml.RZ,
}

_2Q_GATES = {
    "XX": qml.IsingXX,
    "YY": qml.IsingYY,
    "ZZ": qml.IsingZZ,
}



def _apply_2q(term: str, theta, wires) -> None:
    """Apply a 2-qubit rotation exp(-i θ/2 P), P = XX, YY or ZZ (the same convention as the 1-qubit gates)."""
    _2Q_GATES[term](theta, wires=wires)

# Predefined ansatz
_PREDEFINED_ANSATZ = {
    "ZZ_Z_X": ["X", "Z", "ZZ"],
    "ZZ_YY_XX_Z": ["Z", "XX", "YY", "ZZ"],
}


def _get_ansatz_terms() -> list[str]:
    """Get the list of gate terms for the current ansatz config."""
    if CFG.gen_ansatz == "custom":
        if not CFG.custom_ansatz_terms:
            raise ValueError("custom ansatz requires custom_ansatz_terms to be set.")
        return CFG.custom_ansatz_terms
    if CFG.gen_ansatz in _PREDEFINED_ANSATZ:
        return _PREDEFINED_ANSATZ[CFG.gen_ansatz]
    raise ValueError(
        f"Unknown ansatz: {CFG.gen_ansatz}. Expected one of: {list(_PREDEFINED_ANSATZ.keys())} or 'custom'."
    )


def _layer_coupling(layer_idx: int) -> bool:
    """Check if a given layer should have ancilla 1q and 2q couplings.
    Returns:
        True if ancilla 1q and 2q couplings should be applied in this layer.
    """
    if CFG.ancilla_coupling_layers == "all":
        return True
    return layer_idx in CFG.ancilla_coupling_layers


def _ancilla_coupling_terms(terms: list[str]) -> list[str]:
    """2q gate types of the ancilla couplings: the 2q terms of the ansatz, in its order."""
    return [t for t in terms if t in _2Q_GATES]


# -- DEVICE ----------------------------------------------------------------
def _make_device(total_wires: int):
    """Create a PennyLane device for the full Choi and Haar register."""
    return qml.device("default.qubit", wires=total_wires)

# -- UNIFIED ANSATZ --------------------------------------------------------
class Ansatz:
    """Applies gates inside a QNode context.

    All ansatz types follow the same pattern per layer:
        For each term in the term list:
            - 1q terms (X, Y, Z): apply rotation on each system qubit
            - 2q terms (XX, YY, ZZ): apply Ising gate between neighbouring system qubits
        Then ancilla gates (1q or 2q couplings) if configured.
    """

    @staticmethod
    def apply(params, gen_wires: list[int], ancilla_wire: Optional[int]):
        """Apply the ansatz defined in CFG to the generator wires."""
        terms = _get_ansatz_terms()
        system_wires = [w for w in gen_wires if w != ancilla_wire]
        n_sys = len(system_wires)
        idx = 0

        terms_1q = [t for t in terms if t in _1Q_GATES]
        terms_2q = [t for t in terms if t in _2Q_GATES]

        for layer_idx in range(CFG.gen_layers):
            # -- System gates from term list --
            # We apply the gates in this order RX, RZ, RX, RZ...
            for w in system_wires:
                for term in terms_1q:
                    _1Q_GATES[term](params[idx], wires=w)
                    idx += 1

            for term in terms_2q:
                for i in range(n_sys - 1):
                    _apply_2q(term, params[idx], [system_wires[i], system_wires[i + 1]])
                    idx += 1

            # -- Ancilla 1q gates --
            if ancilla_wire is not None and CFG.do_ancilla_1q_gates and _layer_coupling(layer_idx):
                for term in terms_1q:
                    _1Q_GATES[term](params[idx], wires=ancilla_wire)
                    idx += 1

            # -- Ancilla 2q couplings --
            if ancilla_wire is not None and _layer_coupling(layer_idx):
                idx = Ansatz._apply_ancilla_couplings(params, idx, terms, system_wires, ancilla_wire)

    @staticmethod
    def _apply_ancilla_couplings(params, idx: int, terms: list[str], system_wires: list[int], ancilla_wire: int) -> int:
        """Apply ancilla 2q couplings according to topology.

        Uses the same 2q gate typxes from the term list.
        Returns updated param index.
        """
        n_sys = len(system_wires)
        topo = CFG.ancilla_topology

        # Resolve connect_to wire
        connect_to = CFG.ancilla_connect_to
        if connect_to is not None and isinstance(connect_to, int) and connect_to < n_sys:
            connect_wire = system_wires[connect_to]
        else:
            connect_wire = system_wires[-1]

        # 2q gate types of the couplings: the 2q terms of the ansatz
        terms_2q = _ancilla_coupling_terms(terms)

        # Connections of this topology (same order as the upstream numpy code)
        pairs = []
        if topo == "total":
            pairs += [(w, ancilla_wire) for w in system_wires]
        if topo == "bridge":
            pairs.append((system_wires[0], ancilla_wire))
        if topo in ("bridge", "ansatz"):
            pairs.append((connect_wire, ancilla_wire))
        if topo == "fake" and n_sys > 2:
            pairs.append((system_wires[0], connect_wire))

        # Gate type outer loop, connection inner loop (as the system 2q gates)
        order = [(t, p) for t in terms_2q for p in pairs]

        for term, (wire_a, wire_b) in order:
            _apply_2q(term, params[idx], [wire_a, wire_b])
            idx += 1

        return idx


# -- PARAMETER COUNTING ---------------------------------------------------
def count_params(n_system: int, has_ancilla: bool) -> int:
    """Total number of trainable angles for the current ansatz + config."""
    terms = _get_ansatz_terms()
    n = 0

    n_1q_terms = sum(1 for t in terms if t in _1Q_GATES)
    n_2q_terms = sum(1 for t in terms if t in _2Q_GATES)

    for layer_idx in range(CFG.gen_layers):
        n += n_1q_terms * n_system
        n += n_2q_terms * (n_system - 1)

        if has_ancilla and CFG.do_ancilla_1q_gates and _layer_coupling(layer_idx):
            n += n_1q_terms

        if has_ancilla and _layer_coupling(layer_idx):
            n += _count_ancilla_coupling_params(len(_ancilla_coupling_terms(terms)), n_system)

    return n


def _count_ancilla_coupling_params(n_2q_terms: int, n_system: int) -> int:
    """Count params added by one layer of ancilla couplings."""
    topo = CFG.ancilla_topology
    n = 0
    if topo == "total":
        n += n_system * n_2q_terms
    if topo == "bridge":
        n += n_2q_terms
    if topo in ("bridge", "ansatz"):
        n += n_2q_terms
    if topo == "fake" and n_system > 2:
        n += n_2q_terms
    return n


def _coupling_order_matters(saved: dict) -> bool:
    """Whether the order of the ancilla couplings (gate type outer or connection outer) changes the circuit of a saved
    generator: only with two or more connections and two or more 2q gate types (with ZZ_Z_X it never does)."""
    n_system = saved.get("target_size", CFG.system_size)
    n_connections = {"total": n_system, "bridge": 2, "ansatz": 1, "fake": 1 if n_system > 2 else 0}
    terms = saved.get("ancilla_coupling_terms") or _ancilla_coupling_terms(_get_ansatz_terms())  # (older files' key)
    return n_connections.get(saved.get("ancilla_topology"), 0) >= 2 and len(terms) >= 2


# -- WIRE LAYOUT -----------------------------------------------------------
def _wire_layout():
    """Compute wire indices for each register.

    Layout:  [ choi_register | gen_system | (ancilla) ]
    """
    s = CFG.system_size
    if CFG.use_choi:
        choi_wires = list(range(s))
        gen_system_wires = list(range(s, 2 * s))
        if CFG.extra_ancilla:
            ancilla_wire = 2 * s
            gen_wires = gen_system_wires + [ancilla_wire]
            total_wires = 2 * s + 1
        else:
            ancilla_wire = None
            gen_wires = gen_system_wires
            total_wires = 2 * s
    else:
        choi_wires = None  # No choi wire for batching
        if CFG.extra_ancilla:
            ancilla_wire = s
            gen_wires = list(range(s)) + [ancilla_wire]
            total_wires = s + 1
        else:
            ancilla_wire = None
            gen_wires = list(range(s))
            total_wires = s

    return choi_wires, gen_wires, ancilla_wire, total_wires


# -- QNODE -----------------------------------------------------------------
def _build_qnode():
    """Build the QNode for the generator circuit.

    1) Prepare a quantum state from a Gaussian distributed complex vector
    or from a Max. Entangled state (qml.StatePrep)
    2) Apply ansatz on generator register.
    3) Return full statevector (as a torch tensor).
    """
    choi_wires, gen_wires, ancilla_wire, total_wires = _wire_layout()
    dev = _make_device(total_wires)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(params, input_state=None):
        qml.StatePrep(input_state, wires=range(total_wires))
        Ansatz.apply(params, gen_wires, ancilla_wire)
        return qml.state()

    return circuit

def get_circuit_specs(circuit, n_params: int, total_wires: int) -> dict:
    """Run qml.specs on a dummy forward pass and extract depth/gate counts."""
    dummy_params = torch.zeros(n_params, dtype=rdtype())
    dummy_state = torch.zeros(2 ** total_wires, dtype=cdtype())
    dummy_state[0] = 1.0

    specs_fn = qml.specs(circuit)
    raw = specs_fn(dummy_params, dummy_state)

    gate_sizes = raw.get("resources").gate_sizes  # {n_qubits: count}
    return {
        "depth": raw["resources"].depth,
        "total_gates": raw["resources"].num_gates,
        "gates_1q": gate_sizes.get(1, 0),
        "gates_2q": gate_sizes.get(2, 0),
    }

# -- GENERATOR CLASS -------------------------------------------
class Generator:
    """Generator for the Quantum WGAN

    Removed:
        - _grad_theta()             -> replaced by loss.backward()
        - _param_shift_grad_state() -> not needed (backprop through QNode)
        - _apply_momentum_step()    -> replaced by torch.optim.SGD

    Usage:
        gen = Generator()
        gen.update_gen(dis, final_target_state_torch)
        state = gen.get_total_gen_state()
    """

    def __init__(self):
        # -- save/load compatibility metadata --
        self.size: int = CFG.system_size + (1 if CFG.extra_ancilla else 0)
        self.ancilla: bool = CFG.extra_ancilla
        self.ancilla_topology: str = CFG.ancilla_topology
        self.ancilla_coupling_layers = CFG.ancilla_coupling_layers
        self.do_ancilla_1q_gates: bool = CFG.do_ancilla_1q_gates
        self.ansatz: str = CFG.gen_ansatz
        self.custom_ansatz_terms = list(CFG.custom_ansatz_terms) if CFG.gen_ansatz == "custom" else None
        self.layers: int = CFG.gen_layers
        self.target_size: int = CFG.system_size
        self.target_hamiltonian: str = CFG.target_hamiltonian
        self.use_choi: bool = CFG.use_choi

        # -- parameters -------
        n_params = count_params(CFG.system_size, CFG.extra_ancilla)
        self.n_params: int = n_params
        self.params: torch.Tensor = self._init_params(n_params)

        # -- circuit ---------
        self.circuit = _build_qnode()

        # In choi representation we always use the same initial state,
        # so we created once and save it
        # In batching we have different states, thus no need to save the state
        if CFG.use_choi:
            me_state, _ = get_max_entangled_state_with_ancilla_if_needed(CFG.system_size)
            self.choi_input = torch.tensor(np.asarray(me_state).flatten(), dtype=cdtype())
            self.total_gen_state = self.get_total_gen_state()

        # -- optimizer: SGD with momentum, MINIMISING --
        self.optimizer = torch.optim.SGD(
            [self.params],
            lr=CFG.l_rate,
            momentum=CFG.momentum_coeff,
        )
            
    # -- parameter initialisation ------------------------------------------
    def _init_params(self, n_params: int) -> torch.Tensor:
        """Random uniform in [0, 2π), as a torch tensor with requires_grad=True.

        Respects start_ancilla_gates_randomly: if False, ancilla gate params are initialised to 0.
        """
        params = torch.empty(n_params, dtype=rdtype()).uniform_(0, 2 * np.pi)

        if CFG.extra_ancilla and not CFG.start_ancilla_gates_randomly:
            ancilla_indices = self._get_ancilla_param_indices()
            params[ancilla_indices] = 0.0

        params.requires_grad_(True)
        return params

    def _get_ancilla_param_indices(self) -> list[int]:
        """Identify which param indices correspond to ancilla-only gates."""
        terms = _get_ansatz_terms()
        n_1q_terms = sum(1 for t in terms if t in _1Q_GATES)
        n_2q_terms = sum(1 for t in terms if t in _2Q_GATES)
        n_sys = CFG.system_size

        indices = []
        idx = 0

        for layer_idx in range(CFG.gen_layers):
            # System 1q + 2q gates (skip)
            idx += n_1q_terms * n_sys
            idx += n_2q_terms * (n_sys - 1)

            # Ancilla 1q gates
            if CFG.do_ancilla_1q_gates and _layer_coupling(layer_idx):
                for _ in range(n_1q_terms):
                    indices.append(idx)
                    idx += 1

            # Ancilla 2q couplings
            if _layer_coupling(layer_idx):
                n_anc_2q = _count_ancilla_coupling_params(len(_ancilla_coupling_terms(terms)), n_sys)
                for _ in range(n_anc_2q):
                    indices.append(idx)
                    idx += 1

        return indices

    def _effective_params(self) -> torch.Tensor:
        """Params tensor seen by the QNode.

        If ancilla_training is False, returns a tensor whose values match self.params,
        but with ancilla gates detached from the autograd graph. The loss therefore 
        does NOT propagate through ancilla params.
        """
        if not self.ancilla or CFG.ancilla_training:
            return self.params

        # Build a boolean mask: True where the slot is trainable (system),
        # False where it is frozen (ancilla).
        if not hasattr(self, "_train_mask") or self._train_mask is None:
            mask = torch.ones(self.n_params, dtype=torch.bool)
            anc_idx = self._get_ancilla_param_indices()
            mask[anc_idx] = False
            self._train_mask = mask  # cached; layout doesn't change at runtime

        # torch.where keeps system values on the autograd graph and substitutes
        # detached values for ancilla gates. Numerical values are identical to
        # self.params. the only difference is the graph.
        return torch.where(self._train_mask, self.params, self.params.detach())
    
    # -- forward pass ------------------------------------------------------
    def get_total_gen_state(self, input_state=None) -> torch.Tensor:
        """Run the circuit and return the full statevector as a 1D torch tensor."""
        params = self._effective_params()
        if CFG.use_choi:
            return self.circuit(params, self.choi_input)
        else:
            return self.circuit(params, input_state)

    def get_gen_unitary(self) -> np.ndarray:
        """U_gen restricted to the generator wires (system [+ ancilla]), as a numpy array.

        Wire order = generator wires in increasing order, so the ancilla (if any) is the last qubit,
        matching `gen.qc.get_mat_rep()` of the upstream numpy code.
        """
        _, gen_wires, ancilla_wire, _ = _wire_layout()

        def ansatz_only(params):
            Ansatz.apply(params, gen_wires, ancilla_wire)

        with torch.no_grad():
            U = qml.matrix(ansatz_only, wire_order=gen_wires)(self._effective_params().detach())
        return np.asarray(U.detach().numpy() if isinstance(U, torch.Tensor) else U)

    def get_final_gen_state(self, total_gen_state: torch.Tensor) -> torch.Tensor:
        """Apply ancilla post-processing and return the final state for the discriminator.

        Returns a column vector (d, 1) torch tensor.
        """
        return get_final_gen_state_torch(total_gen_state)

    def _get_detached_matrices(self, dis):
        """Detach discriminator matrices safely.
        Even though the generator try to minimize the loss function, here
        we only change the parameters of the Circuit.
        (Computed once per critic update: Discriminator.detached_matrices.)
        """
        return dis.detached_matrices()

    # -- loss computation --------------------------------------------------
    def compute_loss(self, dis, final_target_state):
        """Compute the Wasserstein loss as a differentiable scalar (Choi)"""
        total_gen_state = self.get_total_gen_state()
        final_gen_state = self.get_final_gen_state(total_gen_state)

        g = as_gen_state(final_gen_state)  # 1D state or (n_branches, d) for ancilla_mode "trace"
        t = final_target_state.reshape(-1)

        if CFG.dis_eval == "spectral":  # the critic's terms in its eigenbases (detached), see qgan.dis_spectral
            return spectral_cost(g, t, dis.detached_spectrum())

        dis_matrices = self._get_detached_matrices(dis)
        return _calc_wasserstein(g, t, dis_matrices)

    # -- Loss for batching ----------
    def compute_loss_batch(self, dis, batch_inputs, batch_targets):
        """Compute average Wasserstein loss over a batch of input states (Haar)"""
        spectral = CFG.dis_eval == "spectral"
        critic = dis.detached_spectrum() if spectral else self._get_detached_matrices(dis)
        total_loss = torch.tensor(0.0, dtype=rdtype())

        for input_state, target_state in zip(batch_inputs, batch_targets):
            gen_state = self.get_total_gen_state(input_state)
            g = as_gen_state(get_final_gen_state_torch(gen_state))
            t = target_state.reshape(-1)

            total_loss = total_loss + (spectral_cost(g, t, critic) if spectral else _calc_wasserstein(g, t, critic))

        return total_loss / len(batch_inputs)

    # -- training step --------------------------------
    def update_gen(self, dis, final_target_state=None, batch_inputs=None, batch_targets=None):
        """One generator optimisation step (minimisation).

        Replacement for the old update_gen(). Computes the loss,
        backpropagates through the QNode + ancilla processing + brakets,
        and steps the optimizer.

        Args:
            dis: Discriminator
            final_target_state: Target state as torch tensor, shape (d, 1).
                If you have a numpy array, convert with:
                    torch.tensor(np.asarray(state), dtype=cdtype())
        """
        self.optimizer.zero_grad()
        if CFG.use_choi:
            loss = self.compute_loss(dis, final_target_state)
        else:
            loss = self.compute_loss_batch(dis, batch_inputs, batch_targets)

        loss.backward()
        self.optimizer.step()

        if CFG.use_choi:
            with torch.no_grad():
                self.total_gen_state = self.get_total_gen_state()

    # -- SAVE / LOAD -------------------------------------------------------
    def save_model(self, file_path: str):
        """Save generator state to disk (torch dict format)."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        save_dict = {
            "params": self.params.detach().numpy(),
            "size": self.size,
            "ancilla": self.ancilla,
            "ancilla_topology": self.ancilla_topology,
            "ancilla_coupling_layers": self.ancilla_coupling_layers,
            "do_ancilla_1q_gates": self.do_ancilla_1q_gates,
            "ansatz": self.ansatz,
            "custom_ansatz_terms": self.custom_ansatz_terms,
            "layers": self.layers,
            "target_size": self.target_size,
            "target_hamiltonian": self.target_hamiltonian,
            "n_params": self.n_params,
            "gate_convention": "uniform",  # (the only one now; older code checks it when loading)
            "optimizer_state": self.optimizer.state_dict(),
        }
        torch.save(save_dict, file_path)

    def load_model_params(self, file_path: str) -> bool:
        """Load generator parameters from a saved model"""
        if not os.path.exists(file_path):
            print_and_log("ERROR: Generator model file not found\n", CFG.log_path)
            return False

        try:
            saved = torch.load(file_path, weights_only=False)
        except Exception as e:
            print_and_log(
                f"ERROR: Could not load generator model: {e}\nOld pickle/numpy formats are no longer supported.\n",
                CFG.log_path,
            )
            return False

        if not isinstance(saved, dict) or "params" not in saved:
            print_and_log(
                "ERROR: Unrecognised format. Only torch dict format is supported.\n",
                CFG.log_path,
            )
            return False

        return self._load_from_torch_dict(saved)

    def _load_from_torch_dict(self, saved: dict) -> bool:
        """Load from torch save_model format."""
        cant_load = False
        if saved.get("target_size") != self.target_size:
            print_and_log("ERROR: target size mismatch.\n", CFG.log_path)
            cant_load = True
        if saved.get("target_hamiltonian") != self.target_hamiltonian:
            print_and_log("ERROR: target hamiltonian mismatch.\n", CFG.log_path)
            cant_load = True
        if saved.get("ansatz") != self.ansatz:
            print_and_log("ERROR: ansatz mismatch.\n", CFG.log_path)
            cant_load = True
        elif saved.get("custom_ansatz_terms") is not None and saved["custom_ansatz_terms"] != self.custom_ansatz_terms:
            # (files saved before this key existed: not checked)
            print_and_log(f"ERROR: custom_ansatz_terms mismatch (saved {saved['custom_ansatz_terms']}, current "
                          f"{self.custom_ansatz_terms}).\n", CFG.log_path)
            cant_load = True
        if saved.get("layers") != self.layers:
            print_and_log("ERROR: layer count mismatch.\n", CFG.log_path)
            cant_load = True
        if saved.get("ancilla") and self.ancilla and saved.get("ancilla_topology") != self.ancilla_topology:
            print_and_log("ERROR: ancilla topology mismatch.\n", CFG.log_path)
            cant_load = True
        # (a key absent in files saved before it existed: not checked)
        if saved.get("ancilla") and self.ancilla and "do_ancilla_1q_gates" in saved \
                and saved["do_ancilla_1q_gates"] != self.do_ancilla_1q_gates:
            print_and_log(f"ERROR: do_ancilla_1q_gates mismatch (saved {saved['do_ancilla_1q_gates']}, current "
                          f"{self.do_ancilla_1q_gates}).\n", CFG.log_path)
            cant_load = True
        # Files of the code that had the option ancilla_coupling_terms (None: the ansatz's 2q gates, as now)
        saved_terms = saved.get("ancilla_coupling_terms")
        if saved.get("ancilla") and saved_terms is not None and list(saved_terms) != _ancilla_coupling_terms(
                _get_ansatz_terms()):
            print_and_log(f"ERROR: saved with ancilla_coupling_terms = {list(saved_terms)}, which this code no longer "
                          "has (its ancilla couplings use the 2-qubit gates of the ansatz).\n", CFG.log_path)
            cant_load = True
        # Gates of the file: "uniform" (the gates of this code), or older conventions. Files written by the fork
        # before the merge have no key ("fork": the same gates, but the ancilla couplings in the reverse order,
        # connection outer and gate type inner), so they can be loaded when that order gives the same circuit.
        saved_conv = {"pennylane": "fork"}.get(saved.get("gate_convention", "fork"), saved.get("gate_convention", "fork"))
        if saved_conv == "fork" and saved.get("ancilla") and _coupling_order_matters(saved):
            print_and_log(
                "ERROR: saved with the fork's order of ancilla couplings (connection outer), a different circuit "
                "for this topology and ansatz.\n",
                CFG.log_path,
            )
            cant_load = True
        elif saved_conv not in ("uniform", "fork"):
            print_and_log(f"ERROR: saved with the {saved_conv!r} gates (other XX/YY/ZZ angles).\n", CFG.log_path)
            cant_load = True
        if cant_load:
            return False

        saved_params = saved["params"]
        saved_size = saved.get("size", 0)
        saved_ancilla = saved.get("ancilla", False)

        # Exact match
        if saved_size == self.size and saved_ancilla == self.ancilla:
            if len(saved_params) != self.n_params:
                print_and_log("ERROR: param count mismatch.\n", CFG.log_path)
                return False
            with torch.no_grad():
                self.params.copy_(torch.from_numpy(saved_params).to(self.params.dtype))
            # SGD momentum: restored (fork) unless CFG.reset_optimizer_on_load (upstream behaviour).
            if not CFG.reset_optimizer_on_load and "optimizer_state" in saved:
                self.optimizer.load_state_dict(saved["optimizer_state"])
            self._refresh_state()
            print_and_log("Generator parameters loaded (torch dict format).\n", CFG.log_path)
            return True

        # \pm 1 qubit (ancilla difference)
        if saved_ancilla != self.ancilla and abs(saved_size - self.size) == 1:
            self._partial_load_params(
                saved_params, saved_ancilla, saved.get("ancilla_topology"), saved.get("ancilla_coupling_layers", "all"),
                saved.get("do_ancilla_1q_gates", CFG.do_ancilla_1q_gates),
            )
            self._refresh_state()
            print_and_log("Generator parameters partially loaded (ancilla diff, torch dict).\n", CFG.log_path)
            return True

        print_and_log("ERROR: incompatible generator.\n", CFG.log_path)
        return False

    def _partial_load_params(
        self, saved_params: np.ndarray, saved_ancilla: bool, saved_topology: Optional[str], saved_coupling_layers=None,
        saved_1q_gates: Optional[bool] = None,
    ) -> None:
        """Load system-only params when ancilla is added/removed.

        Identifies ancilla param indices in both saved and current configs,
        extracts system-only params, and copies them. The saved layout uses the saved ancilla settings
        (files without the do_ancilla_1q_gates key: the current one).
        """
        my_ancilla_idx = set(self._get_ancilla_param_indices()) if self.ancilla else set()

        saved_ancilla_idx = set()
        if saved_ancilla:
            # Temporarily swap config to compute saved model's ancilla indices
            keys = ("extra_ancilla", "ancilla_topology", "do_ancilla_1q_gates", "ancilla_coupling_layers")
            orig = {k: getattr(CFG, k) for k in keys}
            CFG.extra_ancilla = saved_ancilla
            CFG.ancilla_topology = saved_topology or CFG.ancilla_topology
            CFG.do_ancilla_1q_gates = CFG.do_ancilla_1q_gates if saved_1q_gates is None else saved_1q_gates
            CFG.ancilla_coupling_layers = saved_coupling_layers if saved_coupling_layers is not None else "all"
            try:
                saved_ancilla_idx = set(self._get_ancilla_param_indices())
            finally:
                for k, v in orig.items():
                    setattr(CFG, k, v)

        saved_system = [p for i, p in enumerate(saved_params) if i not in saved_ancilla_idx]
        my_system_idx = [i for i in range(self.n_params) if i not in my_ancilla_idx]

        n_copy = min(len(saved_system), len(my_system_idx))
        with torch.no_grad():
            for j in range(n_copy):
                self.params[my_system_idx[j]] = float(saved_system[j])

    def _refresh_state(self):
        """Recompute cached statevector from current parameters."""
        if CFG.use_choi:
            with torch.no_grad():
                self.total_gen_state = self.get_total_gen_state()