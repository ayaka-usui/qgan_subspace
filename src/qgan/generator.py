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
_apply_momentum_step) with PyTorch autograd. The QNode uses interface="torch"
and diff_method="backprop", so the statevector is a torch tensor on the
autograd graph. The loss (Wasserstein cost) is computed entirely in torch
from the statevector + discriminator matrices, and loss.backward() gives
exact gradients.

Everything else is preserved:
    - Choi representation: I \otimes G applied to |\Phi^+> (with optional ancilla |0>).
    - Three ansatz modes: ZZ_Z_X, ZZ_YY_XX_Z, and custom.
    - Full ancilla support: topologies and post-processing modes via config.
    - Save/load with backward compatibility.
"""

import os
import pickle
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

from config import CFG
from tools.data_managers import print_and_log
from qgan.ancilla import get_final_gen_state_torch

# Wasserstein cost constants from config
cst1, cst2, cst3, lamb = CFG.cst1, CFG.cst2, CFG.cst3, CFG.lamb


# -- GATE TERM DEFINITIONS ------------------------------------------------
# Same as original generator.py — maps term strings to (PennyLane gate, n_qubits)
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

# Predefined ansatz
_PREDEFINED_ANSATZ = {
    "ZZ_Z_X":     ["X", "Z", "ZZ"],
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
        f"Unknown ansatz: {CFG.gen_ansatz}. "
        f"Expected one of: {list(_PREDEFINED_ANSATZ.keys())} or 'custom'."
    )


# -- DEVICE ----------------------------------------------------------------
def _make_device(total_wires: int):
    """Create a PennyLane device for the full Choi + generator register."""
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

        for _ in range(CFG.gen_layers):
            # -- System gates from term list --
            for term in terms:
                if term in _1Q_GATES:
                    gate_fn = _1Q_GATES[term]
                    for w in system_wires:
                        gate_fn(params[idx], wires=w);  idx += 1
                elif term in _2Q_GATES:
                    gate_fn = _2Q_GATES[term]
                    for i in range(n_sys - 1):
                        gate_fn(params[idx], wires=[system_wires[i], system_wires[i + 1]])
                        idx += 1
                else:
                    raise ValueError(f"Unknown gate term: {term}")

            # -- Ancilla 1q gates --
            if ancilla_wire is not None and CFG.do_ancilla_1q_gates:
                for term in terms:
                    if term in _1Q_GATES:
                        _1Q_GATES[term](params[idx], wires=ancilla_wire);  idx += 1

            # -- Ancilla 2q couplings --
            if ancilla_wire is not None:
                idx = Ansatz._apply_ancilla_couplings(
                    params, idx, terms, system_wires, ancilla_wire
                )

    @staticmethod
    def _apply_ancilla_couplings(params, idx: int, terms: list[str],
                                  system_wires: list[int],
                                  ancilla_wire: int) -> int:
        """Apply ancilla 2q couplings according to topology.

        Uses the same 2q gate types from the term list.
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

        # Collect the 2q gate functions from the term list
        gates_2q = [_2Q_GATES[t] for t in terms if t in _2Q_GATES]

        def _apply_coupling(wire_a, wire_b):
            nonlocal idx
            for gate_fn in gates_2q:
                gate_fn(params[idx], wires=[wire_a, wire_b]);  idx += 1

        if topo == "total":
            for w in system_wires:
                _apply_coupling(w, ancilla_wire)

        if topo == "bridge":
            _apply_coupling(system_wires[0], ancilla_wire)

        if topo in ("bridge", "ansatz"):
            _apply_coupling(connect_wire, ancilla_wire)

        if topo == "fake" and n_sys > 2:
            _apply_coupling(system_wires[0], connect_wire)

        return idx


# -- PARAMETER COUNTING ---------------------------------------------------
def count_params(n_system: int, has_ancilla: bool) -> int:
    """Total number of trainable angles for the current ansatz + config."""
    terms = _get_ansatz_terms()
    n = 0

    n_1q_terms = sum(1 for t in terms if t in _1Q_GATES)
    n_2q_terms = sum(1 for t in terms if t in _2Q_GATES)

    for _ in range(CFG.gen_layers):
        n += n_1q_terms * n_system
        n += n_2q_terms * (n_system - 1)

        if has_ancilla and CFG.do_ancilla_1q_gates:
            n += n_1q_terms

        if has_ancilla:
            n += _count_ancilla_coupling_params(n_2q_terms, n_system)

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


# -- WIRE LAYOUT -----------------------------------------------------------
def _wire_layout():
    """Compute wire indices for each register.

    Layout:  [ choi_register | gen_system | (ancilla) ]
    """
    s = CFG.system_size
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

    return choi_wires, gen_wires, ancilla_wire, total_wires


# -- QNODE -----------------------------------------------------------------
def _build_qnode():
    """Build the QNode for the generator circuit.

    Uses interface="torch" and diff_method="backprop" so the output
    statevector is a torch tensor on the autograd graph.

    1) Prepare maximally-entangled state (Choi) via H + CNOT.
    2) Apply ansatz on generator register.
    3) Return full statevector (as a torch tensor).
    """
    choi_wires, gen_wires, ancilla_wire, total_wires = _wire_layout()
    dev = _make_device(total_wires)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(params):
        gen_system_wires = [w for w in gen_wires if w != ancilla_wire]
        for c_wire, g_wire in zip(choi_wires, gen_system_wires):
            qml.Hadamard(wires=c_wire)
            qml.CNOT(wires=[c_wire, g_wire])

        Ansatz.apply(params, gen_wires, ancilla_wire)
        return qml.state()

    return circuit

# -- GENERATOR CLASS -------------------------------------------
class Generator:
    """Generator for the Quantum WGAN (PyTorch + PennyLane version).

    The circuit parameters are a torch tensor (requires_grad=True).
    The QNode returns a torch statevector, and the full loss is computed
    in torch so autograd handles all gradients.

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
        self.ansatz: str = CFG.gen_ansatz
        self.layers: int = CFG.gen_layers
        self.target_size: int = CFG.system_size
        self.target_hamiltonian: str = CFG.target_hamiltonian

        # -- parameters (torch tensor, requires_grad) --
        n_params = count_params(CFG.system_size, CFG.extra_ancilla)
        self.n_params: int = n_params
        self.params: torch.Tensor = self._init_params(n_params)

        # -- circuit (QNode with torch interface) --
        self.circuit = _build_qnode()

        # -- optimizer: SGD with momentum, MINIMISING --
        self.optimizer = torch.optim.SGD(
            [self.params],
            lr=CFG.l_rate,
            momentum=CFG.momentum_coeff,
        )

        # -- cached state --
        self.total_gen_state: torch.Tensor = self.get_total_gen_state()

    # -- parameter initialisation ------------------------------------------
    def _init_params(self, n_params: int) -> torch.Tensor:
        """Random uniform in [0, 2\pi), as a torch tensor with requires_grad=True.

        Respects start_ancilla_gates_randomly: if False, ancilla gate params
        are initialised to 0.
        """
        params = torch.empty(n_params, dtype=torch.float64).uniform_(0, 2 * np.pi)

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

        for _ in range(CFG.gen_layers):
            # System 1q + 2q gates (skip)
            idx += n_1q_terms * n_sys
            idx += n_2q_terms * (n_sys - 1)

            # Ancilla 1q gates
            if CFG.do_ancilla_1q_gates:
                for _ in range(n_1q_terms):
                    indices.append(idx);  idx += 1

            # Ancilla 2q couplings
            n_anc_2q = _count_ancilla_coupling_params(n_2q_terms, n_sys)
            for _ in range(n_anc_2q):
                indices.append(idx);  idx += 1

        return indices

    # -- forward pass ------------------------------------------------------
    def get_total_gen_state(self) -> torch.Tensor:
        """Run the circuit and return the full statevector as a 1D torch tensor.

        The returned tensor is on the autograd graph .
        """
        return self.circuit(self.params)

    def get_final_gen_state(self, total_gen_state: torch.Tensor) -> torch.Tensor:
        """Apply ancilla post-processing and return the final state for the discriminator.

        Returns a column vector (d, 1) torch tensor.
        """
        return get_final_gen_state_torch(total_gen_state)

    # -- loss computation --------------------------------------------------
    def compute_loss(self, dis, final_target_state: torch.Tensor) -> torch.Tensor:
        """Compute the generator's Wasserstein loss as a differentiable scalar.

        Uses density matrices and torch.trace, same formulation as the
        discriminator's compute_loss:
            <psi|O|psi> = Tr[O · \rho]
            <\Phi|O|psi> · <psi|O'|\Phi> = Tr[O · \rho_psi · O' · \rho_\Phi]

        The generator MINIMISES:
            Loss = psi_term − \Phi_term − reg_term

        Args:
            dis: Discriminator (torch version, from discriminator_torch.py).
            final_target_state: Target state, shape (d,) or (d, 1), complex128.

        Returns:
            torch.Tensor: scalar loss for minimisation.
        """
        # Forward pass through the circuit
        total_gen_state = self.get_total_gen_state()
        self.total_gen_state = total_gen_state  

        # Ancilla post-processing
        final_gen_state = self.get_final_gen_state(total_gen_state)

        # Get discriminator matrices (detached — no gradient flow to discriminator)
        with torch.no_grad():
            A, B, psi, phi = dis.get_dis_matrices_rep()
        A = A.detach()
        B = B.detach()
        psi = psi.detach()
        phi = phi.detach()

        # Build density matrices
        g = final_gen_state.reshape(-1)
        t = final_target_state.reshape(-1)
        rho_g = torch.outer(g, g.conj())  # |gen><gen|
        rho_t = torch.outer(t, t.conj())  # |target><target|

        # psi term: Tr[psi · \rho_t]
        psi_term = torch.trace(psi @ rho_t)

        # \Phi term: Tr[\Phi · \rho_g] 
        phi_term = torch.trace(phi @ rho_g)

        # Regularisation: cross terms collapse into single traces
        t1 = torch.trace(A @ rho_g) * torch.trace(B @ rho_t)
        t2 = torch.trace(A @ rho_g @ B @ rho_t)
        t3 = torch.trace(A @ rho_t @ B @ rho_g)
        t4 = torch.trace(A @ rho_t) * torch.trace(B @ rho_g)
        reg_term = lamb / np.e * (cst1 * t1 - cst2 * (t2 + t3) + cst3 * t4)

        # Full cost
        cost = (psi_term - phi_term - reg_term).real

        # Generator minimises the cost
        return cost

    # -- training step (replaces update_gen) --------------------------------
    def update_gen(self, dis, final_target_state: torch.Tensor):
        """One generator optimisation step (minimisation).

        Drop-in replacement for the old update_gen(). Computes the loss,
        backpropagates through the QNode + ancilla processing + brakets,
        and steps the optimizer.

        Args:
            dis: Discriminator (torch version).
            final_target_state: Target state as torch tensor, shape (d, 1).
                If you have a numpy array, convert with:
                    torch.tensor(np.asarray(state), dtype=torch.complex128)
        """
        self.optimizer.zero_grad()
        loss = self.compute_loss(dis, final_target_state)
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            self.total_gen_state = self.get_total_gen_state()

    # -- pickle support ----------------------------------------------------
    def __getstate__(self):
        """Exclude non-picklable objects (QNode, optimizer) from serialisation."""
        state = self.__dict__.copy()
        state.pop('circuit', None)
        state.pop('optimizer', None)
        # Convert torch tensors to numpy for portability
        if isinstance(state.get('params'), torch.Tensor):
            state['params'] = state['params'].detach().numpy()
        if isinstance(state.get('total_gen_state'), torch.Tensor):
            state['total_gen_state'] = state['total_gen_state'].detach().numpy()
        return state

    def __setstate__(self, state):
        """Restore from pickle: rebuild QNode, optimizer, and torch tensors."""
        self.__dict__.update(state)
        # Rebuild circuit
        self.circuit = _build_qnode()
        # Convert params back to torch
        if isinstance(self.params, np.ndarray):
            self.params = torch.tensor(self.params, dtype=torch.float64, requires_grad=True)
        # Rebuild optimizer
        self.optimizer = torch.optim.SGD(
            [self.params],
            lr=CFG.l_rate,
            momentum=CFG.momentum_coeff,
        )
        # Rebuild cached state
        if isinstance(self.total_gen_state, np.ndarray):
            with torch.no_grad():
                self.total_gen_state = self.get_total_gen_state()

    # -- SAVE / LOAD -------------------------------------------------------
    def save_model(self, file_path: str):
        """Save generator state to disk.

        Saves parameters as numpy (portable), plus all metadata
        for compatibility checks.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        save_dict = {
            "params": self.params.detach().numpy(),
            "size": self.size,
            "ancilla": self.ancilla,
            "ancilla_topology": self.ancilla_topology,
            "ansatz": self.ansatz,
            "layers": self.layers,
            "target_size": self.target_size,
            "target_hamiltonian": self.target_hamiltonian,
            "n_params": self.n_params,
            # Optimizer state (momentum buffers etc.)
            "optimizer_state": self.optimizer.state_dict(),
        }
        torch.save(save_dict, file_path)

    def load_model_params(self, file_path: str) -> bool:
        """Load generator parameters from a saved model.
        """
        if not os.path.exists(file_path):
            print_and_log("ERROR: Generator model file not found\n", CFG.log_path)
            return False

        # Try new torch dict format first
        try:
            saved = torch.load(file_path, weights_only=False)
            if isinstance(saved, dict) and "params" in saved:
                return self._load_from_torch_dict(saved)
        except Exception:
            pass

        # Fall back to old pickle format
        try:
            with open(file_path, "rb") as f:
                saved = pickle.load(f)
        except (OSError, pickle.UnpicklingError) as e:
            print_and_log(f"ERROR: Could not load generator model: {e}\n", CFG.log_path)
            return False

        # Old format: pickled Generator object with .params (numpy)
        if hasattr(saved, 'params') and isinstance(saved.params, np.ndarray):
            return self._load_from_old_numpy(saved)
        # Very old format: pickled Generator with .qc.gates
        if hasattr(saved, 'qc'):
            return self._load_from_old_qc(saved)

        print_and_log("ERROR: Unrecognised saved generator format.\n", CFG.log_path)
        return False

    def _load_from_torch_dict(self, saved: dict) -> bool:
        """Load from new torch save_model format."""
        # Compatibility checks
        cant_load = False
        if saved.get("target_size") != self.target_size:
            print_and_log("ERROR: target size mismatch.\n", CFG.log_path);  cant_load = True
        if saved.get("target_hamiltonian") != self.target_hamiltonian:
            print_and_log("ERROR: target hamiltonian mismatch.\n", CFG.log_path);  cant_load = True
        if saved.get("ansatz") != self.ansatz:
            print_and_log("ERROR: ansatz mismatch.\n", CFG.log_path);  cant_load = True
        if saved.get("layers") != self.layers:
            print_and_log("ERROR: layer count mismatch.\n", CFG.log_path);  cant_load = True
        if (saved.get("ancilla") and self.ancilla
                and saved.get("ancilla_topology") != self.ancilla_topology):
            print_and_log("ERROR: ancilla topology mismatch.\n", CFG.log_path);  cant_load = True
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
                self.params.copy_(torch.from_numpy(saved_params).to(torch.float64))
            if "optimizer_state" in saved:
                self.optimizer.load_state_dict(saved["optimizer_state"])
            self._refresh_state()
            print_and_log("Generator parameters loaded (torch dict format).\n", CFG.log_path)
            return True

        # \pm1 qubit (ancilla difference)
        if saved_ancilla != self.ancilla and abs(saved_size - self.size) == 1:
            self._partial_load_params(saved_params, saved_ancilla,
                                      saved.get("ancilla_topology"))
            self._refresh_state()
            print_and_log("Generator parameters partially loaded (ancilla diff, torch dict).\n",
                          CFG.log_path)
            return True

        print_and_log("ERROR: incompatible generator.\n", CFG.log_path)
        return False

    def _load_from_old_numpy(self, saved) -> bool:
        """Load from old pickle format with .params as numpy array."""
        cant_load = False
        if saved.target_size != self.target_size:
            print_and_log("ERROR: target size mismatch.\n", CFG.log_path);  cant_load = True
        if saved.target_hamiltonian != self.target_hamiltonian:
            print_and_log("ERROR: target hamiltonian mismatch.\n", CFG.log_path);  cant_load = True
        if saved.ansatz != self.ansatz:
            print_and_log("ERROR: ansatz mismatch.\n", CFG.log_path);  cant_load = True
        if saved.layers != self.layers:
            print_and_log("ERROR: layer count mismatch.\n", CFG.log_path);  cant_load = True
        if (saved.ancilla and self.ancilla
                and getattr(saved, 'ancilla_topology', None) != self.ancilla_topology):
            print_and_log("ERROR: ancilla topology mismatch.\n", CFG.log_path);  cant_load = True
        if cant_load:
            return False

        if saved.size == self.size and saved.ancilla == self.ancilla:
            if len(saved.params) != self.n_params:
                print_and_log("ERROR: param count mismatch.\n", CFG.log_path)
                return False
            with torch.no_grad():
                self.params.copy_(torch.from_numpy(saved.params).to(torch.float64))
            self._refresh_state()
            print_and_log("Generator parameters loaded (old numpy -> torch).\n", CFG.log_path)
            return True

        if saved.ancilla != self.ancilla and abs(saved.size - self.size) == 1:
            self._partial_load_params(saved.params, saved.ancilla,
                                      getattr(saved, 'ancilla_topology', None))
            self._refresh_state()
            print_and_log("Generator parameters partially loaded (old numpy, ancilla diff).\n",
                          CFG.log_path)
            return True

        print_and_log("ERROR: incompatible generator.\n", CFG.log_path)
        return False

    def _load_from_old_qc(self, saved) -> bool:
        """Load from very old format with .qc.gates list."""
        cant_load = False
        if saved.target_size != self.target_size:
            print_and_log("ERROR: target size mismatch.\n", CFG.log_path);  cant_load = True
        if saved.target_hamiltonian != self.target_hamiltonian:
            print_and_log("ERROR: target hamiltonian mismatch.\n", CFG.log_path);  cant_load = True
        if saved.ansatz != self.ansatz:
            print_and_log("ERROR: ansatz mismatch.\n", CFG.log_path);  cant_load = True
        if saved.layers != self.layers:
            print_and_log("ERROR: layer count mismatch.\n", CFG.log_path);  cant_load = True
        if cant_load:
            return False

        old_angles = np.array([g.angle for g in saved.qc.gates])

        if saved.size == self.size and saved.ancilla == self.ancilla:
            if len(old_angles) != self.n_params:
                print_and_log("ERROR: gate count mismatch.\n", CFG.log_path)
                return False
            with torch.no_grad():
                self.params.copy_(torch.from_numpy(old_angles).to(torch.float64))
            self._refresh_state()
            print_and_log("Generator parameters loaded (old qc format -> torch).\n", CFG.log_path)
            return True

        if saved.ancilla != self.ancilla and abs(saved.size - self.size) == 1:
            self._partial_load_params(old_angles, saved.ancilla,
                                      getattr(saved, 'ancilla_topology', None))
            self._refresh_state()
            print_and_log("Generator parameters partially loaded (old qc, ancilla diff).\n",
                          CFG.log_path)
            return True

        print_and_log("ERROR: incompatible old-format generator.\n", CFG.log_path)
        return False

    def _partial_load_params(self, saved_params: np.ndarray,
                              saved_ancilla: bool,
                              saved_topology: Optional[str]) -> None:
        """Load system-only params when ancilla is added/removed.

        Same logic as original: identify ancilla param indices in both
        saved and current configs, extract system-only params, and copy.
        """
        my_ancilla_idx = set(self._get_ancilla_param_indices()) if self.ancilla else set()

        saved_ancilla_idx = set()
        if saved_ancilla:
            # Temporarily swap config to compute saved model's ancilla indices
            orig_extra = CFG.extra_ancilla
            orig_topo = CFG.ancilla_topology
            orig_1q = CFG.do_ancilla_1q_gates
            CFG.extra_ancilla = saved_ancilla
            CFG.ancilla_topology = saved_topology or CFG.ancilla_topology
            CFG.do_ancilla_1q_gates = getattr(CFG, 'do_ancilla_1q_gates', True)
            saved_ancilla_idx = set(self._get_ancilla_param_indices())
            CFG.extra_ancilla = orig_extra
            CFG.ancilla_topology = orig_topo
            CFG.do_ancilla_1q_gates = orig_1q

        saved_system = [p for i, p in enumerate(saved_params) if i not in saved_ancilla_idx]
        my_system_idx = [i for i in range(self.n_params) if i not in my_ancilla_idx]

        n_copy = min(len(saved_system), len(my_system_idx))
        with torch.no_grad():
            for j in range(n_copy):
                self.params[my_system_idx[j]] = float(saved_system[j])

    def _refresh_state(self):
        """Recompute cached statevector from current parameters."""
        with torch.no_grad():
            self.total_gen_state = self.get_total_gen_state()