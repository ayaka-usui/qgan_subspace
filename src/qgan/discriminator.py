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
"""Discriminator module — PyTorch-PennyLane version.

The discriminator is NOT a quantum circuit, it is a classical parametrisation
of two Hermitian operators (psi, phi) built from Pauli matrices, of one of two types (CFG.dis_type):
    "local"  : tensor products of single-qubit operators (default),
                   psi = H(alpha_0) ⊗ ... ⊗ H(alpha_{n-1}),   H(c) = c_0 I + c_1 X + c_2 Y + c_3 Z,
               alpha, beta of shape (n, 4);
    "global" : Pauli expansions over the n-qubit Pauli strings P = sigma_{j_0} ⊗ ... ⊗ sigma_{j_{n-1}},
                   psi = sum_P alpha_P P,
               one coefficient per string (4^n, or only the strings of weight <= CFG.dis_global_max_weight),
               i.e. any Hermitian operator: the coefficients of the correlations between qubits are independent
               instead of being products of single-qubit coefficients. The cost function is the same.

Changes from PennyLane/numpy version:
    - alpha and beta are now torch.nn.Parameter -> autograd handles ALL gradients.
    - Manual gradient methods (_compute_grad, _grad_alpha, _grad_beta,
      _grad_psi_or_phi) are REMOVED entirely.
    - Momentum SGD via torch.optim.SGD(momentum=...).
    - Loss computed via torch.vdot()
    - save/load preserved, adapted for torch state_dict.

The discriminator MAXIMISES the Wasserstein cost:
    Loss = psi_term − phi_term − reg_term
We put a minus sign and call optimizer.step() to minimise (−Loss).
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from config import CFG
from qgan.precision import cdtype, rdtype
from qgan.cost_functions import _calc_wasserstein
from qgan.dis_spectral import DisSpectrum, spectral_critic_cost
from tools.data_managers import print_and_log

# -- PAULI MATRICES ---------------------------------
I = torch.eye(2, dtype=torch.complex128)
X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)


PAULIS = [I, X, Y, Z]
_PAULI_STACK = torch.stack(PAULIS)  # (4, 2, 2)

_DIS_TYPES = ("local", "global")


# -- GLOBAL (PAULI-EXPANSION) CRITIC ---------------------------------
# A Pauli string on n qubits is indexed by j = j_0 4^(n-1) + ... + j_(n-1) (j_i = 0, 1, 2, 3 for I, X, Y, Z), with
# qubit 0 the left-most factor, as torch.kron in Discriminator.get_psi_and_phi.
def pauli_string_weights(n: int) -> np.ndarray:
    """Weight (number of non-identity factors) of each of the 4^n Pauli strings."""
    digits = (np.arange(4**n)[:, None] // 4 ** np.arange(n - 1, -1, -1)) % 4
    return (digits != 0).sum(axis=1)


def allowed_pauli_strings(n: int, max_weight: Optional[int]) -> Optional[np.ndarray]:
    """Indices of the strings of weight <= max_weight, or None for all 4^n strings."""
    if max_weight is None or max_weight >= n:
        return None
    return np.flatnonzero(pauli_string_weights(n) <= max_weight)


def pauli_expand(coeffs: torch.Tensor, n: int) -> torch.Tensor:
    r"""\sum_j c_j \sigma_{j_0} \otimes ... \otimes \sigma_{j_{n-1}} for the 4^n coefficients c (real or complex).

    Contracts one qubit at a time, O(n 4^n), without forming the 4^n Pauli matrices; differentiable in c.
    Returns the (2^n, 2^n) operator in the working complex dtype.
    """
    S = _PAULI_STACK.to(cdtype())
    M = coeffs.to(S.dtype).reshape(4, 4 ** (n - 1), 1, 1)  # (j_0, other qubits, row, column)
    for k in range(n):
        M = torch.einsum("jrAB,jab->rAaBb", M, S)  # contract qubit k: (rest, row, a_k, column, b_k)
        r, A = M.shape[0], M.shape[1]
        M = M.reshape(r, 2 * A, 2 * A)
        if k < n - 1:
            M = M.reshape(4, r // 4, 2 * A, 2 * A)
    return M.reshape(2**n, 2**n)


def product_to_pauli(rows: torch.Tensor) -> torch.Tensor:
    r"""Pauli coefficients of the product \otimes_i (r_{i0} I + r_{i1} X + r_{i2} Y + r_{i3} Z) of the rows (n, 4):
    the Kronecker product r_0 \otimes r_1 \otimes ... (4^n)."""
    c = rows[0]
    for i in range(1, rows.shape[0]):
        c = torch.kron(c, rows[i])
    return c


def _check_dis_options() -> None:
    """Combinations of the critic options that are not defined (CFG fields can be set without validation)."""
    if CFG.dis_type not in _DIS_TYPES:
        raise ValueError(f"Unknown dis_type {CFG.dis_type!r}; expected one of {_DIS_TYPES}")
    mw = CFG.dis_global_max_weight
    if mw is not None:
        if CFG.dis_type != "global":
            raise ValueError("dis_global_max_weight only applies to dis_type='global' (set it to None).")
        if not (isinstance(mw, (int, np.integer)) and mw >= 1):
            raise ValueError(f"dis_global_max_weight must be None or an integer >= 1, got {mw!r}")
    if CFG.dis_eval not in ("spectral", "exponentiation"):
        raise ValueError(f"Unknown dis_eval {CFG.dis_eval!r}; expected 'spectral' or 'exponentiation'")
    if CFG.dis_eval == "spectral" and CFG.dis_type != "local":
        raise ValueError("dis_eval='spectral' needs dis_type='local' (set dis_eval='exponentiation' for 'global').")


def _as_state(g: torch.Tensor) -> torch.Tensor:
    """Pure state (d,) or (d, 1) -> (d,); branches (n_branches, d) of a mixed state ("trace" ancilla mode) are kept."""
    return g if (g.dim() == 2 and g.shape[1] != 1) else g.reshape(-1)


# -- DISCRIMINATOR ---------------------------------
# (CFG.lamb is read when the matrices are computed, as the cost functions read CFG.lamb, cst1-3: a change of CFG.lamb
# after this module is imported applies to every term)


def capped_rows(coeffs: torch.Tensor, cap: float, n_scaled: Optional[int] = None) -> torch.Tensor:
    r"""Rows (n, 4) of a local critic scaled down so that its operator norm prod_i (|c_i0| + |\vec c_i|) is at most
    cap (CFG.dis_module_cap): if it is larger, the first n_scaled rows (default all) are all multiplied by the same
    factor; the directions are unchanged."""
    m = coeffs.shape[0] if n_scaled is None else n_scaled
    norm = (coeffs[:, 0].abs() + coeffs[:, 1:].norm(dim=1)).prod()
    if not norm > cap or m == 0:
        return coeffs
    out = coeffs.clone()
    out[:m] = coeffs[:m] * (cap / norm) ** (1.0 / m)
    return out


class Discriminator(nn.Module):
    """Discriminator for the Quantum Wasserstein GAN

    Parameters alpha, beta are nn.Parameters. The full forward pass
    (alpha,beta -> psi, phi -> A,B -> loss) is differentiable via autograd.

    Usage:
        dis = Discriminator()
        loss = dis.compute_loss(final_target_state, final_gen_state)
        loss.backward()
        dis.optimizer.step()
        dis.optimizer.zero_grad()
    """

    def __init__(self):
        super().__init__()

        # Total number of qubits the discriminator acts on
        # Since we are applying the measurement to each qubit, the size changes
        # if we are using Choi method (2* system size) or Haar Batching (system size)
        self.size: int = CFG.system_size * (2 if CFG.use_choi else 1) + (
            1 if CFG.extra_ancilla and CFG.ancilla_mode == "pass" else 0
        )

        # Critic type (CFG.dis_type): "local" tensor products or "global" Pauli expansions
        _check_dis_options()
        self.dis_type: str = CFG.dis_type
        self.max_weight: Optional[int] = CFG.dis_global_max_weight if CFG.dis_type == "global" else None
        if self.dis_type == "global" and not CFG.dis_ancilla_row_trainable and CFG.dis_new_ancilla_row != "identity":
            print_and_log(
                "Note: global critic with dis_ancilla_row_trainable=False: the coefficients of the strings that act on "
                "the ancilla keep their values; only with dis_new_ancilla_row='identity' (all 0) is this psi_sys ⊗ I.\n",
                CFG.log_path,
            )

        # alpha and beta as trainable parameters (real-valued)
        alpha0 = -1 + 2 * torch.rand(self.size, 4, dtype=rdtype())
        beta0 = -1 + 2 * torch.rand(self.size, 4, dtype=rdtype())
        # The last rows as drawn: the factor of an ancilla added when loading (dis_new_ancilla_row = "random")
        self._new_ancilla_rows = (alpha0[-1].clone(), beta0[-1].clone())
        if CFG.dis_module_cap is not None:  # operator norms ||psi||, ||phi|| at most dis_module_cap (see config)
            alpha0, beta0 = (capped_rows(c, CFG.dis_module_cap) for c in (alpha0, beta0))
        if self.dis_type == "local":
            self.alpha = nn.Parameter(alpha0)
            self.beta = nn.Parameter(beta0)
        else:
            # The global critic starts as the Pauli expansion of the local one drawn above (same seed -> the same
            # psi, phi), restricted to the allowed strings.
            self.strings: Optional[np.ndarray] = allowed_pauli_strings(self.size, self.max_weight)
            ca, cb = (self._select(product_to_pauli(c)) for c in (alpha0, beta0))
            self.alpha = nn.Parameter(ca)
            self.beta = nn.Parameter(cb)
            if self.strings is not None:  # the truncation changed the norm
                self.cap_norm()

        # Optimizer: SGD with momentum, MAXIMISING (critic learning rate: CFG.dis_l_rate, or the shared l_rate)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=CFG.l_rate if CFG.dis_l_rate is None else CFG.dis_l_rate,
            momentum=CFG.momentum_coeff,
        )
        if self.dis_type == "global" and self.strings is None and CFG.dis_l_rate is None:
            print_and_log(
                f"Note: global critic with the shared l_rate: its step on psi is l_rate * 2^{self.size} = "
                f"{CFG.l_rate * 2**self.size:g} x dW/dpsi. If the cost falls far below its initial value and the fidelity "
                "collapses, set dis_l_rate (l_rate / 8 was stable at 6 qubits).\n",
                CFG.log_path,
            )

        # Save/load compatibility
        self.ancilla: bool = CFG.extra_ancilla
        self.ancilla_mode: str = CFG.ancilla_mode
        self.target_size: int = CFG.system_size
        self.target_hamiltonian: str = CFG.target_hamiltonian
        # Detached A, B, psi, phi (or eigenbases) of the current parameters, reused until they change
        self._matrices_cache = None
        self._spectrum_cache = None

    # -- global critic: coefficient bookkeeping ---------------------------------
    def _select(self, c_full: torch.Tensor) -> torch.Tensor:
        """4^n coefficients -> the allowed strings (global critic)."""
        return c_full if self.strings is None else c_full[torch.as_tensor(self.strings)]

    def _full(self, c: torch.Tensor) -> torch.Tensor:
        """Allowed-string coefficients -> all 4^n (zeros elsewhere); differentiable."""
        if self.strings is None:
            return c
        return c.new_zeros(4**self.size).index_copy(0, torch.as_tensor(self.strings), c)

    @torch.no_grad()
    def cap_norm(self) -> None:
        """CFG.dis_module_cap: scale psi and phi down to that operator norm if they exceed it (no-op if None).
        Local critic: all rows by the same factor, except a frozen ancilla row (dis_ancilla_row_trainable = False),
        which keeps its value. Global critic: all coefficients (the operator norm from its eigenvalues)."""
        cap = CFG.dis_module_cap
        if cap is None:
            return
        if self.dis_type == "global":
            for p in (self.alpha, self.beta):
                norm = torch.linalg.eigvalsh(pauli_expand(self._full(p), self.size)).abs().max()
                if norm > cap:
                    p.mul_(cap / norm.to(p.dtype))
            return
        frozen = not CFG.dis_ancilla_row_trainable and self.ancilla and self.ancilla_mode == "pass"
        for p in (self.alpha, self.beta):
            p.copy_(capped_rows(p, cap, p.shape[0] - 1 if frozen else None))

    def ancilla_factor_mask(self) -> torch.Tensor:
        """Boolean mask (shaped like alpha) of the coefficients that act on the ancilla (the critic's last qubit):
        its row (local) or the strings with a non-identity factor on it (global)."""
        if self.dis_type == "local":
            mask = torch.zeros(self.alpha.shape, dtype=torch.bool)
            mask[-1] = True
            return mask
        idx = torch.arange(4**self.size) if self.strings is None else torch.as_tensor(self.strings)
        return idx % 4 != 0

    # -- matrix representations ---------------------------------
    def get_psi_and_phi(self) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Build psi and phi matrices via Kronecker product of per-qubit Hermitians.

        For each qubit i:
            H_i = \sum_j coeff[i][j] · \sigma_j    (real coeffs, Hermitian result)

        Full operator: psi = H_0 \otimes H_1 \otimes ... \otimes H_{N-1}

        Global critic: psi = sum_P alpha_P P over the (allowed) Pauli strings (pauli_expand).

        Returns:
            (\psi, \phi) each of shape (2^N, 2^N), on the autograd graph.
        """
        if self.dis_type == "global":
            return pauli_expand(self._full(self.alpha), self.size), pauli_expand(self._full(self.beta), self.size)

        # Cast alpha, beta for matrix multiplication
        alpha_c = self.alpha.to(cdtype())
        beta_c = self.beta.to(cdtype())

        psi = torch.tensor([[1.0]], dtype=cdtype())  # scalar 1 as 1×1 matrix
        phi = torch.tensor([[1.0]], dtype=cdtype())

        paulis = [P.to(alpha_c.dtype) for P in PAULIS]  # master copies are complex128
        for i in range(self.size):
            # Per-qubit Hermitian
            psi_i = sum(alpha_c[i, j] * paulis[j] for j in range(4))
            phi_i = sum(beta_c[i, j] * paulis[j] for j in range(4))
            psi = torch.kron(psi, psi_i)
            phi = torch.kron(phi, phi_i)

        return psi, phi

    def get_dis_matrices_rep(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Compute A = exp(−phi / lambda) and B = exp(+\psi / lambda).

        Returns:
            (A, B, psi, phi)
        """
        psi, phi = self.get_psi_and_phi()
        A = torch.linalg.matrix_exp((-1.0 / CFG.lamb) * phi)
        B = torch.linalg.matrix_exp((1.0 / CFG.lamb) * psi)
        return A, B, psi, phi

    # -- detached representations, computed once per parameter update ---------------------------------
    def _cache_valid(self, cache) -> bool:
        """Whether a cache entry was made with the current alpha, beta (the same tensors, not modified since: every
        in-place change, such as an optimizer step, a load or cap_norm, increments a tensor's _version) and lamb."""
        return (cache is not None and cache[0] is self.alpha and cache[1] == self.alpha._version
                and cache[2] is self.beta and cache[3] == self.beta._version and cache[4] == float(CFG.lamb))

    def _cache_entry(self, value) -> tuple:
        return (self.alpha, self.alpha._version, self.beta, self.beta._version, float(CFG.lamb), value)

    def detached_matrices(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(A, B, psi, phi) of the current parameters, detached (no autograd graph): the generator steps and the
        logged cost use the same critic until its next update, so they are computed once (the same values)."""
        if not self._cache_valid(self._matrices_cache):
            with torch.no_grad():
                matrices = tuple(m.detach() for m in self.get_dis_matrices_rep())
            self._matrices_cache = self._cache_entry(matrices)
        return self._matrices_cache[5]

    def detached_spectrum(self) -> DisSpectrum:
        """The eigenbases and eigenvalues of psi and phi (dis_eval = "spectral"), detached, as detached_matrices."""
        if not self._cache_valid(self._spectrum_cache):
            spectrum = DisSpectrum(self.alpha.detach().cpu().numpy(), self.beta.detach().cpu().numpy(), rdtype(), cdtype())
            self._spectrum_cache = self._cache_entry(spectrum)
        return self._spectrum_cache[5]

    # -- loss computation ---------------------------------
    def compute_loss(self, final_target_state: torch.Tensor, final_gen_state: torch.Tensor) -> torch.Tensor:
        """Compute the Wasserstein loss as a differentiable scalar."""
        # pure state (1D or column) or (n_branches, d) branches of a mixed state ("trace" ancilla mode)
        g = _as_state(final_gen_state)
        t = final_target_state.reshape(-1)

        if CFG.dis_eval == "spectral":  # exp(-phi/lamb), exp(psi/lamb) in the eigenbases; exact gradient
            return -spectral_critic_cost(self.alpha, self.beta, g.unsqueeze(0), t)[0]

        dis_matrices = self.get_dis_matrices_rep()
        return -(_calc_wasserstein(g, t, dis_matrices))

    def compute_loss_batch(self, final_target_states: list, final_gen_states: list) -> torch.Tensor:
        """Mean of compute_loss over a batch of (target, generator state) pairs (batch mode), with the critic's
        matrices exponentiated once (they do not depend on the input state)."""
        n = len(final_gen_states)
        if CFG.dis_eval == "spectral":
            G = torch.stack([_as_state(g) for g in final_gen_states])
            T = torch.stack([t.reshape(-1) for t in final_target_states])
            return -spectral_critic_cost(self.alpha, self.beta, G, T).sum() / n

        dis_matrices = self.get_dis_matrices_rep()
        loss = torch.tensor(0.0, dtype=rdtype())
        for t, g in zip(final_target_states, final_gen_states):
            loss = loss + (-(_calc_wasserstein(_as_state(g), t.reshape(-1), dis_matrices)))
        return loss / n

    # -- save / load ---------------------------------
    def save_model(self, file_path: str):
        """Save discriminator state to disk.

        Saves both the nn.Module state_dict (alpha, beta) and metadata
        needed for compatibility checks.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        save_dict = {
            "state_dict": self.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "size": self.size,
            "ancilla": self.ancilla,
            "ancilla_mode": self.ancilla_mode,
            "target_size": self.target_size,
            "target_hamiltonian": self.target_hamiltonian,
            "dis_type": self.dis_type,
            "dis_global_max_weight": self.max_weight,
        }
        torch.save(save_dict, file_path)

    def load_model_params(self, file_path: str) -> bool:
        """Load discriminator parameters from a saved model (torch format only)."""
        if not os.path.exists(file_path):
            print_and_log("Discriminator model file not found\n", CFG.log_path)
            return False

        try:
            saved = torch.load(file_path, weights_only=False)
        except Exception as e:
            print_and_log(
                f"ERROR: Could not load discriminator model: {e}\nOld pickle/numpy formats are no longer supported.\n",
                CFG.log_path,
            )
            return False

        if not isinstance(saved, dict) or "state_dict" not in saved:
            print_and_log(
                "ERROR: Unrecognised format. Only torch dict format is supported.\n",
                CFG.log_path,
            )
            return False

        return self._load_from_torch_format(saved)

    def _load_from_torch_format(self, saved: dict) -> bool:
        """Load from new torch save format."""
        # Compatibility checks
        if saved.get("target_size") != self.target_size:
            print_and_log("ERROR: target size mismatch.\n", CFG.log_path)
            return False
        if saved.get("target_hamiltonian") != self.target_hamiltonian:
            print_and_log("ERROR: target hamiltonian mismatch.\n", CFG.log_path)
            return False

        saved_type = saved.get("dis_type", "local")  # files saved before dis_type existed are local critics
        if self.dis_type == "global":
            return self._load_into_global(saved, saved_type)
        if saved_type != "local":
            print_and_log("ERROR: a global critic cannot be loaded into a local one (dis_type='local').\n", CFG.log_path)
            return False

        if saved.get("size") == self.size:
            self.load_state_dict(saved["state_dict"])
            # SGD momentum: restored (fork) unless CFG.reset_optimizer_on_load (upstream behaviour).
            if not CFG.reset_optimizer_on_load and "optimizer_state" in saved:
                self.optimizer.load_state_dict(saved["optimizer_state"])
            self.cap_norm()
            print_and_log("Discriminator parameters loaded (torch format).\n", CFG.log_path)
            return True

        # \pm1 qubit (ancilla difference)
        if abs(saved.get("size", 0) - self.size) == 1:
            saved_alpha = saved["state_dict"]["alpha"]
            saved_beta = saved["state_dict"]["beta"]
            min_size = min(saved_alpha.shape[0], self.size)
            with torch.no_grad():
                self.alpha[:min_size] = saved_alpha[:min_size].clone()
                self.beta[:min_size] = saved_beta[:min_size].clone()
                if self.size > min_size and CFG.dis_new_ancilla_row == "identity":
                    # factor I on the inserted ancilla qubit: psi -> psi (x) I, phi -> phi (x) I
                    eye_row = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=self.alpha.dtype)
                    self.alpha[min_size:] = eye_row
                    self.beta[min_size:] = eye_row
                elif self.size > min_size and CFG.dis_new_ancilla_row == "random":
                    # the construction draw (before any dis_module_cap, which applies to the whole critic below)
                    self.alpha[min_size:] = self._new_ancilla_rows[0]
                    self.beta[min_size:] = self._new_ancilla_rows[1]
                elif CFG.dis_new_ancilla_row not in ("random", "identity"):
                    raise ValueError(f"Unknown dis_new_ancilla_row: {CFG.dis_new_ancilla_row}")
            self.cap_norm()
            print_and_log(
                "Discriminator parameters partially loaded (\\pm1 qubit, torch).\n",
                CFG.log_path,
            )
            return True

        print_and_log("ERROR: incompatible discriminator (size mismatch).\n", CFG.log_path)
        return False

    def _load_into_global(self, saved: dict, saved_type: str) -> bool:
        """Load a saved local or global critic into this global one, with the same ±1 qubit rules as the local load.

        local saved : its rows are first adjusted as in the local load (an inserted ancilla gets the row of
                      dis_new_ancilla_row, a removed one loses its row), then expanded: the same psi, phi as loading
                      them into a local critic.
        global saved: an inserted ancilla multiplies the critic by its factor, psi -> psi ⊗ H(r) (coefficients
                      c ⊗ r); a removed one is traced out, c_P <- c_{P ⊗ I} (psi -> Tr_a(psi) / 2).
        The result is restricted to this critic's allowed strings (dis_global_max_weight). The SGD momentum is restored
        only for the same type, size and strings (and not CFG.reset_optimizer_on_load).
        """
        n_saved = saved.get("size", 0)
        if abs(n_saved - self.size) > 1:
            print_and_log("ERROR: incompatible discriminator (size mismatch).\n", CFG.log_path)
            return False
        if CFG.dis_new_ancilla_row not in ("random", "identity"):
            raise ValueError(f"Unknown dis_new_ancilla_row: {CFG.dis_new_ancilla_row}")
        sd = saved["state_dict"]
        dt = self.alpha.dtype
        eye_row = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dt)
        new_rows = (eye_row, eye_row) if CFG.dis_new_ancilla_row == "identity" else self._new_ancilla_rows

        saved_strings = None
        if saved_type == "local":
            rows = [sd["alpha"].to(dt), sd["beta"].to(dt)]
            if n_saved == self.size - 1:
                rows = [torch.cat([r, new[None]]) for r, new in zip(rows, new_rows)]
            elif n_saved == self.size + 1:
                rows = [r[:-1] for r in rows]
            full = [product_to_pauli(r) for r in rows]
        else:
            saved_strings = allowed_pauli_strings(n_saved, saved.get("dis_global_max_weight"))
            full = []
            for key in ("alpha", "beta"):
                c = sd[key].to(dt)
                if saved_strings is not None:
                    c = c.new_zeros(4**n_saved).index_copy(0, torch.as_tensor(saved_strings), c)
                full.append(c)
            if n_saved == self.size - 1:
                full = [torch.kron(c, new) for c, new in zip(full, new_rows)]
            elif n_saved == self.size + 1:
                full = [c.reshape(-1, 4)[:, 0] for c in full]

        with torch.no_grad():
            self.alpha.copy_(self._select(full[0]))
            self.beta.copy_(self._select(full[1]))
        same = saved_type == "global" and n_saved == self.size and (
            (saved_strings is None and self.strings is None)
            or (saved_strings is not None and self.strings is not None and np.array_equal(saved_strings, self.strings))
        )
        if same and not CFG.reset_optimizer_on_load and "optimizer_state" in saved:
            self.optimizer.load_state_dict(saved["optimizer_state"])
        self.cap_norm()
        how = "" if same else f" (converted from a {saved_type} critic on {n_saved} qubits)"
        print_and_log(f"Discriminator parameters loaded into the global critic{how}.\n", CFG.log_path)
        return True
