"""Discriminator module — PyTorch-PennyLane version.

The discriminator is NOT a quantum circuit, it is a classical parametrisation
of two Hermitian operators (psi, phi) built from tensor products of Pauli matrices.

Changes from PennyLane/numpy version:
    - alpha and beta are now torch.nn.Parameter -> autograd handles ALL gradients.
    - Manual gradient methods (_compute_grad, _grad_alpha, _grad_beta,
      _grad_psi_or_phi) are REMOVED entirely.
    - Momentum SGD via torch.optim.SGD(momentum=...).
    - Loss computed via density matrices and torch.trace
      Cross terms like <target|A|gen>·<gen|B|target> become Tr[A·rho_g·B·rho_t].
    - save/load preserved, adapted for torch state_dict.

The discriminator MAXIMISES the Wasserstein cost:
    Loss = psi_term − phi_term − reg_term
We put a minus sign and call optimizer.step() to minimise (−Loss).
"""

import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from config import CFG
from tools.data_managers import print_and_log


# -- PAULI MATRICES ---------------------------------
I = torch.eye(2, dtype=torch.complex128)
X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)

PAULIS = [I, X, Y, Z]


# -- BRAKET (torch version) ---------------------------------
def braket(*args) -> torch.Tensor:
    """Compute <bra| O_1 · O_2 · ... · O_n |ket> using torch tensors.

    Same interface as the numpy version in cost_functions.py, but
    operates on torch tensors so autograd can differentiate through it.

    Args:
        args: (bra, [operators...], ket) — all torch tensors.
              bra and ket are column vectors (shape [d, 1]).
              Operators are square matrices (shape [d, d]).

    Returns:
        torch.Tensor: scalar (complex) inner product.
    """
    bra, *ops, ket = args
    for op in ops:
        ket = torch.matmul(op, ket)
    return torch.matmul(bra.conj().T, ket)


# -- DISCRIMINATOR ---------------------------------
# Wasserstein cost constants from config
cst1, cst2, cst3, lamb = CFG.cst1, CFG.cst2, CFG.cst3, CFG.lamb


class Discriminator(nn.Module):
    """Discriminator for the Quantum Wasserstein GAN (PyTorch version).

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
        self.size: int = (
            CFG.system_size * 2
            + (1 if CFG.extra_ancilla and CFG.ancilla_mode == "pass" else 0)
        )

        # alpha and beta as trainable parameters (real-valued)
        self.alpha = nn.Parameter(
            -1 + 2 * torch.rand(self.size, 4, dtype=torch.float64)
        )
        self.beta = nn.Parameter(
            -1 + 2 * torch.rand(self.size, 4, dtype=torch.float64)
        )

        # Optimizer: SGD with momentum, MAXIMISING
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=CFG.l_rate,
            momentum=CFG.momentum_coeff,
        )

        # Save/load compatibility
        self.ancilla: bool = CFG.extra_ancilla
        self.ancilla_mode: str = CFG.ancilla_mode
        self.target_size: int = CFG.system_size
        self.target_hamiltonian: str = CFG.target_hamiltonian

    # -- matrix representations ---------------------------------
    def get_psi_and_phi(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Build psi and phi matrices via Kronecker product of per-qubit Hermitians.

        For each qubit i:
            H_i = \sum_j coeff[i][j] · \sigma_j    (real coeffs, Hermitian result)

        Full operator: psi = H_0 \otimes H_1 \otimes ... \otimes H_{N-1}

        Returns:
            (\psi, \phi) each of shape (2^N, 2^N), complex128, on the autograd graph.
        """
        # Cast alpha, beta to complex for matrix multiplication
        alpha_c = self.alpha.to(torch.complex128)
        beta_c = self.beta.to(torch.complex128)

        psi = torch.tensor([[1.0]], dtype=torch.complex128)  # scalar 1 as 1×1 matrix
        phi = torch.tensor([[1.0]], dtype=torch.complex128)

        for i in range(self.size):
            # Per-qubit Hermitian
            psi_i = sum(alpha_c[i, j] * PAULIS[j] for j in range(4))
            phi_i = sum(beta_c[i, j] * PAULIS[j] for j in range(4))
            psi = torch.kron(psi, psi_i)
            phi = torch.kron(phi, phi_i)

        return psi, phi

    def get_dis_matrices_rep(self) -> tuple[torch.Tensor, torch.Tensor,
                                            torch.Tensor, torch.Tensor]:
        """Compute A = exp(−phi / lambda) and B = exp(+\psi / lambda).

        Returns:
            (A, B, psi, phi)
        """
        psi, phi = self.get_psi_and_phi()
        A = torch.linalg.matrix_exp((-1.0 / lamb) * phi)
        B = torch.linalg.matrix_exp((1.0 / lamb) * psi)
        return A, B, psi, phi

    # -- loss computation (replaces manual gradients) ---------------------------------
    def compute_loss(self, final_target_state: torch.Tensor,
                     final_gen_state: torch.Tensor) -> torch.Tensor:
        """Compute the Wasserstein cost as a differentiable scalar.

        Uses density matrices and torch.trace instead of brakets:
            <psi|O|psi>= Tr[O · rho]        where rho = |psi><psi|
            <phi|O|psi>· <psi|O'|phi>= Tr[O · rho_psi · O' · rho_phi]  (cross terms)

        Loss = psi_term − phi_term − reg_term

        The discriminator MAXIMISES this, so we return − Loss for
        minimisation with optimizer.step().

        Args:
            final_target_state: Target state, shape (d,) or (d, 1), complex128.
            final_gen_state: Generator state, shape (d,) or (d, 1), complex128.

        Returns:
            torch.Tensor: scalar (real), the negated loss for minimisation.
        """
        A, B, psi, phi = self.get_dis_matrices_rep()

        # Build density matrices
        t = final_target_state.reshape(-1)
        g = final_gen_state.reshape(-1)
        rho_t = torch.outer(t, t.conj())  # |target><target|
        rho_g = torch.outer(g, g.conj())  # |gen><gen|

        # psi term: Tr[psi · rho_t]
        psi_term = torch.trace(psi @ rho_t)

        # phi term: Tr[phi · rho_g]
        phi_term = torch.trace(phi @ rho_g)

        # Regularisation terms:
        t1 = torch.trace(A @ rho_g) * torch.trace(B @ rho_t)
        t2 = torch.trace(A @ rho_g @ B @ rho_t)
        t3 = torch.trace(A @ rho_t @ B @ rho_g)
        t4 = torch.trace(A @ rho_t) * torch.trace(B @ rho_g)
        reg_term = lamb / np.e * (cst1 * t1 - cst2 * (t2 + t3) + cst3 * t4)

        # Full cost (real part to avoid numerical noise)
        cost = (psi_term - phi_term - reg_term).real

        # Negate for minimisation (discriminator maximises)
        return -cost

    # -- training step (replaces update_dis) ---------------------------------
    def update_dis(self, final_target_state: np.ndarray,
                   final_gen_state: np.ndarray):
        """One discriminator training step.

        Drop-in replacement for the old update_dis(). Converts numpy
        states to torch, computes loss, backpropagates, and steps.

        Args:
            final_target_state: numpy column vector (2^N, 1).
            final_gen_state: numpy column vector (2^N, 1).
        """
        # Convert numpy states to torch (no gradient needed for states here)
        target_t = torch.tensor(
            np.asarray(final_target_state), dtype=torch.complex128
        )
        gen_t = torch.tensor(
            np.asarray(final_gen_state), dtype=torch.complex128
        )

        self.optimizer.zero_grad()
        loss = self.compute_loss(target_t, gen_t)
        loss.backward()
        self.optimizer.step()

    # -- numpy interface for cost_functions.py compatibility ---------------------------------
    def get_dis_matrices_rep_numpy(self) -> tuple[np.ndarray, np.ndarray,
                                                   np.ndarray, np.ndarray]:
        """Return (A, B, psi, phi) as numpy arrays (detached from graph).

        Use this when we need the matrices for the generator gradient
        or for compute_cost() in cost_functions.py, which still works
        with numpy.
        """
        with torch.no_grad():
            A, B, psi, phi = self.get_dis_matrices_rep()
        return (
            A.numpy(),
            B.numpy(),
            psi.numpy(),
            phi.numpy(),
        )

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
        }
        torch.save(save_dict, file_path)

    def load_model_params(self, file_path: str) -> bool:
        """Load discriminator parameters from a saved model.

        Supports loading from:
            1. New torch format (state_dict + metadata)
            2. Old numpy/pickle format (for backward compatibility)
        """
        if not os.path.exists(file_path):
            print_and_log("Discriminator model file not found\n", CFG.log_path)
            return False

        # Try torch format first
        try:
            saved = torch.load(file_path, weights_only=False)
            if isinstance(saved, dict) and "state_dict" in saved:
                return self._load_from_torch_format(saved)
        except Exception:
            pass

        # Fall back to old pickle format
        try:
            import pickle
            with open(file_path, "rb") as f:
                saved_dis = pickle.load(f)
            return self._load_from_pickle_format(saved_dis)
        except (OSError, pickle.UnpicklingError) as e:
            print_and_log(
                f"ERROR: Could not load discriminator model: {e}\n", CFG.log_path
            )
            return False

    def _load_from_torch_format(self, saved: dict) -> bool:
        """Load from new torch save format."""
        # Compatibility checks
        if saved.get("target_size") != self.target_size:
            print_and_log("ERROR: target size mismatch.\n", CFG.log_path)
            return False
        if saved.get("target_hamiltonian") != self.target_hamiltonian:
            print_and_log("ERROR: target hamiltonian mismatch.\n", CFG.log_path)
            return False

        if saved.get("size") == self.size:
            self.load_state_dict(saved["state_dict"])
            if "optimizer_state" in saved:
                self.optimizer.load_state_dict(saved["optimizer_state"])
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
            print_and_log(
                "Discriminator parameters partially loaded (\pm1 qubit, torch).\n",
                CFG.log_path,
            )
            return True

        print_and_log("ERROR: incompatible discriminator (size mismatch).\n", CFG.log_path)
        return False

    def _load_from_pickle_format(self, saved_dis) -> bool:
        """Load from old numpy/pickle format (backward compatibility)."""
        cant_load = False
        if saved_dis.target_size != self.target_size:
            print_and_log("ERROR: target size mismatch.\n", CFG.log_path)
            cant_load = True
        if saved_dis.target_hamiltonian != self.target_hamiltonian:
            print_and_log("ERROR: target hamiltonian mismatch.\n", CFG.log_path)
            cant_load = True
        if cant_load:
            return False

        if saved_dis.size == self.size:
            with torch.no_grad():
                self.alpha.copy_(torch.from_numpy(saved_dis.alpha))
                self.beta.copy_(torch.from_numpy(saved_dis.beta))
            print_and_log("Discriminator parameters loaded (pickle -> torch).\n", CFG.log_path)
            return True

        if abs(saved_dis.size - self.size) == 1:
            min_size = min(saved_dis.size, self.size)
            with torch.no_grad():
                self.alpha[:min_size] = torch.from_numpy(
                    saved_dis.alpha[:min_size].copy()
                )
                self.beta[:min_size] = torch.from_numpy(
                    saved_dis.beta[:min_size].copy()
                )
            print_and_log(
                "Discriminator parameters partially loaded (\pm1 qubit, pickle -> torch).\n",
                CFG.log_path,
            )
            return True

        print_and_log("ERROR: incompatible discriminator (size mismatch).\n", CFG.log_path)
        return False