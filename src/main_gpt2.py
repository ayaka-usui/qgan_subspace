import os
import pickle
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

# Define Pauli matrices and Identity for convenience
I2 = np.eye(2, dtype=complex)
X_mat = np.array([[0, 1], [1, 0]], dtype=complex)
Y_mat = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_mat = np.array([[1, 0], [0, -1]], dtype=complex)


class Config:
    def __init__(
        self,
        QUBITS=4,
        LAYERS=10,
        EPOCHS=10,
        ITERATIONS=100,
        INIT_MODE="random",
        ANCILLA_MODE=False,
        COST_FN="original",
        ANSATZ="XYZ",
        TARGET_CHOICE="custom",
        include_Z=False,
        include_ZZ=False,
        include_ZZZ=True,
        include_I_term=False,
    ):
        # Training hyperparameters and settings
        self.QUBITS = QUBITS
        self.LAYERS = LAYERS
        self.EPOCHS = EPOCHS
        self.ITERATIONS = ITERATIONS
        self.INIT_MODE = INIT_MODE  # 'random' or 'zero' (identity initialization)
        self.ANCILLA_MODE = ANCILLA_MODE  # True to include ancilla qubits in generator/discriminator
        self.COST_FN = COST_FN  # 'original' (adversarial) or 'fidelity'
        self.ANSATZ = ANSATZ  # 'XYZ', 'ZZXZ', or 'hardware_efficient'
        self.TARGET_CHOICE = TARGET_CHOICE  # 'clusterH', 'rotated_surface_code', or 'custom'
        # Custom Hamiltonian term inclusion flags
        self.include_Z = include_Z
        self.include_ZZ = include_ZZ
        self.include_ZZZ = include_ZZZ
        self.include_I_term = include_I_term
        # (Optional) optimizer hyperparameters
        self.eta = 0.01  # learning rate (momentum optimizer will use internally)
        self.miu = 0.9  # momentum coefficient
        # Paths for output (will be set in update_timestamp)
        self.curr_timestamp = None
        self.figure_path = None
        self.model_gen_path = None
        self.model_dis_path = None
        self.log_path = None
        self.fid_loss_path = None
        self.theta_path = None

    def update_timestamp(self):
        """Update file paths with a new timestamp (to organize output files)."""
        self.curr_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base = f"./generated_data/{self.curr_timestamp}"
        self.figure_path = f"{base}/figure"
        self.model_gen_path = f"{base}/saved_model/{self.QUBITS}qubit_model-gen.mdl"
        self.model_dis_path = f"{base}/saved_model/{self.QUBITS}qubit_model-dis.mdl"
        self.log_path = f"{base}/logs/{self.QUBITS}qubit_log.txt"
        self.fid_loss_path = f"{base}/fidelities/{self.QUBITS}qubit_log_fidelity_loss.npy"
        self.theta_path = f"{base}/theta/{self.QUBITS}qubit_theta_gen.txt"


class MomentumOptimizer:
    """
    Gradient descent (or ascent) with momentum.
    v_{t+1} = mu * v_t - eta * grad
    theta_{t+1} = theta_t + v_{t+1}
    For ascent, the signs are flipped.
    """

    def __init__(self, eta=0.01, mu=0.9):
        self.eta = eta
        self.mu = mu
        self.v = None

    def compute_grad(self, theta, grad_list, min_or_max):
        # Flatten parameters and gradients to 1D
        theta_flat = np.array(theta, dtype=float).flatten()
        grad_flat = np.array(grad_list, dtype=float).flatten()
        if min_or_max == "min":  # gradient descent
            if self.v is None:
                self.v = -self.eta * grad_flat
            else:
                self.v = self.mu * self.v - self.eta * grad_flat
            new_theta_flat = theta_flat + self.v
        else:  # "max": gradient ascent
            if self.v is None:
                self.v = self.eta * grad_flat
            else:
                self.v = self.mu * self.v + self.eta * grad_flat
            new_theta_flat = theta_flat + self.v
        new_theta = np.reshape(new_theta_flat, np.shape(theta))
        return new_theta


# Functions to construct Hamiltonian terms
def term_XXXX(n, q1, q2, q3, q4):
    mat = 1
    for i in range(n):
        if i in (q1, q2, q3, q4):
            mat = np.kron(mat, X_mat)
        else:
            mat = np.kron(mat, I2)
    return mat


def term_ZZZZ(n, q1, q2, q3, q4):
    mat = 1
    for i in range(n):
        if i in (q1, q2, q3, q4):
            mat = np.kron(mat, Z_mat)
        else:
            mat = np.kron(mat, I2)
    return mat


def term_ZZZ(n, q1, q2, q3):
    mat = 1
    for i in range(n):
        if i in (q1, q2, q3):
            mat = np.kron(mat, Z_mat)
        else:
            mat = np.kron(mat, I2)
    return mat


def term_XZX(n, q1, q2, q3):
    mat = 1
    for i in range(n):
        if i == q2:
            mat = np.kron(mat, Z_mat)
        elif i == q1 or i == q3:
            mat = np.kron(mat, X_mat)
        else:
            mat = np.kron(mat, I2)
    return mat


def term_XX(n, q1, q2):
    mat = 1
    for i in range(n):
        if i == q1 or i == q2:
            mat = np.kron(mat, X_mat)
        else:
            mat = np.kron(mat, I2)
    return mat


def term_ZZ(n, q1, q2):
    mat = 1
    for i in range(n):
        if i == q1 or i == q2:
            mat = np.kron(mat, Z_mat)
        else:
            mat = np.kron(mat, I2)
    return mat


def term_Z_single(n, q1):
    mat = 1
    for i in range(n):
        if i == q1:
            mat = np.kron(mat, Z_mat)
        else:
            mat = np.kron(mat, I2)
    return mat


# Functions to construct target unitary from Hamiltonian
def construct_target_hamiltonian(n, include_Z=False, include_ZZ=False, include_ZZZ=False, include_I=False):
    """Construct target unitary as exp(-i * H) for the specified Hamiltonian terms."""
    dim = 2**n
    H = np.zeros((dim, dim), dtype=complex)
    if include_I:
        # Identity term (use identity Hamiltonian)
        H = np.eye(dim, dtype=complex)
    else:
        if include_Z:
            # Sum of Z on each qubit
            for i in range(n):
                H += term_Z_single(n, i)
        if include_ZZ:
            # Sum of ZZ on neighboring qubits
            for i in range(n - 1):
                H += term_ZZ(n, i, i + 1)
        if include_ZZZ:
            # Sum of ZZZ on neighboring triples
            for i in range(n - 2):
                H += term_ZZZ(n, i, i + 1, i + 2)
    return expm(-1j * H)


def construct_clusterH(n):
    """Construct 1D cluster-state unitary = exp(-i * H_cluster)."""
    dim = 2**n
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(n - 2):
        H += term_XZX(n, i, i + 1, i + 2)
        H += term_Z_single(n, i)
    # Add Z terms on the last two qubits
    if n >= 2:
        H += term_Z_single(n, n - 2)
        H += term_Z_single(n, n - 1)
    return expm(-1j * H)


def construct_RotatedSurfaceCode(n):
    """Construct rotated surface code unitary = exp(-i * H_code). Defined for n=4 or n=9."""
    dim = 2**n
    H = np.zeros((dim, dim), dtype=complex)
    if n == 4:
        H -= term_XXXX(n, 0, 1, 2, 3)
        H -= term_ZZ(n, 0, 1)
        H -= term_ZZ(n, 2, 3)
    elif n == 9:
        H -= term_XXXX(n, 0, 1, 3, 4)
        H -= term_XXXX(n, 4, 5, 7, 8)
        H -= term_XX(n, 2, 5)
        H -= term_XX(n, 3, 6)
        H -= term_ZZZZ(n, 1, 2, 4, 5)
        H -= term_ZZZZ(n, 3, 4, 6, 7)
        H -= term_ZZ(n, 0, 1)
        H -= term_ZZ(n, 7, 8)
    else:
        raise ValueError("rotated_surface_code is only defined for system size 4 or 9")
    return expm(-1j * H)


# Functions to prepare initial states
def get_zero_state(n):
    """Return the |0...0> state vector for n qubits."""
    vec = np.zeros(2**n, dtype=complex)
    vec[0] = 1.0 + 0j
    return vec


def get_maximally_entangled_state(n):
    """Return the maximally entangled state between two registers of n qubits each."""
    dim_sub = 2**n
    state = np.zeros(2 ** (2 * n), dtype=complex)
    for i in range(dim_sub):
        basis = np.zeros(dim_sub, dtype=complex)
        basis[i] = 1.0 + 0j
        state += np.kron(basis, basis)
    state = state / np.sqrt(dim_sub)  # normalize
    return state


def get_maximally_entangled_state_in_subspace(n):
    """Return maximally entangled state between two registers of n+1 qubits each (with an extra ancilla qubit on each side in state |0>)."""
    dim_main = 2**n
    upspin = np.zeros(2, dtype=complex)
    upspin[0] = 1.0 + 0j  # |0> for ancilla
    state = np.zeros(2 ** (2 * n + 2), dtype=complex)
    for i in range(dim_main):
        main_basis = np.zeros(dim_main, dtype=complex)
        main_basis[i] = 1.0 + 0j
        # Attach ancilla |0> to the basis state for each side
        a_side = np.kron(main_basis, upspin)
        b_side = np.kron(main_basis, upspin)
        state += np.kron(a_side, b_side)
    state = state / np.sqrt(dim_main)
    return state


class Generator:
    def __init__(self, system_size):
        self.size = system_size  # number of qubits in the generator's main system
        self.gates = []  # list of gates in the ansatz (tuples of (type, wires))
        self.params = None  # array of gate parameters (angles) for parametrized gates
        self.static_generators = []  # list of static generator matrices for each param gate
        self.optimizer = MomentumOptimizer()  # momentum optimizer for generator parameters

    def build_ansatz(self, ansatz_type, layers):
        """Build the parameterized quantum circuit (ansatz) for the generator."""
        gates = []
        n = self.size
        if ansatz_type == "XYZ":
            for _ in range(layers):
                # Nearest-neighbor entanglers (open chain) + local Z rotations
                for i in range(n - 1):
                    gates.append(("XX", (i, i + 1)))
                    gates.append(("YY", (i, i + 1)))
                    gates.append(("ZZ", (i, i + 1)))
                    gates.append(("Z", (i,)))
                # Close the ring entanglement between last and first qubit
                if n > 1:
                    gates.append(("XX", (0, n - 1)))
                    gates.append(("YY", (0, n - 1)))
                    gates.append(("ZZ", (0, n - 1)))
                # Local Z on the last qubit
                if n > 0:
                    gates.append(("Z", (n - 1,)))
        elif ansatz_type == "ZZXZ":
            for _ in range(layers):
                # Single-qubit X and Z rotations on each qubit
                for i in range(n):
                    gates.append(("X", (i,)))
                    gates.append(("Z", (i,)))
                # Two-qubit ZZ rotations on neighbors (open chain)
                for i in range(n - 1):
                    gates.append(("ZZ", (i, i + 1)))
        elif ansatz_type == "hardware_efficient":
            for _ in range(layers):
                # RY and RZ rotations on each qubit
                for i in range(n):
                    gates.append(("RY", (i,)))
                    gates.append(("RZ", (i,)))
                # CNOT entanglers in a ring topology
                for i in range(n - 1):
                    gates.append(("CNOT", (i, i + 1)))
                if n > 1:
                    gates.append(("CNOT", (n - 1, 0)))
        else:
            raise ValueError(f"Unknown ansatz type: {ansatz_type}")

        # Initialize parameters and static generator matrices for each gate
        params = []
        static_gens = []
        for gate in gates:
            g_type, wires = gate
            if g_type in ["X", "Y", "Z", "RY", "RZ"]:
                # Single-qubit rotation gates
                if g_type == "X" or g_type == "RY":
                    P = X_mat if g_type == "X" else Y_mat
                elif g_type == "Y":
                    P = Y_mat
                else:  # "Z" or "RZ"
                    P = Z_mat
                # Embed the 2x2 Pauli matrix into full system dimension
                P_full = 1
                for q in range(self.size):
                    P_full = np.kron(P_full, P if q == wires[0] else I2)
                static_gens.append(P_full)
                params.append(np.random.randn())
            elif g_type in ["XX", "YY", "ZZ"]:
                # Two-qubit Ising interaction rotations
                P_pair = X_mat if g_type == "XX" else (Y_mat if g_type == "YY" else Z_mat)
                P_full = 1
                for q in range(self.size):
                    if q == wires[0] or q == wires[1]:
                        P_full = np.kron(P_full, P_pair)
                    else:
                        P_full = np.kron(P_full, I2)
                static_gens.append(P_full)
                params.append(np.random.randn())
            elif g_type == "CNOT":
                # CNOT is a fixed gate (no parameter)
                static_gens.append(None)
                params.append(None)
            else:
                raise ValueError(f"Unsupported gate type: {g_type}")
        self.gates = gates
        self.params = np.array(params, dtype=float)
        self.static_generators = static_gens

    def get_unitary_matrix(self):
        """Compute the unitary matrix of the generator's circuit (on its main system only)."""
        if self.params is None:
            return np.eye(2**self.size, dtype=complex)
        U_total = np.eye(2**self.size, dtype=complex)
        param_index = 0
        for idx, gate in enumerate(self.gates):
            g_type, wires = gate
            if g_type == "CNOT":
                # Construct full N-qubit CNOT matrix
                control, target = wires
                dim = 2**self.size
                U_gate = np.zeros((dim, dim), dtype=complex)
                for state in range(dim):
                    # Determine bit values for control and target qubits
                    bits = [(state >> j) & 1 for j in range(self.size)]
                    ctrl_bit = bits[self.size - 1 - control]
                    tgt_bit = bits[self.size - 1 - target]
                    new_state = state
                    if ctrl_bit == 1:
                        # Flip target qubit
                        if tgt_bit == 0:
                            new_state = state | (1 << (self.size - 1 - target))
                        else:
                            new_state = state & ~(1 << (self.size - 1 - target))
                    U_gate[new_state, state] = 1.0 + 0j
            else:
                theta = float(self.params[param_index])
                P = self.static_generators[idx]
                # Rotation unitary: U = exp(-i * theta/2 * P) = cos(theta/2) I - i sin(theta/2) P
                U_gate = np.cos(theta / 2) * np.eye(2**self.size, dtype=complex) - 1j * np.sin(theta / 2) * P
                param_index += 1
            # Multiply into total unitary
            U_total = U_total.dot(U_gate)
        return U_total

    def getGen(self, ancilla_mode=False):
        """Return the full unitary matrix of the generator acting on the total space (generator's system + ancillas/reference)."""
        U_gen = self.get_unitary_matrix()
        # Determine identity on environment (ancillas + reference system)
        if ancilla_mode:
            # A has an ancilla (1 qubit) and B has (self.size+1) qubits
            identity_dim = 2 ** (self.size + 2)  # (1 + (self.size+1)) qubits
        else:
            # B side has self.size qubits
            identity_dim = 2**self.size
        I_env = np.eye(identity_dim, dtype=complex)
        return np.kron(U_gen, I_env)

    def update_gen(self, dis, real_state, input_state, config):
        """Update the generator's parameters by one training step (gradient descent on generator cost)."""
        grad = self._grad_theta(dis, real_state, input_state, config)
        new_params = self.optimizer.compute_grad(self.params, grad, "min")
        # Update only the parameter entries (skip None entries for fixed gates)
        flat_new = new_params.flatten()
        for i in range(len(self.params)):
            if self.params[i] is None:
                continue
            self.params[i] = flat_new[i]

    def _grad_theta(self, dis, real_state, input_state, config):
        """Compute gradient of the adversarial cost with respect to generator parameters."""
        # Compute current fake state
        G = self.getGen(config.ANCILLA_MODE)
        fake_state = G.dot(input_state)
        psi = dis.getPsi()
        phi = dis.getPhi()
        # Compute matrices A and B used in cost (regularization terms)
        A = expm((-1 / dis.lamb) * phi)
        B = expm((1 / dis.lamb) * psi)
        grad_phi_terms = []
        grad_reg_terms = []
        # Pre-compute each gate's unitary for current params
        M = len(self.gates)
        U_list = []
        param_idx = 0
        for idx, gate in enumerate(self.gates):
            g_type, wires = gate
            if g_type == "CNOT":
                # Compute full CNOT matrix
                control, target = wires
                dim = 2**self.size
                U = np.zeros((dim, dim), dtype=complex)
                for state in range(dim):
                    bits = [(state >> j) & 1 for j in range(self.size)]
                    ctrl_bit = bits[self.size - 1 - control]
                    tgt_bit = bits[self.size - 1 - target]
                    new_state = state
                    if ctrl_bit == 1:
                        if tgt_bit == 0:
                            new_state = state | (1 << (self.size - 1 - target))
                        else:
                            new_state = state & ~(1 << (self.size - 1 - target))
                    U[new_state, state] = 1.0 + 0j
                U_list.append(U)
            else:
                theta = float(self.params[param_idx])
                P = self.static_generators[idx]
                U = np.cos(theta / 2) * np.eye(2**self.size, dtype=complex) - 1j * np.sin(theta / 2) * P
                U_list.append(U)
                param_idx += 1
        # Compute prefix and suffix products for quick derivative computation
        pre = [None] * M
        post = [None] * M
        pre[0] = U_list[0]
        for i in range(1, M):
            pre[i] = pre[i - 1].dot(U_list[i])
        post[M - 1] = U_list[M - 1]
        for i in range(M - 2, -1, -1):
            post[i] = U_list[i].dot(post[i + 1])
        # Compute gradient contributions for each parameterized gate
        param_idx = 0
        for idx, gate in enumerate(self.gates):
            g_type, wires = gate
            if g_type == "CNOT":
                continue  # no parameter
            # derivative of gate's unitary: dU_i = -0.5j * P_i * U_i
            P = self.static_generators[idx]
            U_i = U_list[idx]
            dU_i = -0.5j * P.dot(U_i)
            U_pre = np.eye(2**self.size, dtype=complex) if idx == 0 else pre[idx - 1]
            U_post = np.eye(2**self.size, dtype=complex) if idx == M - 1 else post[idx + 1]
            # Full partial derivative of generator unitary on main system
            grad_U_main = U_pre.dot(dU_i).dot(U_post)
            # Embed into full space (including ancillas/reference)
            if config.ANCILLA_MODE:
                I_env = np.eye(2 ** (self.size + 2), dtype=complex)
            else:
                I_env = np.eye(2**self.size, dtype=complex)
            grad_full = np.kron(grad_U_main, I_env)
            # Compute contributions to gradient from cost function terms
            fake_grad = grad_full.dot(input_state)  # ∂|fake>/∂θ_i
            # Gradient of phi-term: Re( fake_grad^H * phi * |fake> + <fake| phi * fake_grad )
            term_phi = np.vdot(fake_grad, phi.dot(fake_state)) + np.vdot(fake_state, phi.dot(fake_grad))
            grad_phi_terms.append(np.real(term_phi))
            # Gradient of regularization term
            term1 = np.vdot(fake_grad, A.dot(fake_state)) * np.vdot(real_state, B.dot(real_state))
            term2 = np.vdot(fake_state, A.dot(fake_grad)) * np.vdot(real_state, B.dot(real_state))
            term3 = np.vdot(fake_grad, B.dot(real_state)) * np.vdot(real_state, A.dot(fake_state))
            term4 = np.vdot(fake_state, B.dot(real_state)) * np.vdot(real_state, A.dot(fake_grad))
            term5 = np.vdot(fake_grad, A.dot(real_state)) * np.vdot(real_state, B.dot(fake_state))
            term6 = np.vdot(fake_state, A.dot(real_state)) * np.vdot(real_state, B.dot(fake_grad))
            term7 = np.vdot(fake_grad, B.dot(fake_state)) * np.vdot(real_state, A.dot(real_state))
            term8 = np.vdot(fake_state, B.dot(fake_state)) * np.vdot(real_state, A.dot(real_state))
            reg_grad = (dis.lamb / np.e) * (
                dis.cst1 * (term1 + term2)
                - dis.cst2 * (term3 + term4)
                - dis.cst2 * (term5 + term6)
                + dis.cst3 * (term7 + term8)
            )
            grad_reg_terms.append(np.real(reg_grad))
            param_idx += 1
        # Total gradient for generator parameters: - (grad_phi + grad_reg) since we minimize generator cost
        grad_total = -(np.array(grad_phi_terms) + np.array(grad_reg_terms))
        return grad_total


class Discriminator:
    def __init__(self, herm_basis, total_qubits, lamb=10):
        # herm_basis: list of single-qubit Hermitian basis matrices [I, X, Y, Z]
        self.size = total_qubits
        self.herm = herm_basis
        self.lamb = float(lamb)
        # Initialize discriminator parameters alpha (for psi) and beta (for phi)
        m = len(herm_basis)  # typically 4
        self.alpha = -1 + 2 * np.random.random((self.size, m))
        self.beta = -1 + 2 * np.random.random((self.size, m))
        # Precompute constants for cost function regularization
        s = np.exp(-1 / (2 * self.lamb)) - 1
        self.cst1 = (s / 2 + 1) ** 2
        self.cst2 = (s / 2) * (s / 2 + 1)
        self.cst3 = (s / 2) ** 2
        # Momentum optimizers for psi (alpha) and phi (beta) parameters
        self.optimizer_psi = MomentumOptimizer()
        self.optimizer_phi = MomentumOptimizer()

    def getPsi(self):
        """Construct the matrix Psi from alpha parameters: Psi = ⊗_{i}(alpha[i,0]*I + alpha[i,1]*X + alpha[i,2]*Y + alpha[i,3]*Z)."""
        psi = 1
        for i in range(self.size):
            psi_i = np.zeros_like(self.herm[0], dtype=complex)
            for j in range(len(self.herm)):
                psi_i += self.alpha[i][j] * self.herm[j]
            psi = np.kron(psi, psi_i)
        return psi

    def getPhi(self):
        """Construct the matrix Phi from beta parameters (similar to Psi)."""
        phi = 1
        for i in range(self.size):
            phi_i = np.zeros_like(self.herm[0], dtype=complex)
            for j in range(len(self.herm)):
                phi_i += self.beta[i][j] * self.herm[j]
            phi = np.kron(phi, phi_i)
        return phi

    def _grad_psi_matrices(self, t_index):
        """Compute ∂Psi/∂alpha[:, t_index] as a list of matrices for each qubit term."""
        grad_list = []
        for i in range(self.size):
            grad_term = 1
            for j in range(self.size):
                if i == j:
                    grad_psij = self.herm[t_index]
                else:
                    psi_j = np.zeros_like(self.herm[0], dtype=complex)
                    for k in range(len(self.herm)):
                        psi_j += self.alpha[j][k] * self.herm[k]
                    grad_psij = psi_j
                grad_term = np.kron(grad_term, grad_psij)
            grad_list.append(grad_term)
        return grad_list

    def _grad_phi_matrices(self, t_index):
        """Compute ∂Phi/∂beta[:, t_index] as a list of matrices for each qubit term."""
        grad_list = []
        for i in range(self.size):
            grad_term = 1
            for j in range(self.size):
                if i == j:
                    grad_phij = self.herm[t_index]
                else:
                    phi_j = np.zeros_like(self.herm[0], dtype=complex)
                    for k in range(len(self.herm)):
                        phi_j += self.beta[j][k] * self.herm[k]
                    grad_phij = phi_j
                grad_term = np.kron(grad_term, grad_phij)
            grad_list.append(grad_term)
        return grad_list

    def _grad_alpha(self, gen, real_state, input_state):
        """Compute gradient of cost w.r.t. discriminator alpha parameters (psi)."""
        # Compute current fake state
        ancilla_mode = self.size == 2 * gen.size + 2
        fake_state = gen.getGen(ancilla_mode).dot(input_state)
        psi = self.getPsi()
        phi = self.getPhi()
        A = expm((-1 / self.lamb) * phi)
        B = expm((1 / self.lamb) * psi)
        cs = 1 / self.lamb
        m = len(self.herm)
        grad_psi_term = np.zeros((self.size, m), dtype=complex)
        grad_phi_term = np.zeros((self.size, m), dtype=complex)
        grad_reg_term = np.zeros((self.size, m), dtype=complex)
        for t in range(m):
            gradpsi_list = []
            gradphi_list = []
            gradreg_list = []
            gradpsi_mats = self._grad_psi_matrices(t)
            for grad_psi in gradpsi_mats:
                gradpsi_list.append(np.vdot(real_state, grad_psi.dot(real_state)))  # <real|gradPsi|real>
                gradphi_list.append(0)
                # Regularization gradient terms
                term1 = (
                    cs * np.vdot(fake_state, A.dot(fake_state)) * np.vdot(real_state, grad_psi.dot(B.dot(real_state)))
                )
                term2 = (
                    cs * np.vdot(fake_state, grad_psi.dot(B.dot(real_state))) * np.vdot(real_state, A.dot(fake_state))
                )
                term3 = (
                    cs * np.vdot(fake_state, A.dot(real_state)) * np.vdot(real_state, grad_psi.dot(B.dot(fake_state)))
                )
                term4 = (
                    cs * np.vdot(fake_state, grad_psi.dot(B.dot(fake_state))) * np.vdot(real_state, A.dot(real_state))
                )
                gradreg_list.append(
                    (self.lamb / np.e) * (self.cst1 * term1 - self.cst2 * term2 - self.cst2 * term3 + self.cst3 * term4)
                )
            grad_psi_term[:, t] = np.array(gradpsi_list)
            grad_phi_term[:, t] = np.array(gradphi_list)
            grad_reg_term[:, t] = np.array(gradreg_list)
        grad = np.real(grad_psi_term - grad_phi_term - grad_reg_term)
        return grad

    def _grad_beta(self, gen, real_state, input_state):
        """Compute gradient of cost w.r.t. discriminator beta parameters (phi)."""
        ancilla_mode = self.size == 2 * gen.size + 2
        fake_state = gen.getGen(ancilla_mode).dot(input_state)
        psi = self.getPsi()
        phi = self.getPhi()
        A = expm((-1 / self.lamb) * phi)
        B = expm((1 / self.lamb) * psi)
        cs = -1 / self.lamb
        m = len(self.herm)
        grad_psi_term = np.zeros((self.size, m), dtype=complex)
        grad_phi_term = np.zeros((self.size, m), dtype=complex)
        grad_reg_term = np.zeros((self.size, m), dtype=complex)
        for t in range(m):
            gradpsi_list = []
            gradphi_list = []
            gradreg_list = []
            gradphi_mats = self._grad_phi_matrices(t)
            for grad_phi in gradphi_mats:
                gradpsi_list.append(0)
                gradphi_list.append(np.vdot(fake_state, grad_phi.dot(fake_state)))  # <fake|gradPhi|fake>
                term1 = (
                    cs * np.vdot(fake_state, grad_phi.dot(A.dot(fake_state))) * np.vdot(real_state, B.dot(real_state))
                )
                term2 = (
                    cs * np.vdot(fake_state, B.dot(real_state)) * np.vdot(real_state, grad_phi.dot(A.dot(fake_state)))
                )
                term3 = (
                    cs * np.vdot(fake_state, grad_phi.dot(A.dot(real_state))) * np.vdot(real_state, B.dot(fake_state))
                )
                term4 = (
                    cs * np.vdot(fake_state, B.dot(fake_state)) * np.vdot(real_state, grad_phi.dot(A.dot(real_state)))
                )
                gradreg_list.append(
                    (self.lamb / np.e) * (self.cst1 * term1 - self.cst2 * term2 - self.cst2 * term3 + self.cst3 * term4)
                )
            grad_psi_term[:, t] = np.array(gradpsi_list)
            grad_phi_term[:, t] = np.array(gradphi_list)
            grad_reg_term[:, t] = np.array(gradreg_list)
        grad = np.real(grad_psi_term - grad_phi_term - grad_reg_term)
        return grad

    def update_dis(self, gen, real_state, input_state):
        """Update the discriminator's parameters by one training step (gradient ascent on discriminator cost)."""
        grad_alpha = self._grad_alpha(gen, real_state, input_state)
        new_alpha = self.optimizer_psi.compute_grad(self.alpha, grad_alpha, "max")
        grad_beta = self._grad_beta(gen, real_state, input_state)
        new_beta = self.optimizer_phi.compute_grad(self.beta, grad_beta, "max")
        self.alpha = new_alpha
        self.beta = new_beta


def train(config):
    """Train the Quantum GAN (QuGAN) with the given configuration. Saves models, logs, and plots."""
    config.update_timestamp()
    # Create necessary directories for output
    os.makedirs(os.path.dirname(config.log_path), exist_ok=True)
    os.makedirs(config.figure_path, exist_ok=True)
    os.makedirs(os.path.dirname(config.model_gen_path), exist_ok=True)
    os.makedirs(os.path.dirname(config.fid_loss_path), exist_ok=True)
    os.makedirs(os.path.dirname(config.theta_path), exist_ok=True)
    # Prepare input (entangled) state and target state
    if config.ANCILLA_MODE:
        input_state = get_maximally_entangled_state_in_subspace(config.QUBITS)
    else:
        input_state = get_maximally_entangled_state(config.QUBITS)
    # Construct target unitary
    if config.TARGET_CHOICE == "clusterH":
        target_unitary = construct_clusterH(config.QUBITS)
    elif config.TARGET_CHOICE == "rotated_surface_code":
        target_unitary = construct_RotatedSurfaceCode(config.QUBITS)
    else:  # "custom"
        target_unitary = construct_target_hamiltonian(
            config.QUBITS, config.include_Z, config.include_ZZ, config.include_ZZZ, config.include_I_term
        )
    # Compute target (real) state = (target_unitary on A) ⊗ I on rest * |input_state>
    if config.ANCILLA_MODE:
        # target_unitary acts on A's main qubits; identity on A's ancilla (1 qubit) and all B qubits (N+1 qubits)
        U_ext = np.kron(np.kron(target_unitary, np.eye(2)), np.eye(2 ** (config.QUBITS + 1)))
    else:
        # target_unitary on A's main qubits; identity on B's main qubits
        U_ext = np.kron(target_unitary, np.eye(2**config.QUBITS))
    real_state = U_ext.dot(input_state)
    # Initialize generator and discriminator
    gen = Generator(config.QUBITS)
    gen.build_ansatz(config.ANSATZ, config.LAYERS)
    if config.INIT_MODE != "random":
        # Initialize generator parameters to zero (identity operation)
        for i in range(len(gen.params)):
            if gen.params[i] is not None:
                gen.params[i] = 0.0
    total_qubits = (config.QUBITS + (1 if config.ANCILLA_MODE else 0)) * 2
    disc = Discriminator([I2, X_mat, Y_mat, Z_mat], total_qubits, lamb=10)
    if config.INIT_MODE != "random":
        disc.alpha[:, :] = 0.0
        disc.beta[:, :] = 0.0
    # Training loop
    fidelities_history = np.array([], dtype=float)
    losses_history = np.array([], dtype=float)
    starttime = datetime.now()
    f = 0.0
    epoch = 0
    while f < 0.99:
        epoch += 1
        fidelities = np.zeros(config.ITERATIONS)
        losses = np.zeros(config.ITERATIONS)
        for it in range(config.ITERATIONS):
            if config.COST_FN == "original":
                # Adversarial training: update generator then discriminator
                gen.update_gen(disc, real_state, input_state, config)
                disc.update_dis(gen, real_state, input_state)
            else:
                # Fidelity training: only update generator to maximize fidelity
                fake_state = gen.getGen(config.ANCILLA_MODE).dot(input_state)
                z = np.vdot(real_state, fake_state)  # overlap <real|fake>
                # Compute gradient of fidelity = |<real|fake>|^2 with respect to generator parameters
                grad_params = []
                # Compute partial derivative of fake state for each generator parameter
                # (We reuse generator's circuit structure for efficiency)
                M = len(gen.gates)
                U_list = []
                param_idx = 0
                for idx, gate in enumerate(gen.gates):
                    g_type, wires = gate
                    if g_type == "CNOT":
                        # Full CNOT matrix
                        control, target = wires
                        dim = 2**gen.size
                        U = np.zeros((dim, dim), dtype=complex)
                        for state in range(dim):
                            bits = [(state >> j) & 1 for j in range(gen.size)]
                            ctrl_bit = bits[gen.size - 1 - control]
                            tgt_bit = bits[gen.size - 1 - target]
                            new_state = state
                            if ctrl_bit == 1:
                                if tgt_bit == 0:
                                    new_state = state | (1 << (gen.size - 1 - target))
                                else:
                                    new_state = state & ~(1 << (gen.size - 1 - target))
                            U[new_state, state] = 1.0 + 0j
                        U_list.append(U)
                    else:
                        theta = float(gen.params[param_idx])
                        P = gen.static_generators[idx]
                        U = np.cos(theta / 2) * np.eye(2**gen.size, dtype=complex) - 1j * np.sin(theta / 2) * P
                        U_list.append(U)
                        param_idx += 1
                pre = [None] * M
                post = [None] * M
                pre[0] = U_list[0]
                for j in range(1, M):
                    pre[j] = pre[j - 1].dot(U_list[j])
                post[M - 1] = U_list[M - 1]
                for j in range(M - 2, -1, -1):
                    post[j] = U_list[j].dot(post[j + 1])
                param_idx = 0
                for idx, gate in enumerate(gen.gates):
                    g_type, wires = gate
                    if g_type == "CNOT":
                        continue
                    P = gen.static_generators[idx]
                    U_i = U_list[idx]
                    dU_i = -0.5j * P.dot(U_i)
                    U_pre = np.eye(2**gen.size, dtype=complex) if idx == 0 else pre[idx - 1]
                    U_post = np.eye(2**gen.size, dtype=complex) if idx == M - 1 else post[idx + 1]
                    grad_U_main = U_pre.dot(dU_i).dot(U_post)
                    if config.ANCILLA_MODE:
                        I_env = np.eye(2 ** (gen.size + 2), dtype=complex)
                    else:
                        I_env = np.eye(2**gen.size, dtype=complex)
                    grad_full = np.kron(grad_U_main, I_env)
                    grad_overlap = np.vdot(real_state, grad_full.dot(input_state))  # <real|∂fake>
                    grad_val = 2 * (z.conjugate() * grad_overlap).real
                    grad_params.append(grad_val)
                    param_idx += 1
                grad_params = np.array(grad_params)
                # Update generator parameters (gradient ascent on fidelity)
                new_params = gen.optimizer.compute_grad(gen.params, grad_params, "max")
                flat_new = new_params.flatten()
                for i in range(len(gen.params)):
                    if gen.params[i] is None:
                        continue
                    gen.params[i] = flat_new[i]
            # Evaluate fidelity and loss after this iteration
            fake_state = gen.getGen(config.ANCILLA_MODE).dot(input_state)
            overlap = np.vdot(real_state, fake_state)
            fidelity = np.abs(overlap) ** 2
            fidelities[it] = fidelity
            if config.COST_FN == "original":
                psi = disc.getPsi()
                phi = disc.getPhi()
                A = expm((-1 / disc.lamb) * phi)
                B = expm((1 / disc.lamb) * psi)
                term1 = np.vdot(fake_state, A.dot(fake_state))
                term2 = np.vdot(real_state, B.dot(real_state))
                term3 = np.vdot(fake_state, B.dot(real_state))
                term4 = np.vdot(real_state, A.dot(fake_state))
                term5 = np.vdot(fake_state, A.dot(real_state))
                term6 = np.vdot(real_state, B.dot(fake_state))
                term7 = np.vdot(fake_state, B.dot(fake_state))
                term8 = np.vdot(real_state, A.dot(real_state))
                regterm = (
                    disc.lamb
                    / np.e
                    * (
                        disc.cst1 * term1 * term2
                        - disc.cst2 * term3 * term4
                        - disc.cst2 * term5 * term6
                        + disc.cst3 * term7 * term8
                    )
                )
                psi_term = np.vdot(real_state, psi.dot(real_state))
                phi_term = np.vdot(fake_state, phi.dot(fake_state))
                loss = (psi_term - phi_term - regterm).real
            else:
                loss = 1 - fidelity
            losses[it] = loss
            # Logging to console and file periodically
            print(f"Epoch {epoch}, Iteration {it+1}, Fidelity: {fidelity:.6f}, Loss: {loss:.6f}")
            if it % 10 == 0:
                current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                duration_hours = (datetime.now() - starttime).seconds / 3600.0
                log_line = f"epoches:{it:4d} | fidelity:{fidelity:8f} | time:{current_time:10s} | duration:{duration_hours:8f}\n"
                with open(config.log_path, "a") as log_file:
                    log_file.write(log_line)
        # Append epoch results to history
        fidelities_history = np.concatenate((fidelities_history, fidelities))
        losses_history = np.concatenate((losses_history, losses))
        f = fidelities[-1]  # final fidelity of this epoch
        # Plot and save fidelity vs iteration and loss vs iteration
        plt.figure()
        plt.plot(range(len(fidelities_history)), fidelities_history, label="Fidelity")
        plt.plot(range(len(losses_history)), losses_history, label="Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Value")
        plt.title(f"Fidelity and Loss vs Iteration (Epoch {epoch})")
        plt.legend()
        plt.savefig(f"{config.figure_path}/fidelity_loss_epoch{epoch}.png")
        plt.close()
        # Check epoch limit
        if epoch >= config.EPOCHS:
            print(f"The number of epochs reached {config.EPOCHS}. Stopping training.")
            break
    # Save final fidelity and loss history
    np.save(config.fid_loss_path, np.vstack([fidelities_history, losses_history]))
    # Save generator and discriminator models
    with open(config.model_gen_path, "wb") as gen_file:
        pickle.dump(gen, gen_file)
    with open(config.model_dis_path, "wb") as dis_file:
        pickle.dump(disc, dis_file)
    # Save generator parameters (angles) to text file
    with open(config.theta_path, "w") as theta_file:
        angles = [str(val) for val in gen.params if val is not None]
        theta_file.write("\n".join(angles))
    endtime = datetime.now()
    print(f"Training finished in {(endtime - starttime).seconds} seconds.")
    return fidelities_history, losses_history


# If run as script, execute training for all combinations of INIT_MODE × ANCILLA_MODE × COST_FN (with reduced epochs for testing)
if __name__ == "__main__":
    init_modes = ["random", "zero"]
    ancilla_modes = [False, True]
    cost_modes = ["original", "fidelity"]
    for init in init_modes:
        for anc in ancilla_modes:
            for cost in cost_modes:
                cfg = Config(
                    QUBITS=3,
                    LAYERS=4,
                    EPOCHS=5,
                    ITERATIONS=50,
                    INIT_MODE=init,
                    ANCILLA_MODE=anc,
                    COST_FN=cost,
                    ANSATZ="XYZ",
                    TARGET_CHOICE="custom",
                    include_Z=False,
                    include_ZZ=False,
                    include_ZZZ=True,
                    include_I_term=False,
                )
                print(f"\nRunning QuGAN training with INIT_MODE={init}, ANCILLA_MODE={anc}, COST_FN={cost}\n")
                train(cfg)
