import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np
from tqdm import tqdm


# ========================
# Configuration
# ========================
class Config:
    QUBITS = 4
    LAYERS = 3
    EPOCHS = 50
    STEPS_PER_EPOCH = 20
    GEN_STEPS = 1
    DISC_STEPS = 1
    LEARNING_RATE = 0.01
    INIT_MODE = "state"  # "state" or "choi"
    INIT_STATE = "zero"  # "zero", "random"
    ANCILLA_MODE = "none"  # "none", "pass", "project", "trace_out"
    COST_FN = "wasserstein"
    TARGET_HAMILTONIAN = "cluster"
    CUSTOM_TERMS = {"ZZ": 0.5, "ZZZ": 0.3}
    GEN_ANSATZ = "zz_xz"
    DISC_ANSATZ = "hardware_efficient"
    SAVE_PATH = f"generated_data/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    PLOT_INTERVAL = 5


# ========================
# Hamiltonian Definitions
# ========================
def get_target_hamiltonian(n_qubits):
    if Config.TARGET_HAMILTONIAN == "cluster":
        coeffs = []
        obs = []
        for i in range(n_qubits - 3):
            coeffs.append(1.0)
            obs.append(qml.PauliZ(i) @ qml.PauliZ(i + 1) @ qml.PauliZ(i + 2) @ qml.PauliZ(i + 3))
        return qml.Hamiltonian(coeffs, obs)
    elif Config.TARGET_HAMILTONIAN == "custom":
        return custom_hamiltonian(n_qubits, Config.CUSTOM_TERMS)
    else:  # Rotated Surface Code (simplified)
        coeffs = [1.0] * (n_qubits // 2)
        obs = [qml.PauliZ(i) @ qml.PauliZ(i + 1) for i in range(0, n_qubits, 2)]
        return qml.Hamiltonian(coeffs, obs)


def custom_hamiltonian(n_qubits, terms):
    coeffs, obs = [], []
    for term, weight in terms.items():
        term_length = len(term)
        for i in range(n_qubits - term_length + 1):
            paulis = [getattr(qml, f"Pauli{term[j]}")(i + j) for j in range(term_length)]
            coeffs.append(weight)
            obs.append(qml.operation.Tensor(*paulis))
    return qml.Hamiltonian(coeffs, obs)


# ========================
# Ansatz Definitions
# ========================
def gen_ansatz(params, wires):
    n_layers = params.shape[0]
    for layer in range(n_layers):
        # Single-qubit rotations
        for i in wires:
            qml.RX(params[layer, i, 0], wires=i)
            qml.RZ(params[layer, i, 1], wires=i)
        # Entangling layers
        for i in range(len(wires) - 1):
            qml.IsingZZ(params[layer, i, 2], wires=[wires[i], wires[i + 1]])


def disc_ansatz(params, wires):
    n_layers = params.shape[0]
    for layer in range(n_layers):
        # Single-qubit rotations
        for i in wires:
            qml.Rot(*params[layer, i, :3], wires=i)
        # Entangling layers
        for i in range(len(wires) - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])


# ========================
# Quantum Circuits
# ========================
def prepare_initial_state(wires):
    if Config.INIT_MODE == "choi":
        qml.Hadamard(wires=wires[0])
        for i in range(1, len(wires) // 2):
            qml.CNOT(wires=[wires[0], wires[i]])
    elif Config.INIT_STATE == "random":
        state = np.random.rand(2 ** len(wires)) + 1j * np.random.rand(2 ** len(wires))
        state /= np.linalg.norm(state)
        qml.StatePrep(state, wires=wires)
    else:
        qml.BasisState(np.zeros(len(wires)), wires=wires)


def create_circuits():
    n_qubits = Config.QUBITS
    ancilla = 1 if Config.ANCILLA_MODE != "none" else 0
    total_wires = n_qubits + ancilla
    dev = qml.device("default.qubit", wires=total_wires)

    @qml.qnode(dev)
    def generator(params):
        prepare_initial_state(range(n_qubits))
        gen_ansatz(params, range(n_qubits))

        if Config.ANCILLA_MODE == "pass":
            qml.CNOT(wires=[n_qubits - 1, n_qubits])
        elif Config.ANCILLA_MODE == "project":
            qml.CRX(params[-1, 0, 0], wires=[n_qubits - 1, n_qubits])
            qml.measure(n_qubits)

        return qml.state()

    @qml.qnode(dev)
    def discriminator(params, state):
        try:
            qml.StatePrep(state, wires=range(total_wires))
        except AttributeError:
            qml.QubitStateVector(state, wires=range(total_wires))

        disc_ansatz(params, range(total_wires))
        return qml.probs(wires=range(total_wires))

    return generator, discriminator


# ========================
# Cost Functions
# ========================
def wasserstein_cost(real_probs, fake_probs):
    return np.sum(real_probs - fake_probs)


def state_fidelity(real_state, fake_state):
    return np.abs(np.vdot(real_state, fake_state)) ** 2


def trace_distance(rho, sigma):
    diff = rho - sigma
    return 0.5 * np.trace(np.sqrt(diff @ diff.conj().T))


# ========================
# Training Loop
# ========================
def train():
    os.makedirs(Config.SAVE_PATH, exist_ok=True)

    # Initialize components
    gen, disc = create_circuits()
    target_H = get_target_hamiltonian(Config.QUBITS)

    # Target state preparation with version handling
    @qml.qnode(qml.device("default.qubit", wires=Config.QUBITS))
    def target_circuit():
        qml.BasisState(np.zeros(Config.QUBITS), wires=range(Config.QUBITS))
        qml.ApproxTimeEvolution(target_H, time=1.0, n=10)
        return qml.state()

    target_state = target_circuit()

    # Handle ancilla in target state
    if Config.ANCILLA_MODE != "none":
        target_state = np.kron(target_state, np.array([1, 0]))  # Add ancilla |0>

    # Parameter initialization
    gen_shape = (Config.LAYERS, Config.QUBITS, 3)
    disc_shape = (Config.LAYERS, Config.QUBITS + (1 if Config.ANCILLA_MODE != "none" else 0), 3)
    gen_params = np.random.uniform(0, 2 * np.pi, gen_shape)
    disc_params = np.random.uniform(0, 2 * np.pi, disc_shape)

    # Optimizers
    gen_opt = qml.AdamOptimizer(Config.LEARNING_RATE)
    disc_opt = qml.AdamOptimizer(Config.LEARNING_RATE)

    # Training history
    history = {"gen_cost": [], "disc_cost": [], "fidelity": [], "trace_dist": []}

    # Main training loop
    for epoch in tqdm(range(Config.EPOCHS)):
        gen_cost_epoch = []
        disc_cost_epoch = []

        for _ in range(Config.STEPS_PER_EPOCH):
            # Discriminator training
            for _ in range(Config.DISC_STEPS):

                def disc_cost(d_params):
                    real_probs = disc(d_params, target_state)
                    fake_state = gen(gen_params)
                    fake_probs = disc(d_params, fake_state)
                    return wasserstein_cost(real_probs, fake_probs)

                disc_params = disc_opt.step(disc_cost, disc_params)
                disc_cost_epoch.append(disc_cost(disc_params))

            # Generator training
            for _ in range(Config.GEN_STEPS):

                def gen_cost(g_params):
                    fake_state = gen(g_params)
                    fake_probs = disc(disc_params, fake_state)
                    real_probs = disc(disc_params, target_state)
                    return -wasserstein_cost(real_probs, fake_probs)

                gen_params = gen_opt.step(gen_cost, gen_params)
                gen_cost_epoch.append(gen_cost(gen_params))

            # Calculate metrics
            fake_state = gen(gen_params)
            history["fidelity"].append(state_fidelity(target_state, fake_state))
            rho = np.outer(target_state, target_state.conj())
            sigma = np.outer(fake_state, fake_state.conj())
            history["trace_dist"].append(trace_distance(rho, sigma))

        # Save history
        history["gen_cost"].extend(gen_cost_epoch)
        history["disc_cost"].extend(disc_cost_epoch)

        # Save checkpoint
        if epoch % Config.PLOT_INTERVAL == 0:
            np.savez(f"{Config.SAVE_PATH}/checkpoint_{epoch}.npz", gen_params=gen_params, disc_params=disc_params)
            plot_training(history, epoch)

    # Save final results
    with open(f"{Config.SAVE_PATH}/config.json", "w") as f:
        json.dump(vars(Config), f, indent=2)

    return gen_params, disc_params, history


# ========================
# Visualization
# ========================
def plot_training(history, epoch):
    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plt.plot(history["gen_cost"], label="Generator Cost")
    plt.plot(history["disc_cost"], label="Discriminator Cost")
    plt.plot(np.array(history["gen_cost"]) - np.array(history["disc_cost"]), label="Gen-Disc Difference")
    plt.xlabel("Training Step")
    plt.ylabel("Cost")
    plt.legend()

    plt.subplot(122)
    plt.plot(history["fidelity"], label="Fidelity")
    plt.plot(history["trace_dist"], label="Trace Distance")
    plt.xlabel("Training Step")
    plt.ylabel("Metric")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{Config.SAVE_PATH}/training_plot_{epoch}.png")
    plt.close()


# ========================
# Main Execution
# ========================
if __name__ == "__main__":
    # Example configuration
    Config.QUBITS = 4
    Config.TARGET_HAMILTONIAN = "cluster"
    Config.COST_FN = "wasserstein"
    Config.ANCILLA_MODE = "none"

    # Run training
    gen_params, disc_params, history = train()

    # Final plot
    plot_training(history, "final")
    print(f"Training complete! Results saved to {Config.SAVE_PATH}")
