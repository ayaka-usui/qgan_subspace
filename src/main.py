import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np
from pennylane import numpy as pnp
from tqdm import tqdm


# ========================
# Configuration
# ========================
class Config:
    QUBITS = 2
    LAYERS = 10
    EPOCHS = 10
    STEPS_PER_EPOCH = 1
    GEN_STEPS = 1
    DISC_STEPS = 5
    LEARNING_RATE = 0.01
    INIT_MODE = "choi"  # choi, state
    INIT_STATE = "zero"  # zero, random
    ANCILLA_MODE = "none"  # none, pass, project
    COST_FN = "wasserstein"  # wasserstein, fidelity, trace_dist, trace_dist_sq
    TARGET_HAMILTONIAN = "cluster"  # cluster, surface_code, custom
    # CUSTOM_TERMS = {"ZZ": 0.5, "ZZZ": 0.3}
    GEN_ANSATZ = "zz_xx_yy_z"  # zz_xz, zz_xx_yy_z, hardware_eff, basic_entangled
    DISC_ANSATZ = "zz_xx_yy_z"  # zz_xz, zz_xx_yy_z, hardware_eff, basic_entangled
    SAVE_PATH = f"generated_data/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    PLOT_INTERVAL = 1


# ========================
# Hamiltonian Definitions
# ========================
def get_target_hamiltonian(n_qubits):
    if Config.TARGET_HAMILTONIAN == "cluster":
        return cluster_hamiltonian(n_qubits)
    elif Config.TARGET_HAMILTONIAN == "surface_code":
        return surface_code_hamiltonian(n_qubits)
    elif Config.TARGET_HAMILTONIAN == "custom":
        return custom_hamiltonian(n_qubits, Config.CUSTOM_TERMS)
    else:
        raise ValueError(f"Unknown Hamiltonian: {Config.TARGET_HAMILTONIAN}")


def cluster_hamiltonian(n_qubits):
    if n_qubits < 4:
        raise ValueError("Cluster Hamiltonian requires at least 4 qubits")
    coeffs = [1.0] * (n_qubits - 3)
    obs = [qml.PauliX(i) @ qml.PauliZ(i + 1) @ qml.PauliZ(i + 2) @ qml.PauliZ(i + 3) for i in range(n_qubits - 3)]
    return qml.Hamiltonian(coeffs, obs)


def surface_code_hamiltonian(n_qubits):
    coeffs = [1.0] * (n_qubits // 2)
    obs = [qml.PauliZ(i) @ qml.PauliZ(i + 1) for i in range(0, n_qubits, 2)]
    return qml.Hamiltonian(coeffs, obs)


def custom_hamiltonian(n_qubits, terms):
    coeffs, obs = [], []
    for term, weight in terms.items():
        term_length = len(term)
        for i in range(n_qubits - term_length + 1):
            paulis = [getattr(qml, f"Pauli{term[j]}")(i + j) for j in range(term_length)]
            # Compute the tensor product using @
            product = paulis[0]
            for p in paulis[1:]:
                product @= p
            coeffs.append(weight)
            obs.append(product)
    return qml.Hamiltonian(coeffs, obs)


# ========================
# Ansatz Definitions
# ========================
ANSATZES = {
    "zz_xz": lambda params, wires: zz_xz_ansatz(params, wires),
    "zz_xx_yy_z": lambda params, wires: zz_xx_yy_z_ansatz(params, wires),
    "hardware_eff": lambda params, wires: hardware_efficient_ansatz(params, wires),
    "basic_entangled": lambda params, wires: basic_entangled_ansatz(params, wires),
}


def zz_xz_ansatz(params, wires):
    n_layers = params.shape[0]
    for layer in range(n_layers):
        for i in wires:
            qml.RX(params[layer, i, 0], wires=i)
            qml.RZ(params[layer, i, 1], wires=i)
        for i in range(len(wires) - 1):
            qml.IsingZZ(params[layer, i, 2], wires=[wires[i], wires[i + 1]])


def zz_xx_yy_z_ansatz(params, wires):
    n_layers = params.shape[0]
    for layer in range(n_layers):
        for i in wires:
            qml.RX(params[layer, i, 0], wires=i)
            qml.RY(params[layer, i, 1], wires=i)
        for i in range(len(wires) - 1):
            qml.IsingZZ(params[layer, i, 2], wires=[wires[i], wires[i + 1]])
            qml.IsingXX(params[layer, i, 3], wires=[wires[i], wires[i + 1]])
            qml.IsingYY(params[layer, i, 4], wires=[wires[i], wires[i + 1]])


def hardware_efficient_ansatz(params, wires):
    n_layers = params.shape[0]
    for layer in range(n_layers):
        for i in wires:
            qml.Rot(*params[layer, i, :3], wires=i)
        for i in range(len(wires) - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])


def basic_entangled_ansatz(params, wires):
    n_layers = params.shape[0]
    for layer in range(n_layers):
        for i in wires:
            qml.RY(params[layer, i, 0], wires=i)
        for i in range(len(wires) - 1):
            qml.CZ(wires=[wires[i], wires[i + 1]])


# ========================
# Cost Functions
# ========================
def get_cost_function():
    return {
        "wasserstein": wasserstein_cost,
        "fidelity": fidelity_cost,
        "trace_dist": trace_distance_cost,
        "trace_dist_sq": trace_distance_squared_cost,
    }[Config.COST_FN]


def wasserstein_cost(real_probs, fake_probs):
    return qml.math.sum(real_probs - fake_probs)


def fidelity_cost(real_state, fake_state):
    return 1 - qml.math.fidelity(real_state, fake_state)


def trace_distance_cost(rho, sigma):
    diff = rho - sigma
    sqrt = qml.math.sqrt(qml.math.dot(diff, qml.math.conj(diff).T))
    return 0.5 * qml.math.trace(sqrt)


def trace_distance_squared_cost(rho, sigma):
    diff = rho - sigma
    return 0.5 * qml.math.trace(qml.math.dot(diff, diff))


# ========================
# Quantum Circuits
# ========================
def create_circuits():
    n_qubits = Config.QUBITS
    ancilla = 1 if Config.ANCILLA_MODE != "none" else 0
    total_wires = n_qubits + ancilla

    # Choose device based on state type
    if Config.INIT_MODE == "choi" or Config.ANCILLA_MODE != "none":
        dev = qml.device("default.mixed", wires=total_wires)
    else:
        dev = qml.device("default.qubit", wires=total_wires)

    @qml.qnode(dev)
    def generator(params):
        prepare_initial_state(range(n_qubits))
        ANSATZES[Config.GEN_ANSATZ](params, range(n_qubits))

        if Config.ANCILLA_MODE == "pass":
            qml.CNOT(wires=[n_qubits - 1, n_qubits])
        elif Config.ANCILLA_MODE == "project":
            qml.CRX(params[-1, 0, 0], wires=[n_qubits - 1, n_qubits])
            qml.measure(n_qubits)

        return qml.state()

    @qml.qnode(dev)
    def discriminator(params, state):
        if dev.name == "default.mixed":
            qml.QubitDensityMatrix(state, wires=range(total_wires))
        else:
            qml.StatePrep(state, wires=range(total_wires))

        ANSATZES[Config.DISC_ANSATZ](params, range(total_wires))
        return qml.probs(wires=range(total_wires))

    return generator, discriminator


def prepare_initial_state(wires):
    if Config.INIT_MODE == "choi":
        # Create a maximally entangled state across all qubits
        n_qubits = len(wires)
        state = np.zeros((2**n_qubits, 2**n_qubits))
        state[0, 0] = 1.0  # |0..0><0..0|
        state[-1, -1] = 1.0  # |1..1><1..1|
        state /= 2.0  # Normalize to ensure trace 1
        qml.QubitDensityMatrix(state, wires=wires)
    elif Config.INIT_STATE == "random":
        vec = np.random.rand(2 ** len(wires)) + 1j * np.random.rand(2 ** len(wires))
        vec /= np.linalg.norm(vec)
        qml.StatePrep(vec, wires=wires)
    else:
        qml.BasisState(np.zeros(len(wires)), wires=wires)


# ========================
# Training Loop
# ========================
def train():
    os.makedirs(Config.SAVE_PATH, exist_ok=True)

    # Initialize components
    gen, disc = create_circuits()
    target_H = get_target_hamiltonian(Config.QUBITS)

    # Determine if using density matrices
    using_density_matrices = isinstance(gen.device, qml.devices.DefaultMixed)

    # Target state preparation
    @qml.qnode(qml.device("default.qubit", wires=Config.QUBITS))
    def target_circuit():
        qml.BasisState(np.zeros(Config.QUBITS), wires=range(Config.QUBITS))
        qml.ApproxTimeEvolution(target_H, time=1.0, n=10)
        return qml.state()

    target_state = target_circuit()

    # Convert to density matrix if needed
    if using_density_matrices:
        target_state = np.outer(target_state, target_state.conj())

    # Handle ancilla
    if Config.ANCILLA_MODE != "none":
        ancilla_state = np.array([1, 0]) if not using_density_matrices else np.outer([1, 0], [1, 0])
        target_state = np.kron(target_state, ancilla_state)

    # Parameter initialization
    gen_params = init_parameters(Config.GEN_ANSATZ, is_generator=True)
    disc_params = init_parameters(Config.DISC_ANSATZ, is_generator=False)

    # Optimizers
    gen_opt = qml.AdamOptimizer(Config.LEARNING_RATE)
    disc_opt = qml.AdamOptimizer(Config.LEARNING_RATE)

    # Training history
    history = {"gen_cost": [], "disc_cost": [], "fidelity": [], Config.COST_FN: []}

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

                disc_params, disc_cost_val = disc_opt.step_and_cost(disc_cost, disc_params)
                disc_cost_epoch.append(disc_cost_val)

            # Generator training
            for _ in range(Config.GEN_STEPS):

                def gen_cost(g_params):
                    fake_state = gen(g_params)
                    fake_probs = disc(disc_params, fake_state)
                    real_probs = disc(disc_params, target_state)
                    return -wasserstein_cost(real_probs, fake_probs)

                gen_params, gen_cost_val = gen_opt.step_and_cost(gen_cost, gen_params)
                gen_cost_epoch.append(gen_cost_val)

            # Calculate metrics
            fake_state = gen(gen_params)

            # Convert states to compatible format
            if using_density_matrices:
                rho = target_state
                sigma = fake_state
            else:
                rho = np.outer(target_state, target_state.conj())
                sigma = np.outer(fake_state, fake_state.conj())

            history["fidelity"].append(qml.math.fidelity(rho, sigma))
            history[Config.COST_FN].append(get_cost_function()(rho, sigma))

        # Save history
        history["gen_cost"].extend(gen_cost_epoch)
        history["disc_cost"].extend(disc_cost_epoch)

        # Save checkpoint
        if epoch % Config.PLOT_INTERVAL == 0:
            np.savez(f"{Config.SAVE_PATH}/checkpoint_{epoch}.npz", gen_params=gen_params, disc_params=disc_params)
            plot_training(history, epoch)

    # Save final results
    with open(f"{Config.SAVE_PATH}/config.json", "w") as f:
        config_dict = {
            k: v
            for k, v in vars(Config).items()
            if not k.startswith("__") and isinstance(v, (int, float, str, list, dict, bool))
        }
        json.dump(config_dict, f, indent=2)

    return gen_params, disc_params, history


def init_parameters(ansatz_type, is_generator=True):
    param_config = {
        "zz_xz": {"gen": (3, 3), "disc": (3, 3)},
        "zz_xx_yy_z": {"gen": (3, 5), "disc": (3, 5)},  # 5 params per gate for generator
        "hardware_eff": {"gen": (3, 3), "disc": (3, 3)},
        "basic_entangled": {"gen": (1, 1), "disc": (1, 1)},
    }
    role = "gen" if is_generator else "disc"
    layers, params_per_gate = param_config[ansatz_type][role]

    # Calculate qubit count (including ancilla if needed)
    n_qubits = Config.QUBITS
    if not is_generator and Config.ANCILLA_MODE != "none":
        n_qubits += 1  # Add ancilla for discriminator

    return pnp.array(np.random.uniform(0, 2 * np.pi, (Config.LAYERS, n_qubits, params_per_gate)), requires_grad=True)


def plot_training(history, epoch):
    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plt.plot(history["gen_cost"], label="Generator Cost")
    plt.plot(history["disc_cost"], label="Discriminator Cost")
    plt.xlabel("Training Step")
    plt.ylabel("Cost Value")
    plt.title(f"{Config.COST_FN} Cost Evolution")
    plt.legend()

    plt.subplot(122)
    plt.plot(history["fidelity"], label="State Fidelity")
    plt.plot(history[Config.COST_FN], label=f"{Config.COST_FN} Value")
    plt.xlabel("Training Step")
    plt.ylabel("Metric Value")
    plt.title("Quality Metrics")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{Config.SAVE_PATH}/training_plot_{epoch}.png")
    plt.close()


if __name__ == "__main__":
    # Test configurations
    gen_params, disc_params, history = train()
