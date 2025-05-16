import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


# Configuration class with all settings
class Config:
    # Default parameters (modify as needed)
    n_qubits = 4  # Number of main qubits (data qubits)
    layers = 10  # Number of layers in each ansatz circuit
    epochs = 10  # Training epochs
    gen_steps = 10  # Generator update steps per epoch
    disc_steps = 50  # Discriminator update steps per epoch
    init_mode = "choi"  # "state" or "choi"
    ancilla_mode = "none"  # "none", "pass", "project", "tracing_out"
    cost_fn = "wasserstein"  # "fidelity", "trace", "wasserstein"
    target_type = "clusterH"  # "clusterH", "rotated_surface_code", or "custom"
    ansatz_gen = "zz_xx_yy_z"  # Ansatz for generator: "zz_xz", "zz_xx_yy_z", "hardware_efficient"
    ansatz_disc = "zz_xx_yy_z"  # Ansatz for discriminator: "zz_xz", "zz_xx_yy_z", "hardware_efficient"
    lr_gen = 0.01  # Learning rate for generator
    lr_disc = 0.01  # Learning rate for discriminator
    custom_hamiltonian = (
        None  # Custom Hamiltonian (if target_type="custom"): can be defined as (coeffs, ops) or np.array
    )
    custom_state = None  # Custom target state vector (optionally set after computing ground state)
    log_interval = 1  # Print log every `log_interval` epochs
    SAVE_PATH = f"generated_data/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"


# Prepare target state vector (and custom_state if needed for custom Hamiltonian)
def compute_target_state_vector():
    # Use a small helper QNode to prepare the target state and get its statevector
    n = Config.n_qubits
    init_mode = Config.init_mode
    target = Config.target_type
    # Determine number of reference qubits and total wires needed for target state preparation
    if init_mode == "choi":
        # Target channel's Choi state preparation (default: identity channel Choi)
        # We'll use 2*n wires for ref+input (no environment for target channel by default)
        total_wires = 2 * n
    else:
        total_wires = n
    dev = qml.device("default.qubit", wires=total_wires)

    @qml.qnode(dev)
    def prep_target_circuit():
        if init_mode == "choi":
            # Prepare maximally entangled state between reference [0..n-1] and output [n..2n-1]
            for i in range(n):
                qml.Hadamard(wires=i)
                qml.CNOT(wires=[i, n + i])
            # If a specific target channel state was desired, we could apply it on output wires here.
            # By default (identity channel), we do nothing further.
        # Non-choi (state targets)
        else:
            if target == "clusterH":
                # Prepare a 1D cluster state on n qubits (open chain)
                for i in range(n):
                    qml.Hadamard(wires=i)
                for i in range(n - 1):
                    qml.CNOT(wires=[i, i + 1])
                    qml.PhaseShift(np.pi, wires=i + 1)  # CZ = CNOT + PhaseShift(pi) on target
                    qml.CNOT(wires=[i, i + 1])
            elif target == "rotated_surface_code":
                # Prepare a 4-qubit GHZ state (logical |0> of a small surface code)
                # (For general n, we connect qubit0 with all others to form GHZ)
                if n >= 1:
                    qml.Hadamard(wires=0)
                for j in range(1, n):
                    qml.CNOT(wires=[0, j])
            elif target == "custom":
                # Prepare custom target state from provided statevector (if available)
                if Config.custom_state is not None:
                    qml.StatePrep(Config.custom_state, wires=range(n))
                else:
                    # If custom_hamiltonian is given and custom_state not yet computed, compute ground state
                    coeffs, ops = Config.custom_hamiltonian
                    H = qml.Hamiltonian(coeffs, ops)
                    # Diagonalize Hamiltonian (dense matrix) to get ground state
                    H_mat = qml.utils.sparse_hamiltonian(H).toarray()
                    eigvals, eigvecs = np.linalg.eigh(H_mat)
                    ground_state = eigvecs[:, np.argmin(eigvals)]
                    Config.custom_state = ground_state  # store for reuse
                    qml.StatePrep(ground_state, wires=range(n))
            else:
                # If target_type is not recognized, default to all zeros state (|00...0>)
                pass
        return qml.state()

    # Execute the QNode to get statevector
    state = prep_target_circuit()
    # Normalize state (should already be normalized, but just in case)
    norm = np.linalg.norm(state)
    if norm == 0:
        raise ValueError("Target state preparation resulted in zero state.")
    state = state / norm
    return state


# Compute target state vector and projector for fidelity calculations
target_state_vec = compute_target_state_vector()
# Determine the wires on which the target state is defined for fidelity measurement
if Config.init_mode == "choi":
    # Target state is on reference+output (2*n_qubits wires)
    target_proj_wires = list(range(0, 2 * Config.n_qubits))
else:
    # Target state is on main wires (n_qubits wires)
    target_proj_wires = list(range(0, Config.n_qubits))
# Projector onto target state
target_proj = np.outer(target_state_vec, target_state_vec.conj())

# Define global device for generator/discriminator circuits.
# We allocate enough wires for the worst-case scenario (Choi with one environment ancilla and one output ancilla).
n = Config.n_qubits
total_wires = 2 * n + 2  # [0..n-1]=ref, [n..2n-1]=input, [2n]=environment, [2n+1]=disc output
dev = qml.device("default.qubit", wires=total_wires)


# Helper functions to apply ansatz layers
def apply_ansatz(params, wires, layers, ansatz_type):
    m = len(wires)
    # Convert wires to list for indexing
    wire_list = list(wires)
    idx = 0  # index in params list
    for l in range(layers):
        if ansatz_type == "zz_xz":
            # Entangling ZZ rotations between adjacent pairs
            for j in range(m - 1):
                qml.IsingZZ(params[idx], wires=[wire_list[j], wire_list[j + 1]])
                idx += 1
            # Single-qubit rotations: X and Z on each wire
            for j in range(m):
                qml.RX(params[idx], wires=wire_list[j])
                idx += 1
                qml.RZ(params[idx], wires=wire_list[j])
                idx += 1
        elif ansatz_type == "zz_xx_yy_z":
            # ZZ couplings
            for j in range(m - 1):
                qml.IsingZZ(params[idx], wires=[wire_list[j], wire_list[j + 1]])
                idx += 1
            # XX couplings
            for j in range(m - 1):
                qml.IsingXX(params[idx], wires=[wire_list[j], wire_list[j + 1]])
                idx += 1
            # YY couplings
            for j in range(m - 1):
                qml.IsingYY(params[idx], wires=[wire_list[j], wire_list[j + 1]])
                idx += 1
            # Single-qubit Z rotations
            for j in range(m):
                qml.RZ(params[idx], wires=wire_list[j])
                idx += 1
        elif ansatz_type == "hardware_efficient":
            # Single-qubit rotations (RY and RZ) on each wire
            for j in range(m):
                qml.RY(params[idx], wires=wire_list[j])
                idx += 1
                qml.RZ(params[idx], wires=wire_list[j])
                idx += 1
            # Entangle adjacent pairs with CNOT
            for j in range(m - 1):
                qml.CNOT(wires=[wire_list[j], wire_list[j + 1]])
        else:
            raise ValueError(f"Unknown ansatz type: {ansatz_type}")
    # Note: Any remaining idle wires are untouched and remain in |0>


# Quantum circuit for discriminator expectation on real (target) state
@qml.qnode(dev, interface="autograd", diff_method="parameter-shift")
def disc_real_circuit(disc_params):
    n = Config.n_qubits
    # Prepare target state on appropriate wires
    if Config.init_mode == "choi":
        # Prepare reference+output target state (identity channel's Choi by default)
        for i in range(n):
            qml.Hadamard(wires=i)
            qml.CNOT(wires=[i, n + i])
        # (If target channel differs, would apply it on output wires here)
        # Environment (if exists) left in |0>, no operation.
    else:
        # Prepare target state on main wires (0..n-1) using same logic as compute_target_state_vector
        if Config.target_type == "clusterH":
            for i in range(n):
                qml.Hadamard(wires=i)
            for i in range(n - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.PhaseShift(np.pi, wires=i + 1)
                qml.CNOT(wires=[i, i + 1])
        elif Config.target_type == "rotated_surface_code":
            if n >= 1:
                qml.Hadamard(wires=0)
            for j in range(1, n):
                qml.CNOT(wires=[0, j])
        elif Config.target_type == "custom":
            if Config.custom_state is not None:
                qml.StatePrep(Config.custom_state, wires=range(n))
            else:
                # Compute ground state if not already computed
                coeffs, ops = Config.custom_hamiltonian
                H = qml.Hamiltonian(coeffs, ops)
                H_mat = qml.utils.sparse_hamiltonian(H).toarray()
                eigvals, eigvecs = np.linalg.eigh(H_mat)
                ground_state = eigvecs[:, np.argmin(eigvals)]
                Config.custom_state = ground_state
                qml.StatePrep(ground_state, wires=range(n))
        # If other targets, initial state |0...0> is already fine (already in that state).
    # Determine discriminator data wires and include environment if needed
    if Config.init_mode == "choi":
        base_count = 2 * n  # ref + output wires count
    else:
        base_count = n  # main wires count
    data_wires = list(range(0, base_count))
    # Include environment in discriminator data if ancilla_mode is pass/project
    env_index = 2 * n  # environment wire index
    if Config.ancilla_mode in ["pass", "project"]:
        data_wires.append(env_index)
    # Define full wires list for discriminator ansatz (data wires plus output ancilla)
    output_index = 2 * n + 1
    disc_wires = data_wires + [output_index]
    # Apply discriminator ansatz gates
    apply_ansatz(disc_params, wires=disc_wires, layers=Config.layers, ansatz_type=Config.ansatz_disc)
    # Measure discriminator output (PauliZ expectation on output qubit)
    return qml.expval(qml.PauliZ(output_index))


# Quantum circuit for discriminator expectation on fake (generator) state
@qml.qnode(dev, interface="autograd", diff_method="parameter-shift")
def disc_fake_circuit(disc_params, gen_params):
    n = Config.n_qubits
    # Prepare generator (fake) state:
    if Config.init_mode == "choi":
        # Prepare maximally entangled pair between reference and input
        for i in range(n):
            qml.Hadamard(wires=i)
            qml.CNOT(wires=[i, n + i])
        # Apply generator ansatz on input wires [n..2n-1] (+ environment if present)
        gen_wires = list(range(n, 2 * n))  # input wires
        env_index = 2 * n
        if Config.ancilla_mode != "none":
            gen_wires.append(env_index)
        apply_ansatz(gen_params, wires=gen_wires, layers=Config.layers, ansatz_type=Config.ansatz_gen)
    else:
        # No reference; prepare state on main wires [0..n-1] (+ environment if present)
        gen_wires = list(range(0, n))
        env_index = 2 * n
        if Config.ancilla_mode != "none":
            gen_wires.append(env_index)
        apply_ansatz(gen_params, wires=gen_wires, layers=Config.layers, ansatz_type=Config.ansatz_gen)
    # Determine discriminator data wires (for fake, same selection as for real)
    if Config.init_mode == "choi":
        base_count = 2 * n
    else:
        base_count = n
    data_wires = list(range(0, base_count))
    if Config.ancilla_mode in ["pass", "project"]:
        data_wires.append(env_index)
    output_index = 2 * n + 1
    disc_wires = data_wires + [output_index]
    # Apply discriminator ansatz gates
    apply_ansatz(disc_params, wires=disc_wires, layers=Config.layers, ansatz_type=Config.ansatz_disc)
    # Measure discriminator output
    return qml.expval(qml.PauliZ(output_index))


# Quantum circuit for generator fidelity (for direct loss mode)
@qml.qnode(dev, interface="autograd", diff_method="parameter-shift")
def fidelity_circuit(gen_params):
    n = Config.n_qubits
    # Prepare generator state (similar to disc_fake but without disc operations, measure fidelity)
    if Config.init_mode == "choi":
        # Prepare maximally entangled pairs between ref and input
        for i in range(n):
            qml.Hadamard(wires=i)
            qml.CNOT(wires=[i, n + i])
        gen_wires = list(range(n, 2 * n))
        env_index = 2 * n
        if Config.ancilla_mode != "none":
            gen_wires.append(env_index)
        apply_ansatz(gen_params, wires=gen_wires, layers=Config.layers, ansatz_type=Config.ansatz_gen)
    else:
        gen_wires = list(range(0, n))
        env_index = 2 * n
        if Config.ancilla_mode != "none":
            gen_wires.append(env_index)
        apply_ansatz(gen_params, wires=gen_wires, layers=Config.layers, ansatz_type=Config.ansatz_gen)
    # Measure fidelity = <psi_target|psi_generated>^2 by projector expectation
    return qml.expval(qml.Hermitian(target_proj, wires=target_proj_wires))


# Initialize generator and discriminator parameters
# Determine number of parameters needed for generator and discriminator ansatz
def count_ansatz_params(n_wires, layers, ansatz_type):
    if ansatz_type == "zz_xz":
        params_per_layer = (n_wires - 1) + 2 * n_wires  # (m-1) ZZ + 2*m singles
    elif ansatz_type == "zz_xx_yy_z":
        params_per_layer = 3 * (n_wires - 1) + n_wires  # (m-1) each for ZZ,XX,YY + m singles
    elif ansatz_type == "hardware_efficient":
        params_per_layer = 2 * n_wires  # 2*m single rotations, entanglers have no parameters
    else:
        raise ValueError(f"Unknown ansatz type: {ansatz_type}")
    return params_per_layer * layers


def train():
    os.makedirs(Config.SAVE_PATH, exist_ok=True)

    # Determine effective number of wires for generator ansatz
    if Config.init_mode == "choi":
        gen_data_wires = Config.n_qubits  # input wires count
    else:
        gen_data_wires = Config.n_qubits
    gen_wire_count = gen_data_wires + (1 if Config.ancilla_mode != "none" else 0)
    # For choi, generator ansatz wires = input + env; for state, main + env if present.
    gen_param_count = count_ansatz_params(gen_wire_count, Config.layers, Config.ansatz_gen)

    # Determine effective number of wires for discriminator ansatz
    if Config.init_mode == "choi":
        disc_data_count = 2 * Config.n_qubits
    else:
        disc_data_count = Config.n_qubits
    if Config.ancilla_mode in ["pass", "project"]:
        disc_data_count += 1  # environment is included as data
    # Discriminator ansatz wires = disc_data_count + 1 output
    disc_param_count = count_ansatz_params(disc_data_count + 1, Config.layers, Config.ansatz_disc)

    # Initialize parameters randomly
    np.random.seed(42)  # for reproducibility
    gen_params = pnp.array(np.random.uniform(0, 2 * np.pi, gen_param_count), requires_grad=True)
    disc_params = pnp.array(np.random.uniform(0, 2 * np.pi, disc_param_count), requires_grad=True)

    # Prepare optimizers for generator and discriminator
    gen_opt = qml.GradientDescentOptimizer(Config.lr_gen)
    disc_opt = qml.GradientDescentOptimizer(Config.lr_disc)

    # Training loop
    fidelity_history = []
    gen_loss_history = []
    disc_loss_history = []

    for epoch in range(1, Config.epochs + 1):
        # Adversarial training (trace or wasserstein cost)
        if Config.cost_fn.lower() in ["trace", "wasserstein"]:
            # Update discriminator
            for _ in range(Config.disc_steps):
                # Define cost for discriminator (with generator params fixed)
                def cost_disc(d_params):
                    E_real = disc_real_circuit(d_params)
                    E_fake = disc_fake_circuit(d_params, gen_params)
                    return -(E_real - E_fake)  # minimize negative difference

                disc_params = disc_opt.step(cost_disc, disc_params)
            # Update generator
            for _ in range(Config.gen_steps):

                def cost_gen(g_params):
                    # Use current disc_params as constant
                    E_fake = disc_fake_circuit(disc_params, g_params)
                    return -E_fake  # generator tries to maximize E_fake (minimize -E_fake)

                gen_params = gen_opt.step(cost_gen, gen_params)
            # Logging: compute fidelity and losses for monitoring
            fid = fidelity_circuit(gen_params) if Config.init_mode == "state" or Config.init_mode == "choi" else None
            E_real = disc_real_circuit(disc_params)
            E_fake = disc_fake_circuit(disc_params, gen_params)
            d_loss = -(E_real - E_fake)
            g_loss = -E_fake
        # Direct fidelity training
        elif Config.cost_fn.lower() == "fidelity":
            # No discriminator updates, optimize generator to maximize fidelity
            for _ in range(Config.gen_steps):

                def cost_fid(g_params):
                    # Cost = 1 - fidelity, to minimize (maximize fidelity)
                    return 1 - fidelity_circuit(g_params)

                gen_params = gen_opt.step(cost_fid, gen_params)
            # Compute metrics
            fid = fidelity_circuit(gen_params)
            g_loss = 1 - fid
            d_loss = None  # no discriminator
        else:
            raise ValueError(f"Unknown cost function: {Config.cost_fn}")
        # Store history
        if fid is not None:
            fidelity_history.append(fid)
        if g_loss is not None:
            gen_loss_history.append(g_loss)
        if d_loss is not None:
            disc_loss_history.append(d_loss)
        # Print log
        if epoch % Config.log_interval == 0 or epoch == Config.epochs:
            if Config.cost_fn.lower() == "fidelity":
                print(f"Epoch {epoch}: Fidelity = {fid:.4f}, Generator loss = {g_loss:.4f}")
            else:
                print(f"Epoch {epoch}: Fidelity = {fid:.4f}, Gen loss = {g_loss:.4f}, Disc loss = {d_loss:.4f}")

        # Plot fidelity and loss curves at each epoch
        plt.figure(figsize=(10, 4))
        # Fidelity plot
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(fidelity_history) + 1), fidelity_history, label="Fidelity", color="blue")
        plt.xlabel("Epoch")
        plt.ylabel("Fidelity")
        plt.title("Fidelity vs Epoch")
        plt.ylim([0, 1.05])
        plt.legend()
        # Loss plot
        plt.subplot(1, 2, 2)
        if gen_loss_history:
            plt.plot(range(1, len(gen_loss_history) + 1), gen_loss_history, label="Generator Loss", color="red")
        if disc_loss_history:
            plt.plot(range(1, len(disc_loss_history) + 1), disc_loss_history, label="Discriminator Loss", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss vs Epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{Config.SAVE_PATH}/training_plot_{epoch}.png")

    # Save final parameters to file
    param_filename = f"{Config.SAVE_PATH}/final_params_{Config.init_mode}_{Config.ancilla_mode}_{Config.cost_fn}.txt"
    with open(param_filename, "w") as f:
        f.write(f"Final generator parameters:\n{gen_params}\n")
        if Config.cost_fn.lower() in ["trace", "wasserstein"]:
            f.write(f"Final discriminator parameters:\n{disc_params}\n")
    print(f"Final parameters saved.")


# Optional: Test multiple configurations automatically (INIT_MODE × ANCILLA_MODE × COST_FN) with 4 qubits and 3 layers
if __name__ == "__main__":
    # If to train or not to train:
    testing = False

    ##################################
    # Testing multiple configurations:
    ##################################
    if testing:
        test_init_modes = ["state", "choi"]
        test_ancilla_modes = ["none", "pass", "project", "tracing_out"]
        test_cost_fns = ["fidelity", "trace", "wasserstein"]
        # Reduce epochs for automated tests to keep runtime reasonable
        Config.epochs = 5
        print("\nTesting multiple configurations:")
        for init_mode in test_init_modes:
            for anc_mode in test_ancilla_modes:
                for cost in test_cost_fns:
                    # Skip non-physical combos if desired (e.g., fidelity+choi might not be usually used, but we will still test it)
                    Config.init_mode = init_mode
                    Config.ancilla_mode = anc_mode
                    Config.cost_fn = cost

                    # Start training with new configuration
                    train()

    #################
    # BASIC EXECUTION
    #################
    else:
        train()
