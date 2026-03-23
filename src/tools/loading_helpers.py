"""Model loading and warm-start helpers — PennyLane-PyTorch version.

Changes from original:
    - Warm start functions operate on gen.params (torch tensor with requires_grad)
      using torch.no_grad() context to modify parameters without breaking autograd.
    - After any perturbation, optimizer momentum is reset (since the parameter
      landscape has changed) and the cached state is refreshed.
    - No dependency on QuantumCircuit or QuantumGate.
    - Everything else (load logic, logging) unchanged.
"""

import math
import os

import numpy as np
import torch

from config import CFG
from data.data_managers import print_and_log


def load_models_if_specified(training_instance):
    """Load generator and discriminator parameters if a load_timestamp is provided.

    Args:
        training_instance (Training): The training instance containing gen and dis.
    """
    if not CFG.load_timestamp:
        print_and_log(
            "\nStarting training from scratch (no timestamp specified).\n",
            CFG.log_path,
        )
        print_and_log("==================================================\n", CFG.log_path)
        return

    print_and_log(
        f"\nAttempting to load model parameters [{CFG.load_timestamp}].\n",
        CFG.log_path,
    )
    gen_model_filename = os.path.basename(CFG.model_gen_path)
    dis_model_filename = os.path.basename(CFG.model_dis_path)
    load_gen_path = os.path.join(
        "generated_data", CFG.load_timestamp, "saved_model", gen_model_filename
    )
    load_dis_path = os.path.join(
        "generated_data", CFG.load_timestamp, "saved_model", dis_model_filename
    )

    print_and_log(
        f"Attempting to load Generator parameters from: {load_gen_path}\n",
        CFG.log_path,
    )
    gen_loaded = training_instance.gen.load_model_params(load_gen_path)

    print_and_log(
        f"\nAttempting to load Discriminator parameters from: {load_dis_path}\n",
        CFG.log_path,
    )
    dis_loaded = training_instance.dis.load_model_params(load_dis_path)

    if gen_loaded and dis_loaded:
        if CFG.type_of_warm_start != "none":
            apply_warm_start(training_instance)
        print_and_log(
            "Model parameter loading complete. Continuing training.\n",
            CFG.log_path,
        )
        print_and_log("==================================================\n", CFG.log_path)
    else:
        raise ValueError(
            "Incompatible or missing model parameters. "
            "Check the load paths or model compatibility."
        )


def perturb_all_gen_params_X_percent(gen):
    """Randomly perturb ALL generator parameters by a small amount.

    Each angle is shifted by a uniform random value in
    [-strength * 2pi, +strength * 2pi], then wrapped to [0, 2pi).

    Used for warm_start type "all".

    Args:
        gen: Generator instance with a .params torch tensor.
    """
    perturbation_strength = CFG.warm_start_strength * 2 * math.pi

    with torch.no_grad():
        noise = torch.empty_like(gen.params).uniform_(
            -perturbation_strength, perturbation_strength
        )
        gen.params.copy_((gen.params + noise) % (2 * math.pi))

    # Reset optimizer momentum (parameter landscape has changed)
    gen.optimizer = torch.optim.SGD(
        [gen.params], lr=CFG.l_rate, momentum=CFG.momentum_coeff
    )

    # Refresh cached state
    gen._refresh_state()


def restart_X_percent_of_gen_params_randomly(gen):
    """Randomly reset a PERCENTAGE of generator parameters to new random values.

    The percentage is determined by CFG.warm_start_strength (0 to 1).
    Selected parameters are replaced with uniform random values in [0, 2pi).

    Used for warm_start type "some".

    Args:
        gen: Generator instance with a .params torch tensor.
    """
    num_params = len(gen.params)

    if CFG.warm_start_strength > 0.0:
        num_perturb = math.ceil(num_params * CFG.warm_start_strength)
        indices = np.random.choice(num_params, size=num_perturb, replace=False)

        with torch.no_grad():
            new_vals = torch.empty(num_perturb, dtype=torch.float64).uniform_(0, 2 * math.pi)
            gen.params[indices] = new_vals

        # Reset optimizer momentum
        gen.optimizer = torch.optim.SGD(
            [gen.params], lr=CFG.l_rate, momentum=CFG.momentum_coeff
        )

        # Refresh cached state
        gen._refresh_state()


def apply_warm_start(training_instance):
    """Apply warm start to the generator if specified in configuration.

    Args:
        training_instance (Training): The training instance containing the generator.
    """
    print_and_log(
        "Warm start enabled. Randomly perturbing generator parameters.\n",
        CFG.log_path,
    )
    if CFG.type_of_warm_start == "all":
        perturb_all_gen_params_X_percent(training_instance.gen)
    elif CFG.type_of_warm_start == "some":
        restart_X_percent_of_gen_params_randomly(training_instance.gen)
    else:
        raise ValueError(f"Unknown type of warm start: {CFG.type_of_warm_start}")