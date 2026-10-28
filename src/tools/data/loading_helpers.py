# Copyright 2025 GIQ, Universitat Autònoma de Barcelona
#
# Licensed under the MIT License. See the LICENSE file in the project root for the full license text.
# SPDX-License-Identifier: MIT
import math
import os
import random

import numpy as np

from config import CFG
from qgan.generator import Generator
from tools.data.data_managers import print_and_log


def load_models_if_specified(training_instance):
    """
    Loads generator and discriminator parameters if a load_timestamp is provided.

    Modifies training_instance.gen and training_instance.dis by calling their load_model_params methods.

    Args:
        training_instance (Training): The training instance containing the generator and discriminator.
    """
    ################################################################
    # Conditions to skip loading models
    ################################################################
    if not CFG.load_timestamp:
        print_and_log("\nStarting training from scratch (no timestamp specified).\n", CFG.log_path)
        print_and_log("==================================================\n", CFG.log_path)
        return

    ################################################################
    # If we reach here, we have a load_timestamp
    ################################################################
    print_and_log(f"\nAttempting to load model parameters [{CFG.load_timestamp}].\n", CFG.log_path)

    # Ensure the load_timestamp is valid
    gen_model_filename = os.path.basename(CFG.model_gen_path)
    dis_model_filename = os.path.basename(CFG.model_dis_path)
    # Path structure: "generated_data/<timestamp>/saved_model/<model_filename>"
    load_gen_path = os.path.join("generated_data", CFG.load_timestamp, "saved_model", gen_model_filename)
    load_dis_path = os.path.join("generated_data", CFG.load_timestamp, "saved_model", dis_model_filename)

    ################################################################
    # Attempt to load generator:
    ################################################################
    print_and_log(
        f"Attempting to load Generator parameters from: {load_gen_path}\n",
        CFG.log_path,
    )
    gen_loaded = training_instance.gen.load_model_params(load_gen_path)

    ################################################################
    # Attempt to load discriminator:
    ################################################################
    print_and_log(
        f"\nAttempting to load Discriminator parameters from: {load_dis_path}\n",
        CFG.log_path,
    )
    dis_loaded = training_instance.dis.load_model_params(load_dis_path)

    ################################################################
    # Final check: if both models are loaded
    ################################################################
    if gen_loaded and dis_loaded:
        ##############################################################
        # Apply warm start if specified
        ##############################################################
        if CFG.type_of_warm_start != "none":
            apply_warm_start(training_instance)

        print_and_log("Model parameter loading complete. Continuing training.\n", CFG.log_path)
        print_and_log("==================================================\n", CFG.log_path)
    else:
        raise ValueError("Incompatible or missing model parameters. Check the load paths or model compatibility.")


def perturb_all_gen_params_X_percent(gen: Generator):
    """
    Randomly perturbs the model parameters by a small amount.
    This is useful for warm starting the training process.

    Args:
        model: The model whose parameters will be perturbed.
        perturbation_strength: The strength of the perturbation.
    """
    perturbation_strength = CFG.warm_start_strength * 2 * math.pi
    for gate in gen.qc.gates:
        # Randomly perturb the angle of each gate
        new_angle = gate.angle + np.random.uniform(-perturbation_strength, perturbation_strength)
        # Ensure the angle is within the range [0, 2π)
        gate.angle = new_angle % (2 * math.pi)


def restart_X_percent_of_gen_params_randomly(gen: Generator):
    """
    Randomly perturbs a percentage of the model parameters to totally new random values.

    Args:
        model: The model whose parameters will be perturbed.
        percent: The percentage of parameters to perturb (0 to 1).
    """
    # Total number of gates in the generator (assume that each gate has a single angle parameter)
    num_params = len(gen.qc.gates)

    if CFG.warm_start_strength > 0.0:
        # Compute the indices of the parameters to perturb
        num_perturb = math.ceil(num_params * CFG.warm_start_strength)
        indices = random.sample(range(num_params), num_perturb)

        # Perturb the selected parameters
        for idx in indices:
            gen.qc.gates[idx].angle = np.random.uniform(0, 2 * np.pi)


def apply_warm_start(training_instance):
    """
    Applies a warm start to the generator if specified in the configuration.

    This function modifies the generator's parameters based on the type of warm start.

    Args:
        training_instance (Training): The training instance containing the generator.
    """
    print_and_log(
        "Warm start enabled. Randomly perturbing model gen parameters.\n",
        CFG.log_path,
    )
    if CFG.type_of_warm_start == "all":
        perturb_all_gen_params_X_percent(training_instance.gen)
    elif CFG.type_of_warm_start == "some":
        restart_X_percent_of_gen_params_randomly(training_instance.gen)
    else:
        raise ValueError(f"Unknown type of warm start: {CFG.type_of_warm_start}")
    training_instance.gen.total_gen_state = training_instance.gen.get_total_gen_state()
