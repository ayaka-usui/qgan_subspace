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

"""Training initialization module for running training instances"""

import itertools
import os
import shutil
import traceback

import numpy as np

from config import CFG, test_configurations
from qgan.training import Training
from tools.data.data_managers import get_last_experiment_idx, print_and_log, print_and_log_with_headers
from tools.plot_hub import generate_all_plots

# ruff: noqa: E226


#################################################################################################################
# SINGLE RUN MODE:
#################################################################################################################
def run_single_training(config=CFG):
    """
    Runs a single training instance.
    """
    print_and_log("Running in SINGLE RUN mode.\n", config.log_path)

    try:
        ##############################################################
        # Run single training instance with specified configuration
        ##############################################################
        training_instance = Training(config=config)
        results = training_instance.run()
        success_msg = "\nDefault configuration run COMPLETED SUCCESSFULLY.\n"
        print_and_log(success_msg, config.log_path)  # Log to file
        return results

    except Exception as e:  # noqa: BLE001
        ##############################################################
        # Handle exceptions during the training run
        ##############################################################
        tb_str = traceback.format_exc()
        error_msg = (
            f"\n{'-' * 60}\n"
            f"FAILED: Default configuration run!\n"
            f"Error Type: {type(e).__name__}\n"
            f"Error Message: {e!s}\n"
            f"Traceback:\n{tb_str}"
            f"{'=' * 60}\n"
        )
        print_and_log(error_msg, config.log_path)  # Log to file


#################################################################################################################
# MULTIPLE RUNS MODE, WITH AND WITHOUT COMMON INITIAL EXPERIMENTS:
#################################################################################################################
def run_multiple_trainings(config=CFG):
    """Runs the multiple training logic, both for common initial plateaus
    with later changes, and for no common initial plateaus.

    This function handles the loading, sets the base path for results, checks for
    previous runs if specified, and executes the training.

    It also generates plots for all runs after completion, and raises the necessary
    exceptions if any errors occur during the training.
    """
    ##############################################################
    # Loading previous MULTIPLE run timestamp if specified:
    ##############################################################
    if config.load_timestamp is not None:
        if config.common_initial_plateaus:
            _check_for_previous_multiple_runs(config)
        config.run_timestamp = config.load_timestamp

    ################################################################
    # Change results directory to MULTIPLE RUNS:
    ################################################################
    base_path = f"./generated_data/{config.run_timestamp}"
    config.base_data_path = base_path
    config.set_results_paths()

    # First log message:
    if config.load_timestamp is not None:
        print_and_log_with_headers("\nFollowing previous MULTIPLE run, in an already existing directory.", config.log_path)
    else:
        print_and_log_with_headers("\nRunning MULTIPLE initial, in a new directory.", config.log_path)
    # Log the changed configuration:
    print_and_log("\nExperiments to execute:\n", config.log_path)
    for config_dict in config.reps_new_config:
        config_str = ", ".join(f"{key}: {value}" for key, value in config_dict.items())
        print_and_log(f"- {config_str}\n", config.log_path)

    ##############################################################
    # Execute multiple training instances (with/out common initial exp.)
    ##############################################################
    try:
        if config.common_initial_plateaus:
            execute_from_common_initial_plateaus(base_path, config)
        else:
            execute_from_no_common_initial_plateaus(base_path, config)

        ##############################################################
        # Generate plots for all runs
        ##############################################################
        generate_all_plots(
            base_path,
            config.log_path,
            n_runs=len(config.reps_new_config),
            max_fidelity=config.max_fidelity,
            common_initial_plateaus=config.common_initial_plateaus,
        )
        print_and_log("\nAll multiple training runs completed.\n", config.log_path)
        print_and_log("\nAnalysis plots (recurrence vs max fidelity, averages, and success rates) generated.\n", config.log_path)
    ##############################################################
    # Handle exceptions during the training run
    ##############################################################
    except Exception as e:
        tb_str = traceback.format_exc()
        error_msg = (
            f"\n{'-' * 60}\n"
            f"FAILED: Multiple training runs!\n"
            f"Error Type: {type(e).__name__}\n"
            f"Error Message: {e!s}\n"
            f"Traceback:\n{tb_str}"
            f"{'=' * 60}\n"
        )
        print_and_log(error_msg, config.log_path)


def _check_for_previous_multiple_runs(config=CFG):
    # Check config compatibility
    prev_log_path = f"./generated_data/{config.load_timestamp}/initial_plateau_1/logs/log.txt"
    if not os.path.exists(prev_log_path):
        raise RuntimeError(f"Previous run log not found: {prev_log_path}")
    with open(prev_log_path, "r") as f:
        log_content = f.read()
    # Only check for a subset of config fields (excluding run_timestamp and load_timestamp.)
    config_str = config.show_data()
    if config_str.split("type_of_warm_start")[1] not in log_content:
        raise RuntimeError("Current config does not match previous initial plateaus. Aborting.")


#############################################################################
# Execute multiple training instances with no common initial plateaus
#############################################################################
def execute_from_no_common_initial_plateaus(base_path, config=CFG):
    """
    Runs multiple experiments from scratch (no common initial plateaus),
    using config.N_reps_if_from_scratch repetitions for each config in config.reps_new_config.

    Results are saved in experimentX/ subfolders.

    If loading a previous multiple run, appends new runs after the last existing index.
    """
    n_reps = getattr(CFG, "N_reps_if_from_scratch", 1)

    # Find the last run index if loading previous multiple runs
    last_idx = 0 if config.load_timestamp is None else get_last_experiment_idx(base_path, common_initial_plateaus=False)
    config.load_timestamp = None  # Clear load_timestamp after using it

    for run_idx, config_dict in enumerate(config.reps_new_config, 1):
        new_run_idx = last_idx + run_idx
        for key, value in config_dict.items():
            setattr(CFG, key, value)
        for rep in range(n_reps):
            out_dir = f"{base_path}/experiment{new_run_idx}/{rep + 1}"
            config.base_data_path = out_dir
            config.set_results_paths()
            print_and_log_with_headers(f"\nExperiment {new_run_idx}, repetition {rep + 1}/{n_reps}", config.log_path)
            Training().run()
            print_and_log(f"\nExperiment {new_run_idx}, repetition {rep + 1} completed.\n", config.log_path)


#############################################################################
# Execute multiple training instances with common initial plateaus
#############################################################################
def execute_from_common_initial_plateaus(base_path, config=CFG):
    """
    Runs multiple training instances, with a change in the middle.

    Loops twice, first for `config.N_initial_plateaus`, then for `config.N_reps_each_init_plateau`,
    starting from each of the last runs, changing what is specified in `config.reps_new_config`.

    Results are saved in initial_plateau_X/repeated_control and initial_plateau_X/repeated_changed_runX/ subfolders.

    If loading a previous multiple runs, appends new runs after the last existing index.
    """
    # Cache loops configuration parameters
    N_initial_plateaus = getattr(CFG, "N_initial_plateaus", 1)
    N_reps_each_init_plateau = getattr(CFG, "N_reps_each_init_plateau", 1)

    # Find the last run index if loading previous multiple runs
    last_idx = 0 if config.load_timestamp is None else get_last_experiment_idx(base_path, common_initial_plateaus=True)

    #############################################################
    # Run initial plateaus
    #############################################################
    if config.load_timestamp is None:
        print_and_log("\nRunning initial plateaus.\n", config.log_path)
        _run_initial_plateaus(N_initial_plateaus, base_path, config)
    else:
        print_and_log("\nFollowing previous MULTIPLE run, initial plateaus will be skipped.\n", config.log_path)

    #############################################################
    # Run controls from each initial plateau
    #############################################################
    if config.load_timestamp is None:
        print_and_log("\nRunning control experiments on plateaus.\n", config.log_path)
        _run_repeated_experiments(N_initial_plateaus, N_reps_each_init_plateau, base_path, "control", config=config)
    else:
        print_and_log("\nFollowing previous MULTIPLE run, control experiments will be skipped.\n", config.log_path)

    #############################################################
    # Run changed experiments from each initial plateau, for each config in reps_new_config:
    #############################################################
    for run_idx, config_dict in enumerate(config.reps_new_config, 1):
        new_run_idx = last_idx + run_idx
        for key, value in config_dict.items():
            setattr(CFG, key, value)
        # Each config gets its own run subdir
        print_and_log(f"\nRunning changed_run{new_run_idx}.\n", config.log_path)
        _run_repeated_experiments(N_initial_plateaus, N_reps_each_init_plateau, base_path, f"changed_run{new_run_idx}", config=config)


def _run_initial_plateaus(N_initial_plateaus: int, base_path: str, config=CFG):
    """
    Run initial plateaus, only keeping those that do NOT reach max_fidelity.
    Continue until N_initial_plateaus such failed initials are found.
    """
    kept = 0
    attempt = 0
    max_fid = getattr(CFG, "max_fidelity", 0.99)
    while kept < N_initial_plateaus:
        attempt += 1
        # Set path for initial plateau (use attempt index for uniqueness)
        exp_dir = f"{base_path}/initial_plateau_{kept + 1}"
        temp_dir = f"{base_path}/initial_plateau_attempt_{attempt}"
        config.base_data_path = temp_dir
        config.set_results_paths()
        print_and_log_with_headers(f"\nInitial Plateau Attempt {attempt} (kept {kept + 1}/{N_initial_plateaus})", config.log_path)
        Training().run()
        print_and_log(f"\nInitial Plateau Attempt {attempt} completed. Checking fidelity...\n", config.log_path)
        # Check final fidelity
        fid_file = os.path.join(temp_dir, "fidelities", "log_fidelity_loss.txt")
        if os.path.exists(fid_file):
            try:
                data = np.loadtxt(fid_file)
                if data.ndim == 1:
                    fids = data
                else:
                    fids = data[0] if data.shape[0] < data.shape[1] else data[:, 0]
                max_found = np.max(fids)
            except Exception:
                max_found = 0
        else:
            max_found = 0
        if max_found < max_fid:
            # Keep this plateau: rename to sequential initial_plateau_X
            os.rename(temp_dir, exp_dir)
            kept += 1
        else:
            # Remove this experiment
            shutil.rmtree(temp_dir, ignore_errors=True)

    # Set the base path to the root directory of the initial plateaus
    config.base_data_path = base_path
    config.set_results_paths()
    print_and_log(f"\nFinished collecting {N_initial_plateaus} initial plateaus below threshold.\n", config.log_path)


def _run_repeated_experiments(N_initial_plateaus: int, N_reps_each_init_plateau: int, base_path: str, changed_or_control: str, config=CFG):
    # changed_or_control can now be 'control' or 'changed_runX'
    is_changed = changed_or_control.startswith("changed_run")
    run_idx = None
    if is_changed:
        run_idx = int(changed_or_control.replace("changed_run", ""))

    # Only 1 control run per initial plateau:
    if changed_or_control == "control":
        N_reps_each_init_plateau = 1

    for i, rep in itertools.product(range(N_initial_plateaus), range(N_reps_each_init_plateau)):
        if changed_or_control == "control":
            out_dir = f"{base_path}/initial_plateau_{i + 1}/repeated_control"
        elif is_changed:
            out_dir = f"{base_path}/initial_plateau_{i + 1}/repeated_changed_run{run_idx}/{rep + 1}"
        else:
            raise ValueError(f"Invalid value for changed_or_control: {changed_or_control} (!= 'control', 'changed_runX').")
        config.load_timestamp = f"{config.run_timestamp}/initial_plateau_{i + 1}"
        config.base_data_path = out_dir
        config.set_results_paths()
        print_and_log_with_headers(
            f"\nRepeated Experiments {changed_or_control} {rep + 1}/{N_reps_each_init_plateau} for Initial Plateau {i + 1}",
            config.log_path,
        )
        Training().run()
        print_and_log(
            f"\nRepeated Experiment {changed_or_control} {rep + 1} for Initial Plateau {i + 1} completed.\n",
            config.log_path,
        )


#################################################################################################################
# TESTING MODE:
#################################################################################################################
def run_test_configurations(config=CFG):
    """
    Runs a suite of test configurations.
    """
    # Change results directory to TESTING:
    config.base_data_path = f"./generated_data/TESTING/{config.run_timestamp}"
    config.set_results_paths()

    all_passed = True
    for i, config_params in enumerate(test_configurations):
        test_header_msg = (
            f"\n{'=' * 60}\nRunning Test Configuration {i + 1}/{len(test_configurations)}: {config_params['label_suffix']}\n{'-' * 60}\n"
        )
        print_and_log(test_header_msg, config.log_path)  # Also log to file

        ##############################################################
        # Set config for the current test run
        ##############################################################
        valid_keys = dir(CFG)  # Get a list of valid attributes of CFG
        for key, value in config_params.items():
            if key in valid_keys or key == "label_suffix":
                setattr(CFG, key, value)
            else:
                error_msg = f"Invalid configuration key: {key}. This key does not exist in config."
                print_and_log(error_msg, config.log_path)  # Log the error
                raise AttributeError(error_msg)

        try:
            ##############################################################
            # Run the training instance with the current configuration
            ##############################################################
            training_instance = Training()
            training_instance.run()
            success_msg = (
                f"\n{'-' * 60}\nTest Configuration {i + 1} ({config_params['label_suffix']}) COMPLETED SUCCESSFULLY.\n{'=' * 60}\n"
            )
            print_and_log(success_msg, config.log_path)

        except Exception as e:  # noqa: BLE001
            ##############################################################
            # Handle exceptions for the current test configuration
            ##############################################################
            all_passed = False
            tb_str = traceback.format_exc()
            error_msg = (
                f"\n{'-' * 60}\n"
                f"FAILED: Test Configuration {i + 1} ({config_params['label_suffix']})!\n"
                f"Error Type: {type(e).__name__}\n"
                f"Error Message: {e!s}\n"
                f"Traceback:\n{tb_str}"
                f"{'=' * 60}\n"
            )
            print_and_log(error_msg, config.log_path)
            # Continue with other test configurations

    ##############################################################
    # Final summary of test configurations
    ##############################################################
    final_summary_msg = ""
    if all_passed:
        final_summary_msg = "\nAll test configurations ran SUCCESSFULLY! No errors detected during these runs.\n"
    else:
        final_summary_msg = "\nSome test configurations FAILED. Please review the logs above and the log file.\n"
    print_and_log(final_summary_msg, config.log_path)
