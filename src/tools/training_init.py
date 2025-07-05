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
import traceback

from config import CFG, test_configurations
from qgan.training import Training
from tools.data.data_managers import print_and_log, print_and_log_with_headers
from tools.plot_hub import generate_all_plots

# ruff: noqa: E226


#################################################################################################################
# SINGLE RUN MODE:
#################################################################################################################
def run_single_training():
    """
    Runs a single training instance.
    """
    try:
        ##############################################################
        # Run single training instance with specified configuration
        ##############################################################
        training_instance = Training()
        training_instance.run()
        success_msg = "\nDefault configuration run COMPLETED SUCCESSFULLY.\n"
        print_and_log(success_msg, CFG.log_path)  # Log to file

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
        print_and_log(error_msg, CFG.log_path)  # Log to file


#################################################################################################################
# MULTIPLE RUNS MODE, WITH AND WITHOUT COMMON INITIAL EXPERIMENTS:
#################################################################################################################
def run_multiple_trainings():
    """Runs the multiple training logic, both for common initial experiments
    with later changes, and for no common initial experiments.

    This function handles the loading, sets the base path for results, checks for
    previous runs if specified, and executes the training.

    It also generates plots for all runs after completion, and raises the necessary
    exceptions if any errors occur during the training.
    """
    ################################################################
    # Change results directory to MULTIPLE RUNS:
    ################################################################
    base_path = f"./generated_data/MULTIPLE_RUNS/{CFG.run_timestamp}"
    CFG.base_data_path = base_path
    CFG.set_results_paths()

    # TODO: Make sure this works for the no common initial experiments case too.
    ##############################################################
    # Loading previous MULTIPLE run timestamp if specified:
    ##############################################################
    if CFG.load_timestamp is not None:
        _check_for_previous_multiple_runs()
        CFG.run_timestamp = CFG.load_timestamp
        print_and_log("Following previous MULTIPLE run, only changed experiments will be run.\n", CFG.log_path)
    else:
        print_and_log("Running MULTIPLE initial, controls and changed experiments.\n", CFG.log_path)

    ##############################################################
    # Execute multiple training instances (with/out common initial exp.)
    ##############################################################
    try:
        if CFG.common_initial_experiment:
            execute_from_common_initial_experiment(base_path)
        else:
            execute_from_no_common_initial_experiment(base_path)

        ##############################################################
        # Generate plots for all runs
        ##############################################################
        generate_all_plots(
            base_path,
            CFG.log_path,
            n_runs=len(CFG.reps_new_config),
            max_fidelity=CFG.max_fidelity,
            common_initial_experiment=CFG.common_initial_experiment,
        )
        print_and_log("\nAll multiple training runs completed.\n", CFG.log_path)
        print_and_log(
            "\nAnalysis plots (recurrence vs max fidelity, averages, and success rates) generated.\n", CFG.log_path
        )

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
        print_and_log(error_msg, CFG.log_path)


def _check_for_previous_multiple_runs():
    # Check config compatibility
    prev_log_path = f"./generated_data/MULTIPLE_RUNS/{CFG.load_timestamp}/initial_exp_1/logs/log.txt"
    if not os.path.exists(prev_log_path):
        raise RuntimeError(f"Previous run log not found: {prev_log_path}")
    with open(prev_log_path, "r") as f:
        log_content = f.read()
    # Only check for a subset of config fields (excluding run_timestamp and load_timestamp.)
    config_str = CFG.show_data()
    if config_str.split("type_of_warm_start")[1] not in log_content:
        raise RuntimeError("Current config does not match previous initial experiments. Aborting.")


#############################################################################
# Execute multiple training instances with no common initial experiments
#############################################################################
def execute_from_no_common_initial_experiment(base_path):
    """
    Runs multiple experiments from scratch (no common initial experiments),
    using CFG.N_reps_no_common_initial_exp repetitions for each config in CFG.reps_new_config.

    Results are saved in experimentX/ subfolders.
    """
    n_reps = getattr(CFG, "N_reps_no_common_initial_exp", 1)

    for run_idx, config_dict in enumerate(CFG.reps_new_config, 1):
        for key, value in config_dict.items():
            setattr(CFG, key, value)
        for rep in range(n_reps):
            out_dir = f"{base_path}/experiment{run_idx}/{rep+1}"
            CFG.base_data_path = out_dir
            CFG.set_results_paths()
            print_and_log_with_headers(f"\nExperiment {run_idx}, repetition {rep+1}/{n_reps}", CFG.log_path)
            Training().run()
            print_and_log(f"\nExperiment {run_idx}, repetition {rep+1} completed.\n", CFG.log_path)


#############################################################################
# Execute multiple training instances with common initial experiments
#############################################################################
def execute_from_common_initial_experiment(base_path):
    """
    Runs multiple training instances, with a change in the middle.

    Loops twice, first for `CFG.N_initial_exp`, then for `CFG.N_reps_each_init_exp`,
    starting from each of the last runs, changing what is specified in `CFG.reps_new_config`.

    Results are saved in initial_exp_X/repeated_controls/ and initial_exp_X/repeated_changed_runX/ subfolders.
    """
    # Cache loops configuration parameters
    n_initial_exp = getattr(CFG, "N_initial_exp", 1)
    n_reps_each_init_exp = getattr(CFG, "N_reps_each_init_exp", 1)

    ##############################################################
    # Run initial experiments
    ##############################################################
    if CFG.load_timestamp is None:
        _run_initial_experiments(n_initial_exp, base_path)
    else:
        print_and_log("\nFollowing previous MULTIPLE run, initial experiments will be skipped.\n", CFG.log_path)

    #############################################################
    # Run controls from each initial experiment
    #############################################################
    if CFG.load_timestamp is None:
        _run_repeated_experiments(n_initial_exp, n_reps_each_init_exp, base_path, "control")
    else:
        print_and_log("\nFollowing previous MULTIPLE run, control experiments will be skipped.\n", CFG.log_path)

    #############################################################
    # Run changed experiments from each initial experiment, for each config in reps_new_config:
    #############################################################
    for run_idx, config_dict in enumerate(CFG.reps_new_config, 1):
        for key, value in config_dict.items():
            setattr(CFG, key, value)
        # Each config gets its own run subdir
        _run_repeated_experiments(n_initial_exp, n_reps_each_init_exp, base_path, f"changed_run{run_idx}")


def _run_initial_experiments(n_initial_exp: int, base_path: str):
    for i in range(n_initial_exp):
        # Set path for initial experiment
        CFG.base_data_path = f"{base_path}/initial_exp_{i+1}"
        CFG.set_results_paths()

        print_and_log_with_headers(f"\nInitial Experiment {i+1}/{n_initial_exp}", CFG.log_path)
        Training().run()
        print_and_log(f"\nInitial Experiment {i+1} completed.\n", CFG.log_path)


def _run_repeated_experiments(n_initial_exp: int, n_reps_each_init_exp: int, base_path: str, changed_or_control: str):
    # changed_or_control can now be 'control' or 'changed_runX'
    is_changed = changed_or_control.startswith("changed_run")
    run_idx = None
    if is_changed:
        run_idx = int(changed_or_control.replace("changed_run", ""))
    for i, rep in itertools.product(range(n_initial_exp), range(n_reps_each_init_exp)):
        if changed_or_control == "control":
            out_dir = f"{base_path}/initial_exp_{i+1}/repeated_controls/{rep+1}"
        elif is_changed:
            out_dir = f"{base_path}/initial_exp_{i+1}/repeated_changed_run{run_idx}/{rep+1}"
        else:
            raise ValueError(
                f"Invalid value for changed_or_control: {changed_or_control} (!= 'control', 'changed_runX')."
            )
        CFG.load_timestamp = f"MULTIPLE_RUNS/{CFG.run_timestamp}/initial_exp_{i+1}"
        CFG.base_data_path = out_dir
        CFG.set_results_paths()
        print_and_log_with_headers(
            f"\nRepeated Experiments {changed_or_control} {rep+1}/{n_reps_each_init_exp} for Initial Exp {i+1}/{n_initial_exp}",
            CFG.log_path,
        )
        Training().run()
        print_and_log(
            f"\nRepeated Experiment {changed_or_control} {rep+1} for Initial Exp {i+1} completed.\n", CFG.log_path
        )


#################################################################################################################
# TESTING MODE:
#################################################################################################################
def run_test_configurations():
    """
    Runs a suite of test configurations.
    """
    # Change results directory to TESTING:
    CFG.base_data_path = f"./generated_data/TESTING/{CFG.run_timestamp}"
    CFG.set_results_paths()

    all_passed = True
    for i, config_params in enumerate(test_configurations):
        test_header_msg = f"\n{'=' * 60}\nRunning Test Configuration {i + 1}/{len(test_configurations)}: {config_params['label_suffix']}\n{'-' * 60}\n"
        print_and_log(test_header_msg, CFG.log_path)  # Also log to file

        ##############################################################
        # Set config for the current test run
        ##############################################################
        valid_keys = dir(CFG)  # Get a list of valid attributes of CFG
        for key, value in config_params.items():
            if key in valid_keys or key == "label_suffix":
                setattr(CFG, key, value)
            else:
                error_msg = f"Invalid configuration key: {key}. This key does not exist in CFG."
                print_and_log(error_msg, CFG.log_path)  # Log the error
                raise AttributeError(error_msg)

        try:
            ##############################################################
            # Run the training instance with the current configuration
            ##############################################################
            training_instance = Training()
            training_instance.run()
            success_msg = f"\n{'-' * 60}\nTest Configuration {i + 1} ({config_params['label_suffix']}) COMPLETED SUCCESSFULLY.\n{'=' * 60}\n"
            print_and_log(success_msg, CFG.log_path)

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
            print_and_log(error_msg, CFG.log_path)
            # Continue with other test configurations

    ##############################################################
    # Final summary of test configurations
    ##############################################################
    final_summary_msg = ""
    if all_passed:
        final_summary_msg = "\nAll test configurations ran SUCCESSFULLY! No errors detected during these runs.\n"
    else:
        final_summary_msg = "\nSome test configurations FAILED. Please review the logs above and the log file.\n"
    print_and_log(final_summary_msg, CFG.log_path)
