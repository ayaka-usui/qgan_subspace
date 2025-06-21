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

import traceback

from config import CFG, test_configurations
from qgan.training import Training
from tools.data.data_managers import print_and_train_log

# Assuming Training class is in src.training (adjust if different)
# from ..training import Training # This relative import might need adjustment based on how you run the script
# For now, let's assume Training will be passed or imported directly in main


##################################################################
# SINGLE RUN mode
##################################################################
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
        print_and_train_log(success_msg, CFG.log_path)  # Log to file

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
        print_and_train_log(error_msg, CFG.log_path)  # Log to file


##################################################################
# MULTIPLE RUNS mode
##################################################################
def run_multiple_trainings():
    """
    Runs multiple training instances, with a change in the middle.

    Loops twice, first for `CFG.N_initial_exp`, then for `CFG.N_reps_each_init_exp`,
    starting from each of the last runs, changing what is specified in `CFG.reps_new_config`.
    Saves and loads results using the generated_data folder structure.
    """
    # Change results directory to MULTIPLE RUNS:
    base_path = f"./generated_data/MULTIPLE_RUNS/{CFG.run_timestamp}"
    CFG.base_data_path = base_path
    CFG.set_results_paths()

    print_and_train_log("Running in MULTIPLE RUNS mode with a change in the middle.\n", CFG.log_path)

    try:
        # Run initial experiments
        for i in range(getattr(CFG, "N_initial_exp", 1)):
            # Set path for initial experiment
            CFG.base_data_path = f"{base_path}/initial_exp_{i+1}"
            CFG.set_results_paths()

            msg = f"\n{'='*60}\nInitial Experiment {i+1}/{CFG.N_initial_exp}\n{'-'*60}"
            print_and_train_log(msg, CFG.log_path)
            training_instance = Training()
            training_instance.run()
            print_and_train_log(f"Initial Experiment {i+1} completed.\n", CFG.log_path)

        # Run repeated experiments, before changes, for controls.
        for i in range(getattr(CFG, "N_initial_exp", 1)):
            for rep in range(getattr(CFG, "N_reps_each_init_exp", 1)):
                msg = f"\n{'='*60}\nRepeated Experiments controls {rep+1}/{CFG.N_reps_each_init_exp} for Initial Exp {i+1}\n{'-'*60}"
                print_and_train_log(msg, CFG.log_path)
                # Set CFG.load_timestamp to the initial experiment's timestamp
                CFG.load_timestamp = f"MULTIPLE_RUNS/{CFG.run_timestamp}/initial_exp_{i+1}"
                CFG.base_data_path = f"{base_path}/initial_exp_{i+1}/repeated_control_{rep+1}"
                CFG.set_results_paths()
                training_instance = Training()
                training_instance.run()
                print_and_train_log(
                    f"Repeated Experiment control {rep+1} for Initial Exp {i+1} completed.\n", CFG.log_path
                )

        # Change config for repeated experiments
        for key, value in getattr(CFG, "reps_new_config", {}).items():
            setattr(CFG, key, value)
        print_and_train_log(
            f"\n{'='*60}\nChanged config for repeated experiments: {CFG.reps_new_config}\n{'='*60}", CFG.log_path
        )

        # Run repeated experiments, loading from each initial experiment's results
        for i in range(getattr(CFG, "N_initial_exp", 1)):
            for rep in range(getattr(CFG, "N_reps_each_init_exp", 1)):
                msg = f"\n{'='*60}\nRepeated Experiment {rep+1}/{CFG.N_reps_each_init_exp} for Initial Exp {i+1}\n{'-'*60}"
                print_and_train_log(msg, CFG.log_path)
                # Set CFG.load_timestamp to the initial experiment's timestamp
                CFG.load_timestamp = f"MULTIPLE_RUNS/{CFG.run_timestamp}/initial_exp_{i+1}"
                CFG.base_data_path = f"{base_path}/initial_exp_{i+1}/repeated_changed_{rep+1}"
                CFG.set_results_paths()
                training_instance = Training()
                training_instance.run()
                print_and_train_log(f"Repeated Experiment {rep+1} for Initial Exp {i+1} completed.\n", CFG.log_path)

        # Analysis/plotting of results: recurrence vs max fidelity for controls and changed
        from tools.plot_recurrence_vs_fidelity import plot_recurrence_vs_fidelity

        plot_recurrence_vs_fidelity(base_path)
        print_and_train_log("\nAnalysis plot (recurrence vs max fidelity) generated.\n", CFG.log_path)

        print_and_train_log("\nAll multiple training runs completed.\n", CFG.log_path)

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
        print_and_train_log(error_msg, CFG.log_path)


###################################################################
# TESTING mode
###################################################################
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
        print_and_train_log(test_header_msg, CFG.log_path)  # Also log to file

        ##############################################################
        # Set config for the current test run
        ##############################################################
        valid_keys = dir(CFG)  # Get a list of valid attributes of CFG
        for key, value in config_params.items():
            if key in valid_keys or key == "label_suffix":
                setattr(CFG, key, value)
            else:
                error_msg = f"Invalid configuration key: {key}. This key does not exist in CFG."
                print_and_train_log(error_msg, CFG.log_path)  # Log the error
                raise AttributeError(error_msg)

        try:
            ##############################################################
            # Run the training instance with the current configuration
            ##############################################################
            training_instance = Training()
            training_instance.run()
            success_msg = f"\n{'-' * 60}\nTest Configuration {i + 1} ({config_params['label_suffix']}) COMPLETED SUCCESSFULLY.\n{'=' * 60}\n"
            print_and_train_log(success_msg, CFG.log_path)

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
            print_and_train_log(error_msg, CFG.log_path)
            # Continue with other test configurations

    ##############################################################
    # Final summary of test configurations
    ##############################################################
    final_summary_msg = ""
    if all_passed:
        final_summary_msg = "\nAll test configurations ran SUCCESSFULLY! No errors detected during these runs.\n"
    else:
        final_summary_msg = "\nSome test configurations FAILED. Please review the logs above and the log file.\n"
    print_and_train_log(final_summary_msg, CFG.log_path)
