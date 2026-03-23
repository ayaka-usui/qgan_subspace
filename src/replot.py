"""Module, for manual replotting of a generated_data/timestamp
No big changed from original"""

import os

from tools.data_managers import get_last_experiment_idx
from tools.plot_hub import find_if_common_initial_plateaus, generate_all_plots


# ------- Parameters for the replotting script --------------
# EXAMPLE TO EDIT PARAMETERS:
time_stamp_to_replot = "2026-03-19__16-47-01" 
max_fidelity = 0.99
x_label = "Ancilla Topology"
run_names = [
    "Ansatz",
    "Bridge No1Q",
    "Short Bridge",
    "Bridge",
    "Total",
]
# STOP EDITING HERE



# -------------- Replotting script for the specified experiment --------------
# Path to the experiment folder
base_path = os.path.join("generated_data", time_stamp_to_replot)
log_path = os.path.join(base_path, "replot_log.txt")

# Extract the number of runs and whether there are common initial plateaus
common_initial_plateaus = find_if_common_initial_plateaus(base_path)
n_runs = get_last_experiment_idx(base_path, common_initial_plateaus)

print(f"Replotting for {time_stamp_to_replot} with {n_runs} experiments")

# Plot:
generate_all_plots(
    base_path,
    log_path,
    n_runs=n_runs,
    max_fidelity=max_fidelity,
    common_initial_plateaus=common_initial_plateaus,
    run_names=run_names,
    x_label=x_label,
)