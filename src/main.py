# Copyright 2025 GIQ, Universitat Autònoma de Barcelona
#
# Licensed under the MIT License. See the LICENSE file in the project root for the full license text.
# SPDX-License-Identifier: MIT
"""Main module for running the quantum GAN training"""

import sys

from config import CFG
from tools.training_init import run_multiple_trainings, run_single_training

# -- SINGLE RUN mode ---------------------------
def main():
    if CFG.run_multiple_experiments:
        run_multiple_trainings()
    else:
        run_single_training()


if __name__ == "__main__":
    # The logs print "═"/"─", which Windows code pages (cp1252) cannot encode when stdout is redirected to a file
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
