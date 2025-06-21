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
"""Main module for running the quantum GAN training and testing."""

from config import CFG
from tools.data.data_managers import print_and_train_log
from tools.training_init import run_multiple_trainings, run_single_training


##############################################################
# SINGLE RUN mode
##############################################################
def main():
    if CFG.run_multiple_experiments:
        run_multiple_trainings()
    else:
        print_and_train_log("Running in SINGLE RUN mode.\n", CFG.log_path)
        run_single_training()


if __name__ == "__main__":
    main()
