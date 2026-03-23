"""Main module for running the quantum GAN training.
No big changed from original """

from config import CFG
from training_init import run_multiple_trainings, run_single_training


# -- SINGLE RUN mode ---------------------------
def main():
    if CFG.run_multiple_experiments:
        run_multiple_trainings()
    else:
        run_single_training()


if __name__ == "__main__":
    main()