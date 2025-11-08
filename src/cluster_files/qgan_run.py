from tools.training_init import run_single_training
from config import CFG 
import copy
import pickle
import os

iters = 10
size = 3
ancilla = False
config = copy.deepcopy(CFG)
config.system_size = size
config.extra_ancilla = ancilla
results = {}
for i in iters:
    res = run_single_training(config)
    results[i] = res
path = f"/data/cvcqml/common/paula/qgan/pennylane_generator/"
os.makedirs(path, exist_ok=True)
name = f'qgan_size{size}_ancilla{ancilla}.pkl'
file_path = os.path.join(path, name)
with open(file_path, 'wb') as f:
    pickle.dump(results, f)