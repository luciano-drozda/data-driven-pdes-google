import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np

from datadrivenpdes.core import grids
from datadrivenpdes.core import integrate
from datadrivenpdes.core import models
from datadrivenpdes.core import tensor_ops
from datadrivenpdes.euler import equations

import tensorflow as tf
tf.enable_eager_execution()
print('\n tensorflow version =', tf.__version__, '\n')
# GPU setup
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

SEED_OFFSET = 1000000

length = 1.
fine_resolution = 256
fine_time_steps = 8000
steps = np.arange(fine_time_steps + 1)

fine_grid = grids.Grid.from_period(size=fine_resolution, length=length)
equation = equations.Weno5Euler(no_dimensions=True)
model = models.FiniteDifferenceModel(equation, fine_grid)

i_seed = int(os.environ['SLURM_ARRAY_TASK_ID'])
print(f'-----------------')
print(f'i_seed = {i_seed}')
print(f'-----------------')
seed = i_seed + SEED_OFFSET
initial_state_fine = \
  equation.random_state_double_shear_layer(fine_grid, seed=seed)

# Solve Euler at high resolution
integrated_fine = integrate.integrate_steps(model, initial_state_fine, steps)
# Write npz file containing fine time-series
npz_fine = { '0': integrated_fine }
np.savez(f'./fine_{fine_resolution}_seed_{i_seed}.npz', **npz_fine)
