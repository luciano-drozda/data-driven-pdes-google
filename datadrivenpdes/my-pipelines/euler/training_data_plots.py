# Make plots of input_ and output_ samples

# Author(s): Luciano Drozda
# Date (mm/dd/yyyy): 03/20/2020

import numpy as np
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.enable_eager_execution()
print('tf.__version__ =',tf.__version__)
print('tf.keras.__version__ =',tf.keras.__version__)
# GPU setup
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

from datadrivenpdes.core import grids
from datadrivenpdes.euler import equations

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

# Euler
length = 1.
fine_resolution = 256
resample_factor = 8
coarse_resolution = fine_resolution // resample_factor
coarse_grid = grids.Grid.from_period(size=coarse_resolution, length=length)
xx, yy = coarse_grid.get_mesh()

# Plot states for a given sample
path = './data/samples'
# i_sample = 9009 # must be < input_['density'].shape[0]
i_sample = 4004 # must be < input_['density'].shape[0]
npz_input = \
  np.load(f'{path}/input_{coarse_resolution}_sample_{i_sample}.npz',
  allow_pickle=True)
npz_output = \
  np.load(f'{path}/output_{coarse_resolution}_sample_{i_sample}.npz',
  allow_pickle=True)
input_sample = npz_input['0'].item()
output_sample = npz_output['0'].item()

print('\n\n input_sample.shape')
for k, v in input_sample.items():
  print(k, v.shape)
print('\n\n output_sample.shape')
for k, v in output_sample.items():
  print(k, v.shape)

# equations.save_plot_state(input_sample, f'{i_sample}_in', xx, yy)
for i in [0, 49, 99]:
  equations.save_plot_state({k: v[i, :, :] for k, v in output_sample.items()}, f'{i_sample}_out_{i + 1}', xx, yy)
