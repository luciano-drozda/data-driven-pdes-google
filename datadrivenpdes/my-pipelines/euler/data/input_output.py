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
print('\n tensorflow version =', tf.__version__,'\n')
# GPU setup
print("\n Num GPUs Available: ", 
      len(tf.config.experimental.list_physical_devices('GPU')),'\n')
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

def make_train_data(
  integrated_coarse, coarse_time_steps, example_time_steps=100
):
  # we need to re-format data so that single-step input maps to multi-step output
  # remove the last several time steps, as training input
  train_input = \
    {k: v[:-example_time_steps] for k, v in integrated_coarse.items()}

  # merge time and sample dimension as required by model
  n_time, n_sample, n_x, n_y = train_input['density'].shape
  for k in train_input:
    train_input[k] = tf.reshape(train_input[k], [n_sample * n_time, n_x, n_y])

  print('\n train_input shape:')
  for k, v in train_input.items():
    print(k, v.shape)  # (merged_sample, x, y)

  # pick the shifted time series, as training output
  train_output = {}
  for k in integrated_coarse.keys():
    output_list = []
    for shift in range(1, example_time_steps+1):
      # output time series, starting from each single time step
      output_slice = \
        integrated_coarse[k][shift:coarse_time_steps - example_time_steps + shift + 1]
      # merge time and sample dimension as required by training
      n_time, n_sample, n_x, n_y = output_slice.shape
      output_slice = tf.reshape(output_slice, [n_sample * n_time, n_x, n_y])
      output_list.append(output_slice)

    train_output[k] = tf.stack(output_list, axis=1)  # concat along shift_time dimension, after sample dimension

  print('\n train_output shape:')
  for k, v in train_output.items():
    print(k, v.shape)  # (merged_sample, shift_time, x, y)

  # sanity check on shapes
  for k in train_output.keys():
    assert train_output[k].shape[0] == train_input[k].shape[0]  # merged_sample
    assert train_output[k].shape[1] == example_time_steps
    assert train_output[k].shape[2] == train_input[k].shape[1]  # x
    assert train_output[k].shape[3] == train_input[k].shape[2]  # y

  return train_input, train_output

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

length = 1.
fine_resolution = 256
fine_time_steps = 8000
resample_factor_list = [8]

fine_grid = grids.Grid.from_period(size=fine_resolution, length=length)
equation = equations.Weno5Euler(no_dimensions=True)
model = models.FiniteDifferenceModel(equation, fine_grid)
key_defs = equation.key_definitions

# Load fine time-series for each seed
coarse_list = [ {} for _ in range(len(resample_factor_list)) ]
n_seeds = 10
for i_seed in range(n_seeds):
  npz_fine = \
    np.load(
      f'./fine_{fine_resolution}_seed_{30 + i_seed}.npz', allow_pickle=True
    )
  integrated_fine = npz_fine['0'].item()
  # Regrid to coarse grid for each resample factor
  for i_factor, resample_factor in enumerate(resample_factor_list):
    coarse_resolution = fine_resolution // resample_factor
    coarse_grid = grids.Grid.from_period(size=coarse_resolution, length=length)
    integrated_coarse = \
      tensor_ops.regrid(
        integrated_fine, key_defs, fine_grid, coarse_grid
      )
    integrated_coarse = \
      {k: v[::resample_factor] for k, v in integrated_coarse.items()}

    for k in key_defs:
      coarse_list[i_factor].setdefault(k, [])
      coarse_list[i_factor][k].append( integrated_coarse[k] )

for i_factor, resample_factor in enumerate(resample_factor_list):
  # Stack seed-samples and revert time and seed-sample dimensions
  integrated_coarse = \
    {k: tf.transpose(tf.stack(v), perm=[1,0,2,3]) 
      for k, v in coarse_list[i_factor].items()}

  # Make training data
  coarse_time_steps = fine_time_steps // resample_factor
  input_, output_ = \
    make_train_data(integrated_coarse, coarse_time_steps)

  # Write npz files containing training input and output for each sample
  coarse_resolution = fine_resolution // resample_factor
  for i_sample in range( len(input_['density']) ):
    npz_input = { '0': {k: v[i_sample] for k, v in input_.items()} }
    npz_output = { '0': {k: v[i_sample] for k, v in output_.items()} }
    np.savez(
      f'./samples/input_{coarse_resolution}_sample_{27030 + i_sample}.npz',
      **npz_input
    )
    np.savez(
      f'./samples/output_{coarse_resolution}_sample_{27030 + i_sample}.npz',
      **npz_output
    )
