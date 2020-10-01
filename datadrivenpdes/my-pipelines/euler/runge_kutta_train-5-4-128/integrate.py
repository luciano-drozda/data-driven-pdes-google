# Integrate pretrained neural net, fine-filtered (reference) and coarse models.

# Author(s): Luciano Drozda
# Date (mm/dd/yyyy): 10/01/2020

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
from datadrivenpdes.core import integrate
from datadrivenpdes.core import models
from datadrivenpdes.core import tensor_ops
from datadrivenpdes.pipelines import model_utils
from datadrivenpdes.euler import equations

# Euler
length = 1.
fine_resolution = 256
resample_factor = 8
coarse_resolution = fine_resolution // resample_factor
fine_grid = grids.Grid.from_period(size=fine_resolution, length=length)
coarse_grid = grids.Grid.from_period(size=coarse_resolution, length=length)
xx, yy = coarse_grid.get_mesh()

# Define initial state to integrate
seed = 600000
initial_state = equation_nn.random_state_double_shear_layer(fine_grid, seed=seed)

# Regrid `initial_state` to coarse resolution
initial_state_coarse = \
  tensor_ops.regrid(
    initial_state, key_defs, fine_grid, coarse_grid
  )

# How many integration time-steps?
num_time_steps = 1001
time_steps = np.arange(num_time_steps)
rk_time_steps = np.arange(3 * num_time_steps)

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

# Get info of the 'stencil_size', 'num_layers', 'filters' from the directory name
folder_name = os.path.split(os.getcwd())[-1] # 'train-{stencil_size}-{num_layers}_{filters}'
str_lst = folder_name.split('-')
stencil_size = int(str_lst[1])
num_layers = int(str_lst[2])
filters = int(str_lst[3])

# Create neural net model instance
equation_nn = equations.FiniteVolumeEuler(no_dimensions=True)
key_defs = equation_nn.key_definitions

NUM_TIME_STEPS = 90
model_nn = models.PseudoLinearModel(
  equation_nn,
  coarse_grid,
  num_time_steps=NUM_TIME_STEPS, # multi-step loss
  stencil_size=stencil_size, kernel_size=(3,3), num_layers=num_layers,filters=filters,
  constrained_accuracy_order=1,
  activation='relu',
  core_model_func=models.RescaledConv2DStack
  )
print('model_nn.learned_keys =',model_nn.learned_keys)
print('model_nn.fixed_keys =',model_nn.fixed_keys)

equations.save_plot_state(initial_state_coarse, 'initial_nn', xx, yy) # plot initial state
initial_state_nn = \
  {k: tf.expand_dims(v, 0) for k, v in initial_state.items()}
# Load pretrained weights on model_nn
model_utils.load_weights(model_nn, f'./weights_trained_{NUM_TIME_STEPS}.h5')

# Neural net model integration
results = \
  integrate.integrate_steps(
    model_nn, initial_state_nn, rk_time_steps, axis=0
  )

# Convert solution to NumPy arrays
for key in results.keys():
  results[key] = results[key].numpy()[2::3, 0, :, :]

# Include pressure in results
GAMMA = 1.4
results['pressure'] = \
  (GAMMA - 1) * ( results['energy']
    - (results['x_momentum'] ** 2 + results['y_momentum'] ** 2) / 2 
      / results['density'] )

# Plot results
for key in results.keys():
  equations.save_plot(results[key], key + '__nn', time_steps, xx, yy)

# Write npz file containing results
npz_results = {'0': results}
np.savez('./results_nn.npz', **npz_results)

print('\n\n NEURAL NET FINISHED ! \n\n')

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

# Create fine-filtered (reference) model instance
equation = equations.Weno5Euler(no_dimensions=True)
key_defs = equation.key_definitions
model = models.FiniteDifferenceModel(equation, fine_grid)
fine_xx, fine_yy = fine_grid.get_mesh()
equations.save_plot_state(initial_state, 'initial_ref', fine_xx, fine_yy) # plot initial state
fine_time_steps = np.arange(num_time_steps * resample_factor)

# Fine-filtered (reference) model integration
results = \
  integrate.integrate_steps(
    model, initial_state, fine_time_steps, axis=0
  )

# Regrid to coarse resolution
results = \
  tensor_ops.regrid(
    results, key_defs, fine_grid, coarse_grid
  )
results = \
  {k: v[::resample_factor] for k, v in results.items()}

# Convert solution to NumPy arrays
for key in results.keys():
  results[key] = results[key].numpy()

# Include pressure in results
GAMMA = 1.4
results['pressure'] = \
  (GAMMA - 1) * ( results['energy']
    - (results['x_momentum'] ** 2 + results['y_momentum'] ** 2) / 2 
      / results['density'] )

# Plot results
for key in results.keys():
  equations.save_plot(results[key], key + '__ref', time_steps, xx, yy)

# Write npz file containing results
npz_results = {'0': results}
np.savez('./results_ref.npz', **npz_results)

print('\n\n FINE-FILTERED FINISHED ! \n\n')

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

# Create coarse model instance
model = models.FiniteDifferenceModel(equation, coarse_grid)

# Coarse model integration
results = \
  integrate.integrate_steps(
    model, initial_state_coarse, time_steps, axis=0
  )

# Convert solution to NumPy arrays
for key in results.keys():
  results[key] = results[key].numpy()

# Include pressure in results
GAMMA = 1.4
results['pressure'] = \
  (GAMMA - 1) * ( results['energy']
    - (results['x_momentum'] ** 2 + results['y_momentum'] ** 2) / 2 
      / results['density'] )

# Plot results
for key in results.keys():
  equations.save_plot(results[key], key + '__coarse', time_steps, xx, yy)

# Write npz file containing results
npz_results = {'0': results}
np.savez('./results_coarse.npz', **npz_results)

print('\n\n COARSE FINISHED ! \n\n')

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

# Write npz file containing grid and integration times
npz_grid = { '0': {'xx': xx, 'yy': yy, 'time': time_steps} }
np.savez('./grid.npz', **npz_grid)
