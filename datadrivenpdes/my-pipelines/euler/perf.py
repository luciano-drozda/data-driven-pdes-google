# Compare performances of each of the trained neural nets.

# Author(s): Luciano Drozda
# Date (mm/dd/yyyy): 09/22/2020

import numpy as np
import time
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from datadrivenpdes.core import grids
from datadrivenpdes.core import integrate
from datadrivenpdes.core import models
from datadrivenpdes.core import tensor_ops
from datadrivenpdes.pipelines import model_utils
from datadrivenpdes.euler import equations

import tensorflow as tf
tf.enable_eager_execution()
print('tf.__version__ =',tf.__version__)
print('tf.keras.__version__ =',tf.keras.__version__)
# GPU setup
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# Something that may help 
# on the multiple outputs training at model_nn.fit() call
def sorted_values(x):
  """Returns the sorted values of a dictionary as a list."""
  return [x[k] for k in sorted(x)]

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# Euler
length = 1.
fine_resolution = 256
resample_factor = 8
coarse_resolution = fine_resolution // resample_factor
coarse_grid = grids.Grid.from_period(size=coarse_resolution, length=length)
xx, yy = coarse_grid.get_mesh()

# Create reference 256 x 256 model
equation = equations.Weno5Euler(no_dimensions=True)
key_defs = equation.key_definitions
fine_grid = grids.Grid.from_period(size=fine_resolution, length=length)
model = models.FiniteDifferenceModel(equation, fine_grid)

# Create baseline 128 x 128 and 64 x 64 models
equation_128 = equations.Weno5Euler(no_dimensions=True)
key_defs_128 = equation_128.key_definitions
resample_factor_128 = 2
coarse_resolution_128 = fine_resolution // resample_factor_128
coarse_grid_128 = grids.Grid.from_period(size=coarse_resolution_128, length=length)
model_128 = models.FiniteDifferenceModel(equation_128, coarse_grid_128)

equation_64 = equations.Weno5Euler(no_dimensions=True)
key_defs_64 = equation_64.key_definitions
resample_factor_64 = 4
coarse_resolution_64 = fine_resolution // resample_factor_64
coarse_grid_64 = grids.Grid.from_period(size=coarse_resolution_64, length=length)
model_64 = models.FiniteDifferenceModel(equation_64, coarse_grid_64)

# Take a sample
initial_state = equation.random_state_double_shear_layer(fine_grid, seed=0)
initial_state_128 = tensor_ops.regrid(
  initial_state, key_defs_128, fine_grid, coarse_grid_128
)
initial_state_64 = tensor_ops.regrid(
  initial_state, key_defs_64, fine_grid, coarse_grid_64
)

# Integrate sample over time to create the reference solution
coarse_time_steps = 100
fine_time_steps = np.arange(coarse_time_steps * resample_factor)
print(f'\n\n Reference solver started ! \n\n')
output_ = integrate.integrate_steps(
  model, initial_state, fine_time_steps, axis=0
)
output_ = tensor_ops.regrid(
  output_, key_defs, fine_grid, coarse_grid
)
output_ = sorted_values(
  {k: v[::resample_factor] for k, v in output_.items()}
)
print(f'output_[0].shape == {output_[0].shape}')
print(f'\n\n Reference solver finished ! \n\n')

# Evaluate the performances of the baseline solvers on this sample
coarse_time_steps_128 = np.arange(
  coarse_time_steps * (resample_factor // resample_factor_128)
)
coarse_time_steps_64 = np.arange(
  coarse_time_steps * (resample_factor // resample_factor_64)
)
loss_object = tf.keras.losses.MeanAbsoluteError()
print(f'\n\n Baseline solvers started ! \n\n')
start = time.time()
preds_128 = integrate.integrate_steps(
  model_128, initial_state_128, coarse_time_steps_128, axis=0
)
preds_128 = tensor_ops.regrid(
  preds_128, key_defs_128, coarse_grid_128, coarse_grid
)
preds_128 = sorted_values(
  {k: v[::resample_factor // resample_factor_128] for k, v in preds_128.items()}
)
end = time.time()
runtime_128 = end - start # in seconds
print(f'preds_128[0].shape == {preds_128[0].shape}')

start = time.time()
preds_64 = integrate.integrate_steps(
  model_64, initial_state_64, coarse_time_steps_64, axis=0
)
preds_64 = tensor_ops.regrid(
  preds_64, key_defs_64, coarse_grid_64, coarse_grid
)
preds_64 = sorted_values(
  {k: v[::resample_factor // resample_factor_64] for k, v in preds_64.items()}
)
end = time.time()
runtime_64 = end - start # in seconds
print(f'preds_64[0].shape == {preds_64[0].shape}')

loss_value_128 = sum(
  [loss_object(output_[i], preds_128[i]) for i in range(len(output_))]
)
loss_value_64 = sum(
  [loss_object(output_[i], preds_64[i]) for i in range(len(output_))]
)
with open(f'stats_128.txt', 'w') as f:
  f.write(f"{runtime_128}, {loss_value_128}")
with open(f'stats_64.txt', 'w') as f:
  f.write(f"{runtime_64}, {loss_value_64}")

print(f'\n\n Baseline solvers finished ! \n\n')

# Evaluate the performances of the trained neural nets on this sample
print(f'\n\n Neural nets started ! \n\n')
for stencil_size in [3, 5]:
  for num_layers in [4, 6, 8]:
    for filters in [64, 128]:
      print(
        f'\n\n stencil_size = {stencil_size}, num_layers = {num_layers}, filters = {filters} \n\n'
      )
      # Create `equation_nn` and `model_nn` instances
      equation_nn = equations.FiniteVolumeEuler(no_dimensions=True)
      key_defs_nn = equation_nn.key_definitions
      model_nn = models.PseudoLinearModel(
        equation_nn, 
        coarse_grid,
        stencil_size=stencil_size, kernel_size=(3,3), num_layers=num_layers, filters=filters, 
        constrained_accuracy_order=1,
        activation='relu',
        core_model_func=models.RescaledConv2DStack
      )
      # Load pretrained weights on `model_nn`
      model_utils.load_weights(
        model_nn, f'runge_kutta_train-{stencil_size}-{num_layers}-{filters}/weights_trained_90.h5'
      )
      # Advance in time
      initial_state_nn = tensor_ops.regrid(
        initial_state, key_defs_nn, fine_grid, coarse_grid
      )
      initial_state_nn = \
        {k: tf.expand_dims(v, 0) for k, v in initial_state_nn.items()}
      start = time.time()
      preds_nn = integrate.integrate_steps(
        model_nn, initial_state_nn, np.arange(3 * coarse_time_steps), axis=0
      )
      preds_nn = sorted_values( {k: tf.squeeze(v)[2::3] for k, v in preds_nn.items()} )
      end = time.time()
      runtime_nn = end - start
      print(f'preds_nn[0].shape == {preds_nn[0].shape}')
      # Compute MAE values
      loss_value_nn = sum(
        [loss_object(output_[i], preds_nn[i]) for i in range(len(output_))]
      ).numpy()
      # Save `runtime_nn` and `loss_value_nn` in a .txt file
      with open(f'stats_{stencil_size}-{num_layers}-{filters}.txt', 'w') as f:
        f.write(f"{runtime_nn}, {loss_value_nn}")

print(f'\n\n Neural nets finished ! \n\n')
