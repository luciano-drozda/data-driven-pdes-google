# Train neural net model and evaluate performances.
# Requires previous creation of training data

# Author(s): Luciano Drozda
# Date (mm/dd/yyyy): 10/01/2020

import numpy as np
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

# Get info of the 'stencil_size', 'num_layers', 'filters' from the directory name
folder_name = os.path.split(os.getcwd())[-1] # 'train-{stencil_size}-{num_layers}_{filters}'
str_lst = folder_name.split('-')
stencil_size = int(str_lst[1])
num_layers = int(str_lst[2])
filters = int(str_lst[3])

print(
  f'\n\n stencil_size = {stencil_size}, num_layers = {num_layers}, filters = {filters} \n\n'
  )

# Create equation instance
equation_nn = equations.FiniteVolumeEuler(no_dimensions=True)
key_defs = equation_nn.key_definitions

# `NUM_TIME_STEPS` should be <= 100
NUM_TIME_STEPS_PREVIOUS = 0
for (NUM_TIME_STEPS, num_epochs) in zip([10, 20, 90], [2, 1, 1]):
  print(f'\n\n NUM_TIME_STEPS = {NUM_TIME_STEPS} \n\n')
  # Create neural net model instance
  model_nn = models.PseudoLinearModel(
    equation_nn, 
    coarse_grid,
    num_time_steps=NUM_TIME_STEPS, # multi-step loss
    stencil_size=stencil_size, kernel_size=(3,3), num_layers=num_layers, filters=filters, 
    constrained_accuracy_order=1,
    activation='relu',
    core_model_func=models.RescaledConv2DStack
  )
  print(f'\n model_nn.learned_keys = {model_nn.learned_keys}')
  print(f'\n model_nn.fixed_keys = {model_nn.fixed_keys}')
  
  print(f'\n\n num_epochs = {num_epochs} \n\n')
  if NUM_TIME_STEPS == 10:
    # Integrate untrained model_nn on a test case
    initial_state = equation_nn.random_state_double_shear_layer(coarse_grid)
    initial_state_nn = \
      {k: tf.expand_dims(v, 0) for k, v in initial_state.items()}
    tf.random.set_random_seed(0)
    _ = model_nn.take_time_step(initial_state_nn) # weights are initialized at the first model call
    model_nn.equation.set_runge_kutta_step(0) # reset for training loop !

  else:
    # Load pretrained weights on model_nn
    model_utils.load_weights(
      model_nn, f'./weights_trained_{NUM_TIME_STEPS_PREVIOUS}.h5'
      )

  # model_nn weights before training
  weights_nn = model_nn.get_weights()
  print(
  *[f'weights[{idx}].shape = {w.shape}' 
    for idx, w in enumerate(weights_nn)],
  sep='\n'
  )

  # Setup eager training
  learning_rate = 1e-4
  batch_size = 8
  if NUM_TIME_STEPS < 40:
    batch_size = 32
  print(f'\n learning_rate = {learning_rate:.2e} \n')
  print(f'\n batch_size = {batch_size} \n')
  optimizer = tf.keras.optimizers.Adam(learning_rate)
  loss_object = tf.keras.losses.MeanAbsoluteError()
  loss_history = []
  def train_step(input_, output_):
    with tf.GradientTape() as tape:
      preds = model_nn(input_, training=True)
      
      # Add asserts to check the shape of the output.
      tf.debugging.assert_equal(
        preds[0].shape,
        (batch_size, NUM_TIME_STEPS, coarse_resolution, coarse_resolution)
      )
      
      loss_value = sum(
        [loss_object(output_[i], preds[i]) for i in range(len(output_))]
      )
    loss_history.append(loss_value.numpy().mean())
    print(f'loss_value == {loss_history[-1]}')
    grads = tape.gradient(loss_value, model_nn.trainable_variables)
    optimizer.apply_gradients(zip(grads, model_nn.trainable_variables))

  # Run eager training
  n_samples = 9010 # if resample_factor == 8
  num_batches = n_samples // batch_size
  num_batches = num_batches // 2 # speed up training
  random_indexes_generator = np.random.RandomState(1000000 + NUM_TIME_STEPS)
  for epoch in range(num_epochs):
    for batch in range(num_batches):
      i_sample_list = random_indexes_generator.randint(
        n_samples, size=batch_size
      )
      input_ = {}
      output_ = {}
      for i_sample in i_sample_list:
        # Recover input_ and output_ from npz files
        path = '../data/samples'
        npz_input = \
          np.load(f'{path}/input_{coarse_resolution}_sample_{i_sample}.npz',
          allow_pickle=True)
        npz_output = \
          np.load(f'{path}/output_{coarse_resolution}_sample_{i_sample}.npz',
          allow_pickle=True)
        input_sample = npz_input['0'].item()
        output_sample = npz_output['0'].item()
        for k in input_sample.keys():
          input_.setdefault(k, []) 
          output_.setdefault(k, []) 
          input_[k].append(input_sample[k])
          output_[k].append(output_sample[k])
      
      input_ = {k: tf.stack(v) for k, v in input_.items()}
      output_ = {k: tf.stack(v) for k, v in output_.items()}
      output_ = {k: v[:, :NUM_TIME_STEPS, :, :] for k, v in output_.items()}
      output_ = sorted_values(output_)

      train_step(input_, output_)
      print(f'\n Batch {batch} finished \n')
    print(f'\n Epoch {epoch} finished \n')

  # Save loss_history in a .txt file
  with open(f'loss_history_{NUM_TIME_STEPS}.txt', 'w') as f:
    for item in loss_history:
      f.write("%s\n" % item)

  # Save trained model_nn weights
  model_utils.save_weights(model_nn, f'weights_trained_{NUM_TIME_STEPS}.h5')

  NUM_TIME_STEPS_PREVIOUS = NUM_TIME_STEPS
