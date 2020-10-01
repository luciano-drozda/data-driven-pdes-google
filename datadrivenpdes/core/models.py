# python3
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Models evaluate spatial state derivatives.

Models encapsulate the machinery that provides all spatial state derivatives
to the governing equation. They can employ different techniques to produce
state derivatives, such as finite difference methods or neural networks.
"""
import collections
from typing import (
    Any, Dict, List, Optional, Mapping, Set, TypeVar, Tuple, Union,
)

import numpy as np
from datadrivenpdes.core import equations
from datadrivenpdes.core import geometry
from datadrivenpdes.core import grids
from datadrivenpdes.core import polynomials
from datadrivenpdes.core import readers
from datadrivenpdes.core import states
from datadrivenpdes.core import tensor_ops
import tensorflow as tf


nest = tf.contrib.framework.nest


T = TypeVar('T')


def sorted_values(x: Dict[Any, T]) -> List[T]:
  """Returns the sorted values of a dictionary."""
  return [x[k] for k in sorted(x)]


def stack_dict(state: Dict[Any, tf.Tensor]) -> tf.Tensor:
  """Stack a dict of tensors along its last axis."""
  return tf.stack(sorted_values(state), axis=-1)


class TimeStepModel(tf.keras.Model):
  """Model that predicts the state at the next time-step."""

  def __init__(
      self,
      equation: equations.Equation,
      grid: grids.Grid,
      num_time_steps: int = 1,
      target: List[str] = None,
      name: str = 'time_step_model',
  ):
    """Initialize a time-step model."""
    super().__init__(name=name)

    if num_time_steps < 1:
      raise ValueError('must use at least one time step')

    self.equation = equation
    self.grid = grid
    self.num_time_steps = num_time_steps

    if target is None:
      target = sorted(list(equation.evolving_keys))
    self.target = target

  def load_data(
      self,
      metadata: Mapping[str, Any],
      prefix: states.Prefix = states.Prefix.EXACT,
  ) -> tf.data.Dataset:
    """Load data into a tf.data.Dataset for inferrence or training."""

    def replace_state_keys_with_names(state):
      return {k: state[equation.key_definitions[k].with_prefix(prefix)]
              for k in equation.base_keys}

    equation = readers.get_equation(metadata)
    grid = readers.get_output_grid(metadata)
    keys = [equation.key_definitions[k].with_prefix(prefix)
            for k in equation.base_keys]
    dataset = readers.initialize_dataset(metadata, [keys], [grid])
    dataset = dataset.map(replace_state_keys_with_names)
    return dataset

  def call(self, inputs: Dict[str, tf.Tensor]) -> List[tf.Tensor]:
    """Predict the target state after multiple time-steps.

    Args:
      inputs: dict of tensors with dimensions [batch, x, y].

    Returns:
      labels: tensor with dimensions [batch, time, x, y], giving the target
        value of the predicted state at steps [1, ..., self.num_time_steps]
        for model training.
    """
    constant_state = {k: v for k, v in inputs.items()
                      if k in self.equation.constant_keys}
    evolving_inputs = {k: v for k, v in inputs.items()
                       if k in self.equation.evolving_keys}

    def advance(evolving_state, _):
      return self.take_time_step({**evolving_state, **constant_state})

    factor = 1
    if self.equation.CONTINUOUS_EQUATION_NAME == 'euler' and self.equation.DISCRETIZATION_NAME == 'finite_volume':
      factor = 3

    advanced = tf.scan(
        advance, tf.range(factor * self.num_time_steps), initializer=evolving_inputs)
    advanced = tensor_ops.moveaxis(advanced, source=0, destination=1)
    
    if self.equation.CONTINUOUS_EQUATION_NAME == 'euler' and self.equation.DISCRETIZATION_NAME == 'finite_volume':
      return [advanced[target_str][:, 2::3] for target_str in self.target]
    else:
      return [advanced[target_str] for target_str in self.target]

  def time_derivative(
      self, state: Mapping[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Compute the time derivative.

    Args:
      state: current state of the solution.

    Returns:
      Updated values for each non-constant term in the state.
    """
    raise NotImplementedError

  def take_time_step(
      self, state: Mapping[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Take a single time-step.

    Args:
      state: current state of the solution.

    Returns:
      Updated values for each non-constant term in the state.
    """
    raise NotImplementedError

  def to_config(self) -> Mapping[str, Any]:
    """Create a configuration dict for this model. Not possible for all models.
    """
    raise NotImplementedError


class SpatialDerivativeModel(TimeStepModel):
  """Model that predicts the next time-step implicitly via spatial derivatives.
  """

  def __init__(self, equation, grid, num_time_steps=1, target=None,
               name='spatial_derivative_model'):
    super().__init__(equation, grid, num_time_steps, target, name)

  def spatial_derivatives(
      self, state: Mapping[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Predict all needed spatial derivatives."""
    raise NotImplementedError

  def time_derivative(
      self, state: Mapping[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """See base class."""
    inputs = self.spatial_derivatives(state)
    outputs = self.equation.time_derivative(self.grid, **inputs)
    return outputs

  def take_time_step(
      self, state: Mapping[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """See base class."""
    inputs = self.spatial_derivatives(state)
    
    if self.equation.CONTINUOUS_EQUATION_NAME == 'euler' and self.equation.DISCRETIZATION_NAME == 'finite_volume':
      rk_step = self.equation.get_runge_kutta_step()
      if rk_step == 0:
        self.equation.set_last_integer_state(state)
      
    outputs = self.equation.take_time_step(self.grid, **inputs)
    
    if self.equation.CONTINUOUS_EQUATION_NAME == 'euler' and self.equation.DISCRETIZATION_NAME == 'finite_volume':
      self.equation.set_runge_kutta_step((rk_step + 1) % 3)
    
    return outputs


class FiniteDifferenceModel(SpatialDerivativeModel):
  """Baseline model with fixed order finite-differences or finite-volumes.

  This model doesn't need to be trained.
  """

  def __init__(
      self, equation, grid, accuracy_order=1, num_time_steps=1, target=None,
      name='finite_difference_model',
  ):
    super().__init__(equation, grid, num_time_steps, target, name)

    self.accuracy_order = accuracy_order
    self.parents = {}  # type: Dict[str, str]
    self.coefficients = {}  # type: Dict[str, Optional[np.ndarray]]
    self.stencils = {}  # type: Dict[str, List[np.ndarray]]

    for key in self.equation.all_keys:
      parent = equation.find_base_key(key)

      key_def = equation.key_definitions[key]
      parent_def = equation.key_definitions[parent]

      stencils = []
      for parent_offset, argument_offset, derivative_order in zip(
          parent_def.offset, key_def.offset, key_def.derivative_orders):
        stencil = polynomials.regular_stencil_1d(
            abs(parent_offset - argument_offset),
            derivative_order,
            accuracy_order,
            grid.step)
        stencils.append(stencil)

      if all(stencil.size == 1 for stencil in stencils):
        # sentinel value indicating that we should just reuse the parent tensor
        # rather than extracting patches and applying coefficients
        coefficients = None
      else:
        coefficients_2d = polynomials.coefficients(
            stencils, equation.METHOD, key_def.derivative_orders,
            accuracy_order, grid.step)
        coefficients = tf.convert_to_tensor(coefficients_2d.ravel(), tf.float32)

      self.parents[key] = parent
      self.stencils[key] = stencils
      self.coefficients[key] = coefficients

  def spatial_derivatives(
      self, inputs: Mapping[str, tf.Tensor], request: Set[str] = None,
  ) -> Dict[str, tf.Tensor]:
    """See base class."""
    if request is None:
      request = self.equation.all_keys

    result = {}
    for key in request:
      coefficients = self.coefficients[key]

      source = inputs[self.parents[key]]
      if coefficients is None:
        result[key] = source
      else:
        sizes = [stencil.size for stencil in self.stencils[key]]

        key_def = self.equation.key_definitions[key]
        parent_def = self.equation.key_definitions[self.parents[key]]
        shifts = [k - p for p, k in zip(parent_def.offset, key_def.offset)]

        patches = tensor_ops.extract_patches_2d(source, sizes, shifts)
        result[key] = tf.tensordot(coefficients, patches, axes=[-1, -1])
        assert result[key].shape[-2:] == source.shape[-2:], (
            result[key], source)

    return result

  def to_config(self) -> Mapping[str, Any]:
    return dict(accuracy_order=self.accuracy_order)


def _round_down_to_odd(x):
  return x if x % 2 else x - 1


def _round_down_to_even(x):
  return x - 1 if x % 2 else x


def build_stencils(
    key: states.StateDefinition,
    parent: states.StateDefinition,
    max_stencil_size: int,
    grid_step: float
) -> List[np.ndarray]:
  """Create stencils for use with learned coefficients."""
  stencils = []

  for parent_offset, key_offset in zip(parent.offset, key.offset):

    if parent_offset == key_offset:
      size = _round_down_to_odd(max_stencil_size)
    else:
      size = _round_down_to_even(max_stencil_size)

    # examples:
    # stencil_size=5 -> [-2, -1, 0, 1, 2]
    # stencil_size=4 -> [-2, -1, 0, 1]
    int_range = np.arange(size) - size // 2

    stencil = grid_step * (0.5 * abs(key_offset - parent_offset) + int_range)

    stencils.append(stencil)

  # we should only be using zero-centered stencils
  if not all(np.allclose(stencil.sum(), 0) for stencil in stencils):
    raise ValueError('stencils are not zero-centered for {} -> {}: {}'
                     .format(parent, key, stencils))

  return stencils


ConstraintLayer = Union[
    polynomials.PolynomialAccuracy, polynomials.PolynomialBias]


class FixedCoefficientsLayer(tf.keras.layers.Layer):
  """Layer representing fixed learned coefficients for a single derivative."""

  def __init__(
      self,
      constraint_layer: ConstraintLayer,
      stencils: List[np.ndarray],
      shifts: List[int],
      input_key: Optional[str] = None,
  ):
    self.constraint_layer = constraint_layer
    self.stencils = stencils
    self.shifts = shifts
    self.input_key = input_key
    super().__init__()

  def build(self, input_shape):
    shape = [self.constraint_layer.input_size]
    self.kernel = self.add_weight('kernel', shape=shape)

  def compute_output_shape(self, input_shape):
    return input_shape[:-1]

  def call(self, inputs):
    coefficients = self.constraint_layer(self.kernel)
    sizes = [stencil.size for stencil in self.stencils]
    patches = tensor_ops.extract_patches_2d(inputs, sizes, self.shifts)
    return tf.einsum('s,bxys->bxy', coefficients, patches)


class VaryingCoefficientsLayer(tf.keras.layers.Layer):
  """Layer representing varying coefficients for a single derivative."""

  def __init__(
      self,
      constraint_layer: ConstraintLayer,
      stencils: List[np.ndarray],
      shifts: List[int],
      input_key: Optional[str] = None,
  ):
    self.constraint_layer = constraint_layer
    self.stencils = stencils
    self.shifts = shifts
    self.input_key = input_key
    self.kernel_size = constraint_layer.input_size
    super().__init__(trainable=False)

  def compute_output_shape(self, input_shape):
    return input_shape[:-1]

  def call(self, inputs):
    (kernel, source) = inputs
    coefficients = self.constraint_layer(kernel)
    sizes = [stencil.size for stencil in self.stencils]
    patches = tensor_ops.extract_patches_2d(source, sizes, self.shifts)
    return tf.einsum('bxys,bxys->bxy', coefficients, patches)


def normalize_learned_and_fixed_keys(
    learned_keys: Optional[Set[str]],
    fixed_keys: Optional[Set[str]],
    equation: equations.Equation,
) -> Tuple[Set[str], Set[str]]:
  """Normalize learned and fixed equation inputs."""
  if learned_keys is None and fixed_keys is None:
    fixed_keys = equation.base_keys
    learned_keys = equation.derived_keys

  elif fixed_keys is None:
    learned_keys = set(learned_keys)
    fixed_keys = equation.all_keys - learned_keys

  elif learned_keys is None:
    fixed_keys = set(fixed_keys)
    learned_keys = equation.all_keys - fixed_keys

  else:
    learned_keys = set(learned_keys)
    fixed_keys = set(fixed_keys)

    if learned_keys.intersection(fixed_keys):
      raise ValueError('learned and fixed inputs must be disjoint sets: '
                       '{} vs {}'.format(learned_keys, fixed_keys))

    missing_inputs = equation.all_keys - learned_keys - fixed_keys
    if missing_inputs:
      raise ValueError(
          'inputs {} not inclued in learned or fixed inputs: {} vs {}'
          .format(missing_inputs, learned_keys, fixed_keys))

  return learned_keys, fixed_keys


def build_output_layers(
    equation, grid, learned_keys,
    stencil_size=5,
    initial_accuracy_order=1,
    constrained_accuracy_order=1,
    layer_cls=FixedCoefficientsLayer,
    predict_permutations=True,
) -> Dict[str, ConstraintLayer]:
  """Build a map of output layers for spatial derivative models."""
  layers = {}
  modeled = set()

  # learned_keys is a set; it can change iteration order per python session
  # need to fix its iteration order to correctly save/load Keras models
  for key in sorted(learned_keys):
    if (not predict_permutations
        and equation.key_definitions[key].swap_xy() in modeled):
      # NOTE(shoyer): this only makes sense if geometric_transforms includes
      # permutations. Otherwise you won't be predicting every needed tensor.
      continue

    parent = equation.find_base_key(key)
    key_def = equation.key_definitions[key]
    modeled.add(key_def)
    parent_def = equation.key_definitions[parent]

    stencils = build_stencils(key_def, parent_def, stencil_size, grid.step)
    shifts = [k - p for p, k in zip(parent_def.offset, key_def.offset)]
    constraint_layer = polynomials.constraint_layer(
        stencils, equation.METHOD, key_def.derivative_orders[:2],
        constrained_accuracy_order, initial_accuracy_order, grid.step,
    )
    layers[key] = layer_cls(
        constraint_layer, stencils, shifts, input_key=parent)

  return layers


def average_over_transforms(func, geometric_transforms, state):
  """Average a function over transformations to achive rotation invariance."""
  result_list = collections.defaultdict(list)

  for transform in geometric_transforms:
    output = transform.inverse(func(transform.forward(state)))
    for k, v in output.items():
      result_list[k].append(v)

  result = {k: tf.add_n(v) / len(v) if len(v) > 1 else v[0]
            for k, v in result_list.items()}
  return result


class LinearModel(SpatialDerivativeModel):
  """Learn constant linear filters for spatial derivatives."""

  def __init__(self, equation, grid, stencil_size=5, initial_accuracy_order=1,
               constrained_accuracy_order=1, learned_keys=None,
               fixed_keys=None, num_time_steps=1, target=None,
               geometric_transforms=None, predict_permutations=True,
               name='linear_model'):
    super().__init__(equation, grid, num_time_steps, target, name)
    self.learned_keys, self.fixed_keys = (
        normalize_learned_and_fixed_keys(learned_keys, fixed_keys, equation))
    self.output_layers = build_output_layers(
        equation, grid, self.learned_keys, stencil_size, initial_accuracy_order,
        constrained_accuracy_order, layer_cls=FixedCoefficientsLayer,
        predict_permutations=predict_permutations)
    self.fd_model = FiniteDifferenceModel(
        equation, grid, initial_accuracy_order)
    self.geometric_transforms = geometric_transforms or [geometry.Identity()]

  def _apply_model(self, state):
    result = {}
    for key, layer in self.output_layers.items():
      input_tensor = state[layer.input_key]
      layer = self.output_layers[key]
      result[key] = layer(input_tensor)
    return result

  def spatial_derivatives(self, inputs):
    """See base class."""
    result = average_over_transforms(
        self._apply_model, self.geometric_transforms, inputs
    )

    if self.fixed_keys:
      result.update(
          self.fd_model.spatial_derivatives(inputs, self.fixed_keys)
      )
    return result


class Conv2DPeriodic(tf.keras.layers.Layer):
  """Conv2D layer with periodic boundary conditions."""

  def __init__(self, filters, kernel_size, **kwargs):
    # Let Conv2D handle argument normalization, e.g., kernel_size -> tuple
    self._layer = tf.keras.layers.Conv2D(
        filters, kernel_size, padding='valid', **kwargs)
    self.filters = self._layer.filters
    self.kernel_size = self._layer.kernel_size

    if any(size % 2 == 0 for size in self.kernel_size):
      raise ValueError('kernel size for conv2d is not odd: {}'
                       .format(self.kernel_size))

    super().__init__()

  def build(self, input_shape):
    self._layer.build(input_shape)
    super().build(input_shape)

  def compute_output_shape(self, input_shape):
    return input_shape[:-1] + (self.filters,)

  def call(self, inputs):
    padded = tensor_ops.pad_periodic_2d(inputs, self.kernel_size)
    result = self._layer(padded)
    assert result.shape[1:3] == inputs.shape[1:3], (result, inputs)
    return result


def conv2d_stack(num_outputs, num_layers=5, filters=32, kernel_size=5,
                 activation='relu', **kwargs):
  """Create a sequence of Conv2DPeriodic layers."""
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Lambda(stack_dict))
  for _ in range(num_layers - 1):
    layer = Conv2DPeriodic(
        filters, kernel_size, activation=activation, **kwargs)
    model.add(layer)
  model.add(Conv2DPeriodic(num_outputs, kernel_size, **kwargs))
  return model


def conv2d_stack_non_periodic(num_outputs, num_layers=5, filters=32, 
                              kernel_size=5, activation='relu', **kwargs):
  """Create a sequence of Conv2D layers."""
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Lambda(stack_dict))
  for _ in range(num_layers - 1):
    layer = tf.keras.layers.Conv2D(
        filters, kernel_size, activation=activation, padding='same', **kwargs)
    model.add(layer)
  model.add(
    tf.keras.layers.Conv2D(num_outputs, kernel_size, padding='same', **kwargs)
  )
  return model


def _rescale_01(array, axis):
  array_max = tf.reduce_max(array, axis, keep_dims=True)
  array_min = tf.reduce_min(array, axis, keep_dims=True)
  return (array - array_min) / (array_max - array_min)


class RescaledConv2DStack(tf.keras.Model):
  """Rescale input fields to stabilize PDE integration."""

  def __init__(self, num_outputs: int, **kwargs):
    super().__init__()
    self.original_model = conv2d_stack(num_outputs, **kwargs)

  def call(self, inputs):
    inputs = inputs.copy()
    for key in inputs.keys():
      inputs[key] = _rescale_01(inputs[key], axis=(-1, -2))

    return self.original_model(inputs)


class RescaledConv2DStackNonPeriodic(tf.keras.Model):
  """Rescale input fields to stabilize PDE integration (using conv2d_stack_non_periodic)."""

  def __init__(self, num_outputs: int, **kwargs):
    super().__init__()
    self.original_model = conv2d_stack_non_periodic(num_outputs, **kwargs)

  def call(self, inputs):
    inputs = inputs.copy()
    for key in inputs.keys():
      inputs[key] = _rescale_01(inputs[key], axis=(-1, -2))

    return self.original_model(inputs)


class ClippedConv2DStack(tf.keras.Model):
  """Clip input fields to stabilize PDE integration."""

  def __init__(self, num_outputs: int, scaled_keys: Set[str], **kwargs):
    super().__init__()
    self.original_model = conv2d_stack(num_outputs, **kwargs)
    self.scaled_keys = scaled_keys

  def call(self, inputs):
    inputs = inputs.copy()
    for key in self.scaled_keys:
      inputs[key] = tf.clip_by_value(inputs[key], 1e-3, 1.0 - 1e-3)

    return self.original_model(inputs)


def block(filters, kernel_size, activation, final_activation=True):
  def f(x):
    # first Conv2DPeriodic (with post-activation)
    h = Conv2DPeriodic(filters, kernel_size, activation=activation)(x)
    # second Conv2DPeriodic
    h = Conv2DPeriodic(filters, kernel_size)(h)
    # skip connection
    h = tf.keras.layers.add([h, x])
    # last activation
    h = tf.keras.layers.Activation(activation)(h)
    return h
  return f

def conv2d_resnet(num_outputs, num_layers=5, filters=32, kernel_size=5,
                  activation='relu', **kwargs):
  # use Keras functional API
  # input data
  # input_tensor = tf.keras.layers.Input((32, 32, 4)) # for Euler
  input_tensor = tf.keras.layers.Input((32, 32, 3)) # for Advection
  # first Conv2DPeriodic to transform data for the loop of residual blocks
  x = Conv2DPeriodic(filters, kernel_size, activation=activation, **kwargs)(input_tensor)
  # compute number of residual blocks
  num_blocks = (num_layers - 2) // 2
  for _ in range(num_blocks):
    x = block(filters, kernel_size, activation)(x)
  # last Conv2DPeriodic to transform data in the right shape 
  # for the output layers
  x = Conv2DPeriodic(num_outputs, kernel_size, **kwargs)(x)

  return tf.keras.Model(inputs=input_tensor, outputs=x)


class Conv2DResNet(tf.keras.Model):
  """Create a sequence of Conv2DPeriodic layers in the form of a ResNet, 
    suitable for very deep neural nets."""

  def __init__(self, num_outputs: int, **kwargs):
    super().__init__()
    self.original_model = conv2d_resnet(num_outputs, **kwargs)

  def call(self, inputs):
    inputs = inputs.copy()
    for key in inputs.keys():
      inputs[key] = _rescale_01(inputs[key], axis=(-1, -2))
    return self.original_model(stack_dict(inputs))


class PseudoLinearModel(SpatialDerivativeModel):
  """Learn pseudo-linear filters for spatial derivatives."""

  def __init__(self, equation, grid, stencil_size=5, initial_accuracy_order=1,
               constrained_accuracy_order=1, learned_keys=None,
               fixed_keys=None, core_model_func=conv2d_stack,
               num_time_steps=1, geometric_transforms=None,
               predict_permutations=True, target=None,
               name='pseudo_linear_model', **kwargs):
    # NOTE(jiaweizhuang): Too many input arguments. Only document important or
    # confusing ones for now.
    # pylint: disable=g-doc-args
    """Initialize class.

    Args:
      core_model_func: callable (function or class object). It should return
        a Keras model (or layer) instance, which contains trainable weights.
        The returned core_model instance should take a dict of tensors as input
        (see the call() method in the base TimeStepModel class).
        Additional kwargs are passed to this callable to specify hyperparameters
        of core_model (such as number of layers and convolutional filters).
    """
    # pylint: enable=g-doc-args

    super().__init__(equation, grid, num_time_steps, target, name)

    self.learned_keys, self.fixed_keys = (
        normalize_learned_and_fixed_keys(learned_keys, fixed_keys, equation))
    self.output_layers = build_output_layers(
        equation, grid, self.learned_keys, stencil_size, initial_accuracy_order,
        constrained_accuracy_order, layer_cls=VaryingCoefficientsLayer,
        predict_permutations=predict_permutations)
    self.fd_model = FiniteDifferenceModel(
        equation, grid, initial_accuracy_order)
    self.geometric_transforms = geometric_transforms or [geometry.Identity()]

    num_outputs = sum(
        layer.kernel_size for layer in self.output_layers.values()
    )
    self.core_model = core_model_func(num_outputs, **kwargs)

  def _apply_model(self, state):
    net = self.core_model(state)

    size_splits = [
        self.output_layers[key].kernel_size for key in self.output_layers
    ]
    heads = tf.split(net, size_splits, axis=-1)

    result = {}
    for (key, layer), head in zip(self.output_layers.items(), heads):
      input_tensor = state[layer.input_key]
      result[key] = layer([head, input_tensor])
    return result

  def spatial_derivatives(self, inputs):
    """See base class."""
    result = average_over_transforms(
        self._apply_model, self.geometric_transforms, inputs
    )
    if self.fixed_keys:
      result.update(
          self.fd_model.spatial_derivatives(inputs, self.fixed_keys)
      )
    return result


class NonlinearModel(SpatialDerivativeModel):
  """Learn spatial derivatives directly."""

  def __init__(self, equation, grid, core_model_func=conv2d_stack,
               learned_keys=None, fixed_keys=None, num_time_steps=1,
               finite_diff_accuracy_order=1, target=None,
               name='nonlinear_model', **kwargs):
    super().__init__(equation, grid, num_time_steps, target, name)
    self.learned_keys, self.fixed_keys = (
        normalize_learned_and_fixed_keys(learned_keys, fixed_keys, equation))
    self.core_model = core_model_func(
        num_outputs=len(self.learned_keys), **kwargs)
    self.fd_model = FiniteDifferenceModel(
        equation, grid, finite_diff_accuracy_order)

  def spatial_derivatives(
      self, inputs: Mapping[str, tf.Tensor],
  ) -> Dict[str, tf.Tensor]:
    """See base class."""
    net = self.core_model(inputs)
    heads = tf.unstack(net, axis=-1)
    result = dict(zip(self.learned_keys, heads))

    if self.fixed_keys:
      result.update(
          self.fd_model.spatial_derivatives(inputs, self.fixed_keys)
      )
    return result


class DirectModel(TimeStepModel):
  """Learn time-evolution directly, ignoring the equation."""

  def __init__(self, equation, grid, core_model_func=conv2d_stack,
               num_time_steps=1, finite_diff_accuracy_order=1, target=None,
               name='direct_model', **kwargs):
    super().__init__(equation, grid, num_time_steps, target, name)
    self.keys = equation.evolving_keys
    self.core_model = core_model_func(num_outputs=len(self.keys), **kwargs)

  def take_time_step(self, inputs):
    """See base class."""
    net = self.core_model(inputs)
    heads = tf.unstack(net, axis=-1)
    return dict(zip(self.keys, heads))
