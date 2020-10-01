"""Reaction-diffusion (aka Turing) equations."""
import enum
import functools
import operator
from typing import Any, Dict, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter
import xarray as xr
from matplotlib import pyplot as plt
from datadrivenpdes.core import equations
from datadrivenpdes.core import grids
from datadrivenpdes.core import polynomials
from datadrivenpdes.core import states
from datadrivenpdes.core import tensor_ops
import tensorflow as tf

StateDef = states.StateDefinition

X = states.Dimension.X
Y = states.Dimension.Y

NO_DERIVATIVES = (0, 0, 0)
D_X = (1, 0, 0)
D_Y = (0, 1, 0)
D_XX = (2, 0, 0)
D_YY = (0, 2, 0)

NO_OFFSET = (0, 0)
X_PLUS_HALF = (1, 0)
Y_PLUS_HALF = (0, 1)

# numpy.random.RandomState uses uint32 for seeds
MAX_SEED_PLUS_ONE = 2**32

# Useful for tensor_ops.roll calls
X_AXIS = -2
Y_AXIS = -1

class _TuringBase(equations.Equation):
  """Base class for 2D Turing equations.
  
  This base class defines the state and common methods.

  Subclasses must implement the time_derivative() method.

  Attributes:
  """

  CONTINUOUS_EQUATION_NAME = 'turing'

  def __init__(self, alpha, beta, d_A, d_B):
    self.alpha = alpha
    self.beta = beta
    self.d_A = d_A
    self.d_B = d_B
    super().__init__()

  def reaction_A(self, A, B):
    return A - (A ** 3) - B * self.alpha

  def reaction_B(self, A, B):
    return (A - B) * self.beta

  def get_time_step(
    self, 
    grid: grids.Grid
    ) -> float:
    raise NotImplementedError

  @property
  def cfl_safety_factor(self) -> float:
    raise NotImplementedError

  def random_state(
    self,
    grid: grids.Grid,
    seed: int = None,
    dtype: Any = np.float32,
  ) -> Dict[str, np.ndarray]:
    """Returns a state with random initial conditions 
    for two-component reaction-diffusion.

    Args:
      grid: Grid object holding discretization parameters.
      seed: random seed to use for random number generator.
      dtype: dtype for generated tensors.

    Returns:
      State with random initial values.
    """
    if seed is not None:
      random = np.random.RandomState(seed=seed)
      noise = random.randn(grid.size_x, grid.size_y)
      A = 0.5 * gaussian_filter(noise, sigma=1.)
      noise = random.randn(grid.size_x, grid.size_y)
      B = 0.5 * gaussian_filter(noise, sigma=1.)
      noise = random.randn(grid.size_x, grid.size_y)
      Source = 0.5 * gaussian_filter(noise, sigma=1.)

    state = {
      'A': A - np.amin(A), # only positive values for concentration fields
      'B': B - np.amin(B),
      'Source': Source - np.amin(Source),
    }
    state = {k: tf.cast(v, dtype) for k, v in state.items()}
    return state

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

def flux_to_time_derivative(x_flux_edge_x, y_flux_edge_y, grid_step):
  """Use continuity to convert from fluxes to a time derivative."""
  # right - left + top - bottom
  numerator = tf.add_n([
      x_flux_edge_x,
      -tensor_ops.roll_2d(x_flux_edge_x, (1, 0)),
      y_flux_edge_y,
      -tensor_ops.roll_2d(y_flux_edge_y, (0, 1)),
  ])
  return -(1 / grid_step) * numerator

def _minimum(*args):
  return functools.reduce(tf.minimum, args)

def _maximum(*args):
  return functools.reduce(tf.maximum, args)

def _sum(array, axis = None):
  return tf.reduce_sum(array, axis = axis)

def _abs(array):
  return tf.abs(array)

def _amax(array, axis = None):
  return tf.reduce_max(array, axis = axis)

def _transpose(array):
  return tf.linalg.matrix_transpose(array)

def _sqrt(array):
  return tf.sqrt(array)

def _ones(shape):
  return tf.ones(shape)

def _roll_minus_one(array, axis):
  return tensor_ops.roll(array, 1, axis)

def _roll_minus_two(array, axis):
  return tensor_ops.roll(array, 2, axis)

def _roll_plus_one(array, axis):
  return tensor_ops.roll(array, -1, axis)

def _roll_plus_two(array, axis):
  return tensor_ops.roll(array, -2, axis)

def _roll_plus_three(array, axis):
  return tensor_ops.roll(array, -3, axis)

# Method for plotting time-series
def save_plot(data, filename, time_steps, xx, yy):
  kw = dict(dims=['time', 'x'], coords={'time': time_steps, 'x': xx[:,0]})
  if yy.shape[1] > 1:
    kw['dims'].append('y')
    kw['coords']['y'] = yy[0]
  else:
    data = data.squeeze()
  data = xr.DataArray(data, **kw)
  plt.close('all')
  kw = dict(col='time', aspect=1)
  if yy.shape[1] > 1:
    kw['robust'] = True
  if len(data) < 4:
    data.plot(**kw)
  else:
    data[: : len(data) // 4].plot(**kw)
  plt.suptitle(filename)
  plt.savefig(filename + '.png')

# Method for plotting state
def save_plot_state(state, filename, xx, yy):
  plt.close('all')
  fig, axs = plt.subplots(
    nrows=1, ncols=len(state.keys()), sharey=True, figsize=(10,5)
  )
  axs[0].set_ylabel('x')
  if (yy.shape[1] > 1):
    for i, key in enumerate(state.keys()):
      plot = axs[i].pcolor(yy, xx, state[key], cmap='RdBu')
      fig.colorbar(plot, ax=axs[i], orientation='horizontal')
      axs[i].set_title(key)
      axs[i].set_xlabel('y')
      axs[i].set_aspect('equal')
  else:
    for i, key in enumerate(state.keys()):
      axs[i].plot(state[key], xx.squeeze())
      axs[i].set_xlabel(key)
  plt.savefig(filename + '.png')

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

class FTCS(_TuringBase):
  """
      Finite-volume Forward-Time Central-Space (FTCS) 
      scheme for 2D Turing equations.
  """

  DISCRETIZATION_NAME = 'ftcs_turing'
  METHOD = polynomials.Method.FINITE_DIFFERENCE
  MONOTONIC = False

  def __init__(
    self,
    cfl_safety_factor: float = 0.2,
    alpha: float = 1.,
    beta: float = 1.,
    d_A: float = 1.,
    d_B: float = 1.,
    ):
    self.key_definitions = {
      'A': states.StateDefinition('A', (), NO_DERIVATIVES, NO_OFFSET),
      'B': states.StateDefinition('B', (), NO_DERIVATIVES, NO_OFFSET),
      'Source' : states.StateDefinition('Source', (), NO_DERIVATIVES, NO_OFFSET)
    }
    self.evolving_keys = {'A', 'B'}
    self.constant_keys = {'Source'}
    self.alpha = alpha
    self.beta = beta
    self.d_A = d_A
    self.d_B = d_B
    self._cfl_safety_factor = cfl_safety_factor
    super().__init__(alpha, beta, d_A, d_B)
  
  @property
  def cfl_safety_factor(self) -> float:
    return self._cfl_safety_factor

  def get_time_step(self, grid):
    return self.cfl_safety_factor * 0.5 * grid.step ** 2 / max(self.d_A, self.d_B)

  def time_derivative(self, grid, A, B, Source):
    """See base class."""
    rA = self.reaction_A(A, B)
    rB = self.reaction_B(A, B)
    derivatives_A = []
    derivatives_B = []
    for axis in [X_AXIS, Y_AXIS]:
      derivatives_A.append(
        (_roll_plus_one(A, axis) + 
        _roll_minus_one(A, axis) - 2 * A) / grid.step ** 2
        )
      derivatives_B.append(
        (_roll_plus_one(B, axis) + 
        _roll_minus_one(B, axis) - 2 * B) / grid.step ** 2
        )
    diff_A = self.d_A * sum(derivatives_A)
    diff_B = self.d_B * sum(derivatives_B)
    return {'A': rA + diff_A + Source, 'B': rB + diff_B,}


# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

class FiniteDifferenceTuring(_TuringBase):

  DISCRETIZATION_NAME = 'finite_difference'
  METHOD = polynomials.Method.FINITE_DIFFERENCE
  MONOTONIC = False

  def __init__(
    self,
    cfl_safety_factor: float = 0.2,
    alpha: float = 1.,
    beta: float = 1.,
    d_A: float = 1.,
    d_B: float = 1.,
    ):
    self.key_definitions = {
      'A': states.StateDefinition('A', (), NO_DERIVATIVES, NO_OFFSET),
      'A_xx': states.StateDefinition('A', (), D_XX, NO_OFFSET),
      'A_yy': states.StateDefinition('A', (), D_YY, NO_OFFSET),
      'B': states.StateDefinition('B', (), NO_DERIVATIVES, NO_OFFSET),
      'B_xx': states.StateDefinition('B', (), D_XX, NO_OFFSET),
      'B_yy': states.StateDefinition('B', (), D_YY, NO_OFFSET),
      'Source' : states.StateDefinition('Source', (), NO_DERIVATIVES, NO_OFFSET)
    }
    self.evolving_keys = {'A', 'B'}
    self.constant_keys = {'Source'}
    self.alpha = alpha
    self.beta = beta
    self.d_A = d_A
    self.d_B = d_B
    self._cfl_safety_factor = cfl_safety_factor
    super().__init__(alpha, beta, d_A, d_B)
  
  @property
  def cfl_safety_factor(self) -> float:
    return self._cfl_safety_factor

  def get_time_step(self, grid):
    return self._cfl_safety_factor * 0.5 * grid.step ** 2 / max(self.d_A, self.d_B)

  def time_derivative(self, grid, A, A_xx, A_yy, B, B_xx, B_yy, Source):
    """See base class."""
    rA = self.reaction_A(A, B)
    rB = self.reaction_B(A, B)
    diff_A = self.d_A * (A_xx + A_yy)
    diff_B = self.d_B * (B_xx + B_yy)
    return {'A': rA + diff_A + Source, 'B': rB + diff_B,}
