"""Euler equations."""
import enum
import functools
import operator
from typing import Any, Dict, Tuple, Union

import numpy as np
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

# Useful for SSP 3rd order Runge-Kutta time-stepping scheme
RUNGE_KUTTA_FACTOR = [[1, 0, 1], [3/4, 1/4, 1/4], [1/3, 2/3, 2/3]]

# Physical constants
MACH_INFTY = 0.5
GAMMA = 1.4
R_GAS = 287.15
P_INFTY = 1e5
T_INFTY = 300.
CP = R_GAS / (1. - 1. / GAMMA)
SPEED_OF_SOUND_INFTY = (GAMMA * R_GAS * T_INFTY) ** 0.5

# To be defined in the case of shock-vortex interaction
TARGET_PRESSURE_OUTLET = None

class _EulerBase(equations.Equation):
  """Base class for 2D unsteady Euler equations.
  
  This base class defines the state and common methods.

  Subclasses must implement the time_derivative() method.

  Attributes:
  """

  CONTINUOUS_EQUATION_NAME = 'euler'

  def __init__(self):
    super().__init__()

  def get_time_step(
    self, 
    grid: grids.Grid
    ) -> float:
    raise NotImplementedError

  @property
  def cfl_safety_factor(self) -> float:
    raise NotImplementedError

  @property
  def no_dimensions(self) -> bool:
    raise NotImplementedError

  @property
  def characteristic_density(self) -> float:
    return P_INFTY / R_GAS / T_INFTY

  @property
  def characteristic_velocity(self) -> float:
    return MACH_INFTY * SPEED_OF_SOUND_INFTY

  def random_state(
    self,
    grid: grids.Grid,
    seed: int = None,
    dtype: Any = np.float32,
  ) -> Dict[str, np.ndarray]:
    """Returns a state with random initial conditions for a vortex convection.

    Args:
      grid: Grid object holding discretization parameters.
      seed: random seed to use for random number generator.
      dtype: dtype for generated tensors.

    Returns:
      State with random initial values.
    """

    beta = 1/5
    vortex_radius = 5e-3

    if seed is not None:
      random = np.random.RandomState(seed=seed)
      beta = random.uniform(low=0.1, high=1.)
      vortex_radius = random.uniform(low=5e-3, high=1e-2)
      print(f'\n\n beta = {beta}')
      print(f'\n\n vortex_radius = {vortex_radius} \n\n')

    xx_vortex = 0.05 * np.ones(grid.shape)
    yy_vortex = 0.05 * np.ones(grid.shape)

    density_infty = P_INFTY / (R_GAS * T_INFTY)
    velocity_infty = \
      np.stack(
        [MACH_INFTY * SPEED_OF_SOUND_INFTY * np.ones(grid.shape),
        np.zeros(grid.shape)]
      )

    xx, yy = grid.get_mesh()
    Z = np.stack((yy, xx))
    Z_vortex = np.stack((yy_vortex, xx_vortex))
    distance_to_center = (Z - Z_vortex) / vortex_radius

    aux_perturbation = \
      velocity_infty[0] * beta \
        * np.exp(- np.sum(distance_to_center**2, axis=0) / 2)
    
    velocity_perturbation = \
      np.stack((-aux_perturbation, aux_perturbation)) * distance_to_center
    
    T_perturbation = -aux_perturbation**2 / 2 / CP

    velocity_0 = velocity_infty + velocity_perturbation
    T_0 = T_INFTY + T_perturbation
    density_0 = density_infty * (T_0 / T_INFTY) ** (1/(GAMMA - 1))
    
    momentum_0 = density_0 * velocity_0
    energy_0 = \
      density_0 * R_GAS * T_0 / (GAMMA - 1) \
        + np.sum(momentum_0**2, axis=0) / 2 / density_0

    state = {
      'density': density_0,
      'x_momentum': momentum_0[0],
      'y_momentum': momentum_0[1],
      'energy': energy_0,
    }
    state = {k: tf.cast(v, dtype) for k, v in state.items()}
    return state

  def random_state_double_shear_layer(
    self, 
    grid: grids.Grid, 
    seed: int = None, 
    dtype: Any = np.float32,
  ) -> Dict[str, np.ndarray]:

    # Define an homogeneous state over the whole domain
    aux_density = P_INFTY / R_GAS / T_INFTY
    aux_velocity = MACH_INFTY * SPEED_OF_SOUND_INFTY
    density_0 = aux_density * np.ones(grid.shape)
    x_velocity_0 = aux_velocity * np.ones(grid.shape)
    
    # Set an harmonic wave-like y_velocity
    mode = 1
    phase = 0.
    factor_amplitude = 0.01 # must be around 0.01 (for numerical stability)
    amplitude = factor_amplitude * aux_velocity
    xx, _ = grid.get_mesh()
    y_velocity_0 = \
      amplitude * np.sin(2 * np.pi * mode * xx / grid.length_x + phase)

    n_waves = 10
    if seed is not None:
      y_velocity_0 = 0.
      for i_wave in range(1, n_waves + 1):
        random = np.random.RandomState(seed=i_wave * seed)
        mode = random.randint(1, 3)
        phase = random.uniform(low=0.0, high=2*np.pi)
        factor_amplitude = random.uniform(low=0.001, high=0.01)
        amplitude = factor_amplitude * aux_velocity
        y_velocity_0 += \
          amplitude * np.sin(2 * np.pi * mode * xx / grid.length_x + phase)

    # Set inner density and x_velocity
    factor_density = 1.2 # must be greater than unity (proper to flow's physics)
    factor_velocity = 0.1 # must be lower than unity (for numerical stability)
    # if seed is not None:
    #   random = np.random.RandomState(seed=(n_waves + 1) * seed)
    #   factor_density = random.uniform(low=1., high=2.)
    #   factor_velocity = random.uniform(low=0.1, high=0.5)
    offset = grid.size_y // 4
    density_0[:, offset:-offset - 1] = factor_density * aux_density
    x_velocity_0[:, offset:-offset - 1] = - factor_velocity * aux_velocity

    print(f'\n factor_density = {factor_density}')
    print(f'\n factor_velocity = {factor_velocity} \n')

    # Set fluid's energy
    energy_0 = \
      P_INFTY / (GAMMA - 1) \
        + 0.5 * density_0 * (x_velocity_0**2 + y_velocity_0**2)
    
    # Non-dimensionalization?
    if self.no_dimensions:
      density_0 /= self.characteristic_density
      x_velocity_0 /= self.characteristic_velocity
      y_velocity_0 /= self.characteristic_velocity
      energy_0 /= self.characteristic_density * self.characteristic_velocity ** 2

    state = {
      'density': density_0,
      'x_momentum': density_0 * x_velocity_0,
      'y_momentum': density_0 * y_velocity_0,
      'energy': energy_0,
    }
    state = {k: tf.cast(v, dtype) for k, v in state.items()}
    return state

  def random_state_circular_shock(
    self, 
    grid: grids.Grid, 
    seed: int = None, 
    dtype: Any = np.float32,
  ) -> Dict[str, np.ndarray]:

    # Define an homogeneous state over the whole domain
    aux_pressure = P_INFTY
    aux_density = aux_pressure / R_GAS / T_INFTY

    pressure_0 = aux_pressure * np.ones(grid.shape)
    density_0 = aux_density * np.ones(grid.shape)
    x_velocity_0 = 0.
    y_velocity_0 = 0.
    
    # Set circular region's pressure and density
    xx_shock = 0.05 * np.ones(grid.shape)
    yy_shock = 0.05 * np.ones(grid.shape)
    xx, yy = grid.get_mesh()
    Z = np.stack((yy, xx))
    Z_shock = np.stack((yy_shock,xx_shock))
    distance_to_center = Z - Z_shock # idx {0, 1} == distance along {y, x}
    r = np.sqrt( np.sum(distance_to_center**2, axis=0) )
    near_shock = (r < 0.02)
    pressure_0[near_shock] = 5 * aux_pressure
    density_0[near_shock] = 5 * aux_density

    # Set fluid's energy
    energy_0 = \
      pressure_0 / (GAMMA - 1) \
        + 0.5 * density_0 * (x_velocity_0**2 + y_velocity_0**2)

    state = {
      'density': density_0,
      'x_momentum': density_0 * x_velocity_0,
      'y_momentum': density_0 * y_velocity_0,
      'energy': energy_0,
    }
    state = {k: tf.cast(v, dtype) for k, v in state.items()}
    return state

  def random_state_circular_shock_vortex_interaction(
    self, 
    grid: grids.Grid, 
    seed: int = None, 
    dtype: Any = np.float32,
  ) -> Dict[str, np.ndarray]:

    beta = 1/5
    vortex_radius = 5e-3

    if seed is not None:
      random = np.random.RandomState(seed=seed)
      beta = random.uniform(low=0.1, high=1.)
      vortex_radius = random.uniform(low=5e-3, high=1e-2)
      print(f'\n\n beta = {beta}')
      print(f'\n\n vortex_radius = {vortex_radius} \n\n')

    xx_vortex = 0.05 * np.ones(grid.shape)
    yy_vortex = 0.05 * np.ones(grid.shape)

    # Define an homogeneous state over the whole domain
    aux_density = P_INFTY / R_GAS / T_INFTY
    density_infty = aux_density * np.ones(grid.shape)
    velocity_infty = \
      np.stack(
        [MACH_INFTY * SPEED_OF_SOUND_INFTY * np.ones(grid.shape),
        np.zeros(grid.shape)]
      )

    xx, yy = grid.get_mesh()
    Z = np.stack((yy, xx))

    # Setup circular shock
    xx_shock = 0.075 * np.ones(grid.shape)
    yy_shock = 0.05 * np.ones(grid.shape)
    Z_shock = np.stack((yy_shock, xx_shock))
    distance_to_center_shock = Z - Z_shock # idx {0, 1} == distance along {y, x}
    r_shock = np.sqrt( np.sum(distance_to_center_shock**2, axis=0) )
    near_shock = (r_shock < 0.01)
    density_infty[near_shock] = 1.1 * aux_density
    
    # Setup vortex
    Z_vortex = np.stack((yy_vortex, xx_vortex))
    distance_to_center_vortex = (Z - Z_vortex) / vortex_radius

    aux_perturbation = \
      velocity_infty[0] * beta \
        * np.exp(- np.sum(distance_to_center_vortex**2, axis=0) / 2)
    
    velocity_perturbation = \
      np.stack((-aux_perturbation, aux_perturbation))*distance_to_center_vortex
    
    T_perturbation = -aux_perturbation**2 / 2 / CP

    velocity_0 = velocity_infty + velocity_perturbation
    T_0 = T_INFTY + T_perturbation
    density_0 = density_infty * (T_0 / T_INFTY) ** (1/(GAMMA - 1))
    
    momentum_0 = density_0 * velocity_0
    energy_0 = \
      density_0 * R_GAS * T_0 / (GAMMA - 1) \
        + np.sum(momentum_0**2, axis=0) / 2 / density_0

    state = {
      'density': density_0,
      'x_momentum': momentum_0[0],
      'y_momentum': momentum_0[1],
      'energy': energy_0,
    }
    state = {k: tf.cast(v, dtype) for k, v in state.items()}
    return state

  def random_state_acoustic_1d(
    self, 
    grid: grids.Grid, 
    seed: int = None, 
    dtype: Any = np.float32,
  ) -> Dict[str, np.ndarray]:

    # Define a mean state
    density_infty = P_INFTY / (R_GAS * T_INFTY)

    # Define state's perturbation - gaussian pulse in acoustic pressure field
    # (linear acoustics - see H4 of section 8.2.1 of the TNC book)
    xx, _ = grid.get_mesh()
    mean = grid.length_x / 2
    std_dev = grid.length_x / 10
    P_1 = 1e-3 * P_INFTY * np.exp(- (xx - mean)**2 / std_dev**2 / 2) / std_dev / np.sqrt(2*np.pi)
    x_velocity_1 = 1e-2 * SPEED_OF_SOUND_INFTY
    
    n_waves = 10
    if seed is not None:
      P_1 = 0.
      x_velocity_1 = 0.
      for i_wave in range(1, n_waves + 1):
        random = np.random.RandomState(seed=i_wave * seed)
        mode = random.randint(3, 7)
        phase = random.uniform(low=0.0, high=2*np.pi)
        factor_amplitude = random.uniform(low=0.001, high=0.01)
        amplitude = factor_amplitude * P_INFTY
        P_1 += \
          amplitude * np.sin(2 * np.pi * mode * xx / grid.length_x + phase)
        random = np.random.RandomState(seed=(i_wave + n_waves) * seed)
        mode = random.randint(3, 7)
        phase = random.uniform(low=0.0, high=2*np.pi)
        factor_amplitude = random.uniform(low=0.0001, high=0.001)
        amplitude = factor_amplitude * SPEED_OF_SOUND_INFTY
        x_velocity_1 += \
          amplitude * np.sin(2 * np.pi * mode * xx / grid.length_x + phase)
    density_1 = P_1 / SPEED_OF_SOUND_INFTY ** 2

    # Define state
    P_0 = P_INFTY + P_1
    density_0 = density_infty + density_1
    x_momentum_0 = density_0 * x_velocity_1
    y_momentum_0 = np.zeros(grid.shape)
    energy_0 = P_0 / (GAMMA - 1) + 0.5 * x_momentum_0 ** 2 / density_0 

    state = {
      'density': density_0,
      'x_momentum': x_momentum_0,
      'y_momentum': y_momentum_0,
      'energy': energy_0,
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

class Weno5ComponentEuler(_EulerBase):
  """Finite-differences WENO5 (component-wise) scheme for Euler equations.
  
  References:
    [1] Shu CW. (1999) High Order ENO and WENO Schemes for Computational Fluid Dynamics. In: Barth T.J., Deconinck H. (eds) High-Order Methods for Computational Physics. Lecture Notes in Computational Science and Engineering, vol 9. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-662-03882-6_5
    --------------- ALGORTIHMS 3.2 (page 463), 4.3 (page 469) and 4.5 (page 475)
  """

  DISCRETIZATION_NAME = 'weno5_component'
  METHOD = polynomials.Method.FINITE_DIFFERENCE
  MONOTONIC = False

  def __init__(
    self, 
    cfl_safety_factor: float = 0.8,
    no_dimensions: bool = False,
    ):
    self.key_definitions = {
      'density': StateDef('density', (), NO_DERIVATIVES, NO_OFFSET),
      'x_momentum': StateDef('momentum', (X,), NO_DERIVATIVES, NO_OFFSET),
      'y_momentum': StateDef('momentum', (Y,), NO_DERIVATIVES, NO_OFFSET),
      'energy': StateDef('energy', (), NO_DERIVATIVES, NO_OFFSET),
    }
    self.evolving_keys = {'density', 'x_momentum', 'y_momentum', 'energy'}
    self.constant_keys = set()
    self._cfl_safety_factor = cfl_safety_factor
    self._no_dimensions = no_dimensions
    super().__init__()

  @property
  def cfl_safety_factor(self) -> float:
    return self._cfl_safety_factor
  
  @property
  def no_dimensions(self) -> bool:
    return self._no_dimensions
  
  def get_time_step(self, grid):
    dt = self._cfl_safety_factor * grid.step / SPEED_OF_SOUND_INFTY / (MACH_INFTY + 1.)
    if self._no_dimensions:
      dt /= grid.length_y / self.characteristic_velocity
    return dt

  def flux_splitting(
    self, grid, f, velocity, speed_of_sound, density, x_momentum, y_momentum, 
    energy, eigenvalues
  ):
    w_dict = {}
    w_dict['density'] = density
    w_dict['x_momentum'] = x_momentum
    w_dict['y_momentum'] = y_momentum
    w_dict['energy'] = energy

    f_left = {}
    f_right = {}
    for i_axis, axis in enumerate([X_AXIS, Y_AXIS]):
      a = _maximum( _abs(velocity[i_axis] - speed_of_sound), 
                    _abs(velocity[i_axis]), 
                    _abs(velocity[i_axis] + speed_of_sound) )
      a = _amax(a)
      for key in self.evolving_keys:
        f_left.setdefault(key, [])
        f_right.setdefault(key, [])
        f_left[key].append(
          0.5 * (f[key][i_axis] + a * w_dict[key])
        )
        f_right[key].append(
          _roll_plus_one(0.5 * (f[key][i_axis] - a * w_dict[key]), axis)
        )

    return f_left, f_right

  def neighbors(self, f):
    f_m2 = {}
    f_m1 = {}
    f_p1 = {}
    f_p2 = {}
    for i, axis in enumerate([X_AXIS, Y_AXIS]):
      for key in self.evolving_keys:
        f_m2.setdefault(key, [])
        f_m1.setdefault(key, [])
        f_p1.setdefault(key, [])
        f_p2.setdefault(key, [])
        f_ = f[key][i]
        f_m2[key].append(_roll_minus_two(f_,axis))
        f_m1[key].append(_roll_minus_one(f_,axis))
        f_p1[key].append(_roll_plus_one(f_,axis))
        f_p2[key].append(_roll_plus_two(f_,axis))
    
    return f_m2, f_m1, f_p1, f_p2

  def smoothness_indicators_tau(self, f, f_m2, f_m1, f_p1, f_p2):
    """ Compute WENO5 smoothness indicators and \tau parameter.
    
    Returns:
      dict of 
      [ tf.stack([BETA_1_X, BETA_2_X, BETA_3_X]), 
        tf.stack([BETA_1_Y, BETA_2_Y, BETA_3_Y]) ]
      lists;
      
      dict of [TAU_X, TAU_Y] lists.
    """
    beta = {}
    tau = {}
    for i_axis in range(2):
      for key in self.evolving_keys:
        f_j = f[key][i_axis]
        f_jm2 = f_m2[key][i_axis]
        f_jm1 = f_m1[key][i_axis]
        f_jp1 = f_p1[key][i_axis]
        f_jp2 = f_p2[key][i_axis]
        beta_1 = (13/12) * (f_jm2 - 2 * f_jm1 + f_j) ** 2 \
                + (1/4) * (f_jm2 - 4 * f_jm1 + 3 * f_j) ** 2 
        beta_2 = (13/12) * (f_jm1 - 2 * f_j + f_jp1) ** 2 \
                + (1/4) * (f_jm1 - f_jp1) ** 2
        beta_3 = (13/12) * (f_j - 2 * f_jp1 + f_jp2) ** 2 \
                + (1/4) * (3 * f_j - 4 * f_jp1 + f_jp2) ** 2
        beta.setdefault(key, [])
        beta[key].append( tf.stack([beta_1, beta_2, beta_3]) )
        tau.setdefault(key, [])
        tau[key].append(
          (f_jm2 - 4 * f_jm1 + 6 * f_j - 4 * f_jp1 + f_jp2) ** 2
        ) 

    return beta, tau

  def weights(self, grid, beta, tau, epsilon, c, p):
    """ Compute WENO5 weights.
    
    Returns:
      dicts of 
      [ [WEIGHT_1_X, WEIGHT_2_X, WEIGHT_3_X], 
        [WEIGHT_1_Y, WEIGHT_2_Y, WEIGHT_3_Y] ]
      lists.
    """
    # Convert c to a tf Tensor with shape (len(c),grid.size_x,grid.size_y)
    c_1 = tf.constant( c[0], shape=grid.shape )
    c_2 = tf.constant( c[1], shape=grid.shape )
    c_3 = tf.constant( c[2], shape=grid.shape )
    c = tf.stack( [c_1, c_2, c_3] )

    weights = {}
    for i_axis in range(2):
      for key in self.evolving_keys:
        alpha = c \
          * ( 1 + ( tau[key][i_axis] / ( epsilon + beta[key][i_axis] ) ) ** p )
        weights.setdefault(key, [])
        weights[key].append( alpha / _sum(alpha, axis=0) )

    return weights
  
  def reconstruction(self, grid, f_left, f_right):
    list_f = [f_left, f_right]
    list_fluxes = [{}, {}]
    for i_bias, bias in enumerate(['left','right']):
      f = list_f[i_bias]

      f_m2, f_m1, f_p1, f_p2 = self.neighbors(f)

      beta, tau = self.smoothness_indicators_tau(f, f_m2, f_m1, f_p1, f_p2)

      epsilon = 1e-6
      c = [0.1, 0.6, 0.3]
      if bias == 'right':
        c = c[::-1] # reverse order
      p = 2
      weights = self.weights(grid, beta, tau, epsilon, c, p)

      for i_axis in range(2):
        for key in self.evolving_keys:
          f_j = f[key][i_axis]
          f_jm2 = f_m2[key][i_axis]
          f_jm1 = f_m1[key][i_axis]
          f_jp1 = f_p1[key][i_axis]
          f_jp2 = f_p2[key][i_axis]
          [omega_1, omega_2, omega_3] = \
            [ weight for weight in weights[key][i_axis] ]
          if bias == 'left':
            flux_biased = (1/3) * omega_1 * f_jm2 \
                          - (1/6) * (7 * omega_1 + omega_2) * f_jm1 \
                          + (1/6) * (11*omega_1 + 5*omega_2 + 2*omega_3) * f_j \
                          + (1/6) * (2 * omega_2 + 5 * omega_3) * f_jp1 \
                          - (1/6) * omega_3 * f_jp2 
          else:
            flux_biased = - (1/6) * omega_1 * f_jm2 \
                          + (1/6) * (5 * omega_1 + 2 * omega_2) * f_jm1 \
                          + (1/6) * (2*omega_1 + 5*omega_2 + 11*omega_3) * f_j \
                          - (1/6) * (omega_2 + 7 * omega_3) * f_jp1 \
                          + (1/3) * omega_3 * f_jp2
          list_fluxes[i_bias].setdefault(key, [])
          list_fluxes[i_bias][key].append( flux_biased )

    return list_fluxes        

  def flux(
    self, grid, velocity, pressure, speed_of_sound, density, x_momentum, 
    y_momentum, energy):
    """Compute velocity fluxes for all evolving keys.
    
    Returns:
      dict of [X_FLUX_EDGE_X, Y_FLUX_EDGE_Y] lists.
    """
    f = {}
    f['density'] = [x_momentum, y_momentum]
    f['x_momentum'] = [x_momentum**2 / density + pressure,
                       x_momentum * y_momentum / density]
    f['y_momentum'] = [x_momentum * y_momentum / density, 
                       y_momentum**2 / density + pressure]
    f['energy'] = [x_momentum * (energy + pressure) / density, 
                   y_momentum * (energy + pressure) / density]

    # Global Lax-Friedrichs flux splitting
    eigenvalues = {}
    eigenvalues['density'] = \
      [velocity[0] - speed_of_sound, velocity[1] - speed_of_sound]
    eigenvalues['x_momentum'] = \
      [velocity[0], velocity[1]]
    eigenvalues['y_momentum'] = \
      [velocity[0] + speed_of_sound, velocity[1] + speed_of_sound]
    eigenvalues['energy'] = \
      [velocity[0], velocity[1]]
    f_left, f_right = self.flux_splitting(grid, f, velocity, speed_of_sound,
                                          density, x_momentum, y_momentum,
                                          energy, eigenvalues)

    # Reconstruct left- and right-biased fluxes
    list_fluxes = self.reconstruction(grid, f_left, f_right)

    # Sum left- and right-biased fluxes to recover total flux
    flux = {}
    for i_axis in range(2):
      for key in self.evolving_keys:
        flux.setdefault(key, [])
        flux[key].append( 
          list_fluxes[0][key][i_axis] + list_fluxes[1][key][i_axis] 
        )

    return flux

  def time_derivative(
    self, grid, density, x_momentum, y_momentum, energy):

    velocity = [x_momentum / density, y_momentum / density]
    pressure = (GAMMA - 1) * (energy \
                              - density * (velocity[0]**2 + velocity[1]**2) / 2)
    speed_of_sound = _sqrt(GAMMA * pressure / density)

    fluxes = self.flux(grid, velocity, pressure, speed_of_sound, density, x_momentum, y_momentum, energy)
    
    grid_step = grid.step
    if self._no_dimensions:
      grid_step /= grid.length_y
    time_derivs = {}
    for key in self.evolving_keys:
      x_flux_edge_x = fluxes[key][0]
      y_flux_edge_y = fluxes[key][1]
      time_derivs[key] = \
        flux_to_time_derivative(x_flux_edge_x, y_flux_edge_y, grid_step)

    return time_derivs

  def take_time_step(
    self, grid, density, x_momentum, y_momentum, energy):
    
    dt = self.get_time_step(grid)
    w_dict_0 = {}
    w_dict_0['density'] = density
    w_dict_0['x_momentum'] = x_momentum
    w_dict_0['y_momentum'] = y_momentum
    w_dict_0['energy'] = energy

    w_dict = {}
    w_dict['density'] = density
    w_dict['x_momentum'] = x_momentum
    w_dict['y_momentum'] = y_momentum
    w_dict['energy'] = energy
    
    # SSP 3rd order Runge-Kutta (TVD-RK3)
    factor = tf.constant([[1, 0, 1], [3/4, 1/4, 1/4], [1/3, 2/3, 2/3]])
    for i in range(3):
      time_derivs = self.time_derivative(grid, **w_dict)
      for key in self.evolving_keys:
        w_dict[key] = factor[i,0] * w_dict_0[key] + factor[i,1] * w_dict[key] \
                      + factor[i,2] * dt * time_derivs[key]

    # Perform sanity check on solution
    for key in self.evolving_keys:
      if np.isfinite(w_dict[key]).all() == False:
        raise ValueError('\n\n NaN or +/- inf values appeared ! \n\n')

    return w_dict

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

class JamesonEuler(_EulerBase):
  """Finite-volume JST scheme for 2D unsteady Euler equations.
  
  References:
    [1] Jameson, A. (2017). "Origins and Further Development of the Jameson-Schmidt-Turkel Scheme," AIAA Journal, 55, 1487-1510. doi:10.2514/1.J055493.
  """

  DISCRETIZATION_NAME = 'jameson'
  METHOD = polynomials.Method.FINITE_VOLUME
  MONOTONIC = False

  def __init__(
    self,
    cfl_safety_factor: float = 0.8,
    no_dimensions: bool = False,
    ):
    self.key_definitions = {
      'density': StateDef('density',(),NO_DERIVATIVES,NO_OFFSET),
      'x_momentum': StateDef('momentum', (X,), NO_DERIVATIVES, NO_OFFSET),
      'y_momentum': StateDef('momentum', (Y,), NO_DERIVATIVES, NO_OFFSET),
      'energy': StateDef('energy',(),NO_DERIVATIVES,NO_OFFSET), # rho * E
    }
    self.evolving_keys = {'density','x_momentum', 'y_momentum','energy'}
    self.constant_keys = set()
    self._cfl_safety_factor = cfl_safety_factor
    self._no_dimensions = no_dimensions
    super().__init__()
  
  @property
  def cfl_safety_factor(self) -> float:
    return self._cfl_safety_factor
  
  @property
  def no_dimensions(self) -> bool:
    return self._no_dimensions

  def get_time_step(self, grid):
    dt = self._cfl_safety_factor * 2 ** 1.5 * grid.step / SPEED_OF_SOUND_INFTY / (MACH_INFTY + 1.)
    if self._no_dimensions:
      dt /= grid.length_y / self.characteristic_velocity
    return dt

  def flux(
    self, pressure, 
    density, x_momentum, y_momentum, energy):
    """Compute velocity fluxes for all evolving keys.
    
    Returns:
      dict of [X_FLUX_EDGE_X, Y_FLUX_EDGE_Y] lists.
    """
    f = {}
    f['density'] = [x_momentum, y_momentum]
    f['x_momentum'] = [x_momentum**2 / density + pressure,
                       x_momentum * y_momentum / density]
    f['y_momentum'] = [x_momentum * y_momentum / density, 
                       y_momentum**2 / density + pressure]
    f['energy'] = [x_momentum * (energy + pressure) / density, 
                   y_momentum * (energy + pressure) / density]

    flux = {}
    for i, axis in enumerate([X_AXIS, Y_AXIS]):
      for key in self.evolving_keys:
        flux.setdefault(key, [])
        flux[key].append( 0.5 * (_roll_plus_one(f[key][i],axis) + f[key][i]) )

    return flux

  def dissipation(
    self, velocity, pressure, speed_of_sound,
    density, x_momentum, y_momentum, energy):
    """Compute artificial dissipation fluxes for all evolving keys.
    
    Returns:
      dict of [X_DISSIPATION_FLUX_EDGE_X, Y_DISSIPATION_FLUX_EDGE_Y] lists.
    """
    kappa2 = 1
    kappa4 = 1/32

    w_dict = {}
    w_dict['density'] = density
    w_dict['x_momentum'] = x_momentum
    w_dict['y_momentum'] = y_momentum
    w_dict['energy'] = energy

    dissipation = {}
    for axis in [X_AXIS,Y_AXIS]:
      pressure_left = _roll_minus_one(pressure,axis)
      pressure_right = _roll_plus_one(pressure,axis)
      nu = abs( (pressure_right - 2 * pressure + pressure_left) / (pressure_right + 2 * pressure + pressure_left) )
      r = velocity + speed_of_sound
      nu_max = _maximum(nu, _roll_plus_one(nu, axis))
      r_max = _maximum(r, _roll_plus_one(r, axis))
      epsilon2 = kappa2 * nu_max * r_max
      epsilon4 = _maximum(0,kappa4 * r_max - 2 * epsilon2)

      for key, w in w_dict.items():
        delta_w = _roll_plus_one(w, axis) - w
        dissipation.setdefault(key, [])
        dissipation[key].append(
          epsilon2 * delta_w \
            - epsilon4 * (
              _roll_plus_one(delta_w, axis) - 2 * delta_w + _roll_minus_one(delta_w, axis)
            ) 
        )

    return dissipation

  def time_derivative(
    self, grid, density, x_momentum, y_momentum, energy):

    velocity = _sqrt(x_momentum**2 + y_momentum**2) / density
    pressure = (GAMMA - 1) * (energy - density * velocity**2 / 2)
    speed_of_sound = _sqrt(GAMMA * pressure / density)

    fluxes = self.flux(pressure, density, x_momentum, y_momentum, energy)
    dissipations = self.dissipation(velocity, pressure, speed_of_sound,
                                    density, x_momentum, y_momentum, energy)
    
    grid_step = grid.step
    if self._no_dimensions:
      grid_step /= grid.length_y
    time_derivs = {}
    for key in self.evolving_keys:
      x_flux_edge_x = fluxes[key][0] - dissipations[key][0]
      y_flux_edge_y = fluxes[key][1] - dissipations[key][1]
      time_derivs[key] = flux_to_time_derivative(x_flux_edge_x, y_flux_edge_y,
                                                 grid_step)

    return time_derivs

  def take_time_step(
    self, grid, density, x_momentum, y_momentum, energy):
    
    dt = self.get_time_step(grid)
    w_dict_0 = {}
    w_dict_0['density'] = density
    w_dict_0['x_momentum'] = x_momentum
    w_dict_0['y_momentum'] = y_momentum
    w_dict_0['energy'] = energy
    all_time_derivs = [self.time_derivative(grid,**w_dict_0)]

    ## Classic 4th order Runge-Kutta
    # Steps 1 to 3
    w_dict = {}
    for i, factor in enumerate([1/2, 1/2, 1]):
      for key in self.evolving_keys:
        w_dict[key] = w_dict_0[key] + factor * dt * all_time_derivs[i][key]
      all_time_derivs.append(self.time_derivative(grid,**w_dict))
    
    # Step 4
    for key in self.evolving_keys:
      w_dict[key] = w_dict_0[key]
      for i, factor in enumerate([1/6, 1/3, 1/3, 1/6]):
        w_dict[key] += factor * dt * (all_time_derivs[i][key])
    
    # Perform sanity check on solution
    for key in self.evolving_keys:
      if np.isfinite(w_dict[key]).all() == False:
        raise ValueError('\n\n NaN or +/- inf values appeared ! \n\n')

    return w_dict
  

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

class Weno5Euler(_EulerBase):
  """Finite-differences WENO5 (characteristic-wise) scheme for Euler equations.
  
  References:
    [1] Shu CW. (1999) High Order ENO and WENO Schemes for Computational Fluid Dynamics. In: Barth T.J., Deconinck H. (eds) High-Order Methods for Computational Physics. Lecture Notes in Computational Science and Engineering, vol 9. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-662-03882-6_5
    --------------- ALGORTIHMS 3.2 (page 463), 4.3 (page 469) and 4.8 (page 478)
    
    [2] Fernandez-Fidalgo FJ. (2019) Computational simulation of compressible flows: A family of very efficient and highly accurate numerical methods based on Finite Differences. PhD thesis at Universidade da Coruna.
    --------------- EULER EQUATIONS EIGENSYSTEM (appendix B.2., page 160-161) 
  """

  DISCRETIZATION_NAME = 'weno_5'
  METHOD = polynomials.Method.FINITE_DIFFERENCE
  MONOTONIC = False

  def __init__(
    self, 
    cfl_safety_factor: float = 0.8,
    no_dimensions: bool = False,
    ):
    self.key_definitions = {
      'density': StateDef('density', (), NO_DERIVATIVES, NO_OFFSET),
      'x_momentum': StateDef('momentum', (X,), NO_DERIVATIVES, NO_OFFSET),
      'y_momentum': StateDef('momentum', (Y,), NO_DERIVATIVES, NO_OFFSET),
      'energy': StateDef('energy', (), NO_DERIVATIVES, NO_OFFSET),
    }
    self.evolving_keys = {'density', 'x_momentum', 'y_momentum', 'energy'}
    self.constant_keys = set()
    self._cfl_safety_factor = cfl_safety_factor
    self._no_dimensions = no_dimensions
    super().__init__()

  @property
  def cfl_safety_factor(self) -> float:
    return self._cfl_safety_factor
  
  @property
  def no_dimensions(self) -> bool:
    return self._no_dimensions

  def get_time_step(self, grid):
    dt = self._cfl_safety_factor * grid.step / SPEED_OF_SOUND_INFTY / (MACH_INFTY + 1.)
    if self._no_dimensions:
      dt /= grid.length_y / self.characteristic_velocity
    return dt

  def average_eigensystem(
    self, grid, w_avg
  ):
    x_velocity_edge_x = w_avg['x_momentum'][0] / w_avg['density'][0]
    x_velocity_edge_y = w_avg['x_momentum'][1] / w_avg['density'][1]
    y_velocity_edge_x = w_avg['y_momentum'][0] / w_avg['density'][0]
    y_velocity_edge_y = w_avg['y_momentum'][1] / w_avg['density'][1]
    e_edge_x = 0.5 * (x_velocity_edge_x ** 2 + y_velocity_edge_x ** 2)
    e_edge_y = 0.5 * (x_velocity_edge_y ** 2 + y_velocity_edge_y ** 2)
    pressure_edge_x = \
      (GAMMA - 1) * (w_avg['energy'][0] - w_avg['density'][0] * e_edge_x )
    speed_of_sound_edge_x = _sqrt(GAMMA * pressure_edge_x / w_avg['density'][0])
    pressure_edge_y = \
      (GAMMA - 1) * (w_avg['energy'][1] - w_avg['density'][1] * e_edge_y )
    speed_of_sound_edge_y = _sqrt(GAMMA * pressure_edge_y / w_avg['density'][1])

    R = {}
    R['density'] = \
      [ tf.stack(
          [tf.ones(grid.shape),
          tf.ones(grid.shape),
          tf.ones(grid.shape),
          tf.zeros(grid.shape)]
        ),
        tf.stack(
          [tf.ones(grid.shape),
          tf.ones(grid.shape),
          tf.ones(grid.shape),
          tf.zeros(grid.shape)]
        )
      ]
    
    R['x_momentum'] = \
      [ tf.stack(
          [x_velocity_edge_x - speed_of_sound_edge_x,
          x_velocity_edge_x,
          x_velocity_edge_x + speed_of_sound_edge_x,
          tf.zeros(grid.shape)]
        ),
        tf.stack(
          [x_velocity_edge_y,
          x_velocity_edge_y,
          x_velocity_edge_y,
          tf.ones(grid.shape)]
        )
      ]
    
    R['y_momentum'] = \
      [ tf.stack(
          [y_velocity_edge_x,
          y_velocity_edge_x,
          y_velocity_edge_x,
          - tf.ones(grid.shape)]
        ),
        tf.stack(
          [y_velocity_edge_y - speed_of_sound_edge_y,
          y_velocity_edge_y,
          y_velocity_edge_y + speed_of_sound_edge_y,
          tf.zeros(grid.shape)]
        )
      ]

    R['energy'] = \
      [ tf.stack(
          [e_edge_x + speed_of_sound_edge_x ** 2 / (GAMMA - 1) \
            - speed_of_sound_edge_x * x_velocity_edge_x,
          e_edge_x,
          e_edge_x + speed_of_sound_edge_x ** 2 / (GAMMA - 1) \
            + speed_of_sound_edge_x * x_velocity_edge_x,
          - y_velocity_edge_x]
        ),
        tf.stack(
          [e_edge_y + speed_of_sound_edge_y ** 2 / (GAMMA - 1) \
            - speed_of_sound_edge_y * y_velocity_edge_y,
          e_edge_y,
          e_edge_y + speed_of_sound_edge_y ** 2 / (GAMMA - 1) \
            + speed_of_sound_edge_y * y_velocity_edge_y,
          x_velocity_edge_y]
        )
      ]
    
    L = {}
    L['density'] = \
      [ tf.stack(
          [((GAMMA - 1)*e_edge_x + speed_of_sound_edge_x * x_velocity_edge_x) \
            / 2 / speed_of_sound_edge_x ** 2,
          ((1 - GAMMA) * x_velocity_edge_x - speed_of_sound_edge_x) \
            / 2 / speed_of_sound_edge_x ** 2,
          ((1 - GAMMA) * y_velocity_edge_x) \
            / 2 / speed_of_sound_edge_x ** 2,
          (GAMMA - 1) / 2 / speed_of_sound_edge_x ** 2]
        ),
        tf.stack(
          [((GAMMA - 1)*e_edge_y + speed_of_sound_edge_y * y_velocity_edge_y) \
            / 2 / speed_of_sound_edge_y ** 2,
          ((1 - GAMMA) * x_velocity_edge_y) \
            / 2 / speed_of_sound_edge_y ** 2,
          ((1 - GAMMA) * y_velocity_edge_y - speed_of_sound_edge_y) \
            / 2 / speed_of_sound_edge_y ** 2,
          (GAMMA - 1) / 2 / speed_of_sound_edge_y ** 2]
        )
      ]
    
    L['x_momentum'] = \
      [ tf.stack(
          [tf.ones(grid.shape) - (GAMMA-1)*e_edge_x / speed_of_sound_edge_x**2,
          (GAMMA - 1) * x_velocity_edge_x / speed_of_sound_edge_x ** 2,
          (GAMMA - 1) * y_velocity_edge_x / speed_of_sound_edge_x ** 2,
          (1 - GAMMA) / speed_of_sound_edge_x ** 2]
        ),
        tf.stack(
          [tf.ones(grid.shape) - (GAMMA-1)*e_edge_y / speed_of_sound_edge_y**2,
          (GAMMA - 1) * x_velocity_edge_y / speed_of_sound_edge_y ** 2,
          (GAMMA - 1) * y_velocity_edge_y / speed_of_sound_edge_y ** 2,
          (1 - GAMMA) / speed_of_sound_edge_y ** 2]
        )
      ]
    
    L['y_momentum'] = \
      [ tf.stack(
          [((GAMMA - 1)*e_edge_x - speed_of_sound_edge_x * x_velocity_edge_x) \
            / 2 / speed_of_sound_edge_x ** 2,
          ((1 - GAMMA) * x_velocity_edge_x + speed_of_sound_edge_x) \
            / 2 / speed_of_sound_edge_x ** 2,
          ((1 - GAMMA) * y_velocity_edge_x) \
            / 2 / speed_of_sound_edge_x ** 2,
          (GAMMA - 1) / 2 / speed_of_sound_edge_x ** 2]
        ),
        tf.stack(
          [((GAMMA - 1)*e_edge_y - speed_of_sound_edge_y * y_velocity_edge_y) \
            / 2 / speed_of_sound_edge_y ** 2,
          ((1 - GAMMA) * x_velocity_edge_y) \
            / 2 / speed_of_sound_edge_y ** 2,
          ((1 - GAMMA) * y_velocity_edge_y + speed_of_sound_edge_y) \
            / 2 / speed_of_sound_edge_y ** 2,
          (GAMMA - 1) / 2 / speed_of_sound_edge_y ** 2]
        )
      ]

    L['energy'] = \
      [ tf.stack(
          [y_velocity_edge_x,
          tf.zeros(grid.shape),
          - tf.ones(grid.shape),
          tf.zeros(grid.shape)]
        ),
        tf.stack(
          [- x_velocity_edge_y,
          tf.ones(grid.shape),
          tf.zeros(grid.shape),
          tf.zeros(grid.shape)]
        )
      ]

    return R, L

  def flux_splitting(
    self, grid, f, eigenvalues, w
  ):
    f_left = {}
    f_right = {}
    for i_axis, axis in enumerate([X_AXIS, Y_AXIS]):
      for key in self.evolving_keys:
        a = _amax(_abs(eigenvalues[key][i_axis]))
        f_left.setdefault(key, [])
        f_right.setdefault(key, [])
        f_left[key].append(
          0.5 * (f[key][i_axis] + a * w[key][i_axis])
        )
        f_right[key].append(
          0.5 * (f[key][i_axis] - a * w[key][i_axis])
        )

    return f_left, f_right

  def neighbors(self, f):
    f_m2 = {}
    f_m1 = {}
    f_p1 = {}
    f_p2 = {}
    f_p3 = {}
    for i_axis, axis in enumerate([X_AXIS, Y_AXIS]):
      for key in self.evolving_keys:
        f_m2.setdefault(key, [])
        f_m1.setdefault(key, [])
        f_p1.setdefault(key, [])
        f_p2.setdefault(key, [])
        f_p3.setdefault(key, [])
        f_ = f[key][i_axis]
        f_m2[key].append(_roll_minus_two(f_,axis))
        f_m1[key].append(_roll_minus_one(f_,axis))
        f_p1[key].append(_roll_plus_one(f_,axis))
        f_p2[key].append(_roll_plus_two(f_,axis))
        f_p3[key].append(_roll_plus_three(f_,axis))
    
    return f_m2, f_m1, f_p1, f_p2, f_p3

  def smoothness_indicators_tau(self, f, f_m2, f_m1, f_p1, f_p2):
    """ Compute WENO5 smoothness indicators and \tau parameter.
    
    Returns:
      dict of 
      [ tf.stack([BETA_1_X, BETA_2_X, BETA_3_X]), 
        tf.stack([BETA_1_Y, BETA_2_Y, BETA_3_Y]) ]
      lists;
      
      dict of [TAU_X, TAU_Y] lists.
    """
    beta = {}
    tau = {}
    for i_axis in range(2):
      for key in self.evolving_keys:
        f_j = f[key][i_axis]
        f_jm2 = f_m2[key][i_axis]
        f_jm1 = f_m1[key][i_axis]
        f_jp1 = f_p1[key][i_axis]
        f_jp2 = f_p2[key][i_axis]
        beta_1 = (13/12) * (f_jm2 - 2 * f_jm1 + f_j) ** 2 \
                + (1/4) * (f_jm2 - 4 * f_jm1 + 3 * f_j) ** 2 
        beta_2 = (13/12) * (f_jm1 - 2 * f_j + f_jp1) ** 2 \
                + (1/4) * (f_jm1 - f_jp1) ** 2
        beta_3 = (13/12) * (f_j - 2 * f_jp1 + f_jp2) ** 2 \
                + (1/4) * (3 * f_j - 4 * f_jp1 + f_jp2) ** 2
        beta.setdefault(key, [])
        beta[key].append( tf.stack([beta_1, beta_2, beta_3]) )
        tau.setdefault(key, [])
        tau[key].append( 
          (f_jm2 - 4 * f_jm1 + 6 * f_j - 4 * f_jp1 + f_jp2) ** 2
        ) 

    return beta, tau

  def weights(self, grid, beta, tau, epsilon, c, p):
    """ Compute WENO5 weights.
    
    Returns:
      dicts of 
      [ [WEIGHT_1_X, WEIGHT_2_X, WEIGHT_3_X], 
        [WEIGHT_1_Y, WEIGHT_2_Y, WEIGHT_3_Y] ]
      lists.
    """
    # Convert c to a tf Tensor with shape (len(c),grid.size_x,grid.size_y)
    c_1 = tf.constant( c[0], shape=grid.shape )
    c_2 = tf.constant( c[1], shape=grid.shape )
    c_3 = tf.constant( c[2], shape=grid.shape )
    c = tf.stack( [c_1, c_2, c_3] )

    weights = {}
    for i_axis in range(2):
      for key in self.evolving_keys:
        alpha = c \
          * ( 1 + ( tau[key][i_axis] / ( epsilon + beta[key][i_axis] ) ) ** p )
        weights.setdefault(key, [])
        weights[key].append( alpha / _sum(alpha, axis=0) )

    return weights
  
  def reconstruction(
    self,
    grid, 
    f_left, f_right,
    f_m2_left, f_m2_right, 
    f_m1_left, f_m1_right,
    f_p1_left, f_p1_right,
    f_p2_left, f_p2_right,
    f_p3_left, f_p3_right
  ):
    # Left reconstruction (Stencil I_{i-2} to I_{i+2})
    beta, tau = self.smoothness_indicators_tau(
                  f_left, f_m2_left, f_m1_left, f_p1_left, f_p2_left)
    epsilon = 1e-6
    c = [0.1, 0.6, 0.3]
    p = 2
    weights = self.weights(grid, beta, tau, epsilon, c, p)

    flux_left = {}
    for i_axis in range(2):
      for key in self.evolving_keys:
        f_j = f_left[key][i_axis]
        f_jm2 = f_m2_left[key][i_axis]
        f_jm1 = f_m1_left[key][i_axis]
        f_jp1 = f_p1_left[key][i_axis]
        f_jp2 = f_p2_left[key][i_axis]
        [omega_1, omega_2, omega_3] = \
          [ weight for weight in weights[key][i_axis] ]
        flux_biased = \
          (1/3) * omega_1 * f_jm2 \
          - (1/6) * (7 * omega_1 + omega_2) * f_jm1 \
          + (1/6) * (11*omega_1 + 5*omega_2 + 2*omega_3) * f_j \
          + (1/6) * (2 * omega_2 + 5 * omega_3) * f_jp1 \
          - (1/6) * omega_3 * f_jp2
        flux_left.setdefault(key, [])
        flux_left[key].append( flux_biased )
    
    # Right reconstruction (Stencil I_{i-1} to I_{i+3})
    beta, tau = self.smoothness_indicators_tau(
                  f_p1_right, f_m1_right, f_right, f_p2_right, f_p3_right)
    c = c[::-1] # reverse order
    weights = self.weights(grid, beta, tau, epsilon, c, p)

    flux_right = {}
    for i_axis in range(2):
      for key in self.evolving_keys:
        f_j = f_p1_right[key][i_axis]
        f_jm2 = f_m1_right[key][i_axis]
        f_jm1 = f_right[key][i_axis]
        f_jp1 = f_p2_right[key][i_axis]
        f_jp2 = f_p3_right[key][i_axis]
        [omega_1, omega_2, omega_3] = \
          [ weight for weight in weights[key][i_axis] ]
        flux_biased = \
          - (1/6) * omega_1 * f_jm2 \
          + (1/6) * (5 * omega_1 + 2 * omega_2) * f_jm1 \
          + (1/6) * (2*omega_1 + 5*omega_2 + 11*omega_3) * f_j \
          - (1/6) * (omega_2 + 7 * omega_3) * f_jp1 \
          + (1/3) * omega_3 * f_jp2
        flux_right.setdefault(key, [])
        flux_right[key].append( flux_biased )

    return flux_left, flux_right

  def average_state(
    self, pressure, w, roe=True
  ):
    w_avg = {}
    if roe:
      for i_axis, axis in enumerate([X_AXIS, Y_AXIS]):
        enthalpy = \
          (w['energy'][i_axis] + pressure) / w['density'][i_axis]
        sqrt_density = _sqrt( w['density'][i_axis] )
        sqrt_density_p1 = _roll_plus_one(sqrt_density, axis)
        x_velocity = w['x_momentum'][i_axis] / w['density'][i_axis]
        y_velocity = w['y_momentum'][i_axis] / w['density'][i_axis]
        prod_x_vel = x_velocity * sqrt_density
        prod_y_vel = y_velocity * sqrt_density
        prod_enthalpy = enthalpy * sqrt_density
        sum_sqrt_density = sqrt_density + sqrt_density_p1
        sum_prod_x_vel = prod_x_vel + _roll_plus_one(prod_x_vel, axis)
        sum_prod_y_vel = prod_y_vel + _roll_plus_one(prod_y_vel, axis)
        sum_prod_enthalpy = prod_enthalpy + _roll_plus_one(prod_enthalpy, axis)

        density_avg = sqrt_density * sqrt_density_p1
        x_velocity_avg = sum_prod_x_vel / sum_sqrt_density
        y_velocity_avg = sum_prod_y_vel / sum_sqrt_density
        enthalpy_avg = sum_prod_enthalpy / sum_sqrt_density

        w_avg.setdefault('density', [])
        w_avg.setdefault('x_momentum', [])
        w_avg.setdefault('y_momentum', [])
        w_avg.setdefault('energy', [])
        w_avg['density'].append( density_avg )
        w_avg['x_momentum'].append( density_avg * x_velocity_avg )
        w_avg['y_momentum'].append( density_avg * y_velocity_avg )
        w_avg['energy'].append(
          (enthalpy_avg 
            + 0.5 * (GAMMA - 1) * (x_velocity_avg**2 + y_velocity_avg**2)) \
              * density_avg / GAMMA
        )
    else:
      for i_axis, axis in enumerate([X_AXIS, Y_AXIS]):
        for key in self.evolving_keys:
          w_avg.setdefault(key, [])
          w_avg[key].append(
            0.5 * ( w[key][i_axis] + _roll_plus_one(w[key][i_axis], axis) )
          )

    return w_avg

  def project(
    self, L, f
  ):
    f_proj = {}
    for i_axis in range(2):
      f_all = tf.stack(
        [f['density'][i_axis],
        f['x_momentum'][i_axis],
        f['y_momentum'][i_axis],
        f['energy'][i_axis]]
      )
      for key in self.evolving_keys:
        f_proj.setdefault(key, [])
        f_proj[key].append( _sum(L[key][i_axis] * f_all, axis=0) )

    return f_proj

  def flux(
    self, grid, velocity, pressure, speed_of_sound, density, x_momentum, y_momentum, energy
  ):
    """Compute velocity fluxes for all evolving keys.
    
    Returns:
      dict of [X_FLUX_EDGE_X, Y_FLUX_EDGE_Y] lists.
    """
    
    # Store state
    w = {}
    w['density'] = [density, density]
    w['x_momentum'] = [x_momentum, x_momentum]
    w['y_momentum'] = [y_momentum, y_momentum]
    w['energy'] = [energy, energy]

    # Compute fluxes
    f = {}
    f['density'] = [x_momentum, y_momentum]
    f['x_momentum'] = [x_momentum**2 / density + pressure,
                       x_momentum * y_momentum / density]
    f['y_momentum'] = [x_momentum * y_momentum / density, 
                       y_momentum**2 / density + pressure]
    f['energy'] = [x_momentum * (energy + pressure) / density, 
                   y_momentum * (energy + pressure) / density]

    # Compute neighbors
    f_m2, f_m1, f_p1, f_p2, f_p3 = self.neighbors(f)
    w_m2, w_m1, w_p1, w_p2, w_p3 = self.neighbors(w)

    # Compute the average state
    w_avg = self.average_state(pressure, w)

    # Compute the average eigensystem
    R, L = self.average_eigensystem(grid, w_avg)

    # Project in the characteristic space
    f    = self.project(L, f)
    f_m2 = self.project(L, f_m2)
    f_m1 = self.project(L, f_m1)
    f_p1 = self.project(L, f_p1)
    f_p2 = self.project(L, f_p2)
    f_p3 = self.project(L, f_p3)
    w    = self.project(L, w)
    w_m2 = self.project(L, w_m2)
    w_m1 = self.project(L, w_m1)
    w_p1 = self.project(L, w_p1)
    w_p2 = self.project(L, w_p2)
    w_p3 = self.project(L, w_p3)

    # Global Lax-Friedrichs flux splitting in the characteristic space
    eigenvalues = {}
    eigenvalues['density'] = \
      [velocity[0] - speed_of_sound, velocity[1] - speed_of_sound]
    eigenvalues['x_momentum'] = \
      [velocity[0], velocity[1]]
    eigenvalues['y_momentum'] = \
      [velocity[0] + speed_of_sound, velocity[1] + speed_of_sound]
    eigenvalues['energy'] = \
      [velocity[0], velocity[1]]
    f_left, f_right = \
      self.flux_splitting(grid, f, eigenvalues, w)
    f_m2_left, f_m2_right = \
      self.flux_splitting(grid, f_m2, eigenvalues, w_m2)
    f_m1_left, f_m1_right = \
      self.flux_splitting(grid, f_m1, eigenvalues, w_m1)
    f_p1_left, f_p1_right = \
      self.flux_splitting(grid, f_p1, eigenvalues, w_p1)
    f_p2_left, f_p2_right = \
      self.flux_splitting(grid, f_p2, eigenvalues, w_p2)
    f_p3_left, f_p3_right = \
      self.flux_splitting(grid, f_p3, eigenvalues, w_p3)
    
    # Reconstruct left- and right-biased fluxes in the characteristic space
    flux_left, flux_right = self.reconstruction(grid, 
                                                f_left, f_right,
                                                f_m2_left, f_m2_right, 
                                                f_m1_left, f_m1_right,
                                                f_p1_left, f_p1_right,
                                                f_p2_left, f_p2_right,
                                                f_p3_left, f_p3_right)

    # Project left- and right-biased fluxes back in the component space
    flux_left =  self.project(R, flux_left)
    flux_right = self.project(R, flux_right)

    # Sum left- and right-biased fluxes to recover total flux
    flux = {}
    for i_axis in range(2):
      for key in self.evolving_keys:
        flux.setdefault(key, [])
        flux[key].append( 
          flux_left[key][i_axis] + flux_right[key][i_axis] 
        )

    return flux

  def time_derivative(
    self, grid, density, x_momentum, y_momentum, energy
  ):
    velocity = [x_momentum / density, y_momentum / density]
    pressure = (GAMMA - 1) * (energy \
                              - density * (velocity[0]**2 + velocity[1]**2) / 2)
    speed_of_sound = _sqrt(GAMMA * pressure / density)

    fluxes = self.flux(grid, velocity, pressure, speed_of_sound, density, x_momentum, y_momentum, energy)
    
    grid_step = grid.step
    if self._no_dimensions:
      grid_step /= grid.length_y
    time_derivs = {}
    for key in self.evolving_keys:
      x_flux_edge_x = fluxes[key][0]
      y_flux_edge_y = fluxes[key][1]
      time_derivs[key] = \
        flux_to_time_derivative(x_flux_edge_x, y_flux_edge_y, grid_step)

    return time_derivs

  def take_time_step(
    self, grid, density, x_momentum, y_momentum, energy):
    
    dt = self.get_time_step(grid)
    w_dict_0 = {}
    w_dict_0['density'] = density
    w_dict_0['x_momentum'] = x_momentum
    w_dict_0['y_momentum'] = y_momentum
    w_dict_0['energy'] = energy
    
    w_dict = {}
    w_dict['density'] = density
    w_dict['x_momentum'] = x_momentum
    w_dict['y_momentum'] = y_momentum
    w_dict['energy'] = energy
    
    # SSP 3rd order Runge-Kutta (TVD-RK3)
    factor = tf.constant([[1, 0, 1], [3/4, 1/4, 1/4], [1/3, 2/3, 2/3]])
    for i in range(3):
      # print('\t Runge-Kutta step ' + str(i + 1) + ' ... \n')
      time_derivs = self.time_derivative(grid, **w_dict)
      for key in self.evolving_keys:
        w_dict[key] = factor[i,0] * w_dict_0[key] + factor[i,1] * w_dict[key] \
                      + factor[i,2] * dt * time_derivs[key]

    # Perform sanity check on solution
    for key in self.evolving_keys:
      if np.isfinite(w_dict[key]).all() == False:
        raise ValueError('\n\n NaN or +/- inf values appeared ! \n\n')

    return w_dict

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

class FiniteDifferenceEuler(_EulerBase):
  """Finite-difference scheme for Euler equations."""

  DISCRETIZATION_NAME = 'finite_difference'
  METHOD = polynomials.Method.FINITE_DIFFERENCE
  MONOTONIC = False

  def __init__(
    self, 
    cfl_safety_factor: float = 0.8,
    no_dimensions: bool = False,
    ):
    self.key_definitions = {
      'density': StateDef('density', (), NO_DERIVATIVES, NO_OFFSET),
      'x_momentum': StateDef('momentum', (X,), NO_DERIVATIVES, NO_OFFSET),
      'y_momentum': StateDef('momentum', (Y,), NO_DERIVATIVES, NO_OFFSET),
      'energy': StateDef('energy', (), NO_DERIVATIVES, NO_OFFSET),
      'density_x': StateDef('density', (), D_X, NO_OFFSET),
      'density_y': StateDef('density', (), D_Y, NO_OFFSET),
      'x_momentum_x': StateDef('momentum', (X,), D_X, NO_OFFSET),
      'x_momentum_y': StateDef('momentum', (X,), D_Y, NO_OFFSET),
      'y_momentum_x': StateDef('momentum', (Y,), D_X, NO_OFFSET),
      'y_momentum_y': StateDef('momentum', (Y,), D_Y, NO_OFFSET),
      'energy_x': StateDef('energy', (), D_X, NO_OFFSET),
      'energy_y': StateDef('energy', (), D_Y, NO_OFFSET),
    }
    self.evolving_keys = {'density', 'x_momentum', 'y_momentum', 'energy'}
    self.constant_keys = set()
    self._cfl_safety_factor = cfl_safety_factor
    self._no_dimensions = no_dimensions
    super().__init__()

  @property
  def cfl_safety_factor(self) -> float:
    return self._cfl_safety_factor
  
  @property
  def no_dimensions(self) -> bool:
    return self._no_dimensions

  def get_time_step(self, grid):
    dt = self._cfl_safety_factor * grid.step / SPEED_OF_SOUND_INFTY / (MACH_INFTY + 1.)
    if self._no_dimensions:
      dt /= grid.length_y / self.characteristic_velocity
    return dt

  def time_derivative(
    self, grid, 
    density, x_momentum, y_momentum, energy,
    density_x, x_momentum_x, y_momentum_x, energy_x,
    density_y, x_momentum_y, y_momentum_y, energy_y,
  ):
    del grid # unused

    pressure = \
      (GAMMA - 1) * (energy - (x_momentum ** 2 + y_momentum ** 2) / 2 / density)

    pressure_x = \
      (GAMMA - 1) * \
        (energy_x 
          - (x_momentum * x_momentum_x + y_momentum * y_momentum_x) / density
          + density_x * (x_momentum ** 2 + y_momentum ** 2) / 2 / density ** 2
        )

    pressure_y = \
      (GAMMA - 1) * \
        (energy_y 
          - (x_momentum * x_momentum_y + y_momentum * y_momentum_y) / density
          + density_y * (x_momentum ** 2 + y_momentum ** 2) / 2 / density ** 2
        )

    density_t = -(x_momentum_x + y_momentum_y)
    x_momentum_t = -(x_momentum_x * 2 * x_momentum / density 
                      - density_x * x_momentum ** 2 / density ** 2
                      + pressure_x
                      + x_momentum_y * y_momentum / density
                      + y_momentum_y * x_momentum / density
                      - density_y * x_momentum * y_momentum / density ** 2
                    )
    y_momentum_t = -(y_momentum_y * 2 * y_momentum / density 
                      - density_y * y_momentum ** 2 / density ** 2
                      + pressure_y
                      + x_momentum_x * y_momentum / density
                      + y_momentum_x * x_momentum / density
                      - density_x * x_momentum * y_momentum / density ** 2
                    )
    energy_t = -(x_momentum_x * (energy + pressure) / density
                  + (energy_x + pressure_x) * x_momentum / density
                  - density_x * x_momentum * (energy + pressure) / density ** 2
                  + y_momentum_y * (energy + pressure) / density
                  + (energy_y + pressure_y) * y_momentum / density
                  - density_y * y_momentum * (energy + pressure) / density ** 2
                )

    return {'density': density_t, 'x_momentum': x_momentum_t, 'y_momentum': y_momentum_t, 'energy': energy_t, }
  
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

class FiniteVolumeEuler(_EulerBase):
  """Finite-volume scheme for Euler equations."""

  DISCRETIZATION_NAME = 'finite_volume'
  METHOD = polynomials.Method.FINITE_VOLUME
  MONOTONIC = False

  def __init__(
    self, 
    cfl_safety_factor: float = 0.8,
    no_dimensions: bool = False,
    ):
    self.key_definitions = {
      'density': StateDef('density', (), NO_DERIVATIVES, NO_OFFSET),
      'x_momentum': StateDef('momentum', (X,), NO_DERIVATIVES, NO_OFFSET),
      'y_momentum': StateDef('momentum', (Y,), NO_DERIVATIVES, NO_OFFSET),
      'energy': StateDef('energy', (), NO_DERIVATIVES, NO_OFFSET),
      'density_edge_x': StateDef('density', (), NO_DERIVATIVES, X_PLUS_HALF),
      'density_edge_y': StateDef('density', (), NO_DERIVATIVES, Y_PLUS_HALF),
      'x_momentum_edge_x': StateDef('momentum', (X,), NO_DERIVATIVES, X_PLUS_HALF),
      'x_momentum_edge_y': StateDef('momentum', (X,), NO_DERIVATIVES, Y_PLUS_HALF),
      'y_momentum_edge_x': StateDef('momentum', (Y,), NO_DERIVATIVES, X_PLUS_HALF),
      'y_momentum_edge_y': StateDef('momentum', (Y,), NO_DERIVATIVES, Y_PLUS_HALF),
      'energy_edge_x': StateDef('energy', (), NO_DERIVATIVES, X_PLUS_HALF),
      'energy_edge_y': StateDef('energy', (), NO_DERIVATIVES, Y_PLUS_HALF),
    }
    self.evolving_keys = {'density', 'x_momentum', 'y_momentum', 'energy'}
    self.constant_keys = set()
    self._cfl_safety_factor = cfl_safety_factor
    self._no_dimensions = no_dimensions
    self.runge_kutta_step = 0
    self.last_integer_state = None
    super().__init__()

  @property
  def cfl_safety_factor(self) -> float:
    return self._cfl_safety_factor
  
  @property
  def no_dimensions(self) -> bool:
    return self._no_dimensions

  def get_runge_kutta_step(self) -> bool:
    return self.runge_kutta_step

  def set_runge_kutta_step(self, rk_step: int):
    self.runge_kutta_step = rk_step
  
  def set_last_integer_state(self, state):
    self.last_integer_state = state
  
  def get_time_step(self, grid):
    dt = self._cfl_safety_factor * grid.step / SPEED_OF_SOUND_INFTY / (MACH_INFTY + 1.)
    if self._no_dimensions:
      dt /= grid.length_y / self.characteristic_velocity
    return dt

  def time_derivative(
    self, grid, 
    density, x_momentum, y_momentum, energy,
    density_edge_x, x_momentum_edge_x, y_momentum_edge_x, energy_edge_x,
    density_edge_y, x_momentum_edge_y, y_momentum_edge_y, energy_edge_y,
  ):
    del density, x_momentum, y_momentum, energy  # unused

    pressure_edge_x = \
      (GAMMA - 1) \
        * (energy_edge_x - 
        (x_momentum_edge_x ** 2 + y_momentum_edge_x ** 2) / 2 / density_edge_x)
    
    pressure_edge_y = \
      (GAMMA - 1) \
        * (energy_edge_y - 
        (x_momentum_edge_y ** 2 + y_momentum_edge_y ** 2) / 2 / density_edge_y)

    # Compute fluxes
    f = {}
    f['density'] = \
      [x_momentum_edge_x, y_momentum_edge_y]
    f['x_momentum'] = \
      [x_momentum_edge_x ** 2 / density_edge_x + pressure_edge_x,
       x_momentum_edge_y * y_momentum_edge_y / density_edge_y]
    f['y_momentum'] = \
      [x_momentum_edge_x * y_momentum_edge_x / density_edge_x, 
       y_momentum_edge_y ** 2 / density_edge_y + pressure_edge_y]
    f['energy'] = \
      [x_momentum_edge_x * (energy_edge_x + pressure_edge_x) / density_edge_x, 
       y_momentum_edge_y * (energy_edge_y + pressure_edge_y) / density_edge_y]

    grid_step = grid.step
    if self._no_dimensions:
      grid_step /= grid.length_y
    time_derivs = {}
    for key in self.evolving_keys:
      x_flux_edge_x = f[key][0]
      y_flux_edge_y = f[key][1]
      time_derivs[key] = \
        flux_to_time_derivative(x_flux_edge_x, y_flux_edge_y, grid_step)

    return time_derivs

  def take_time_step(
    self, grid, density, x_momentum, y_momentum, energy,
    density_edge_x, x_momentum_edge_x, y_momentum_edge_x, energy_edge_x,
    density_edge_y, x_momentum_edge_y, y_momentum_edge_y, energy_edge_y,
  ):
    
    dt = self.get_time_step(grid)
    
    w_dict = {}
    w_dict['density'] = density
    w_dict['x_momentum'] = x_momentum
    w_dict['y_momentum'] = y_momentum
    w_dict['energy'] = energy
    
    w_dict_derivs = {}
    w_dict_derivs['density_edge_x'] = density_edge_x
    w_dict_derivs['density_edge_y'] = density_edge_y
    w_dict_derivs['x_momentum_edge_x'] = x_momentum_edge_x
    w_dict_derivs['x_momentum_edge_y'] = x_momentum_edge_y
    w_dict_derivs['y_momentum_edge_x'] = y_momentum_edge_x
    w_dict_derivs['y_momentum_edge_y'] = y_momentum_edge_y
    w_dict_derivs['energy_edge_x'] = energy_edge_x
    w_dict_derivs['energy_edge_y'] = energy_edge_y

    # SSP 3rd order Runge-Kutta (TVD-RK3)
    i = self.runge_kutta_step
    time_derivs = self.time_derivative(grid, **{**w_dict, **w_dict_derivs})
    for key in self.evolving_keys:
      w_dict[key] = RUNGE_KUTTA_FACTOR[i][0] * self.last_integer_state[key] \
                    + RUNGE_KUTTA_FACTOR[i][1] * w_dict[key] \
                    + RUNGE_KUTTA_FACTOR[i][2] * dt * time_derivs[key]

    return w_dict
