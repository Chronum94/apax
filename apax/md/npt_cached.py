"""jax-md's NPT Nose-Hoover integrator, with dU/dV carried in the state.

Upstream ``inner_step`` starts with a full energy+gradient evaluation (stress_fn) at the
positions and box the previous step already evaluated in force_stress_fn, so every step
runs the model twice. Here that dU/dV is stored in the state and reused: one model
evaluation per step, same trajectory.
Copied from jax_md 0.2.29 ``simulate.npt_nose_hoover``; only the dU/dV caching differs.
"""
from typing import Callable, Dict, Tuple

import jax.numpy as jnp
from jax import value_and_grad
from jax_md import dataclasses, quantity, util
from jax_md.simulate import (
    Array,
    NPTNoseHooverState,
    ShiftFn,
    Simulator,
    _npt_box_info,
    canonicalize_mass,
    canonicalize_momenta,
    default_nhc_kwargs,
    initialize_momenta,
    kinetic_energy,

)

from apax.md.nhc_fused import nose_hoover_chain_fused

f32 = util.f32


@dataclasses.dataclass
class NPTCachedState(NPTNoseHooverState):
  """NPTNoseHooverState plus dU/dV at the current positions and box."""

  dUdV: Array


def npt_nose_hoover_cached(
  energy_fn: Callable[..., Array],
  shift_fn: ShiftFn,
  dt: float,
  pressure: float,
  kT: float,
  barostat_kwargs: Dict | None = None,
  thermostat_kwargs: Dict | None = None,
) -> Simulator:
  """Simulation in the NPT ensemble using a pair of Nose Hoover Chains.

  Samples from the canonical ensemble in which the number of particles (N),
  the system pressure (P), and the temperature (T) are held constant.
  We use a pair of Nose Hoover Chains (NHC) described in
  [#martyna92]_ [#martyna98]_ [#tuckerman]_ coupled to the
  barostat and the thermostat respectively. We follow the direct translation
  method outlined in Tuckerman et al. [#tuckerman]_ and the interested reader
  might want to look at that paper as a reference.

  Args:
    energy_fn: A function that produces either an energy from a set of particle
      positions specified as an ndarray of shape `[n, spatial_dimension]`.
    shift_fn: A function that displaces positions, `R`, by an amount `dR`. Both
      `R` and `dR` should be ndarrays of shape `[n, spatial_dimension]`.
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
    pressure: Floating point number specifying the target pressure. To update
      the pressure dynamically during a simulation one should pass `pressure`
      as a keyword argument to the step function.
    kT: Floating point number specifying the temperature in units of Boltzmann
      constant. To update the temperature dynamically during a simulation one
      should pass `kT` as a keyword argument to the step function.
    barostat_kwargs: A dictionary of keyword arguments passed to the barostat
      NHC. Any parameters not set are drawn from a relatively robust default
      set.
    thermostat_kwargs: A dictionary of keyword arguments passed to the
      thermostat NHC. Any parameters not set are drawn from a relatively robust
      default set.

  Returns:
    See above.

  """

  t = f32(dt)
  dt_2 = f32(dt / 2)

  force_fn = quantity.force(energy_fn)

  barostat_kwargs = default_nhc_kwargs(1000 * dt, barostat_kwargs)
  barostat = nose_hoover_chain_fused(dt, **barostat_kwargs)

  thermostat_kwargs = default_nhc_kwargs(100 * dt, thermostat_kwargs)
  thermostat = nose_hoover_chain_fused(dt, **thermostat_kwargs)

  def init_fn(key, R, box, mass=f32(1.0), momenta=None, **kwargs):
    N, dim = R.shape

    _kT = kT if 'kT' not in kwargs else kwargs['kT']

    # The box position is defined via pos = (1 / d) log V / V_0.
    zero = jnp.zeros((), dtype=R.dtype)
    one = jnp.ones((), dtype=R.dtype)
    box_position = zero
    box_momentum = zero
    box_mass = dim * (N + 1) * kT * barostat_kwargs['tau'] ** 2 * one
    KE_box = quantity.kinetic_energy(momentum=box_momentum, mass=box_mass)

    if jnp.isscalar(box) or box.ndim == 0:
      # TODO(schsam): This is necessary because of JAX issue #5849.
      box = jnp.eye(R.shape[-1]) * box

    _, F, dUdV = force_stress_fn(R, box, **kwargs)
    state = NPTCachedState(
      R,
      None,
      F,
      mass,
      box,
      box_position,
      box_momentum,
      box_mass,
      barostat.initialize(1, KE_box, _kT),
      None,
      dUdV,
    )  # pytype: disable=wrong-arg-count
    state = canonicalize_mass(state)
    if momenta is None:
      state = initialize_momenta(state, key, _kT)
    else:
      state = canonicalize_momenta(state, momenta)
    KE = kinetic_energy(state)
    return state.set(
      thermostat=thermostat.initialize(quantity.count_dof(R), KE, _kT)
    )

  def update_box_mass(state, kT):
    N, dim = state.position.shape
    dtype = state.position.dtype
    box_mass = jnp.array(dim * (N + 1) * kT * state.barostat.tau**2, dtype)
    return state.set(box_mass=box_mass)

  def stress_fn(position, box, **kwargs) -> Tuple[Array, Array]:
    def U(eps):
      return energy_fn(position, box=box, perturbation=(1 + eps), **kwargs)

    E, dUdV = value_and_grad(U)(0.0)
    return E, dUdV

  def force_stress_fn(position, box, **kwargs) -> Tuple[Array, Array, Array]:
    def U(position, eps):
      return energy_fn(position, box=box, perturbation=(1 + eps), **kwargs)

    E, grad = value_and_grad(U, argnums=(0, 1))(position, 0.0)
    F = -grad[0]
    dUdV = grad[1]
    return E, F, dUdV

  def box_force(
    alpha,
    vol,
    dUdV,
    position,
    momentum,
    mass,
    pressure,
  ) -> Array:
    N, dim = position.shape

    KE2 = util.high_precision_sum(momentum**2 / mass)

    return alpha * KE2 - dUdV - pressure * vol * dim

  def sinhx_x(x):
    """Taylor series for sinh(x) / x as x -> 0."""
    return (
      1
      + x**2 / 6
      + x**4 / 120
      + x**6 / 5040
      + x**8 / 362_880
      + x**10 / 39_916_800
    )

  def exp_iL1(box, R, V, V_b, **kwargs):
    x = V_b * dt
    x_2 = x / 2
    sinhV = sinhx_x(x_2)  # jnp.sinh(x_2) / x_2
    return shift_fn(
      R, R * (jnp.exp(x) - 1) + dt * V * jnp.exp(x_2) * sinhV, box=box, **kwargs
    )  # pytype: disable=wrong-keyword-args

  def exp_iL2(alpha, P, F, V_b):
    x = alpha * V_b * dt_2
    x_2 = x / 2
    sinhP = sinhx_x(x_2)  # jnp.sinh(x_2) / x_2
    return P * jnp.exp(-x) + dt_2 * F * sinhP * jnp.exp(-x_2)

  def inner_step(state, **kwargs):
    _pressure = kwargs.pop('pressure', pressure)

    R, P, M, F = state.position, state.momentum, state.mass, state.force
    R_b, P_b, M_b = state.box_position, state.box_momentum, state.box_mass

    N, dim = R.shape

    vol, box_fn = _npt_box_info(state)

    alpha = 1 + 1 / N
    # dU/dV at (R, box) was computed by the previous step's force_stress_fn
    # (or init_fn); nothing between steps moves R or the box.
    dUdV = state.dUdV
    G_e = box_force(alpha, vol, dUdV, R, P, M, _pressure)
    P_b = P_b + dt_2 * G_e
    P = exp_iL2(alpha, P, F, P_b / M_b)

    R_b = R_b + P_b / M_b * dt
    state = state.set(box_position=R_b)

    vol, box_fn = _npt_box_info(state)

    box = box_fn(vol)
    R = exp_iL1(box, R, P / M, P_b / M_b)
    E, F, dUdV = force_stress_fn(R, box, **kwargs)
    P = exp_iL2(alpha, P, F, P_b / M_b)
    G_e = box_force(alpha, vol, dUdV, R, P, M, _pressure)
    P_b = P_b + dt_2 * G_e

    return state.set(
      position=R,
      momentum=P,
      mass=M,
      force=F,
      box_position=R_b,
      box_momentum=P_b,
      box_mass=M_b,
      dUdV=dUdV,
    )

  def apply_fn(state, **kwargs):
    S = state
    _kT = kT if 'kT' not in kwargs else kwargs['kT']

    bc = barostat.update_mass(S.barostat, _kT)
    tc = thermostat.update_mass(S.thermostat, _kT)
    S = update_box_mass(S, _kT)

    P_b, bc = barostat.half_step(S.box_momentum, bc, _kT)
    P, tc = thermostat.half_step(S.momentum, tc, _kT)

    S = S.set(momentum=P, box_momentum=P_b)
    S = inner_step(S, **kwargs)

    KE = quantity.kinetic_energy(momentum=S.momentum, mass=S.mass)
    tc = tc.set(kinetic_energy=KE)

    KE_box = quantity.kinetic_energy(momentum=S.box_momentum, mass=S.box_mass)
    bc = bc.set(kinetic_energy=KE_box)

    P, tc = thermostat.half_step(S.momentum, tc, _kT)
    P_b, bc = barostat.half_step(S.box_momentum, bc, _kT)

    S = S.set(thermostat=tc, barostat=bc, momentum=P, box_momentum=P_b)

    return S

  return init_fn, apply_fn
