"""jax-md's NVT Nose-Hoover integrator using the fused chain from ``nhc_fused``.

Copied from jax_md 0.2.29 ``simulate.nvt_nose_hoover``; only the chain differs.
"""
from typing import Callable

from jax import jit
from jax_md import quantity, util
from jax_md.simulate import (
    Array,
    NVTNoseHooverState,
    ShiftFn,
    Simulator,
    canonicalize_mass,
    canonicalize_momenta,
    initialize_momenta,
    kinetic_energy,
    velocity_verlet,
)

from apax.md.nhc_fused import nose_hoover_chain_fused

f32 = util.f32


def nvt_nose_hoover_fused(
  energy_or_force_fn: Callable[..., Array],
  shift_fn: ShiftFn,
  dt: float,
  kT: float,
  chain_length: int = 5,
  chain_steps: int = 2,
  sy_steps: int = 3,
  tau: float | None = None,
  **sim_kwargs,
) -> Simulator:
  """Simulation in the NVT ensemble using a Nose Hoover Chain thermostat.

  Samples from the canonical ensemble in which the number of particles (N),
  the system volume (V), and the temperature (T) are held constant. We use a
  Nose Hoover Chain (NHC) thermostat described in [#martyna92]_ [#martyna98]_
  [#tuckerman]_. We follow the direct translation method outlined in
  Tuckerman et al. [#tuckerman]_ and the interested reader might want to look
  at that paper as a reference.

  Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      `[n, spatial_dimension]`.
    shift_fn: A function that displaces positions, `R`, by an amount `dR`.
      Both `R` and `dR` should be ndarrays of shape `[n, spatial_dimension]`.
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
    kT: Floating point number specifying the temperature in units of Boltzmann
      constant. To update the temperature dynamically during a simulation one
      should pass `kT` as a keyword argument to the step function.
    chain_length: An integer specifying the number of particles in
      the Nose-Hoover chain.
    chain_steps: An integer specifying the number, :math:`n_c`, of outer
      substeps.
    sy_steps: An integer specifying the number of Suzuki-Yoshida steps. This
      must be either `1`, `3`, `5`, or `7`.
    tau: A floating point timescale over which temperature equilibration
      occurs. Measured in units of `dt`. The performance of the Nose-Hoover
      chain thermostat can be quite sensitive to this choice.
  Returns:
    See above.

  .. rubric:: References
  .. [#martyna92] Martyna, Glenn J., Michael L. Klein, and Mark Tuckerman.
    "Nose-Hoover chains: The canonical ensemble via continuous dynamics."
    The Journal of chemical physics 97, no. 4 (1992): 2635-2643.
  .. [#martyna98] Martyna, Glenn, Mark Tuckerman, Douglas J. Tobias, and Michael L. Klein.
    "Explicit reversible integrators for extended systems dynamics."
    Molecular Physics 87. (1998) 1117-1157.
  .. [#tuckerman] Tuckerman, Mark E., Jose Alejandre, Roberto Lopez-Rendon,
    Andrea L. Jochim, and Glenn J. Martyna.
    "A Liouville-operator derived measure-preserving integrator for molecular
    dynamics simulations in the isothermal-isobaric ensemble."
    Journal of Physics A: Mathematical and General 39, no. 19 (2006): 5629.
  """
  force_fn = quantity.canonicalize_force(energy_or_force_fn)
  dt = f32(dt)
  dt_2 = f32(dt / 2)
  if tau is None:
    tau = dt * 100
  tau = f32(tau)

  thermostat = nose_hoover_chain_fused(dt, chain_length, chain_steps, sy_steps, tau)

  @jit
  def init_fn(key, R, mass=f32(1.0), momenta=None, **kwargs):
    _kT = kT if 'kT' not in kwargs else kwargs['kT']

    dof = quantity.count_dof(R)

    state = NVTNoseHooverState(R, None, force_fn(R, **kwargs), mass, None)
    state = canonicalize_mass(state)
    if momenta is None:
      state = initialize_momenta(state, key, _kT)
    else:
      state = canonicalize_momenta(state, momenta)
    KE = kinetic_energy(state)
    return state.set(chain=thermostat.initialize(dof, KE, _kT))

  @jit
  def apply_fn(state, **kwargs):
    _kT = kT if 'kT' not in kwargs else kwargs['kT']

    chain = state.chain

    chain = thermostat.update_mass(chain, _kT)

    p, chain = thermostat.half_step(state.momentum, chain, _kT)
    state = state.set(momentum=p)

    state = velocity_verlet(force_fn, shift_fn, dt, state, **kwargs)

    chain = chain.set(kinetic_energy=kinetic_energy(state))

    p, chain = thermostat.half_step(state.momentum, chain, _kT)
    state = state.set(momentum=p, chain=chain)

    return state

  return init_fn, apply_fn
