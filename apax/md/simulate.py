import logging
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from ase import units
from ase.io import read
from jax.experimental import io_callback
from jax_md import partition, quantity, simulate, space
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm

from apax.config import Config, MDConfig, parse_config
from apax.config.md_config import Integrator
from apax.md.ase_calc import make_ensemble, maybe_vmap
from apax.md.bias import (
    BiasEnergies,
    apply_bias_auxiliary,
    apply_bias_energy,
)
from apax.md.constraints import Constraint, ConstraintBase
from apax.md.dynamics_checks import DynamicsCheckBase, DynamicsChecks
from apax.md.io import H5TrajHandler, TrajHandler, truncate_trajectory_to_checkpoint
from apax.md.md_checkpoint import load_md_state
from apax.md.replicas import (
    add_replica_axis,
    allocate_neighbors,
    make_neighbor_update,
    replica_slice,
    select_replicas,
    shard_replicas,
)
from apax.md.sim_utils import SimulationFunctions, System
from apax.train.checkpoints import (
    canonicalize_energy_model_parameters,
    restore_parameters,
)
from apax.train.run import setup_logging
from apax.utils.space import perpendicular_widths

log = logging.getLogger(__name__)


def create_energy_fn(
    model,
    params,
    numbers,
    n_models,
    shallow=False,
):
    def full_ensemble(params, R, Z, neighbor, box, offsets, perturbation=None, **kwargs):
        vmodel = jax.vmap(model, (0, None, None, None, None, None, None), 0)
        energies, _ = vmodel(params, R, Z, neighbor, box, offsets, perturbation)
        energy = jnp.mean(energies)
        return energy

    def shallow_ensemble(
        params, R, Z, neighbor, box, offsets, perturbation=None, **kwargs
    ):
        energies, _ = model(params, R, Z, neighbor, box, offsets, perturbation)
        energy = jnp.mean(energies)
        return energy

    def single_model(params, R, Z, neighbor, box, offsets, perturbation=None, **kwargs):
        energy, _ = model(params, R, Z, neighbor, box, offsets, perturbation)
        return energy

    if n_models > 1:
        if shallow:
            energy_fn = shallow_ensemble
        else:
            energy_fn = full_ensemble
    else:
        energy_fn = single_model

    energy_fn = partial(
        energy_fn,
        params,
        Z=numbers,
        offsets=jnp.array([0.0, 0.0, 0.0]),
    )

    return energy_fn


def check_minimum_image(box, r_max):
    """jax_md's periodic displacement uses the minimum image convention,
    which only finds every neighbor if the cutoff is below half of each
    perpendicular cell width.
    """
    half_width = np.min(perpendicular_widths(box)) / 2
    if r_max >= half_width:
        raise ValueError(
            f"Cutoff radius {r_max} is not smaller than half the smallest perpendicular "
            f"cell width ({half_width:.3f}). jax_md's minimum image convention would "
            "miss neighbors; use a larger supercell."
        )


def nbr_update_options_default(state):
    return {}


def nbr_update_options_npt(state):
    box = simulate.npt_box(state)
    return {"box": box}


def remove_constrained_dof(init_fn, n_constrained_dof):
    """jax_md sizes the thermostat chain for all 3N degrees of freedom.
    Constrained ones carry no kinetic energy, so they are removed here.
    The chain masses are recomputed from the corrected DOF at every step.
    """

    def wrapped_init_fn(*args, **kwargs):
        state = init_fn(*args, **kwargs)
        dof = state.chain.degrees_of_freedom - n_constrained_dof
        return state.set(chain=state.chain.set(degrees_of_freedom=dof))

    return wrapped_init_fn


def nvt_langevin_baoab(energy_fn, shift_fn, dt, kT, gamma, constrain_fn):
    """BAOAB Langevin built from jax_md's primitives.
    `jax_md.simulate.nvt_langevin` gives the O-step noise to constrained atoms,
    which then move in the second A step before the force call. Applying the
    constraints right after O keeps them in place.
    """
    init_fn, _ = simulate.nvt_langevin(energy_fn, shift_fn, dt, kT, gamma=gamma)
    force_fn = quantity.canonicalize_force(energy_fn)

    def apply_fn(state, **kwargs):
        _kT = kwargs.pop("kT", kT)
        dt_2 = dt / 2
        state = simulate.momentum_step(state, dt_2)
        state = simulate.position_step(state, shift_fn, dt_2, **kwargs)
        state = simulate.stochastic_step(state, dt, _kT, gamma)
        state = constrain_fn(state)
        state = simulate.position_step(state, shift_fn, dt_2, **kwargs)
        state = state.set(force=force_fn(state.position, **kwargs))
        return simulate.momentum_step(state, dt_2)

    return init_fn, apply_fn


def get_ensemble(
    ensemble: Integrator, sim_fns, n_constrained_dof=0, constrain_fn=lambda state: state
):
    energy, shift = sim_fns.energy_fn, sim_fns.shift_fn

    dt = ensemble.dt * units.fs
    nbr_options = nbr_update_options_default

    kT = ensemble.temperature_schedule.get_schedule()
    if ensemble.name == "nve":
        init_fn, apply_fn = simulate.nve(energy, shift, dt)
        init_fn = partial(init_fn, kT=kT(0))
    elif ensemble.name == "nvt":
        thermostat_chain = dict(ensemble.thermostat_chain)
        thermostat_chain["tau"] *= dt

        init_fn, apply_fn = simulate.nvt_nose_hoover(
            energy,
            shift,
            dt,
            kT(0),
            **thermostat_chain,
        )
        if n_constrained_dof:
            init_fn = remove_constrained_dof(init_fn, n_constrained_dof)

    elif ensemble.name == "langevin":
        init_fn, apply_fn = nvt_langevin_baoab(
            energy, shift, dt, kT(0), ensemble.friction / units.fs, constrain_fn
        )

    elif ensemble.name == "npt":
        if n_constrained_dof:
            raise NotImplementedError(
                "Constraining atoms in NPT simulations is not implemented."
            )
        pressure = ensemble.pressure * units.bar
        thermostat_chain = dict(ensemble.thermostat_chain)
        barostat_chain = dict(ensemble.barostat_chain)
        thermostat_chain["tau"] *= dt
        barostat_chain["tau"] *= dt

        init_fn, apply_fn = simulate.npt_nose_hoover(
            energy,
            shift,
            dt,
            pressure,
            kT(0),
            thermostat_kwargs=thermostat_chain,
            barostat_kwargs=barostat_chain,
        )
        nbr_options = nbr_update_options_npt
    else:
        raise NotImplementedError(
            "Only NVE, Langevin NVT and Nose Hoover NVT/NPT are currently interfaced."
        )

    return init_fn, apply_fn, kT, nbr_options


def handle_checkpoints(state, step, system, load_momenta, ckpt_dir, should_load_ckpt):
    if load_momenta and not should_load_ckpt:
        log.info("loading momenta from starting configuration")
        momenta = jnp.broadcast_to(system.momenta, state.momentum.shape)
        state = state.set(momentum=momenta)

    elif should_load_ckpt:
        state, step = load_md_state(state, ckpt_dir.resolve())
    return state, step


def create_evaluation_functions(traj_handlers, aux_fn, Z, dynamics_checks):
    """Evaluate every replica and hand each its own slice of the results."""

    def evaluate(state, neighbor, box):
        # sized per call: a reallocated neighbor list has a different capacity
        offsets = jnp.zeros((neighbor.idx.shape[1], 3))
        positions = state.position
        predictions = aux_fn(positions, Z, neighbor, box, offsets)
        all_checks_passed = jnp.array(True)

        for check in dynamics_checks:
            check_passed = check.check(predictions, positions, box)
            all_checks_passed = all_checks_passed & check_passed

        return predictions, all_checks_passed

    def write(samples):
        for k, handler in enumerate(traj_handlers):
            handler.step(replica_slice(samples, k))

    def on_eval(state, neighbor, box, nbr_kwargs):
        predictions, all_checks_passed = jax.vmap(evaluate)(state, neighbor, box)
        io_callback(write, None, (state, predictions, nbr_kwargs))
        return all_checks_passed

    def no_eval(state, neighbor, box, nbr_kwargs):
        return jnp.ones(len(traj_handlers), dtype=bool)

    return on_eval, no_eval


def check_unique_idxs(constraind_idxs):
    unique_idxs = []
    seen_idxs = set()

    for idxs in constraind_idxs:
        for val in idxs:
            val = int(val)
            if val not in seen_idxs:
                seen_idxs.add(val)
                unique_idxs.append(val)

    return unique_idxs


def create_constraint_function(constraints: list[ConstraintBase], system):
    constrain_fns = []
    constraind_idxs = []

    for constraint in constraints:
        constrain_fn, idx = constraint.create(system)
        constrain_fns.append(constrain_fn)
        constraind_idxs.append(idx)

    if constraind_idxs:
        constraind_idxs = check_unique_idxs(constraind_idxs)

    def apply_constraints(state):
        for fn in constrain_fns:
            state = fn(state)

        return state

    return apply_constraints, constraind_idxs


def check_for_nans(state, step):
    position, velocity = np.asarray(state.position), np.asarray(state.velocity)
    has_nan = np.isnan(position).any(axis=(1, 2)) | np.isnan(velocity).any(axis=(1, 2))
    if np.any(has_nan):
        replicas = np.flatnonzero(has_nan).tolist()
        raise ValueError(
            f"NaN encountered in replica(s) {replicas}, "
            f"simulation aborted after {step + 1} steps."
        )


def needs_reallocation(neighbor):
    # A shrinking NPT box makes the cells smaller than the cutoff,
    # which silently drops neighbors without overflowing any buffer.
    return bool(np.any(neighbor.did_buffer_overflow)) or bool(
        np.any(neighbor.cell_size_too_small)
    )


def neighbor_kwargs(state, nbr_options, is_npt):
    return jax.vmap(nbr_options)(state) if is_npt else {}


def handle_overflow(
    neighbor_fn, state, nbr_options, is_npt, extra_capacity, fractional, step
):
    with logging_redirect_tqdm():
        log.warning("step %d: neighbor list invalid, reallocating.", step)
    nbr_kwargs = neighbor_kwargs(state, nbr_options, is_npt)
    neighbors = allocate_neighbors(
        neighbor_fn, state.position, nbr_kwargs, extra_capacity, fractional
    )
    return shard_replicas(neighbors, len(state.position))


def maybe_save_checkpoint(mngr, state, step, checkpoint_interval, sim_time_per_step):
    if step % checkpoint_interval == 0:
        with logging_redirect_tqdm():
            log.info(
                "saving checkpoint at %.1f ps - step: %d",
                step * sim_time_per_step,
                step,
            )
        mngr.save(step, args=ocp.args.StandardSave({"state": state, "step": step}))


def maybe_update_pbar(
    sim_pbar, step, pbar_update_freq, pbar_increment, current_temperature
):
    if step % pbar_update_freq == 0:
        sim_pbar.set_postfix(T=f"{current_temperature:.1f} K")
        sim_pbar.update(pbar_increment)


def run_sim(
    system: System,
    sim_fns: SimulationFunctions,
    ensemble,
    sim_dir: Path,
    n_steps: int,
    n_inner: int,
    extra_capacity: int,
    rng_key: int,
    traj_handler: TrajHandler | list[TrajHandler],
    sampling_rate: int = 10,
    load_momenta: bool = False,
    restart: bool = True,
    checkpoint_interval: int = 50_000,
    dynamics_checks: list[DynamicsCheckBase] = [],
    constraints: list[ConstraintBase] = [],
    disable_pbar: bool = False,
    n_replicas: int = 1,
):
    """
    Performs MD on `n_replicas` independent copies of `system`.

    Parameters
    ----------
    ensemble :
        Thermodynamic ensemble.
    n_steps : int
        Total time steps.
    n_inner : int
        JIT compiled inner loop. Also determines atoms buffer size.
    extra_capacity : int
        Extra capacity for the neighborlist.
    rng_key : int
        RNG key used to initialize the simulation. Replicas get keys split
        from it; a single replica uses it as is.
    traj_handler : TrajHandler | list[TrajHandler]
        One trajectory handler per replica.
    restart : bool, default = True
        Whether a checkpoint should be loaded. No implemented yet.
    checkpoint_interval : int, default = 50_000
        Number of time steps between saving
        full simulation state checkpoints.
    sim_dir : Path
        Directory where the trajectory and simulation checkpoints will be saved.
    n_replicas : int, default = 1
        Number of independent replicas, run together in one jitted loop.
    """
    traj_handlers = traj_handler if isinstance(traj_handler, list) else [traj_handler]
    if len(traj_handlers) != n_replicas:
        raise ValueError(
            f"Got {len(traj_handlers)} trajectory handlers for {n_replicas} replicas."
        )
    if load_momenta and n_replicas > 1 and ensemble.name != "langevin":
        raise ValueError(
            "Replicas loading the same momenta in a deterministic ensemble "
            "would all follow the same trajectory."
        )
    fractional = bool(np.any(system.box > 1e-6))
    neighbor_fn = sim_fns.neighbor_fn
    ckpt_dir = sim_dir / "ckpts"
    ckpt_dir.mkdir(exist_ok=True)

    apply_constraints, constrained_idxs = create_constraint_function(
        constraints,
        system,
    )

    log.info("initializing simulation")
    # Each constrained index stands for 3 removed degrees of freedom.
    n_constrained_dof = 3 * len(constrained_idxs)
    n_dof = system.positions.size - n_constrained_dof
    init_fn, apply_fn, kT, nbr_options = get_ensemble(
        ensemble, sim_fns, n_constrained_dof, apply_constraints
    )

    positions = jnp.broadcast_to(system.positions, (n_replicas, *system.positions.shape))
    neighbor = allocate_neighbors(neighbor_fn, positions, {}, extra_capacity, fractional)

    def init_replica(key, neighbor):
        return init_fn(
            key, system.positions, box=system.box, mass=system.masses, neighbor=neighbor
        )

    keys = rng_key[None] if n_replicas == 1 else jax.random.split(rng_key, n_replicas)
    state = jax.vmap(init_replica)(keys, neighbor)

    # Checkpoints of a single replica keep the unbatched layout.
    def to_checkpoint(state):
        return replica_slice(state, 0) if n_replicas == 1 else state

    def from_checkpoint(state):
        return add_replica_axis(state) if n_replicas == 1 else state

    step = 0

    options = ocp.CheckpointManagerOptions(max_to_keep=1, save_interval_steps=1)
    mngr = ocp.CheckpointManager(ckpt_dir.resolve(), options=options)

    ckpts_exist = mngr.latest_step() is not None

    should_load_ckpt = restart and ckpts_exist
    state, step = handle_checkpoints(
        to_checkpoint(state), step, system, load_momenta, ckpt_dir, should_load_ckpt
    )
    state = from_checkpoint(state)
    if should_load_ckpt:
        length = step * n_inner
        for handler in traj_handlers:
            truncate_trajectory_to_checkpoint(handler.traj_path, length)

    initial_step = step  # used for measuring time correctly
    is_npt = isinstance(state, simulate.NPTNoseHooverState)
    state, neighbor = shard_replicas((state, neighbor), n_replicas)
    update_neighbors = make_neighbor_update(
        n_replicas, fractional, sim_fns.dr_threshold, sim_fns.r_max
    )
    boxes = jnp.broadcast_to(system.box, (n_replicas, *jnp.shape(system.box)))

    n_outer = int(np.ceil(n_steps / n_inner))
    pbar_update_freq = int(np.ceil(500 / n_inner))
    pbar_increment = n_inner * pbar_update_freq

    on_eval, no_eval = create_evaluation_functions(
        traj_handlers,
        sim_fns.auxiliary_fn,
        system.atomic_numbers,
        dynamics_checks,
    )

    def advance(state, neighbor, kT):
        apply_fn_kwargs = {"kT": kT}
        if not is_npt:
            apply_fn_kwargs["box"] = system.box
        state = apply_fn(state, neighbor=neighbor, **apply_fn_kwargs)
        return apply_constraints(state)

    def temperature(state):
        kinetic_energy = quantity.kinetic_energy(velocity=state.velocity, mass=state.mass)
        return 2 * kinetic_energy / n_dof / units.kB

    @jax.jit
    def sim(state, outer_step, neighbor):  # TODO make more modular
        def body_fn(i, state):
            state, outer_step, neighbor, all_checks_passed = state
            step = i + outer_step * n_inner

            state = jax.vmap(advance, (0, 0, None))(state, neighbor, kT(step))
            box = jax.vmap(lambda s: s.box)(state) if is_npt else boxes

            nbr_kwargs = neighbor_kwargs(state, nbr_options, is_npt)
            nbr_boxes = nbr_kwargs["box"] if is_npt else boxes
            neighbor = update_neighbors(neighbor, state.position, nbr_kwargs, nbr_boxes)

            condition = step % sampling_rate == 0
            checks_passed = jax.lax.cond(
                condition, on_eval, no_eval, state, neighbor, box, nbr_kwargs
            )

            all_checks_passed = all_checks_passed & checks_passed
            return state, outer_step, neighbor, all_checks_passed

        all_checks_passed = jnp.ones(n_replicas, dtype=bool)
        state, outer_step, neighbor, all_checks_passed = jax.lax.fori_loop(
            0, n_inner, body_fn, (state, outer_step, neighbor, all_checks_passed)
        )
        current_temperature = jax.vmap(temperature)(state)

        return state, neighbor, current_temperature, all_checks_passed

    start = time.time()
    total_sim_time = n_steps * ensemble.dt / 1000
    log.info("running simulation for %.1f ps", total_sim_time)
    initial_time = step * n_inner
    sim_pbar = trange(
        initial_time,
        n_steps,
        initial=initial_time,
        total=n_steps,
        desc="Simulation",
        ncols=100,
        disable=disable_pbar,
        leave=True,
    )
    sim_time_per_step = n_inner * ensemble.dt / 1000
    with mngr:
        while step < n_outer:
            new_state, neighbor, current_temperature, all_checks_passed = sim(
                state, step, neighbor
            )

            if needs_reallocation(neighbor):
                # Size the list for where the chunk ended up (densest
                # configuration / smallest NPT box), then redo the chunk.
                # Sizing it from `state` would overflow again on the retry.
                finite = np.isfinite(np.asarray(new_state.position)).all(axis=(1, 2))
                neighbor = handle_overflow(
                    neighbor_fn,
                    select_replicas(finite, new_state, state),
                    nbr_options,
                    is_npt,
                    extra_capacity,
                    fractional,
                    step,
                )
                for handler in traj_handlers:
                    handler.reset_buffer()
                continue

            check_for_nans(new_state, step)

            if not np.all(all_checks_passed):
                failed = np.flatnonzero(~np.asarray(all_checks_passed)).tolist()
                with logging_redirect_tqdm():
                    log.critical(
                        "One or more dynamics checks failed at step %d in replica(s) %s",
                        step + 1,
                        failed,
                    )
                break

            state = new_state
            step += 1
            if is_npt and sim_fns.r_max is not None:
                for box in np.asarray(jax.vmap(simulate.npt_box)(state)):
                    check_minimum_image(box, sim_fns.r_max)
            maybe_save_checkpoint(
                mngr, to_checkpoint(state), step, checkpoint_interval, sim_time_per_step
            )
            maybe_update_pbar(
                sim_pbar,
                step,
                pbar_update_freq,
                pbar_increment,
                float(np.mean(current_temperature)),
            )

        # In case of mismatch update freq and n_steps, we can set it to 100% manually
        sim_pbar.update(n_steps - sim_pbar.n)
        sim_pbar.close()

        ckpt = {"state": to_checkpoint(state), "step": step}
        mngr.save(step, args=ocp.args.StandardSave(ckpt))

    for handler in traj_handlers:
        handler.write()
        handler.close()
    end = time.time()
    elapsed_wall_time = end - start
    elapsed_sim_time = (step - initial_step) * n_inner * ensemble.dt / 1000

    ps_per_s = elapsed_sim_time / elapsed_wall_time
    nanosec_per_day = ps_per_s / 1e3 * 60 * 60 * 24

    sec_per_step = elapsed_wall_time / n_steps
    n_atoms = system.positions.shape[0] * n_replicas
    musec_per_step_per_atom = sec_per_step * 1e6 / n_atoms

    log.info("simulation finished after: %.2f s", elapsed_wall_time)
    log.info(
        "performance summary: %.2f ns/day, %.2f mu s/step/atom",
        nanosec_per_day,
        musec_per_step_per_atom,
    )


def md_setup(model_config: Config, md_config: MDConfig):
    """
    Sets up the energy and neighborlist functions for an MD simulation,
    loads the initial structure.

    Parameters
    ----------
    model_config : Config
        Configuration of the model used as an interatomic potential.
    md_config : MDConfig
        configuration of the MD simulation.

    Returns
    -------
    R:
        Initial positions in Angstrom.
    atomic_numbers:
        Atomic numbers of the system.
    masses:
        Atomic masses in ASE units.
    box:
        Side length of the cubic box.
    energy_fn:
        Interatomic potential.
    neighbor_fn:
        Neighborlist function.
    shift_fn:
        Shift function for the integrator.
    """
    log.info("reading structure")
    atoms = read(md_config.initial_structure)
    system = System.from_atoms(atoms)

    r_max = model_config.model.basis.r_max
    log.info("initializing model")
    if np.all(system.box < 1e-6):
        frac_coords = False
        displacement_fn, shift_fn = space.free()
    else:
        frac_coords = True
        check_minimum_image(system.box, r_max)

        displacement_fn, shift_fn = space.periodic_general(
            system.box,
            fractional_coordinates=frac_coords,
            wrapped=md_config.wrapped,
        )

    Builder = model_config.model.get_builder()
    builder = Builder(model_config.model.model_dump())
    energy_model = builder.build_energy_model(
        apply_mask=True,
        init_box=np.array(system.box),
        inference_disp_fn=displacement_fn,
    )
    disable_cell_list = md_config.disable_cell_list or np.all(system.box < 1e-6)
    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        system.box,
        r_max,
        md_config.dr_threshold,
        fractional_coordinates=frac_coords,
        format=partition.Sparse,
        disable_cell_list=disable_cell_list,
    )

    _, gradient_model_params = restore_parameters(model_config.data.model_version_path)
    params = canonicalize_energy_model_parameters(gradient_model_params)

    n_models = 1
    shallow = False
    if (
        "ensemble" in model_config.model.model_dump().keys()
        and model_config.model.ensemble is not None
        and model_config.model.ensemble.n_members > 1
    ):
        n_models = model_config.model.ensemble.n_members
        if model_config.model.ensemble.kind == "shallow":
            shallow = True

    energy_fn = create_energy_fn(
        energy_model.apply,
        params,
        system.atomic_numbers,
        n_models,
        shallow,
    )

    biases = []
    if md_config.biases:
        bias_list = [BiasEnergies(b.model_dump()) for b in md_config.biases]
        biases.extend(bias_list)

    force_variance = "forces_uncertainty" in md_config.properties
    if md_config.dynamics_checks:
        for check in md_config.dynamics_checks:
            if check.name == "forces_uncertainty":
                force_variance = True

    auxiliary_fn = builder.build_energy_derivative_model(
        apply_mask=True,
        init_box=np.array(system.box),
        inference_disp_fn=displacement_fn,
        calc_stress="stress" in md_config.properties,
        calc_hessian="hessian" in md_config.properties,
        force_variance=force_variance,
    ).apply

    if n_models > 1 and not shallow:
        auxiliary_fn = maybe_vmap(auxiliary_fn, gradient_model_params)
        auxiliary_fn = make_ensemble(auxiliary_fn)
    else:
        auxiliary_fn = partial(
            auxiliary_fn,
            gradient_model_params,
        )

    for bias in biases:
        energy_fn = apply_bias_energy(bias, energy_fn)
        auxiliary_fn = apply_bias_auxiliary(bias, auxiliary_fn)

    sim_fns = SimulationFunctions(
        energy_fn, auxiliary_fn, shift_fn, neighbor_fn, r_max, md_config.dr_threshold
    )
    return system, sim_fns


def run_md(model_config: Config, md_config: MDConfig, log_level="error"):
    """
    Utiliy function to start NVT molecualr dynamics simulations from
    a previously trained model.

    Parameters
    ----------
    model_config : Config
        Configuration of the model used as an interatomic potential.
    md_config : MDConfig
        configuration of the MD simulation.
    """

    model_config = parse_config(model_config)
    md_config = parse_config(md_config, mode="md")

    sim_dir = Path(md_config.sim_dir)
    sim_dir.mkdir(parents=True, exist_ok=True)
    log_file = sim_dir / "md.log"
    setup_logging(log_file, log_level)
    traj_path = sim_dir / md_config.traj_name

    system, sim_fns = md_setup(model_config, md_config)

    dynamics_checks = []
    if md_config.dynamics_checks:
        check_list = [
            DynamicsChecks(check.model_dump()) for check in md_config.dynamics_checks
        ]
        dynamics_checks.extend(check_list)

    constraints = []
    if md_config.constraints:
        constraint_list = [Constraint(c.model_dump()) for c in md_config.constraints]
        constraints.extend(constraint_list)

    n_steps = int(np.ceil(md_config.duration / md_config.ensemble.dt))

    n_replicas = md_config.n_replicas
    if n_replicas > 1:
        traj_paths = [
            traj_path.with_name(f"{traj_path.stem}_{k}{traj_path.suffix}")
            for k in range(n_replicas)
        ]
    else:
        traj_paths = [traj_path]
    traj_handlers = [
        H5TrajHandler(
            system,
            md_config.buffer_size,
            path,
            md_config.ensemble.dt,
            properties=md_config.properties,
            h5md_options=md_config.h5md_options.model_dump(),
        )
        for path in traj_paths
    ]
    # TODO implement correct chunking

    run_sim(
        system,
        sim_fns,
        md_config.ensemble,
        n_steps=n_steps,
        n_inner=md_config.n_inner,
        extra_capacity=md_config.extra_capacity,
        sampling_rate=md_config.sampling_rate,
        load_momenta=md_config.load_momenta,
        traj_handler=traj_handlers,
        rng_key=jax.random.PRNGKey(md_config.seed),
        restart=md_config.restart,
        checkpoint_interval=md_config.checkpoint_interval,
        sim_dir=sim_dir,
        dynamics_checks=dynamics_checks,
        constraints=constraints,
        disable_pbar=md_config.disable_pbar,
        n_replicas=n_replicas,
    )
