import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ase import data
from ase.calculators.singlepoint import SinglePointCalculator

from apax.layers.empirical import LatentEwald, PairEnvelope, fit_pair_envelope


@pytest.fixture
def setup_data():
    n_atoms = 4

    R = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    dr_vec = None
    Z = jnp.array([1, 1, 1, 1])
    idx = None
    box = jnp.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])
    properties = {
        "charges": jnp.array([1.0, -1.0, 1.0, -1.0])[:, None]
    }  # Charges on atoms

    return R, dr_vec, Z, idx, box, properties


def test_latent_ewald(setup_data):
    R, dr_vec, Z, idx, box, properties = setup_data

    latent_ewald = LatentEwald(apply_mask=True, sigma=1.0, kgrid=[2, 2, 2])

    # Check for KeyError if "charge" property is missing
    with pytest.raises(KeyError, match="property 'charges' not found"):
        latent_ewald(R, dr_vec, Z, idx, box, {})

    energy = latent_ewald(R, dr_vec, Z, idx, box, properties)

    # Validate the energy output
    assert isinstance(energy, jnp.ndarray), "Output should be a JAX array"
    assert energy.ndim == 0, "Energy should be a scalar value"

    # Test with a larger kgrid
    latent_ewald = LatentEwald(apply_mask=False, sigma=1.5, kgrid=[4, 4, 4])
    energy = latent_ewald(R, dr_vec, Z, idx, box, properties)
    assert isinstance(energy, jnp.ndarray)


def _labelled(atoms_list, amp, r_max):
    """Energies from the PairEnvelope layer itself, so the fit is checked against what it initialises."""
    from ase.neighborlist import neighbor_list

    layer = PairEnvelope(r_max=r_max, amp_init=amp, apply_mask=False)
    for a in atoms_list:
        i, j, D = neighbor_list("ijD", a, r_max)
        args = (None, jnp.asarray(D), jnp.asarray(a.numbers), jnp.asarray(np.stack([i, j])), None, {})
        params = layer.init(jax.random.PRNGKey(0), *args)
        a.calc = SinglePointCalculator(a, energy=float(layer.apply(params, *args)))
    return atoms_list


def test_fit_pair_envelope_recovers_amplitudes():
    from ase.build import bulk

    r_max = 4.0
    amp = np.zeros(len(data.covalent_radii))
    amp[[6, 14]] = [0.3, 0.8]
    structs = []
    for k, (n_c, scale) in enumerate([(0, 1.0), (8, 0.98), (16, 1.02), (4, 1.01), (12, 0.99)]):
        a = bulk("Si", "diamond", a=5.43 * scale).repeat(2)
        a.numbers[:n_c] = 6
        a.rattle(0.05, seed=k)
        structs.append(a)
    structs = _labelled(structs, amp, r_max)
    fit = fit_pair_envelope(structs, np.zeros(len(amp)), r_max, ridge=0.0)
    assert np.allclose(fit[[6, 14]], [0.3, 0.8], atol=1e-6)  # layer runs in fp32


def test_fit_pair_envelope_degenerate_data_is_finite():
    from ase import Atoms
    from ase.build import bulk

    shift = np.zeros(len(data.covalent_radii))
    isolated = [Atoms("Si", cell=[20.0] * 3, pbc=True) for _ in range(3)]
    for a in isolated:
        a.calc = SinglePointCalculator(a, energy=-1.0)
    assert np.array_equal(fit_pair_envelope(isolated, shift, 5.0), np.zeros(len(shift)))

    amp = np.full(len(shift), 0.5)
    single_element = _labelled([bulk("Si", "diamond", a=5.43).repeat(2) for _ in range(3)], amp, 4.0)
    fixed_stoich = _labelled([bulk("NaCl", "rocksalt", a=5.6).repeat(n) for n in (1, 2)], amp, 4.0)
    for structs in (single_element, fixed_stoich, single_element[:1]):
        for ridge in (0.0, 1e-2):
            assert np.isfinite(fit_pair_envelope(structs, shift, 4.0, ridge=ridge)).all()
