"""Keep `indices_are_sorted` / `unique_indices` through the transpose of scatter-add.

JAX's transpose of scatter_add (and scatter_sub) builds its gather without these
flags. For a sorted segment sum that is harmless in the first derivative (a gather),
but the second derivative transposes that gather back into a scatter, which then
also lacks the flag. XLA:GPU emits an unflagged scatter-add as one atomic per
element, and with ~100 pairs per centre atom those atomics contend heavily (on AMD
consumer GPUs this made the double-backward moment scatters ~45% of a training
step). The rule below is JAX's own, with the two flags passed on.
"""
from jax._src import ad_util
from jax._src.interpreters import ad
from jax._src.lax import lax
from jax._src.lax import slicing


def _scatter_addsub_transpose_rule(
    prim, t, operand, indices, updates, *, update_jaxpr, update_consts,
    dimension_numbers, indices_are_sorted, unique_indices, mode,
):
    assert not ad.is_undefined_primal(indices)
    if ad.is_undefined_primal(updates):
        updates_shape = updates.aval.shape
    else:
        updates_shape = updates.shape
    if type(t) is ad_util.Zero:
        operand_t = ad_util.Zero(operand.aval) if ad.is_undefined_primal(operand) else None
        update_t = ad_util.Zero(updates.aval) if ad.is_undefined_primal(updates) else None
    else:
        operand_t = update_t = None
        if ad.is_undefined_primal(operand):
            operand_t = t

        if ad.is_undefined_primal(updates):
            gather_dnums = slicing.GatherDimensionNumbers(
                offset_dims=dimension_numbers.update_window_dims,
                collapsed_slice_dims=dimension_numbers.inserted_window_dims,
                start_index_map=dimension_numbers.scatter_dims_to_operand_dims,
                operand_batching_dims=dimension_numbers.operand_batching_dims,
                start_indices_batching_dims=dimension_numbers.scatter_indices_batching_dims,
            )
            slice_sizes = []
            pos = 0
            for i in range(len(t.shape)):
                if (
                    i in dimension_numbers.inserted_window_dims
                    or i in dimension_numbers.operand_batching_dims
                ):
                    slice_sizes.append(1)
                else:
                    slice_sizes.append(updates_shape[dimension_numbers.update_window_dims[pos]])
                    pos += 1
            update_t = slicing.gather(
                t,
                indices,
                dimension_numbers=gather_dnums,
                slice_sizes=slice_sizes,
                mode=mode,
                fill_value=0,
                indices_are_sorted=indices_are_sorted,
                unique_indices=unique_indices,
            )
            if prim is slicing.scatter_sub_p:
                update_t = lax.neg(update_t)
    return [operand_t, None, update_t]


def keep_scatter_flags_in_transpose():
    for prim in (slicing.scatter_add_p, slicing.scatter_sub_p):
        ad.primitive_transposes[prim] = (
            lambda *args, _prim=prim, **kw: _scatter_addsub_transpose_rule(_prim, *args, **kw)
        )
