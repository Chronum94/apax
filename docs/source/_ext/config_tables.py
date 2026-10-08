"""Generate Markdown reference tables for apax configs from the pydantic models.

Defaults and types come from the models themselves (never stale); descriptions come from the
numpydoc "Parameters" sections of the class docstrings. Written to reference/_generated/ at build start.
"""

import re
import types
import typing
from pathlib import Path

from pydantic import BaseModel
from pydantic_core import PydanticUndefined


def _doc_params(cls) -> dict:
    """name -> one-line description from a numpydoc Parameters section (searching the MRO)."""
    out = {}
    for klass in reversed(cls.__mro__):
        doc = klass.__doc__ or ""
        lines = doc.splitlines()
        for i, line in enumerate(lines):
            m = re.match(r"^\s{4}(\w+)\s*:", line)
            if not m:
                continue
            desc = []
            same_line = line.split(":", 1)[1].strip()
            for nxt in lines[i + 1 :]:
                if re.match(r"^\s{4}\w+\s*:", nxt) or not nxt.strip():
                    break
                desc.append(nxt.strip().lstrip("|").strip())
            if desc:
                out[m.group(1)] = " ".join(desc)
            elif same_line and not re.match(r"^[\w\[\],. |]+(,|$)", same_line):
                out[m.group(1)] = same_line  # "name: description" style docstrings
    return out


def _unwrap(ann):
    return typing.get_args(ann)[0] if hasattr(ann, "__metadata__") else ann


def _models_in(ann):
    ann = _unwrap(ann)
    origin = typing.get_origin(ann)
    if origin in (typing.Union, types.UnionType, list, typing.List):
        res = []
        for a in typing.get_args(ann):
            res += _models_in(a)
        return res
    if isinstance(ann, type) and issubclass(ann, BaseModel):
        return [ann]
    return []


def _type_str(ann) -> str:
    ann = _unwrap(ann)
    origin = typing.get_origin(ann)
    if origin is typing.Literal:
        return " \\| ".join(f"`{v}`" for v in typing.get_args(ann))
    if origin in (typing.Union, types.UnionType):
        parts = [_type_str(a) for a in typing.get_args(ann) if a is not type(None)]
        return " \\| ".join(dict.fromkeys(parts))
    if origin in (list, typing.List):
        args = typing.get_args(ann)
        return f"list[{_type_str(args[0])}]" if args else "list"
    if isinstance(ann, type) and issubclass(ann, BaseModel):
        return "section"
    return getattr(ann, "__name__", str(ann)).replace("Positive", "positive ").replace("NonNegative", "≥0 ")


def _default_str(field) -> str:
    if field.is_required():
        return "**required**"
    val = field.default
    if val is PydanticUndefined and field.default_factory is not None:
        try:
            val = field.default_factory()
        except TypeError:
            return "—"
    if isinstance(val, BaseModel):
        name = getattr(val, "name", None) or getattr(val, "processing", None) or getattr(val, "kind", None)
        return f"`{name}`" if name else "(section defaults)"
    if isinstance(val, list) and val and isinstance(val[0], BaseModel):
        return "[" + ", ".join(f"`{getattr(v, 'name', type(v).__name__)}`" for v in val) + "]"
    text = repr(val)
    return f"`{text}`" if len(text) <= 40 else f"`{text[:37]}...`"


def _variant(cls):
    for key in ("name", "processing", "kind"):
        f = cls.model_fields.get(key)
        if f is not None and typing.get_origin(_unwrap(f.annotation)) is typing.Literal:
            return key, typing.get_args(_unwrap(f.annotation))[0]
    return None, None


# Corrections where class docstrings are wrong/missing (keyed by class name, field name).
DESC_OVERRIDES = {
    ("MetricsConfig", "reductions"): "Any of `mae`, `mse`, `rmse`, `cosine_sim` (vector labels), `per_atom_mae`, "
    "`per_atom_mse`, `per_atom_rmse` (per-atom variants divide the per-structure error by N; `per_atom_rmse` = RMSE/√N). "
    "Logged as `{train,val}_{name}_{reduction}`.",
    ("LossConfig", "loss_type"): "`mse` (default), `huber` (needs `parameters: {delta: ...}`), `cosine_sim`, "
    "`cosine_sim_div_magnitude`, `cosine_sim_exp_magnitude` (force direction), `tril` (stress), `nll`, `crps` "
    "(shallow ensembles), `nll_3x3`, `mw_hessian`.",
    ("LossConfig", "atoms_exponent"): "Each structure's term is divided by N_atoms^atoms_exponent.",
    ("Config", "transfer_learning"): "Fine-tuning from a trained model. Only add this section when fine-tuning: its "
    "presence enables transfer learning.",
    ("Config", "progress_bar"): "Progress bar options.",
    ("WeightAverage", "alpha"): "Weight of the *new* parameters in the EMA update, applied twice per epoch (not per "
    "step; MACE's `ema_decay` has no direct equivalent). EMA parameters are used for validation only.",
    ("MDConfig", "n_inner"): "Time steps per compiled block (one jitted loop). Dynamics checks and checkpoints act at "
    "block boundaries.",
    ("MDConfig", "checkpoint_interval"): "Checkpoint every this many *blocks* of `n_inner` steps.",
    ("MDConfig", "restart"): "Continue from `sim_dir/ckpts` if present.",
    ("MDConfig", "properties"): "Properties to save to the trajectory; listing a derivative (e.g. `stress`) also makes "
    "MD compute it.",
    ("MDConfig", "sampling_rate"): "Save a frame every this many time steps.",
    ("MDConfig", "extra_capacity"): "Spare neighbour-list slots. Set > 0 when density can increase (NPT, heating): "
    "overflow triggers a reallocation that can crash.",
    ("NHCOptions", "tau"): "Thermostat/barostat time constant in units of `dt` (time steps).",
}

# Fields shared by all model variants: shown once in a common table.
_SHARED_MODEL_TABLE = "model (fields common to all models)"


def _row(cls, name, field, docs, default=None):
    models = _models_in(field.annotation)
    tstr = _type_str(field.annotation)
    if models:
        variants = [m for m in models if _variant(m)[1] is not None]
        if variants:
            key, _ = _variant(variants[0])
            tstr = f"one of {key}: " + ", ".join(f"`{_variant(m)[1]}`" for m in variants)
        else:
            tstr = "section"
        if typing.get_origin(_unwrap(field.annotation)) in (list, typing.List):
            tstr = "list of " + tstr
    desc = DESC_OVERRIDES.get((cls.__name__, name))
    if desc is None:
        desc = re.sub(r":\w+:`\.?([^`]+)`", r"\1", docs.get(name, ""))
    dstr = _default_str(field) if default is None else default
    return (name, tstr, dstr, desc), models


def _tables(cls, path, out, seen, instance=None, skip=()):
    docs = _doc_params(cls)
    rows, sub = [], []
    for name, field in cls.model_fields.items():
        if name in skip:
            continue
        default = None
        if instance is not None:
            val = getattr(instance, name)
            default = f"`{val!r}`" if not isinstance(val, BaseModel) else "(section defaults)"
        if name == _variant(cls)[0]:
            default = "**required** (selects the variant)"
        row, models = _row(cls, name, field, docs, default)
        rows.append(row)
        child_instance = field.default if isinstance(field.default, BaseModel) else None
        if instance is not None and isinstance(getattr(instance, name), BaseModel):
            child_instance = getattr(instance, name)
        for m in models:
            key, v = _variant(m)
            label = f"{path}{name}" + (f" ({key}: {v})" if v is not None else "")
            sub.append((label, m, child_instance if type(child_instance) is m else None))
    out.append((path.rstrip(".") or "top level", cls, rows))
    for p, m, inst in sub:
        if (p, m) in seen:
            continue
        seen.add((p, m))
        if p.startswith("model (name:"):
            # model variants: shared fields once, then only the variant-specific ones
            shared = _shared_fields(m)
            if _SHARED_MODEL_TABLE not in seen:
                seen.add(_SHARED_MODEL_TABLE)
                base = m.__mro__[1]
                idx = len(out)
                _tables(base, "model.", out, seen, skip=())
                out[idx] = (_SHARED_MODEL_TABLE, base, out[idx][2])
            _tables(m, p + ".", out, seen, inst, skip=shared)
        else:
            _tables(m, p + ".", out, seen, inst)


def _shared_fields(model_cls):
    base = model_cls.__mro__[1]
    if isinstance(base, type) and issubclass(base, BaseModel) and base is not BaseModel:
        return set(base.model_fields)
    return set()


def render(cls, title_level="##") -> str:
    out = []
    _tables(cls, "", out, set())
    md = []
    for path, klass, rows in out:
        if not rows:
            continue
        md.append(f"{title_level} `{path}`\n")
        md.append("| Key | Type | Default | Description |\n|---|---|---|---|")
        for name, t, d, desc in rows:
            md.append(f"| `{name}` | {t} | {d} | {desc} |")
        md.append("")
    return "\n".join(md)


def generate(app):
    from apax.config.md_config import MDConfig
    from apax.config.train_config import Config

    target = Path(app.srcdir) / "reference" / "_generated"
    target.mkdir(parents=True, exist_ok=True)
    (target / "train_config_tables.md").write_text(render(Config))
    (target / "md_config_tables.md").write_text(render(MDConfig))


def setup(app):
    app.connect("builder-inited", generate)
    return {"parallel_read_safe": True}
