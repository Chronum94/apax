import os
import warnings
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

import jax

try:
    from ._version import __version__, __version_tuple__
except (ImportError, ModuleNotFoundError):
    try:
        __version__ = _pkg_version("apax")
    except PackageNotFoundError:
        __version__ = "0.0.0"
    __version_tuple__ = (0, 0, 0)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
jax.config.update("jax_enable_x64", True)

from apax.utils.jax_patches import keep_scatter_flags_in_transpose  # noqa: E402

keep_scatter_flags_in_transpose()

warnings.filterwarnings("ignore", message=".*os.fork()*")
