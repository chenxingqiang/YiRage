import os
import ctypes
import z3

def preload_so(lib_path, name_hint):
    try:
        ctypes.CDLL(lib_path)
    except OSError as e:
        raise ImportError(f"Could not preload {name_hint} ({lib_path}): {e}")

_z3_libdir = os.path.join(os.path.dirname(z3.__file__), "lib")
# Try different z3 library names for different platforms
for lib_name in ["libz3.so", "libz3.dylib", "libz3.dll"]:
    _z3_so_path = os.path.join(_z3_libdir, lib_name)
    if os.path.exists(_z3_so_path):
        preload_so(_z3_so_path, lib_name)
        break

_this_dir = os.path.dirname(__file__)
_yirage_root = os.path.abspath(os.path.join(_this_dir, "..", ".."))

# Try different library extensions for cross-platform compatibility
import platform
lib_ext = ".dylib" if platform.system() == "Darwin" else ".so"

_subexpr_so_path = os.path.join(_yirage_root, "build", "abstract_subexpr", "release", f"libabstract_subexpr{lib_ext}")
_formal_verifier_so_path = os.path.join(_yirage_root, "build", "formal_verifier", "release", f"libformal_verifier{lib_ext}")

# Only preload if files exist
if os.path.exists(_subexpr_so_path):
    preload_so(_subexpr_so_path, f"libabstract_subexpr{lib_ext}")
if os.path.exists(_formal_verifier_so_path):
    preload_so(_formal_verifier_so_path, f"libformal_verifier{lib_ext}")

from .core import *
from .kernel import *
from .persistent_kernel import PersistentKernel
from .threadblock import *

# Backend API
from .backend_api import (
    get_available_backends,
    is_backend_available,
    get_default_backend,
    get_backend_info,
    set_default_backend,
    list_backends,
    available_backends,
    default_backend,
)

# Auto-initialize backends on module import
try:
    if hasattr(core, 'init_backends'):
        core.init_backends()
except Exception as e:
    import sys
    print(f"Warning: Backend initialization failed: {e}", file=sys.stderr)


class InputNotFoundError(Exception):
    """Raised when cannot find input tensors"""

    pass


def set_gpu_device_id(device_id: int):
    global_config.gpu_device_id = device_id
    core.set_gpu_device_id(device_id)


def bypass_compile_errors(value: bool = True):
    global_config.bypass_compile_errors = value


def new_kernel_graph():
    kgraph = core.CyKNGraph()
    return KNGraph(kgraph)


def new_threadblock_graph(
    grid_dim: tuple, block_dim: tuple, forloop_range: int, reduction_dimx: int
):
    bgraph = core.CyTBGraph(grid_dim, block_dim, forloop_range, reduction_dimx)
    return TBGraph(bgraph)


# Other Configurations
from .global_config import global_config

# Graph Datasets
from .graph_dataset import graph_dataset
from .version import __version__