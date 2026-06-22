# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
#
# Standalone Cython extension build helper (e.g. python cython_setup.py build_ext --inplace).
# Full wheel / editable installs should use the repository root: pip install -e . --no-build-isolation

import importlib.util
import os
import re
import sys
from os import path

from setuptools import find_packages

# need to use distutils.core for correct placement of cython dll
if "--inplace" in sys.argv:
    from distutils.core import setup
    from distutils.extension import Extension
else:
    from setuptools import setup
    from setuptools.extension import Extension

import z3

# Single source of truth (same as repository root setup.py and pyproject dynamic version)
_version_file = path.join(path.dirname(__file__), "yirage", "version.py")
with open(_version_file, encoding="utf-8") as _vf:
    exec(_vf.read())  # defines __version__


def _repo_root() -> str:
    return path.abspath(path.join(path.dirname(__file__), ".."))


def _config_use_cuda(repo: str) -> bool:
    cfg = path.join(repo, "config.cmake")
    if not path.isfile(cfg):
        return False
    with open(cfg, encoding="utf-8") as f:
        text = f.read()
    return re.search(r"set\s*\(\s*USE_CUDA\s+ON\s*\)", text, re.IGNORECASE) is not None


def _cuda_home() -> str:
    return os.environ.get("CUDA_HOME", "/usr/local/cuda")


def _cuda_dirs(repo: str) -> tuple[list[str], list[str]]:
    """Include / library dirs for CUDA; empty on macOS or when CUDA is not used / missing."""
    if sys.platform == "darwin":
        return [], []
    if not _config_use_cuda(repo):
        return [], []
    ch = _cuda_home()
    inc = path.join(ch, "include")
    if not path.isdir(inc):
        return [], []
    lib_candidates = [
        path.join(ch, "lib64"),
        path.join(ch, "lib"),
        path.join(ch, "lib64", "stubs"),
    ]
    lib_dirs = [p for p in lib_candidates if path.isdir(p)]
    return [inc], lib_dirs


def config_cython():
    try:
        from Cython.Build import cythonize

        repo = _repo_root()
        cm_path = path.join(repo, "python", "yirage", "cmake_macros.py")
        spec = importlib.util.spec_from_file_location("_yirage_cmake_macros", cm_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {cm_path}")
        cm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cm)

        z3_path = path.dirname(z3.__file__)
        cython_path = path.join(path.dirname(__file__), "yirage/_cython")
        cuda_includes, cuda_lib_dirs = _cuda_dirs(repo)

        cfg = path.join(repo, "config.cmake")
        if not path.isfile(cfg):
            raise FileNotFoundError(
                f"Missing {cfg}; generate it with pip install / setup.py "
                "(YIRAGE_BACKEND=...) before building Cython extensions."
            )
        define_macros = cm.macros_from_config(cfg)

        include_dirs = [
            path.join(repo, "include"),
            path.join(repo, "deps", "json", "include"),
            path.join(repo, "deps", "cutlass", "include"),
            path.join(repo, "deps", "cutlass", "tools", "util", "include"),
            path.join(repo, "build", "abstract_subexpr", "release"),
            path.join(repo, "build", "formal_verifier", "release"),
            path.join(z3_path, "include"),
        ]
        include_dirs.extend(cuda_includes)

        library_dirs = [
            path.join(repo, "build"),
            path.join(z3_path, "lib"),
            path.join(repo, "build", "abstract_subexpr", "release"),
            path.join(repo, "build", "formal_verifier", "release"),
        ]
        library_dirs.extend(cuda_lib_dirs)

        libraries = ["z3"]
        if sys.platform == "darwin":
            libraries.append("omp")
        else:
            libraries.append("gomp")
            libraries.append("rt")

        if cuda_lib_dirs:
            libraries.extend(["cudart", "cuda"])
        if any(m[0] == "YIRAGE_BACKEND_MACA_ENABLED" for m in define_macros):
            libraries.append("mcruntime")

        extra_compile_args = ["-std=c++17"]
        extra_compile_args += (
            ["-Xpreprocessor", "-fopenmp"] if sys.platform == "darwin" else ["-fopenmp"]
        )

        extra_link_args = ["-fPIC", f"-L{path.join(z3_path, 'lib')}", "-lz3"]
        if sys.platform == "darwin":
            extra_link_args += [
                f"-L{path.join(repo, 'build')}",
                "-lyirage_runtime",
                f"-L{path.join(repo, 'build', 'abstract_subexpr', 'release')}",
                "-labstract_subexpr",
                f"-L{path.join(repo, 'build', 'formal_verifier', 'release')}",
                "-lformal_verifier",
                "-Wl,-rpath,@loader_path/../../build/abstract_subexpr/release",
                "-Wl,-rpath,@loader_path/../../build/formal_verifier/release",
                f"-Wl,-rpath,{path.join(z3_path, 'lib')}",
            ]
        else:
            extra_link_args += [
                "-fopenmp",
                "-Wl,--no-as-needed",
                f"-L{path.join(repo, 'build')}",
                "-lyirage_runtime",
                "-Wl,--as-needed",
                f"-L{path.join(repo, 'build', 'abstract_subexpr', 'release')}",
                "-labstract_subexpr",
                f"-L{path.join(repo, 'build', 'formal_verifier', 'release')}",
                "-lformal_verifier",
                f"-Wl,-rpath,{path.join(z3_path, 'lib')}",
            ]
            if cuda_lib_dirs:
                extra_link_args += [
                    f"-L{cuda_lib_dirs[0]}",
                    "-lcudart",
                    f"-Wl,-rpath,{cuda_lib_dirs[0]}",
                ]

        setup_py_dir = path.abspath(path.dirname(__file__))

        def _relativize_extension_sources(extensions):
            root = setup_py_dir
            prefix = root + path.sep
            for ext in extensions:
                rel_sources = []
                for src in ext.sources:
                    if path.isabs(src):
                        abs_src = path.abspath(src)
                        if abs_src.startswith(prefix):
                            rel = path.relpath(abs_src, root)
                            rel_sources.append(rel.replace("\\", "/"))
                        else:
                            rel_sources.append(src.replace("\\", "/"))
                    else:
                        rel_sources.append(src.replace("\\", "/"))
                ext.sources = rel_sources
            return extensions

        ret = []
        for fn in os.listdir(cython_path):
            if not fn.endswith(".pyx"):
                continue
            pyx_src = path.join("yirage", "_cython", fn).replace("\\", "/")
            ret.append(
                Extension(
                    "yirage.%s" % fn[:-4],
                    [pyx_src],
                    include_dirs=include_dirs,
                    libraries=libraries,
                    library_dirs=library_dirs,
                    define_macros=define_macros,
                    extra_compile_args=extra_compile_args,
                    extra_link_args=extra_link_args,
                    language="c++",
                )
            )
        exts = cythonize(ret, compiler_directives={"language_level": 3})
        return _relativize_extension_sources(exts)
    except ImportError:
        print("WARNING: cython is not installed!!!")
        raise SystemExit(1) from None


setup_args = {}

setup(
    name="yirage",
    version=__version__,
    description="YiRage: A Multi-Level Superoptimizer for Tensor Algebra",
    zip_safe=False,
    install_requires=[],
    packages=find_packages(),
    url="https://github.com/chenxingqiang/YiRage",
    ext_modules=config_cython(),
)
