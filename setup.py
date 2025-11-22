# Copyright 2024 CMU
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import shutil
from os import path
from pathlib import Path
import sys
import sysconfig
from setuptools import find_packages, setup, Command
from contextlib import contextmanager
import subprocess
import re

# need to use distutils.core for correct placement of cython dll
if "--inplace" in sys.argv:
    from distutils.core import setup
    from distutils.extension import Extension
else:
    from setuptools import setup
    from setuptools.extension import Extension

import z3

nvcc_path = shutil.which("nvcc")
if nvcc_path:
    cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
else:
    cuda_home = "/usr/local/cuda"

cuda_include_dir = os.path.join(cuda_home, "include")
cuda_library_dirs = [
    os.path.join(cuda_home, "lib"),
    os.path.join(cuda_home, "lib64"),
    os.path.join(cuda_home, "lib64", "stubs"),
]

z3_path = path.dirname(z3.__file__)

# Use version.py to get package version
version_file = os.path.join(os.path.dirname(__file__), "python/yirage/version.py")
with open(version_file, "r") as f:
    exec(f.read())  # This will define __version__

def get_backend_macros(config_file):
    """Read config.cmake and return macros for all enabled backends"""
    backend_flags = {
        # GPU Backends
        "USE_CUDA": None,
        "USE_CUDNN": None,
        "USE_MPS": None,
        "USE_CUSPARSELT": None,
        # CPU Backends
        "USE_CPU": None,
        "USE_MKL": None,
        "USE_MKLDNN": None,
        "USE_OPENMP": None,
        "USE_XEON": None,
        # Specialized Backends
        "USE_NKI": None,
        "USE_TRITON": None,
        "USE_MHA": None,
        "USE_NNPACK": None,
        "USE_OPT_EINSUM": None,
    }

    pattern = re.compile(r'^\s*set\s*\(\s*([A-Z_]+)\s+(ON|OFF)\s*\)', re.IGNORECASE)
    
    with open(config_file, 'r') as f:
        for line in f:
            match = pattern.match(line)
            if match:
                var, val = match.groups()
                if var in backend_flags:
                    backend_flags[var] = (val.upper() == "ON")
    
    macros = []
    enabled_backends = []
    
    # Add macro for each enabled backend
    for flag_name, enabled in backend_flags.items():
        if enabled:
            backend_name = flag_name.replace("USE_", "")
            macros.append((f"YIRAGE_BACKEND_{backend_name}_ENABLED", None))
            enabled_backends.append(backend_name)
            
            # Backward compatibility macros
            if flag_name == "USE_CUDA":
                macros.append(("YIRAGE_BACKEND_USE_CUDA", None))
                macros.append(("YIRAGE_FINGERPRINT_USE_CUDA", None))
            elif flag_name == "USE_NKI":
                macros.append(("YIRAGE_BACKEND_USE_NKI", None))
                macros.append(("YIRAGE_FINGERPRINT_USE_CPU", None))
    
    # At least one backend must be enabled
    if not enabled_backends:
        raise ValueError("At least one backend must be enabled in config.cmake file")
    
    print(f"Enabled backends: {', '.join(enabled_backends)}")
    return macros

def config_cython():
    sys_cflags = sysconfig.get_config_var("CFLAGS")
    try:
        from Cython.Build import cythonize

        ret = []
        yirage_path = ''
        config_path = path.join(yirage_path, "config.cmake")
        macros = get_backend_macros(config_path)
        cython_path = path.join(yirage_path, "python/yirage/_cython")
        for fn in os.listdir(cython_path):
            if not fn.endswith(".pyx"):
                continue
            ret.append(
                Extension(
                    "yirage.%s" % fn[:-4],
                    ["%s/%s" % (cython_path, fn)],
                    include_dirs=[
                        path.join(yirage_path, "include"),
                        path.join(yirage_path, "deps", "json", "include"),
                        path.join(yirage_path, "deps", "cutlass", "include"),
                        path.join(yirage_path, "deps", "cutlass", "tools", "util", "include"),
                        path.join(yirage_path, "build", "abstract_subexpr", "release"),
                        path.join(yirage_path, "build", "formal_verifier", "release"),
                        path.join(z3_path, "include"),
                        cuda_include_dir,
                    ],
                    libraries=[
                        "yirage_runtime",
                        "z3",
                        "abstract_subexpr",
                        "formal_verifier",
                    ] + (["cudadevrt", "cudart_static", "cudart", "cuda"] if macros and any("CUDA" in str(m) for m in macros) else []),
                    library_dirs=[
                        path.join(yirage_path, "build"),
                        path.join(z3_path, "lib"),
                        path.join(yirage_path, "build", "abstract_subexpr", "release"),
                        path.join(yirage_path, "build", "formal_verifier", "release"),
                    ]
                    + cuda_library_dirs,
                    define_macros=macros,
                    extra_compile_args=["-std=c++17"],
                    extra_link_args=[
                        "-fPIC",
                        f"-Wl,-rpath,{path.join('$ORIGIN', '..', '..', 'build', 'abstract_subexpr', 'release')}",
                        f"-Wl,-rpath,{path.join('$ORIGIN', '..', '..', 'build', 'formal_verifier', 'release')}",
                    ],
                    language="c++",
                )
            )
        return cythonize(ret, compiler_directives={"language_level": 3})
    except ImportError:
        print("WARNING: cython is not installed!!!")
        raise SystemExit(1)
    
# Install Rust if not yet available
try:
    # Attempt to run a Rust command to check if Rust is installed
    subprocess.check_output(['cargo', '--version'])
except FileNotFoundError:
    print("Rust/Cargo not found, installing it...")
    # Rust is not installed, so install it using rustup
    try:
        subprocess.run("curl https://sh.rustup.rs -sSf | sh -s -- -y", shell=True, check=True)
        print("Rust and Cargo installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    # Add the cargo binary directory to the PATH
    os.environ["PATH"] = f"{os.path.join(os.environ.get('HOME', '/root'), '.cargo', 'bin')}:{os.environ.get('PATH', '')}"

yirage_path = path.dirname(__file__)
# z3_path = os.path.join(yirage_path, 'deps', 'z3', 'build')
# os.environ['Z3_DIR'] = z3_path
if yirage_path == '':
    yirage_path = '.'

try:
    subprocess.check_output(['cargo', 'build', '--release', '--target-dir', '../../../../build/abstract_subexpr'], cwd='src/search/abstract_expr/abstract_subexpr')
except subprocess.CalledProcessError as e:
    print("Failed to build abstract_subexpr Rust library, building it ...")
    try:
        subprocess.run(['cargo', 'build', '--release', '--target-dir', '../../../../build/abstract_subexpr'], cwd='src/search/abstract_expr/abstract_subexpr', check=True)
        print("Abstract_subexpr Rust library built successfully.")
    except subprocess.CalledProcessError as e:
        print("Failed to build abstract_subexpr Rust library.")
    os.environ['ABSTRACT_SUBEXPR_LIB'] = os.path.join(yirage_path,'build', 'abstract_subexpr', 'release', 'libabstract_subexpr.so')

try:
    subprocess.check_output(['cargo', 'build', '--release', '--target-dir', '../../../../build/formal_verifier'], cwd='src/search/verification/formal_verifier_equiv')
except subprocess.CalledProcessError as e:
    print("Failed to build formal_verifier Rust library, building it ...")
    try:
        subprocess.run(['cargo', 'build', '--release', '--target-dir', '../../../../build/formal_verifier'], cwd='src/search/verification/formal_verifier_equiv', check=True)
        print("formal_verifier Rust library built successfully.")
    except subprocess.CalledProcessError as e:
        print("Failed to build formal_verifier Rust library.")
    os.environ['FORMAL_VERIFIER_LIB'] = os.path.join(yirage_path,'build', 'formal_verifier', 'release', 'libformal_verifier.so')


# build YiRage runtime library
try:
    os.environ["CUDACXX"] = nvcc_path if nvcc_path else os.path.join(
        cuda_home, "bin", "nvcc"
    )
    yirage_path = path.dirname(__file__)
    # z3_path = os.path.join(yirage_path, 'deps', 'z3', 'build')
    # os.environ['Z3_DIR'] = z3_path
    if yirage_path == "":
        yirage_path = "."
    os.makedirs(yirage_path, exist_ok=True)
    os.chdir(yirage_path)
    build_dir = os.path.join(yirage_path, "build")

    cc_path = shutil.which("gcc")
    os.environ["CC"] = cc_path if cc_path else "/usr/bin/gcc"
    cxx_path = shutil.which("g++")
    os.environ["CXX"] = cxx_path if cxx_path else "/usr/bin/g++"
    print(f"CC: {os.environ['CC']}, CXX: {os.environ['CXX']}", flush=True)

    # Create the build directory if it does not exist
    os.makedirs(build_dir, exist_ok=True)
    subprocess.check_call(
        [
            "cmake",
            "..",
            "-DCMAKE_BUILD_TYPE=Debug",
            "-DZ3_CXX_INCLUDE_DIRS=" + z3_path + "/include/",
            "-DZ3_LIBRARIES=" + path.join(z3_path, "lib", "libz3.so"),
            '-DABSTRACT_SUBEXPR_LIB=' + path.join(yirage_path, 'build', 'abstract_subexpr', 'release'),
            '-DABSTRACT_SUBEXPR_LIBRARIES=' + path.join(yirage_path, 'build', 'abstract_subexpr', 'release', 'libabstract_subexpr.so'),
            '-DFORMAL_VERIFIER_LIB=' + path.join(yirage_path, 'build', 'formal_verifier', 'release'),
            '-DFORMAL_VERIFIER_LIBRARIES=' + path.join(yirage_path, 'build', 'formal_verifier', 'release', 'libformal_verifier.so'),
            "-DCMAKE_C_COMPILER=" + os.environ["CC"],
            "-DCMAKE_CXX_COMPILER=" + os.environ["CXX"],
        ],
        cwd=build_dir,
        env=os.environ.copy(),
    )
    subprocess.check_call(["make", "-j8"], cwd=build_dir, env=os.environ.copy())
    print("YiRage runtime library built successfully.")
except subprocess.CalledProcessError as e:
    print("Failed to build runtime library.")
    raise SystemExit(e.returncode)

setup_args = {}

# Create requirements list from requirements.txt
with open(Path(__file__).parent / "requirements.txt", "r") as reqs_file:
    requirements = reqs_file.read().strip().split("\n")
print(f"Requirements: {requirements}")

INCLUDE_BASE = "python/yirage/include"


@contextmanager
def copy_include():
    if not path.exists(INCLUDE_BASE):
        src_dirs = ["deps/cutlass/include", "deps/json/include"]
        for src_dir in src_dirs:
            shutil.copytree(src_dir, path.join(INCLUDE_BASE, src_dir))
        # copy yirage/transpiler/runtime/*
        # to python/yirage/include/yirage/transpiler/runtime/*
        # instead of python/yirage/include/include/yirage/transpiler/runtime/*
        include_yirage_dirs = [
            "include/yirage/transpiler/runtime",
            "include/yirage/triton_transpiler/runtime",
            "include/yirage/persistent_kernel",
        ]
        include_yirage_dsts = [
            path.join(INCLUDE_BASE, "yirage/transpiler/runtime"),
            path.join(INCLUDE_BASE, "yirage/triton_transpiler/runtime"),
            path.join(INCLUDE_BASE, "yirage/persistent_kernel"),
        ]
        for include_yirage_dir, include_yirage_dst in zip(
            include_yirage_dirs, include_yirage_dsts
        ):
            shutil.copytree(include_yirage_dir, include_yirage_dst)

        config_h_src = path.join(
            yirage_path, "include/yirage/config.h"
        )  # Needed by transpiler/runtime/threadblock/utils.h
        config_h_dst = path.join(INCLUDE_BASE, "yirage/config.h")
        shutil.copy(config_h_src, config_h_dst)
        yield True
    else:
        yield False
    shutil.rmtree(INCLUDE_BASE)


with copy_include() as copied:
    if not copied:
        print(
            "WARNING: include directory already exists. Not copying again. "
            f"This may cause issues. Please remove {INCLUDE_BASE} and rerun setup.py",
            flush=True,
        )

    setup(
        name="yirage",
        version=__version__,
        description="YiRage: A Multi-Level Superoptimizer for Tensor Algebra",
        zip_safe=False,
        install_requires=requirements,
        packages=find_packages(where="python"),
        package_dir={"": "python"},
        url="https://github.com/yirage-project/yirage",
        ext_modules=config_cython(),
        include_package_data=True,
        # **setup_args,
    )
