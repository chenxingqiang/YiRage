# YiRage Installation

The quickest way to try YiRage is installing the latest stable release from Pypi:
```bash
pip install yirage-project
```

YiRage can also be built from source code using the following instructions.

## Intall from pre-built wheel
We provide some pre-built binary wheels in [Release Page](https://github.com/yirage-project/yirage/releases/latest). For example, to install yirage 0.2.2 compiled with CUDA 12.2 for python 3.10, using the following command:
```bash
pip install https://github.com/yirage-project/yirage/releases/download/v0.2.2/yirage_project-0.2.2+cu122-cp310-cp310-linux_x86_64.whl
```

## Install from source code

### Prerequisties

* CMAKE 3.24 or higher
* Cython 0.28 or higher
* CUDA 11.0 or higher and CUDNN 8.0 or higher

### Install the YiRage python package from source code
To get started, you can clone the YiRage source code from github.
```bash
git clone --recursive https://www.github.com/yirage-project/yirage
cd yirage
```

Then, you can simple build the YiRage runtime library from source code using the following command line
```bash
pip install -e . -v 
```
All dependenices will be automatically installed.

### Check your installation
Just try to import yirage in Python. If there is no output, then YiRage and all dependencies have been successfully installed.
```bash
python -c 'import yirage'
```

## Build Standalone C++ library
If you want to build standalone c++ library, you can follow the steps below.
Given that YIRAGE_ROOT points to top-level yirage project folder.
* Build the Z3 from source.
```bash
cd $YIRAGE_ROOT/deps/z3
mkdir build; cd build
cmake ..
make -j
```
* Export Z3 build directory.
```bash
export Z3_DIR=$YIRAGE_ROOT/deps/z3/build
```
* Build yirage from source.
```bash
cd $YIRAGE_ROOT
mkdir build; cd build
cmake ..
make -j
make install
```
By default, yirage build process will generate a static library. To install yirage in your directory of choice
specify -CMAKE_INSTALL_PREFIX=path/to/your/directory as a cmake option.

## Docker images

We require [docker](https://docs.docker.com/engine/installation/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker/) to run the YiRage [docker images](https://hub.docker.com/r/mlso/yirage).

* First, clone the YiRage gitpub repository to obtain necessary scripts.
```bash
git clone --recursive https://www.github.com/yirage-project/yirage
```

* Second, use the following command to run a YiRage docker image. The default CUDA version is 12.4.
```bash
/path-to-yirage/docker/run_docker.sh mlso/yirage
```

* You are ready to use YiRage now. Try some of our demos to superoptimize DNNs.
```python
python demo/demo_group_query_attention_spec_decode.py --checkpoint demo/checkpoint_group_query_attn_spec_decode.json
```

## Modal

[Modal](https://modal.com) provides GPUs to try YPK with minimal setup.

```
pip install modal
modal setup
modal run demo/qwen3/demo_modal.py
```

