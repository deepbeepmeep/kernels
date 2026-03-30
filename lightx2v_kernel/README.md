# lightx2v_kernel

### Preparation

In Windows you need to install cmake. For example you run this in a MSVC developer console.


Create a python virtual environment then install torch, uv and scikit:

```

# Install torch, at least version 2.7 e.g. 
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

pip install scikit_build_core uv
```

### Build whl

```
#clone cutlass
git clone https://github.com/NVIDIA/cutlass.git
#clone this repo
git clone https://github.com/deepbeepmeep/kernels

cd kernels
cd lightx2v_kernel
```

Set the max Jobs and parallel level to the amount of cores that you have.
Set the /path/to/cutlass below to the absolute path of cutlass you download.


in windows
```bash
set MAX_JOBS=6 
set CMAKE_BUILD_PARALLEL_LEVEL=6
``` 

in Linux
```bash
set MAX_JOBS=6 
set CMAKE_BUILD_PARALLEL_LEVEL=6
``` 

no build the wheel
``` 
uv build --wheel -Cbuild-dir=build . -Ccmake.define.CUTLASS_PATH=/path/to/cutlass --verbose --color=always --no-build-isolation
```


### Install whl
```
pip install dist/*whl --force-reinstall --no-deps
```

### Test

##### cos and speed test, mm without bias
```
python test/nvfp4_nvfp4/test_bench2.py
```

##### cos and speed test, mm with bias
```
python test/nvfp4_nvfp4/test_bench3_bias.py
```

##### Bandwidth utilization test for quant
```
python test/nvfp4_nvfp4/test_quant_mem_utils.py
```

##### tflops test for mm
```
python test/nvfp4_nvfp4/test_mm_tflops.py
```
