# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE-WanGP.txt)
"""Header-free NVRTC compilation and CUDA driver launches on the PyTorch stream."""
import ctypes as ct
import functools
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import time
import uuid

import torch


@functools.cache
def _libraries():
    major = torch.version.cuda.split('.')[0]
    if sys.platform == 'win32':
        rtc = ct.CDLL(str(Path(torch.__file__).parent / 'lib' / f'nvrtc64_{major}0_0.dll'))
        driver = ct.CDLL('nvcuda.dll')
    else:
        filename = f'libnvrtc.so.{major}'
        locations = [Path(torch.__file__).parent / 'lib' / filename]
        try:
            package = importlib.util.find_spec('nvidia.cuda_nvrtc')
        except ModuleNotFoundError:
            package = None
        if package is not None:
            locations.extend(Path(root) / 'lib' / filename for root in package.submodule_search_locations)
        rtc = ct.CDLL(str(next((path for path in locations if path.is_file()), filename)))
        driver = ct.CDLL('libcuda.so.1')
    rtc.nvrtcGetErrorString.restype = ct.c_char_p
    return rtc, driver


def _check(status, library, compiler=False):
    if status:
        if compiler:
            message = library.nvrtcGetErrorString(status).decode()
        else:
            detail = ct.c_char_p()
            library.cuGetErrorString(status, ct.byref(detail))
            message = detail.value.decode() if detail.value else str(status)
        raise RuntimeError(f'CUDA kernel: {message}')


def compile_ptx(source, architecture=75):
    rtc, _ = _libraries()
    major, minor = ct.c_int(), ct.c_int()
    _check(rtc.nvrtcVersion(ct.byref(major), ct.byref(minor)), rtc, True)
    options = [f'--gpu-architecture=compute_{architecture}', '--std=c++17', '--fmad=false']
    identity = repr((major.value, minor.value, options)).encode() + source.encode()
    key = hashlib.sha256(identity).hexdigest()
    path = Path.home() / '.cache' / 'wangp' / 'cuda' / (key + '.ptx')
    try:
        return path.read_bytes()
    except OSError:
        pass
    started = time.perf_counter()
    print(f'[GGUF CUDA] Compiling SM{architecture} Packed Kernel...', flush=True)
    program = ct.c_void_p()
    _check(rtc.nvrtcCreateProgram(ct.byref(program), source.encode(), b'wangp_gguf.cu', 0, None, None), rtc, True)
    try:
        encoded = (ct.c_char_p * len(options))(*(value.encode() for value in options))
        status = rtc.nvrtcCompileProgram(program, len(options), encoded)
        if status:
            size = ct.c_size_t()
            _check(rtc.nvrtcGetProgramLogSize(program, ct.byref(size)), rtc, True)
            log = ct.create_string_buffer(size.value)
            _check(rtc.nvrtcGetProgramLog(program, log), rtc, True)
            raise RuntimeError('GGUF CUDA compilation failed:\n' + log.value.decode())
        size = ct.c_size_t()
        _check(rtc.nvrtcGetPTXSize(program, ct.byref(size)), rtc, True)
        ptx = ct.create_string_buffer(size.value)
        _check(rtc.nvrtcGetPTX(program, ptx), rtc, True)
        data = ptx.value
    finally:
        _check(rtc.nvrtcDestroyProgram(ct.byref(program)), rtc, True)
    print(f'[GGUF CUDA] Compiled in {time.perf_counter() - started:.2f} s.', flush=True)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.' + uuid.uuid4().hex + '.tmp')
        temporary.write_bytes(data)
        os.replace(temporary, path)
    except OSError as error:
        print(f'[GGUF CUDA] Could not cache compiled kernel: {error}')
    return data


class Kernel:
    def __init__(self, ptx, name):
        _, self.driver = _libraries()
        self.module, self.function = ct.c_void_p(), ct.c_void_p()
        # Keep the module alive with the cached kernel, including graph replays.
        _check(self.driver.cuModuleLoadData(ct.byref(self.module), ct.c_char_p(ptx)), self.driver)
        _check(self.driver.cuModuleGetFunction(ct.byref(self.function), self.module, name.encode()), self.driver)

    def launch(self, grid, threads, values):
        arguments = (ct.c_void_p * len(values))(*(ct.addressof(value) for value in values))
        stream = ct.c_void_p(torch.cuda.current_stream().cuda_stream)
        _check(self.driver.cuLaunchKernel(self.function, grid[0], grid[1], 1, threads, 1, 1,
                                         0, stream, arguments, None), self.driver)
