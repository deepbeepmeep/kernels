"""Compile four SM120 attention binaries offline; runtime does not require Triton."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import subprocess


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--ptxas', type=Path, required=True)
parser.add_argument('--cuda-major', choices=('12', '13'), required=True)
args = parser.parse_args()
# Triton on Windows names the executable-specific override with an EXE suffix.
for key in ('TRITON_PTXAS_PATH', 'TRITON_PTXAS.EXE_PATH', 'TRITON_PTXAS_BLACKWELL_PATH', 'TRITON_PTXAS_BLACKWELL.EXE_PATH'):
    os.environ[key] = str(args.ptxas.resolve())
import triton
from triton.backends.compiler import GPUTarget
from triton.experimental.gluon._runtime import GluonASTSource

source = ROOT / 'csrc/sm120_async.py'
spec = importlib.util.spec_from_file_location('sm120_async', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
output = ROOT / 'src/llamacpp_gguf_cuda/kernels' / f'sm120_cu{args.cuda_major}'
output.mkdir(parents=True, exist_ok=True)
manifest = {'ptxas_version': subprocess.check_output([str(args.ptxas.resolve()), '--version'], text=True).strip(), 'ptx_version': 87 if args.cuda_major == '12' else 90, 'triton': triton.__version__, 'target': 'sm_120', 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'kernels': {}}
for name in ('q8_prefill_async_kernel', 'q8_grouped_async_kernel'):
    kernel = getattr(module, name)
    for dtype in ('fp16', 'bf16'):
        signature = {}
        for arg in kernel.arg_names:
            if arg == 'D':
                signature[arg] = 'constexpr'
            elif arg in ('q_ptr', 'out_ptr'):
                signature[arg] = '*' + dtype
            elif arg in ('k_ptr', 'v_ptr'):
                signature[arg] = '*i8'
            elif arg in ('ks_ptr', 'vs_ptr'):
                signature[arg] = '*fp16'
            elif arg in ('tables_ptr', 'cu_q_ptr', 'cu_k_ptr', 'lengths_ptr'):
                signature[arg] = '*i32'
            elif arg.endswith('_ptr'):
                signature[arg] = '*fp32'
            elif arg == 'SCALE':
                signature[arg] = 'fp32'
            else:
                signature[arg] = 'i32'
        attrs = {(index,): [['tt.divisibility', 16]] for index, arg in enumerate(kernel.arg_names) if signature[arg].startswith('*')}
        ast = GluonASTSource(kernel, signature, constexprs={'D': 256}, attrs=attrs)
        started = time.perf_counter()
        print(f'AOT build {name}/{dtype} for SM120, CUDA {args.cuda_major}', flush=True)
        compiled = triton.compile(ast, target=GPUTarget('cuda', 120, 32), options={'num_warps': 4, 'num_stages': 1, 'ptx_version': 87 if args.cuda_major == '12' else 90})
        assert compiled.metadata.global_scratch_size == 0 and compiled.metadata.profile_scratch_size == 0
        key = name.replace('_kernel', '') + '_' + dtype
        binary = compiled.asm['cubin']
        (output / f'{key}.cubin').write_bytes(binary)
        (output / f'{key}.ptx').write_text(compiled.asm['ptx'], encoding='utf-8')
        manifest['kernels'][key] = {'function': compiled.metadata.name, 'shared': compiled.metadata.shared, 'threads': 128, 'sha256': hashlib.sha256(binary).hexdigest(), 'signature': {k:v for k,v in signature.items() if v != 'constexpr'}, 'scratch_pointer_arguments': 2}
        print(f'AOT artifact ready in {time.perf_counter()-started:.2f}s: {key} ({len(binary)} bytes)', flush=True)
(output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
