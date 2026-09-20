"""Graph-replay timing of unchanged formats, for before/after comparisons."""
import argparse
import json
import statistics
from pathlib import Path
import torch
import llamacpp_gguf_cuda as kernels
from validate_release import packed_fixture


@torch.inference_mode()
def main(output):
    records=[]
    for name in ('Q4_K', 'Q6_K', 'IQ3_S', 'Q8_0'):
        packed, _ = packed_fixture(name)
        raw=torch.from_numpy(packed).reshape(128,-1).repeat(40,10).flatten().cuda()
        for batch in (1,3,64,256):
            x=torch.randn(batch,5120,device='cuda',dtype=torch.bfloat16)*.1
            for _ in range(5):
                kernels.linear(raw,name,(5120,5120),x,None,x.dtype)
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                result=kernels.linear(raw,name,(5120,5120),x,None,x.dtype)
            samples=[]
            for _ in range(7):
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record()
                for _ in range(50):
                    g.replay()
                b.record(); b.synchronize()
                samples.append(a.elapsed_time(b)/50)
            records.append(dict(qtype=name,batch=batch,milliseconds=statistics.median(samples)))
            del g, result
    Path(output).write_text(json.dumps(records,indent=2))
    print(json.dumps(records))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    main(p.parse_args().output)
