"""Numerical/graph checks and launch sweep for fused PTQ1 decode."""
import argparse, json, statistics
from pathlib import Path
import torch
import llamacpp_gguf_cuda as kernels
from test_ptq1 import fixture


def load_candidate():
    from llamacpp_gguf_cuda import _prism
    return _prism


def timer(fn):
    for _ in range(4): fn()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(32): y = fn()
    samples = []
    for _ in range(5):
        a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(10): g.replay()
        b.record(); b.synchronize()
        samples.append(a.elapsed_time(b)/320)
    return statistics.median(samples)


@torch.inference_mode()
def main(args):
    module = load_candidate()
    torch.manual_seed(234)
    records=[]
    if args.bench:
        shapes=[(5120,5120),(34816,5120),(5120,17408),(12288,5120),(5120,6144),(256,5120)]
    else:
        shapes=[(130,1024),(256,5120),(130,6144),(128,17408)]
    for rows,width in shapes:
        raw,dense = fixture(rows,width)
        raw=torch.from_numpy(raw).cuda()
        for dtype in ((torch.bfloat16,) if args.bench else (torch.float16,torch.bfloat16,torch.float32)):
            x=torch.randn((1,width*2),device='cuda',dtype=dtype)[:,::2] * .1
            signs=(torch.randint(0,2,(width,),device='cuda',dtype=torch.int8)*2-1)
            groups=[(0,0,0)] + ([(16,3,128)] if width==6144 and not args.bench else [])
            for group in groups:
                for bias in ([None] if args.bench else [None,torch.randn(rows,device='cuda',dtype=dtype)*.01]):
                    def reference(): return kernels.linear(raw,'PTQ1_0',(rows,width),kernels.prism_hadamard(x,signs,False,group),bias,dtype)
                    expected=reference()
                    entry=dict(rows=rows,width=width,dtype=str(dtype),group=group,bias=bias is not None)
                    if args.bench: entry['baseline_ms']=timer(reference)
                    results=[]
                    for nw in (1,2,4,8):
                        for nr in (1,2,4):
                            def run(): return module.decode(x,raw,signs,bias,rows,*group,nw,nr)
                            actual=run()
                            if nw==4:
                                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                            else:
                                relative=((actual.float()-expected.float()).norm()/expected.float().norm()).item()
                                assert relative < .002, (nw,nr,relative)
                            g=torch.cuda.CUDAGraph()
                            with torch.cuda.graph(g): replayed=run()
                            for _ in range(2):
                                x.normal_(std=.1)
                                if bias is not None: bias.normal_(std=.01)
                                g.replay()
                                torch.testing.assert_close(replayed,run(),rtol=0,atol=0)
                                if nw==4: torch.testing.assert_close(replayed,reference(),rtol=0,atol=0)
                            expected=reference()
                            results.append(dict(warps=nw,rows=nr,**({'ms':timer(run)} if args.bench else {})))
                            del g,replayed
                    entry['candidates']=results
                    records.append(entry)
                    print(json.dumps(entry),flush=True)
    Path(args.output).write_text(json.dumps(records,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True);p.add_argument('--bench',action='store_true')
    main(p.parse_args())
