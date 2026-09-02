import torch

import llamacpp_gguf_cuda


def run_case(rows):
    out_features, in_features = 128, 5120
    packed_row_bytes = in_features // 256 * 210
    raw_weight = torch.zeros(out_features * packed_row_bytes, dtype=torch.uint8, device="cuda")
    hidden = torch.randn(rows, in_features, dtype=torch.bfloat16, device="cuda")
    callback = lambda: llamacpp_gguf_cuda.linear(raw_weight, "Q6_K", (out_features, in_features), hidden, None, torch.bfloat16)
    for _ in range(3):
        callback()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = callback()
    for _ in range(100):
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch.zeros_like(output), atol=0, rtol=0)


if __name__ == "__main__":
    for row_count in (1, 2, 3):
        run_case(row_count)
        print(f"passed rows={row_count}")
