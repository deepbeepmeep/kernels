"""Prepare the pinned Bonsai MTP sidecar without rewriting the target model."""
import argparse
import hashlib
import json
from pathlib import Path

import gguf
import numpy as np
from safetensors import safe_open

REPO = "ProCreations/Ternary-Bonsai-2-27B-MTP"
REVISION = "6f66852e436806f0db2b6ff5351bf30a0eaa95f9"
SOURCE_SHA256 = "c7d477c1dff218744069dbf0a8e287fbc29c2a0bab38183c03dc1798bb2b0643"
NAMES = {
    "fc": "fc", "pre_fc_norm_embedding": "pre_fc_norm_embedding",
    "pre_fc_norm_hidden": "pre_fc_norm_hidden", "norm": "norm",
    "layers.0.input_layernorm": "block.attn_norm",
    "layers.0.post_attention_layernorm": "block.post_attention_norm",
    "layers.0.self_attn.q_proj": "block.attn_q",
    "layers.0.self_attn.k_proj": "block.attn_k",
    "layers.0.self_attn.v_proj": "block.attn_v",
    "layers.0.self_attn.o_proj": "block.attn_output",
    "layers.0.self_attn.q_norm": "block.attn_q_norm",
    "layers.0.self_attn.k_norm": "block.attn_k_norm",
    "layers.0.mlp.gate_proj": "block.ffn_gate",
    "layers.0.mlp.up_proj": "block.ffn_up",
    "layers.0.mlp.down_proj": "block.ffn_down",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.source.open("rb") as stream:
        source_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if source_sha256 != SOURCE_SHA256:
        raise ValueError("Source does not match the pinned ProCreations MTP checkpoint.")
    writer = gguf.GGUFWriter(str(args.output), "qwen35")
    writer.add_name("Bonsai 2 27B adapted MTP Q8_0 sidecar")
    writer.add_string("bonsai.mtp.source_repository", REPO)
    writer.add_string("bonsai.mtp.source_revision", REVISION)
    prepared = {}
    with safe_open(args.source, framework="pt", device="cpu") as source:
        assert len(source.keys()) == len(NAMES)
        for name in source.keys():
            target = "mtp." + NAMES[name.removeprefix("mtp.").removesuffix(".weight")] + ".weight"
            value = source.get_tensor(name).float().numpy()
            # Qwen stores zero-centered norms; WanGP/llama.cpp use effective scales.
            if value.ndim == 1:
                data, qtype = value + np.float32(1), gguf.GGMLQuantizationType.F32
            else:
                data, qtype = gguf.quantize(value, gguf.GGMLQuantizationType.Q8_0), gguf.GGMLQuantizationType.Q8_0
            writer.add_tensor(target, data, raw_dtype=qtype)
            prepared[target] = (data, qtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    reader = gguf.GGUFReader(str(args.output))
    assert len(reader.tensors) == len(prepared)
    for tensor in reader.tensors:
        data, qtype = prepared[tensor.name]
        assert tensor.tensor_type == qtype
        assert np.array_equal(tensor.data, data), tensor.name
    def sha(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    report = dict(repository=REPO, revision=REVISION, source_sha256=source_sha256,
                  output_sha256=sha(args.output), tensors=len(prepared), matrices="Q8_0",
                  norms="F32 effective multipliers", target_weights_modified=False)
    args.output.with_suffix(".provenance.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
