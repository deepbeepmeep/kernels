"""New draft exports must retain their trained rotary frequencies on Transformers 4/5."""
import json

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from shared.prompt_enhancer.block_draft import _load_draft_config


def load_config(tmp_path, **rope):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(dict(hidden_size=5120, num_attention_heads=32, head_dim=128,
                                    max_position_embeddings=262144, **rope)), encoding="utf-8")
    return _load_draft_config(path)


@pytest.mark.parametrize("theta", [10000., 10000000.])
def test_modern_default_rope_uses_checkpoint_frequencies(tmp_path, theta):
    config = load_config(tmp_path, rope_parameters={"rope_type": "default", "rope_theta": theta})
    rotary = Qwen3RotaryEmbedding(config, device="cpu")
    expected = 1.0 / (theta ** (torch.arange(0, 128, 2, device="cpu", dtype=torch.float32) / 128))
    torch.testing.assert_close(rotary.inv_freq, expected, rtol=0, atol=0)
    assert config.rope_theta == theta


def test_modern_yarn_matches_explicit_legacy_configuration(tmp_path):
    scaling = dict(rope_type="yarn", factor=32., original_max_position_embeddings=8192,
                   beta_fast=32., beta_slow=1.)
    modern = load_config(tmp_path, rope_parameters=dict(rope_theta=10000000., **scaling))
    legacy = load_config(tmp_path, rope_theta=10000000., rope_scaling=scaling)
    a, b = (Qwen3RotaryEmbedding(config, device="cpu") for config in (modern, legacy))
    torch.testing.assert_close(a.inv_freq, b.inv_freq, rtol=0, atol=0)
    assert a.attention_scaling == b.attention_scaling


def test_legacy_draft_export_is_unchanged(tmp_path):
    config = load_config(tmp_path, rope_theta=10000000.)
    assert config.rope_theta == 10000000.
    assert config.rope_scaling is None


def test_modern_schema_is_authoritative_when_legacy_fields_are_stale(tmp_path):
    config = load_config(tmp_path, rope_theta=10000., rope_scaling={"rope_type": "linear", "factor": 2.},
                         rope_parameters={"rope_type": "default", "rope_theta": 10000000.})
    assert config.rope_theta == 10000000.
    assert config.rope_scaling == {"rope_type": "default"}
