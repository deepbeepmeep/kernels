import pytest

from shared.prompt_enhancer.config import (
    PROMPT_ENHANCER_SPECULATIVE_DECODING_CHOICES,
    normalize_prompt_enhancer_speculative_decoding,
    resolve_prompt_enhancer_speculative_decoding,
    validate_prompt_enhancer_speculative_decoding,
    speculative_decoding_config,
    speculative_decoding_runtime,
    speculative_decoding_ui_state,
)


@pytest.mark.parametrize("mode", [0, 1, 2, 3, 4, "dspark", "dflash2"])
def test_saved_speculative_modes_round_trip(mode):
    assert normalize_prompt_enhancer_speculative_decoding(mode) == mode
    assert mode in [value for _label, value in PROMPT_ENHANCER_SPECULATIVE_DECODING_CHOICES]
    assert validate_prompt_enhancer_speculative_decoding(5, mode) == mode
    if mode != 2:
        assert resolve_prompt_enhancer_speculative_decoding(5, mode, total_vram_gb=32) == (mode, "")


def test_automatic_speculation_retains_vram_policy_and_legacy_two_token_mode():
    assert resolve_prompt_enhancer_speculative_decoding(5, "auto", total_vram_gb=32)[0] == 1
    assert resolve_prompt_enhancer_speculative_decoding(5, "auto", total_vram_gb=16)[0] == 0
    assert resolve_prompt_enhancer_speculative_decoding(4, "auto", total_vram_gb=12)[0] == 1
    assert normalize_prompt_enhancer_speculative_decoding("yes") == 1
    assert normalize_prompt_enhancer_speculative_decoding(False) == 0


@pytest.mark.parametrize("mode", [1, 3, 4])
def test_explicit_speculation_requires_supported_model(mode):
    with pytest.raises(ValueError, match="not available with Qwen3.5-4B"):
        validate_prompt_enhancer_speculative_decoding(3, mode)


@pytest.mark.parametrize("mode", [3, 4])
def test_explicit_draft_count_accepts_text_config(mode):
    assert normalize_prompt_enhancer_speculative_decoding(str(mode)) == mode


@pytest.mark.parametrize("method", ["dspark", "dflash2"])
def test_parallel_drafts_require_27b_and_preserve_explicit_selection(method):
    assert normalize_prompt_enhancer_speculative_decoding(method.upper()) == method
    assert resolve_prompt_enhancer_speculative_decoding(5, method, total_vram_gb=8) == (method, "")
    with pytest.raises(ValueError, match="require Qwen3.8"):
        validate_prompt_enhancer_speculative_decoding(4, method)


@pytest.mark.parametrize("legacy,method,tokens", [
    (0, "disabled", None), (1, "mtp", 2), (2, "auto", None),
    (3, "mtp", 3), (4, "mtp", 4), ("dspark", "dspark", 7), ("dflash2", "dflash2", 5),
])
def test_version_123_migration_preserves_selection_and_is_idempotent(legacy, method, tokens):
    from shared.utils.wgp_config_migration import migrate_extension_defaults

    config = {"extensions_defaults_version": "1.22", "prompt_enhancer_speculative_decoding": legacy}
    assert migrate_extension_defaults(config)
    assert config["extensions_defaults_version"] == "1.23"
    assert config["prompt_enhancer_speculative_decoding"] == {"method": method, "tokens": tokens}
    assert not migrate_extension_defaults(config)


@pytest.mark.parametrize("model,quantization,engine,methods", [
    (3, "int8", "vllm", ["auto", "disabled"]),
    (4, "int8", "vllm", ["auto", "disabled", "mtp"]),
    (5, "gguf", "vllm", ["auto", "disabled", "mtp", "dspark", "dflash2"]),
    (5, "gguf_q3", "vllm", ["auto", "disabled", "mtp", "dspark", "dflash2"]),
    (5, "gguf_q2", "vllm", ["auto", "disabled", "mtp", "dspark", "dflash2"]),
    (5, "gguf_ptq1", "", ["auto", "disabled", "mtp", "dspark", "dflash2"]),
    (5, "gguf_ptq1", "cg", ["auto", "disabled", "mtp"]),
    (5, "gguf_ptq1", "legacy", ["auto", "disabled", "mtp"]),
])
def test_method_availability_and_vram_labels(model, quantization, engine, methods):
    choices, method, counts, count = speculative_decoding_ui_state(model, quantization, engine, "auto")
    assert [value for label, value in choices] == methods
    assert all("[+" in label and "GiB VRAM]" in label for label, value in choices)
    assert (method, counts, count) == ("auto", [], None)


@pytest.mark.parametrize("method,maximum,runtime_mode", [("mtp", 8, 1), ("dspark", 7, "dspark"), ("dflash2", 7, "dflash2")])
def test_counts_are_bounded_and_reach_runtime(method, maximum, runtime_mode):
    for count in range(1, maximum + 1):
        saved = speculative_decoding_config(method, count)
        assert resolve_prompt_enhancer_speculative_decoding(5, saved, 32) == (saved, "")
        assert speculative_decoding_runtime(saved) == (runtime_mode, count)
        _, selected, choices, tokens = speculative_decoding_ui_state(5, "gguf", "vllm", saved)
        assert (selected, choices, tokens) == (method, list(range(1, maximum + 1)), count)
    with pytest.raises(ValueError, match="supports 1 to"):
        speculative_decoding_config(method, maximum + 1)


def test_model_and_method_switches_reset_or_clamp():
    assert speculative_decoding_ui_state(4, "int8", "vllm", "dspark", 7)[1:] == ("auto", [], None)
    assert speculative_decoding_ui_state(5, "gguf", "cg", "dflash2", 5)[1:] == ("auto", [], None)
    assert speculative_decoding_ui_state(5, "gguf_ptq1", "vllm", "dflash2", 8)[-1] == 5
    assert speculative_decoding_ui_state(5, "gguf_ptq1", "vllm", "disabled", 8)[1:] == ("disabled", [], None)


@pytest.mark.parametrize("backend,folder,maximum", [
    ("gguf", "Qwen3_8_27B_DFlash2", 7),
    ("gguf_q3", "Qwen3_8_27B_DFlash2", 7),
    ("gguf_q2", "Qwen3_8_27B_DFlash2", 7),
    ("gguf_ptq1", "Bonsai_2_27B_DFlash2", 5),
])
def test_dflash_checkpoint_and_token_limits_follow_target(backend, folder, maximum):
    from shared.prompt_enhancer.block_draft import ensure_block_draft_assets, block_draft_spec
    downloads = []
    ensure_block_draft_assets(lambda **kwargs: downloads.append(kwargs), "dflash2", "27b", backend)
    assert downloads[0]["sourceFolderList"] == [folder]
    assert block_draft_spec("dflash2", bonsai=backend == "gguf_ptq1")["drafts"] == maximum
    assert speculative_decoding_ui_state(5, backend, "vllm", "dflash2")[2:] == (list(range(1, maximum + 1)), maximum)
    assert speculative_decoding_ui_state(5, backend, "vllm", "dflash2", 7)[-1] == maximum
