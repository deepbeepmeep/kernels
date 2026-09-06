import pytest

from shared.prompt_enhancer.config import (
    PROMPT_ENHANCER_SPECULATIVE_DECODING_CHOICES,
    normalize_prompt_enhancer_speculative_decoding,
    resolve_prompt_enhancer_speculative_decoding,
    validate_prompt_enhancer_speculative_decoding,
)


@pytest.mark.parametrize("mode", [0, 1, 2, 3, 4])
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
