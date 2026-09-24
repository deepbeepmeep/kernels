import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from plugins.configuration import plugin
from shared.deepy import config
from shared.prompt_enhancer import loader
from shared.remote_llm.config import ENGINE_CODEX, ENGINE_LOCAL_1, ENGINE_LOCAL_2, ENGINE_QWEN38_27B


@pytest.fixture
def save_config(monkeypatch):
    source = ast.parse(Path(plugin.__file__).read_text(encoding="utf-8-sig"))
    method = next(node for node in ast.walk(source) if isinstance(node, ast.FunctionDef) and node.name == "_save_changes")
    unpack = next(node for node in ast.walk(method) if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name) and node.value.id == "fixed_args")
    names = [node.id for node in unpack.targets[0].elts]
    monkeypatch.setattr(plugin.notifications, "prepare_config_update", lambda *args: {})
    monkeypatch.setattr(plugin.gr, "Info", lambda *args: None)

    class Written(Exception):
        pass

    def save(engine, mode, saved=None, **overrides):
        old = dict(saved or {})
        values = dict.fromkeys(names, "")
        values.update(deepy_llm_engine_choice=engine, deepy_type_choice=mode, enhancer_enabled_choice=5,
                      enhancer_quantization_choice="gguf", enhancer_speculative_decoding_choice=0,
                      deepy_context_tokens_choice=32000, deepy_compaction_type_choice="summarize",
                      deepy_prime_mcp_servers_choice="{}", deepy_allow_read_file_system_choice="disabled",
                      video_container_choice="mp4", video_output_codec_choice="libx264_8", audio_output_codec_choice="aac_128",
                      prompt_enhancer_temperature_choice=0.7, prompt_enhancer_top_p_choice=0.9,
                      codex_executable_choice="codex", voice_mode_choice="auto", int8_kernels_choice="disabled", kernel_precision_choice="strict")
        values.update(overrides)
        host = SimpleNamespace(server_config=old, server_config_filename="unused.json", args=SimpleNamespace(lock_config=False),
                               is_generation_in_progress=lambda: False, fl=SimpleNamespace(default_checkpoints_paths=[], set_checkpoints_paths=lambda paths: None),
                               audio_processor_config_bindings=[], temporal_upsampler_config_bindings=[], upsampler_config_bindings=[])
        captured = {}

        def write(server_config, filename, updates, *, remove):
            captured.update(server_config)
            captured.update(updates)
            for key in remove:
                captured.pop(key)
            raise Written

        monkeypatch.setattr(plugin, "update_config", write)
        try:
            result = plugin.ConfigTabPlugin._save_changes(host, {}, *(values[name] for name in names))
        except Written:
            return captured
        raise ValueError(result[0])

    return save


@pytest.mark.parametrize("engine", [ENGINE_LOCAL_1, ENGINE_LOCAL_2])
def test_florence_save_ignores_hidden_qwen_options(save_config, engine):
    saved = {"prompt_enhancer_speculative_decoding": 4, "prompt_enhancer_quantization": "gguf_q3",
             "deepy_context_tokens": 8192, "deepy_compaction_type": "summarize", "deepy_compaction_thinking": True,
             "deepy_kv_cache_quantization": "int8", "deepy_repetition_penalty": True}
    result = save_config(engine, "disabled", saved, enhancer_speculative_decoding_choice=4,
                         enhancer_quantization_choice="invalid", deepy_context_tokens_choice="invalid",
                         deepy_compaction_type_choice="summarize_thinking", deepy_kv_cache_quantization_choice="invalid")
    assert all(result[key] == value for key, value in saved.items())


def test_disabled_deepy_save_preserves_hidden_values_without_parsing(save_config):
    saved = {"deepy_context_tokens": 64000, "deepy_compaction_type": "summarize", "deepy_compaction_thinking": True,
             "deepy_prime_mcp_servers": {"saved": {"command": "existing"}}, "deepy_file_system_paths": ["saved path"],
             "deepy_vram_mode": "always_loaded", "deepy_prime_custom_system_prompt": "saved guidance"}
    result = save_config(ENGINE_QWEN38_27B, "disabled", saved, deepy_context_tokens_choice=8192,
                         deepy_compaction_type_choice="summarize_thinking", deepy_prime_mcp_servers_choice="{broken json",
                         deepy_file_system_paths_choice='"unclosed', deepy_prime_custom_system_prompt_choice="edited hidden value")
    assert result["deepy_enabled"] == 0
    assert result["deepy_context_tokens"] == 64000
    assert all(result[key] == value for key, value in saved.items())


@pytest.mark.parametrize("engine", [ENGINE_LOCAL_1, ENGINE_LOCAL_2])
@pytest.mark.parametrize("mode", ["zero", "prime"])
def test_florence_cannot_be_saved_with_deepy_enabled(save_config, engine, mode):
    with pytest.raises(ValueError, match="Florence 2 is not compatible with Deepy"):
        save_config(engine, mode, deepy_prime_mcp_servers_choice="{invalid hidden JSON")


@pytest.mark.parametrize("enabled", [False, True])
def test_prompt_penalty_can_be_saved_with_deepy_disabled(save_config, enabled):
    result = save_config(ENGINE_QWEN38_27B, "disabled", {"deepy_repetition_penalty": not enabled}, deepy_repetition_penalty_choice=enabled)
    assert result["deepy_repetition_penalty"] is enabled
    runtime = config.normalize_deepy_runtime_config(result)
    assert runtime["deepy_repetition_penalty"] is enabled


def test_prompt_penalty_is_visible_with_deepy_disabled():
    source = ast.parse(Path(plugin.__file__).read_text(encoding="utf-8-sig"))
    callback = next(node for node in ast.walk(source) if isinstance(node, ast.FunctionDef) and node.name == "update_remote_engine_ui")
    namespace = {**vars(plugin), "self": SimpleNamespace(server_config={})}
    exec(compile(ast.Module(body=[callback], type_ignores=[]), plugin.__file__, "exec"), namespace)
    updates = namespace["update_remote_engine_ui"](ENGINE_QWEN38_27B, "gguf", "disabled")
    assert updates[-1]["visible"] is True
    assert updates[-2]["visible"] is False


def test_hidden_filesystem_paths_are_not_parsed(save_config):
    result = save_config(ENGINE_QWEN38_27B, "prime", {"deepy_file_system_paths": ["saved path"]},
                         deepy_file_system_paths_choice='"unclosed')
    assert result["deepy_file_system_paths"] == ["saved path"]


def test_remote_save_preserves_hidden_local_options(save_config):
    saved = {"deepy_context_tokens": 8192, "deepy_compaction_type": "summarize", "deepy_compaction_thinking": True,
             "prompt_enhancer_temperature": 0.8, "prompt_enhancer_speculative_decoding": 4}
    result = save_config(ENGINE_CODEX, "prime", saved, deepy_context_tokens_choice="invalid", enhancer_speculative_decoding_choice="invalid")
    assert all(result[key] == value for key, value in saved.items())


@pytest.mark.parametrize("method,tokens,quantization,expected", [
    ("mtp", 8, "gguf", {"method": "mtp", "tokens": 8}),
    ("dspark", 3, "gguf", {"method": "dspark", "tokens": 3}),
    ("dspark", 7, "gguf_ptq1", {"method": "dspark", "tokens": 7}),
    ("dflash2", 3, "gguf_ptq1", {"method": "dflash2", "tokens": 3}),
    ("dflash2", 7, "gguf", {"method": "dflash2", "tokens": 7}),
    ("dflash2", 7, "gguf_q3", {"method": "dflash2", "tokens": 7}),
    ("dflash2", 7, "gguf_q2", {"method": "dflash2", "tokens": 7}),
    ("dflash2", 7, "gguf_ptq1", {"method": "dflash2", "tokens": 5}),
])
def test_save_separate_speculative_controls(save_config, method, tokens, quantization, expected):
    result = save_config(ENGINE_QWEN38_27B, "disabled", enhancer_speculative_decoding_choice=method,
                         enhancer_speculative_tokens_choice=tokens, enhancer_quantization_choice=quantization,
                         lm_decoder_engine_choice="vllm")
    assert result["prompt_enhancer_speculative_decoding"] == expected


@pytest.mark.parametrize("overrides, message", [
    ({"deepy_context_tokens_choice": 8192}, "32,000"),
    ({"deepy_prime_mcp_servers_choice": "{broken json"}, "Expecting property name"),
])
def test_visible_options_still_validate(save_config, overrides, message):
    with pytest.raises(ValueError, match=message):
        save_config(ENGINE_QWEN38_27B, "prime", **overrides)


def test_disabled_deepy_runtime_does_not_process_hidden_options(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A hidden Deepy option was processed")

    for name in ("normalize_deepy_context_tokens", "normalize_deepy_compaction_type", "normalize_deepy_compaction_thinking",
                 "normalize_deepy_vram_mode", "normalize_deepy_prime_mcp_servers", "normalize_deepy_file_system_paths"):
        monkeypatch.setattr(config, name, forbidden)
    saved = {"deepy_enabled": 0, "deepy_compaction_thinking": "unchanged", "deepy_prime_mcp_servers": "{broken json",
             "deepy_file_system_paths": '"unclosed'}
    result = config.normalize_deepy_runtime_config(saved)
    assert all(result[key] == value for key, value in saved.items())
    assert not config.deepy_available(saved)


def test_disabled_filesystem_access_does_not_read_hidden_paths(monkeypatch):
    from shared.deepy.filesystem import build_file_access_policy

    monkeypatch.setattr(config, "parse_deepy_file_system_paths", lambda *args: pytest.fail("Hidden paths were parsed"))
    policy = build_file_access_policy({"deepy_allow_read_file_system": "disabled", "deepy_file_system_paths": '"unclosed'})
    assert policy.selected_roots == ()
    assert not policy.read_everywhere


@pytest.mark.parametrize("engine", [ENGINE_LOCAL_1, ENGINE_LOCAL_2, ENGINE_CODEX])
def test_non_qwen_runtime_does_not_process_hidden_cache_options(monkeypatch, engine):
    def forbidden(*args):
        pytest.fail("A hidden Qwen option was processed")

    for name in ("normalize_deepy_context_tokens", "normalize_deepy_kv_cache_quantization", "normalize_deepy_compaction_type", "normalize_deepy_repetition_penalty"):
        monkeypatch.setattr(config, name, forbidden)
    saved = {"llm_engines": {"deepy": engine}, "deepy_enabled": 1, "deepy_type": "prime",
             "deepy_context_tokens": "unchanged", "deepy_compaction_type": "unchanged"}
    result = config.normalize_deepy_runtime_config(saved)
    assert result["deepy_context_tokens"] == result["deepy_compaction_type"] == "unchanged"


def test_disabled_controller_does_not_read_hidden_requirements_or_vram(monkeypatch):
    from shared.deepy import controller

    def forbidden(*args):
        pytest.fail("Disabled Deepy read hidden options")

    monkeypatch.setattr(controller, "deepy_requirement_met", forbidden)
    monkeypatch.setattr(controller, "normalize_deepy_vram_mode", forbidden)
    host = SimpleNamespace(_server_config=lambda: {"deepy_enabled": 0})
    assert controller.DeepyController.requirement_error_text(host) == controller._DEEPY_DISABLED_TEXT
    assert controller.DeepyController.get_vram_mode(host) == config.DEEPY_VRAM_MODE_UNLOAD


@pytest.mark.parametrize("enhancer", [1, 2])
def test_florence_assets_and_loading_never_resolve_qwen_options(monkeypatch, enhancer):
    def forbidden(*args, **kwargs):
        pytest.fail("Florence tried to process a Qwen option")

    monkeypatch.setattr(loader, "resolve_prompt_enhancer_speculative_decoding", forbidden)
    monkeypatch.setattr(loader, "resolve_deepy_kv_cache_quantization", forbidden)
    monkeypatch.setattr(loader.fl, "locate_folder", lambda *args: "unused")
    monkeypatch.setattr(loader, "load_florence2", lambda *args, **kwargs: (Mock(), Mock()))
    monkeypatch.setattr(loader, "_load_llama32_prompt_enhancer", lambda: (Mock(), Mock(), 10000))
    monkeypatch.setattr(loader, "_load_joycaption_prompt_enhancer", lambda: (Mock(), Mock(), 10000))
    downloads = Mock()
    loader.ensure_prompt_enhancer_assets(downloads, enhancer, speculative_decoding=4)
    runtime = loader.load_prompt_enhancer_runtime(downloads, enhancer, qwen_backend="invalid", speculative_decoding=4, deepy_kv_cache_quantization="invalid")
    assert runtime.llm_model is not None
    assert downloads.call_count == 2
