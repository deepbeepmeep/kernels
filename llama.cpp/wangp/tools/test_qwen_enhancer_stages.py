"""Real-checkpoint validation of vision -> captions -> prompt enhancement.

Run with --engine legacy/cg/vllm and --variant 3/4/5. Outputs include caption
and prompt text, MMGP load order, and graph/weight reuse assertions.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests
import torch
from PIL import Image
from mmgp import offload

from shared.prompt_enhancer.images import ImageContext
from shared.prompt_enhancer.loader import load_prompt_enhancer_runtime
from shared.prompt_enhancer.prompt_enhance_utils import generate_cinematic_prompt
from shared.prompt_enhancer.progress import EnhancementProgress
from shared.prompt_enhancer import qwen35_vl
from shared.utils import files_locator as fl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("legacy", "cg", "vllm"), default="cg")
    parser.add_argument("--variant", type=int, default=4)
    parser.add_argument("--backend", default="quanto_int8")
    parser.add_argument("--multi-reference", action="store_true")
    parser.add_argument("--draft", type=int, choices=range(5), default=0)
    parser.add_argument("--output", default="tmp/qwen_enhancer_stages")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    fl.set_checkpoints_paths(json.loads(Path("wgp_config.json").read_text())["checkpoints_paths"])
    fixture = output / "cats.jpg"
    if not fixture.exists():
        response = requests.get("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png", timeout=60)
        response.raise_for_status()
        fixture.write_bytes(response.content)
    image = Image.open(fixture).convert("RGB")
    detail = image.crop((0, 0, image.width // 2, image.height))
    contexts = [ImageContext([image, detail], ["start image", "end image"], 5), ImageContext([detail, image], ["start image", "Image reference no 2 (Frame 2s/5s)"], 5)]
    if args.multi_reference:
        dog_fixture = output / "dog.jpg"
        if not dog_fixture.exists():
            response = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg", timeout=60)
            response.raise_for_status()
            dog_fixture.write_bytes(response.content)
        dog = Image.open(dog_fixture).convert("RGB")
        other_cat = image.crop((image.width // 2, 0, image.width, image.height))
        contexts[0] = ImageContext([image, dog, other_cat, detail], ["start image", "Image reference no 1", "Image reference no 2", "end image"], 5)

    def require_assets(repoId, sourceFolderList, fileList, targetFolderList=None):
        for folder, files in zip(targetFolderList or sourceFolderList, fileList):
            for filename in files:
                fl.locate_file(f"{folder}/{filename}" if folder else filename)

    runtime = load_prompt_enhancer_runtime(require_assets, args.variant, lm_decoder_engine=args.engine, qwen_backend=args.backend, speculative_decoding=args.draft)
    manager = offload.profile(runtime.pipe_models, profile_no=4, budgets=runtime.budgets, coTenantsMap=runtime.co_tenants, pinnedMemory=False, quantizeTransformer=False, verboseLevel=1)
    events, captions = [], []
    original_load = manager.gpu_load_blocks
    original_unload = manager.unload_all
    original_caption = runtime.image_caption_model.generate_image_captions
    original_generate = runtime.llm_model.generate_messages

    def load(model_id, *args, **kwargs):
        events.append(("load", model_id))
        return original_load(model_id, *args, **kwargs)

    def unload():
        events.append(("unload_all", list(manager.active_models_ids)))
        return original_unload()

    def caption(*args, **kwargs):
        result = original_caption(*args, **kwargs)
        for context, description in zip(kwargs["image_contexts"], result):
            assert all(label in description for label in context.labels), description
        captions.extend(result)
        engine = runtime.llm_model._prompt_enhancer_vllm_engine
        events.append(("captions_done", id(engine)))
        runtime.llm_model._test_caption_engine = engine
        runtime.llm_model._test_caption_runner = engine._llm.model_runner
        runtime.llm_model._test_caption_pointers = [parameter.data_ptr() for parameter in runtime.llm_model.parameters()]
        return result

    def generate(*args, **kwargs):
        events.append(("enhancement_start",))
        result = original_generate(*args, **kwargs)
        assert runtime.llm_model._prompt_enhancer_vllm_engine is runtime.llm_model._test_caption_engine
        assert runtime.llm_model._prompt_enhancer_vllm_engine._llm.model_runner is runtime.llm_model._test_caption_runner
        assert runtime.llm_model._test_caption_pointers == [parameter.data_ptr() for parameter in runtime.llm_model.parameters()]
        events.append(("enhancement_done",))
        return result

    manager.gpu_load_blocks, manager.unload_all = load, unload
    runtime.image_caption_model.generate_image_captions = caption
    runtime.llm_model.generate_messages = generate
    results = []
    progress_events = []
    progress = EnhancementProgress(lambda value, desc, total, unit: progress_events.append(dict(value=value, title=desc, unit=unit)))
    generation_callbacks = dict(enhancement_progress=progress)
    original_features = runtime.image_caption_model.model.get_image_features
    original_prepare = qwen35_vl._prepare_multimodal_vllm_prompt

    def cpu_features(pixels, grids, **kwargs):
        assert pixels.device.type == grids.device.type == "cpu"
        return original_features(pixels, grids, **kwargs)

    def cpu_prepare(model, inputs, image_features=None):
        assert all(feature.device.type == "cpu" for feature in image_features)
        return original_prepare(model, inputs, image_features=image_features)

    runtime.image_caption_model.model.get_image_features = cpu_features
    qwen35_vl._prepare_multimodal_vllm_prompt = cpu_prepare
    try:
        for default_device in ("cpu", "cuda"):
            torch.set_default_device(default_device)
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                prompts = generate_cinematic_prompt(runtime.image_caption_model, runtime.image_caption_processor, runtime.llm_model, runtime.llm_tokenizer, ["The cats relax on the sofa.", "The cat looks around."], image_contexts=contexts, offload_manager=manager, max_new_tokens=96, do_sample=False, thinking_enabled=False, generation_callbacks=generation_callbacks)
            assert len(prompts) == 2 and all(len(prompt.strip()) > 15 for prompt in prompts)
            assert any(event["title"] == "Enhancing Prompt 1/2" and event["value"][0] > 0 for event in progress_events)
            assert any(event["title"] == "Enhancing Prompt 2/2" and event["value"][0] > 0 for event in progress_events)
            assert any(event["title"].startswith("Encoding Image 2/") and event["value"][0] > 0 for event in progress_events)
            assert any(event["title"] == "Captioning Images 1/2" and event["value"][0] > 0 for event in progress_events)
            results.append(dict(default_device=default_device, prompts=prompts, peak_vram=torch.cuda.max_memory_allocated()))
        # Cancel inside the real tower, then prove a fresh generation succeeds.
        cancelled = [False]
        handle = runtime.image_caption_model.vision_tower_model.blocks[0].register_forward_hook(lambda *args: cancelled.__setitem__(0, True))
        try:
            with torch.inference_mode():
                runtime.image_caption_model.generate_image_captions([], image_contexts=contexts, offload_manager=manager, stop_requested=lambda: cancelled[0])
            raise AssertionError("Vision cancellation was not propagated")
        except InterruptedError:
            assert not manager.active_models_ids
        finally:
            handle.remove()
        with torch.inference_mode():
            recovered = generate_cinematic_prompt(runtime.image_caption_model, runtime.image_caption_processor, runtime.llm_model, runtime.llm_tokenizer, ["The cats rest."], image_contexts=contexts[:1], offload_manager=manager, max_new_tokens=48, do_sample=False, thinking_enabled=False)
        results.append(dict(recovered=recovered))
        # Decoder cancellation uses the same scoped cancellation checks. Do not
        # cancel until vision has finished and the real text model is executing.
        with torch.inference_mode():
            try:
                runtime.image_caption_model.generate_image_captions([], image_contexts=contexts[:1], offload_manager=manager, stop_requested=lambda: "prompt_enhancer_llm_model" in manager.active_models_ids)
                raise AssertionError("Decoder cancellation was not propagated")
            except InterruptedError:
                assert not manager.active_models_ids
            recovered = generate_cinematic_prompt(runtime.image_caption_model, runtime.image_caption_processor, runtime.llm_model, runtime.llm_tokenizer, ["The cats rest."], image_contexts=contexts[:1], offload_manager=manager, max_new_tokens=48, do_sample=False, thinking_enabled=False)
        results.append(dict(recovered_after_decoder_cancel=recovered))
        # No MMGP unload or tower load between captioning and enhancement.
        in_handoff = False
        for event in events:
            if event[0] == "captions_done":
                in_handoff = True
            elif event[0] == "enhancement_done":
                in_handoff = False
            elif in_handoff:
                assert event[0] != "unload_all"
                assert not (event[0] == "load" and "vision_tower" in event[1])
        report = dict(engine=args.engine, variant=args.variant, backend=args.backend, results=results, captions=captions, events=events, progress=progress_events)
        suffix = "_multi_reference" if args.multi_reference else ""
        (output / f"result_{args.variant}_{args.backend}_{args.engine}{suffix}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(dict(results=results, captions=captions), indent=2))
    finally:
        runtime.image_caption_model.model.get_image_features = original_features
        qwen35_vl._prepare_multimodal_vllm_prompt = original_prepare
        torch.set_default_device("cpu")
        runtime.llm_model.unload()
        manager.release()


if __name__ == "__main__":
    main()
