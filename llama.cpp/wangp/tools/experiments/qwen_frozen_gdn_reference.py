"""Load a literal Git HEAD GDN reference for benchmark captures only.

The original module bytes are saved outside production source. Its imports,
Triton kernel definition, line numbers, and launch parameters remain unchanged.
Only the Python call adapter accepts the current caller's new layout keywords.

Usage in a benchmark, after loading the model:
    reference, metadata = load_frozen_reference(repo_root, proof_directory)
    block._gdn_direct_layout = False
    block._gdn_recurrent_raw = reference

Save and restore each block's original callable when capturing the candidate.
Running this file directly only freezes source and prints hashes; it imports no
GPU libraries and launches no kernels.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_PATH = "shared/kernels/qwen_gdn.py"


def _git_bytes(*arguments: str, repo_root: Path = _REPO_ROOT) -> bytes:
    return subprocess.run(
        ["git", *arguments], cwd=repo_root, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout


def freeze_head_source(output_dir: str | Path, *, repo_root: Path = _REPO_ROOT) -> dict:
    """Save exact committed module bytes, without importing Torch or Triton."""
    repo_root = Path(repo_root).resolve()
    revision = _git_bytes("rev-parse", "HEAD", repo_root=repo_root).decode("ascii").strip()
    source_ref = f"{revision}:{_SOURCE_PATH}"
    source = _git_bytes("show", source_ref, repo_root=repo_root)
    blob = _git_bytes("rev-parse", source_ref, repo_root=repo_root).decode("ascii").strip()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = output_dir / f"qwen_gdn_head_{revision}.py"
    if frozen_path.exists():
        if frozen_path.read_bytes() != source:
            raise RuntimeError(f"Frozen reference differs from Git HEAD: {frozen_path}")
    else:
        frozen_path.write_bytes(source)
    metadata = {
        "repository": str(repo_root),
        "git_revision": revision,
        "source_git_path": _SOURCE_PATH,
        "git_blob": blob,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "frozen_path": str(frozen_path),
        "literal_committed_source": True,
        "monkeypatch_target": "model.blk[*]._gdn_recurrent_raw",
        "required_direct_layout": False,
    }
    frozen_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_frozen_reference(repo_root: Path, output_dir: Path):
    """Return (compatible reference callable, source metadata) for capture."""
    metadata = freeze_head_source(output_dir, repo_root=repo_root)
    # Retain a normal file-backed module so Triton's source inspection and
    # compilation cache see the literal original kernel, with its old lines.
    name = f"_qwen_frozen_gdn_{metadata['source_sha256']}"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, metadata["frozen_path"])
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load frozen GDN reference: {metadata['frozen_path']}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise

    def reference(q, k, v, a, b, ssm_a, ssm_dt, initial, snapshots=None, *,
                  v_heads_tiled=False, ssm_params_tiled=False, interleave_ab=False):
        if v_heads_tiled or ssm_params_tiled or interleave_ab:
            raise AssertionError("Frozen GDN reference requires all direct-layout flags to be false.")
        return module.recurrent_raw_gates(q, k, v, a, b, ssm_a, ssm_dt, initial, snapshots)

    reference.frozen_module = module
    reference.source_metadata = metadata
    return reference, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze_head_source(args.output), indent=2))
