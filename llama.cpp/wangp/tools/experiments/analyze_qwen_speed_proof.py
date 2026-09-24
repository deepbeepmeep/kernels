"""CPU-only analysis of balanced Qwen target-graph timing quartets.

Example:
  python tools/experiments/analyze_qwen_speed_proof.py run1.json run2.json \
      --output-dir D:/AMD/qwen-speed-proof/analysis

No rounds are trimmed. Each ABBA/BAAB quartet, rather than each graph replay,
is one resampling unit. Confidence intervals describe these tested workloads;
they do not establish performance on other prompts, GPUs or generation paths.
Uses only the Python standard library and never imports torch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path


COMPARISONS = {
    "main": ("installed", "candidate"),
    "duplicate_control": ("installed", "installed_copy"),
    "build_control": ("installed", "baseline"),
}
SCOPE = (
    "Fixed-input, full target GPU graph time on resident weights and caches. "
    "Excludes prefill, CPU scheduling, generation throughput, MTP drafting, "
    "acceptance and cache refresh. This experiment is not a peak-VRAM audit."
)


def quantile(values, probability):
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def derived_seed(seed, label):
    payload = f"{seed}:{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def resample_mean(values, rng, chunk_length):
    """Circular moving-block bootstrap; length 1 resamples whole quartets."""
    selected = []
    while len(selected) < len(values):
        start = rng.randrange(len(values))
        selected.extend(values[(start + offset) % len(values)]
                        for offset in range(min(chunk_length, len(values) - len(selected))))
    return statistics.fmean(selected)


def estimate(groups, *, draws, seed, chunk_length=1):
    """Equal-weight tested strata; resample chronological blocks within strata."""
    series = list(groups.values())
    estimate_log = statistics.fmean(statistics.fmean(values) for values in series)
    point = math.exp(estimate_log)
    # One block contains no information about between-block variability.
    ci = None
    if all(len(values) >= 2 for values in series):
        rng = random.Random(seed)
        boot = [statistics.fmean(resample_mean(values, rng, chunk_length)
                                 for values in series) for _ in range(draws)]
        ci = [math.exp(quantile(boot, .025)), math.exp(quantile(boot, .975))]
    return dict(ratio=point, speed_gain_percent=100 * (point - 1),
                time_reduction_percent=100 * (1 - 1 / point), ci95=ci)


def positive_number(value, description):
    if isinstance(value, bool):
        raise ValueError(f"{description}: expected a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{description}: expected a positive finite number")
    return result


def load_inputs(paths):
    samples, pairs, blocks, sources, warnings = [], [], [], [], []
    seen_runs = set()
    for path in paths:
        payload = path.read_bytes()
        report = json.loads(payload)
        run_id = str(report["run_id"])
        if run_id in seen_runs:
            raise ValueError(f"Repeated run_id {run_id!r}; refuse to double-count a run")
        seen_runs.add(run_id)
        replays = positive_number(report["replays"], f"{path}: replays")
        if int(replays) != replays:
            raise ValueError(f"{path}: replays must be an integer")
        records = report["records"]
        if not records:
            raise ValueError(f"{path}: no timing records")
        source = dict(path=str(path.resolve()), sha256=hashlib.sha256(payload).hexdigest(),
                      run_id=run_id, context=report["context"], replays=int(replays),
                      completed=report.get("completed"), expected_blocks=report.get("rounds"),
                      exact_logits_and_state=report.get("exact_logits_and_state"),
                      parity_replays=report.get("parity_replays"),
                      prompt_sha256=report.get("prompt_sha256"),
                      gpu=report.get("gpu"), torch=report.get("torch"),
                      benchmark_sha256=next((value for key, value in report.get("source_sha256", {}).items()
                                             if key.replace("\\", "/").endswith("/benchmark_qwen_fusion_graph.py")), None),
                      wrapper_reference=report.get("wrapper_reference", "Not recorded: cross-wrapper parity is recorded separately"),
                      records=len(records))
        sources.append(source)
        if report.get("completed") is not True:
            warnings.append(f"{run_id}: run is not marked completed; this is a partial result.")
        if report.get("exact_logits_and_state") is not True:
            warnings.append(f"{run_id}: exact_logits_and_state is not true; no validated speed claim.")
        wrapper_parity = report.get("wrapper_parity", [])
        if len(wrapper_parity) != 8 or any(row.get("exact") is not True for row in wrapper_parity):
            warnings.append(f"{run_id}: missing complete exact-parity evidence for eight timed wrappers.")
        if int(replays) not in (report.get("parity_replays") or []):
            warnings.append(f"{run_id}: timed replay count was not included in the parity check.")
        if report.get("error") or report.get("errors"):
            warnings.append(f"{run_id}: raw report records an error; inspect original JSON.")
        seen_records = set()
        for record_index, record in enumerate(records):
            comparison = record["comparison"]
            if comparison not in COMPARISONS:
                raise ValueError(f"Unknown comparison {comparison!r}")
            reference, candidate = COMPARISONS[comparison]
            order = record["order"]
            order_name = "".join("A" if item == reference else "B" if item == candidate else "?"
                                 for item in order)
            if order_name not in ("ABBA", "BAAB"):
                raise ValueError(f"{run_id}/{record_index}: unbalanced quartet {order!r}")
            block_id = record["block_id"]
            unique = (comparison, str(block_id))
            if unique in seen_records:
                raise ValueError(f"{run_id}: duplicate comparison/block {unique!r}")
            seen_records.add(unique)
            rows = record["samples"]
            if len(rows) != 4:
                raise ValueError(f"{run_id}/{record_index}: expected all four raw samples")
            common = dict(source=str(path.resolve()), run_id=run_id, context=report["context"],
                          prompt_sha256=report.get("prompt_sha256"), record_index=record_index,
                          comparison=comparison, block_id=block_id, quartet_order=order_name)
            by_variant = defaultdict(list)
            measured = []
            for position, row in enumerate(rows):
                if row["variant"] != order[position] or row["position"] != position:
                    raise ValueError(f"{run_id}/{record_index}: sample/order mismatch at {position}")
                total = positive_number(row["gpu_ms_total"], "gpu_ms_total")
                duration = positive_number(row["gpu_ms"], "gpu_ms")
                if not math.isclose(duration, total / replays, rel_tol=1e-8, abs_tol=1e-10):
                    raise ValueError(f"{run_id}/{record_index}: gpu_ms is not total/replays")
                by_variant[row["variant"]].append(duration)
                measured.append(duration)
                samples.append(dict(**common, position=position, variant=row["variant"],
                                    slot=row.get("slot"), gpu_ms_total=total, gpu_ms=duration,
                                    replays=int(replays), enqueue_ns=record.get("enqueue_ns")))
            pair_logs = {}
            for pair_index, start in enumerate((0, 2)):
                if order[start] == reference:
                    ref_ms, candidate_ms, pair_order = measured[start], measured[start + 1], "AB"
                else:
                    candidate_ms, ref_ms, pair_order = measured[start], measured[start + 1], "BA"
                log_ratio = math.log(ref_ms) - math.log(candidate_ms)
                pair_logs[pair_order] = log_ratio
                pairs.append(dict(**common, pair_index=pair_index, pair_order=pair_order,
                                  reference_ms=ref_ms, candidate_ms=candidate_ms,
                                  log_ratio=log_ratio, ratio=math.exp(log_ratio)))
            log_ratio = statistics.fmean(pair_logs.values())
            blocks.append(dict(**common, reference_variant=reference, candidate_variant=candidate,
                               reference_geomean_ms=math.exp(statistics.fmean(map(math.log, by_variant[reference]))),
                               candidate_geomean_ms=math.exp(statistics.fmean(map(math.log, by_variant[candidate]))),
                               reference_mean_ms=statistics.fmean(by_variant[reference]),
                               candidate_mean_ms=statistics.fmean(by_variant[candidate]),
                               log_ratio=log_ratio, ratio=math.exp(log_ratio),
                               ab_log_ratio=pair_logs["AB"], ba_log_ratio=pair_logs["BA"],
                               order_log_effect=pair_logs["AB"] - pair_logs["BA"]))
    return samples, pairs, blocks, sources, warnings


def summarize(blocks, *, draws, seed, label):
    groups = defaultdict(list)
    order_groups = defaultdict(list)
    for row in blocks:
        # run_id is unique and each raw run contains a single context/prompt.
        groups[row["run_id"]].append(row["log_ratio"])
        order_groups[row["run_id"]].append(row["order_log_effect"])
    summary = estimate(groups, draws=draws, seed=derived_seed(seed, label))
    summary.update(blocks=len(blocks), launches=4 * len(blocks), strata=len(groups),
                   median_block_ratio=statistics.median(row["ratio"] for row in blocks),
                   min_block_ratio=min(row["ratio"] for row in blocks),
                   max_block_ratio=max(row["ratio"] for row in blocks),
                   blocks_ratio_below_one=sum(row["ratio"] < 1 for row in blocks),
                   p05_block_ratio=quantile([row["ratio"] for row in blocks], .05),
                   p95_block_ratio=quantile([row["ratio"] for row in blocks], .95),
                   reference_mean_ms=statistics.fmean(row["reference_mean_ms"] for row in blocks),
                   candidate_mean_ms=statistics.fmean(row["candidate_mean_ms"] for row in blocks))
    summary["ab_vs_ba_order_effect"] = estimate(order_groups, draws=draws,
                                                  seed=derived_seed(seed, label + ":order"))
    # Sensitivity to correlation between consecutive timing quartets. This is
    # still conditional on the measured runs, not a new independent replicate.
    summary["adjacent_two_quartet_ci95"] = estimate(
        groups, draws=draws, seed=derived_seed(seed, label + ":moving2"), chunk_length=2)["ci95"]
    summary["quartet_orders"] = {}
    for order in ("ABBA", "BAAB"):
        selected = [row for row in blocks if row["quartet_order"] == order]
        by_run = defaultdict(list)
        for row in selected:
            by_run[row["run_id"]].append(row["log_ratio"])
        summary["quartet_orders"][order] = dict(
            blocks=len(selected), ratio=math.exp(statistics.fmean(
                statistics.fmean(values) for values in by_run.values())) if selected else None)
    return summary


def analyze(blocks, sources, warnings, *, draws, seed):
    results = {}
    for comparison in COMPARISONS:
        selected = [row for row in blocks if row["comparison"] == comparison]
        if not selected:
            warnings.append(f"Missing comparison: {comparison}.")
            continue
        result = summarize(selected, draws=draws, seed=seed, label=comparison)
        result["runs"] = {}
        for source in sources:
            run_id = source["run_id"]
            subset = [row for row in selected if row["run_id"] == run_id]
            if not subset:
                warnings.append(f"{run_id}: missing {comparison}.")
                continue
            run = summarize(subset, draws=draws, seed=seed, label=f"{comparison}:{run_id}")
            run.update(context=source["context"], prompt_sha256=source["prompt_sha256"])
            result["runs"][run_id] = run
            orders = run["quartet_orders"]
            if orders["ABBA"]["blocks"] != orders["BAAB"]["blocks"]:
                warnings.append(f"{run_id}/{comparison}: ABBA/BAAB counts are unbalanced; all retained.")
            if run["blocks"] < 10:
                warnings.append(f"{run_id}/{comparison}: fewer than ten quartets; interval is fragile.")
            if source["expected_blocks"] is not None and run["blocks"] != source["expected_blocks"]:
                warnings.append(f"{run_id}/{comparison}: {run['blocks']} of {source['expected_blocks']} expected quartets are present.")
        results[comparison] = result
    repeated_workloads = []
    workload_sources = defaultdict(list)
    for source in sources:
        if source["prompt_sha256"]:
            workload_sources[(str(source["context"]), source["prompt_sha256"])].append(source)
    for (context, prompt_sha256), matched in workload_sources.items():
        if len(matched) < 2:
            continue
        run_ids = [source["run_id"] for source in matched]
        entry = dict(context=context, prompt_sha256=prompt_sha256, run_ids=run_ids, comparisons={})
        for comparison in COMPARISONS:
            selected = [row for row in blocks if row["comparison"] == comparison and row["run_id"] in run_ids]
            if selected:
                item = summarize(selected, draws=draws, seed=seed, label=f"repeat:{context}:{prompt_sha256}:{comparison}")
                run_ratios = {run: results[comparison]["runs"][run]["ratio"] for run in run_ids
                              if run in results[comparison]["runs"]}
                item.update(run_ratios=run_ratios,
                            max_min_run_ratio=max(run_ratios.values()) / min(run_ratios.values()))
                entry["comparisons"][comparison] = item
        repeated_workloads.append(entry)
    return dict(scope=SCOPE, bootstrap_draws=draws, seed=seed, confidence_level=.95,
                method="Percentile bootstrap of whole ABBA/BAAB quartets within each run; equal weight per tested run.",
                interval_scope="Conditional on tested run/context/prompt strata; not a cross-run or cross-workload prediction interval.",
                exclusions="None: every supplied timing quartet and sample is retained.",
                sources=sources, warnings=warnings, comparisons=results,
                repeated_workloads=repeated_workloads)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ratio_text(value):
    return f"{value:.5f}×" if value is not None else "unavailable"


def ci_text(value):
    return f"{value[0]:.5f}–{value[1]:.5f}×" if value is not None else "unavailable"


def markdown(report):
    lines = ["# Qwen target-graph timing evidence", "", report["scope"], "",
             "Ratios are reference time / compared time: above 1 means the compared graph was faster. "
             "Main = installed/candidate; duplicate control = installed/installed_copy; "
             "build control = installed/baseline (new binary with optimizations disabled).", "",
             "All raw launches, adjacent pairs and quartets are retained in "
             "[samples.csv](samples.csv), [pairs.csv](pairs.csv) and [blocks.csv](blocks.csv). "
             "No outlier removal or post-hoc timing exclusions were applied. "
             "The overall estimate weights each supplied process equally; a workload repeated in "
             "two processes receives twice the weight of a workload tested in one process.", "",
             "| Comparison | Quartets | Geometric speed ratio | 95% CI | Speed gain | Time reduction | Median quartet |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, result in report["comparisons"].items():
        lines.append(f"| {name} | {result['blocks']} | {ratio_text(result['ratio'])} | "
                     f"{ci_text(result['ci95'])} | {result['speed_gain_percent']:+.2f}% | "
                     f"{result['time_reduction_percent']:+.2f}% | {ratio_text(result['median_block_ratio'])} |")
    lines += ["", "## Each tested workload", "",
              "| Run | Context | Comparison | Quartets | Geometric ratio | 95% CI | Mean ref / compared ms | Range |",
              "|---|---:|---|---:|---:|---:|---:|---:|"]
    for name, result in report["comparisons"].items():
        for run_id, run in result["runs"].items():
            lines.append(f"| {run_id} | {run['context']} | {name} | {run['blocks']} | "
                         f"{ratio_text(run['ratio'])} | {ci_text(run['ci95'])} | "
                         f"{run['reference_mean_ms']:.4f} / {run['candidate_mean_ms']:.4f} | "
                         f"{run['min_block_ratio']:.5f}–{run['max_block_ratio']:.5f}× |")
    lines += ["", "## Timing-order and control diagnostics", "",
              "AB/BA compares the speed ratio when the reference was first in its adjacent pair "
              "with the ratio when it was second. A value of 1 indicates no measured pair-order effect. "
              "ABBA/BAAB are separate quartet-order estimates, not additional trials.", "",
              "| Comparison | AB/BA effect (95% CI) | ABBA ratio (n) | BAAB ratio (n) | Adjacent-two-quartet CI | Quartets below 1 | 5th–95th percentile |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, result in report["comparisons"].items():
        effect, orders = result["ab_vs_ba_order_effect"], result["quartet_orders"]
        lines.append(f"| {name} | {ratio_text(effect['ratio'])} ({ci_text(effect['ci95'])}) | "
                     f"{ratio_text(orders['ABBA']['ratio'])} ({orders['ABBA']['blocks']}) | "
                     f"{ratio_text(orders['BAAB']['ratio'])} ({orders['BAAB']['blocks']}) | "
                     f"{ci_text(result['adjacent_two_quartet_ci95'])} | {result['blocks_ratio_below_one']}/{result['blocks']} | "
                     f"{result['p05_block_ratio']:.5f}–{result['p95_block_ratio']:.5f}× |")
    lines += ["", "The duplicate control should be interpreted as a noise/bias diagnostic, "
              "not subtracted from the main result. Build control separates changed binary behavior "
              "from the optional graph optimizations. Control intervals containing 1 do not prove "
              "equivalence or rule out all bias.", "", "## Method and limits", "",
              "Each quartet contributes the mean of two log reference/candidate ratios. "
              "The reported geometric ratio averages quartet log ratios within each run, then "
              "weights each tested run equally. The median is over all observed quartet ratios. "
              "Means in milliseconds are descriptive arithmetic means, not the ratio estimator.", "",
              f"{report['method']} {report['bootstrap_draws']:,} draws; seed {report['seed']}. "
              "The adjacent-two-quartet sensitivity interval resamples circular chunks of two "
              "chronologically adjacent quartets to allow short-range serial correlation. "
              "Different comparison/stratum streams use deterministic SHA-256-derived seeds.", "",
              report["interval_scope"] + " Separate processes at different contexts/prompts do not "
              "establish repeated-process reproducibility at one fixed workload. Identical-workload "
              "repetitions, when supplied, are shown separately below. CIs do not prove "
              "the absence of regressions; per-workload results and controls remain necessary.", "",
              "Speed gain = 100 × (ratio − 1); elapsed-time reduction = 100 × (1 − 1/ratio). "
              "Neither number is an end-to-end tokens/s improvement."]
    if report["repeated_workloads"]:
        lines += ["", "## Identical-workload process repetitions", "",
                  "These processes share context and prompt hash. Their separate point estimates "
                  "show observed reproducibility; with only a few processes, the within-process "
                  "bootstrap CI is not a reliable estimate of process-to-process uncertainty.", "",
                  "| Context | Comparison | Process ratios | Equal-process ratio | Within-process 95% CI | Max/min process ratio |",
                  "|---|---|---|---:|---:|---:|"]
        for workload in report["repeated_workloads"]:
            for name, item in workload["comparisons"].items():
                ratios = "; ".join(f"{run}: {ratio_text(ratio)}" for run, ratio in item["run_ratios"].items())
                lines.append(f"| {workload['context']} | {name} | {ratios} | {ratio_text(item['ratio'])} | "
                             f"{ci_text(item['ci95'])} | {ratio_text(item['max_min_run_ratio'])} |")
    lines += ["", "## Input provenance", ""]
    for source in report["sources"]:
        lines += [f"- `{source['run_id']}`: `{source['path']}`; SHA-256 `{source['sha256']}`; "
                  f"completed: `{source['completed']}`; exact logits/state: `{source['exact_logits_and_state']}`; "
                  f"parity replays: `{source['parity_replays']}`; prompt SHA-256: `{source['prompt_sha256']}`. "
                  f"Harness SHA-256: `{source['benchmark_sha256']}`. "
                  f"Timed-wrapper reference: {source['wrapper_reference']}."]
    lines += ["", "Source JSON files retain the binary/source manifest and full run metadata."]
    if report.get("analyzer"):
        analyzer = report["analyzer"]
        lines += ["", f"Analyzer: `{analyzer['path']}`; SHA-256 `{analyzer['sha256']}`; "
                  f"Python `{analyzer['python']}`."]
    if report.get("setup_failures"):
        lines += ["", "## Setup attempts with no timing data", "",
                  "These attempts failed before any timing quartet was recorded. They are documented "
                  "separately and contribute no observations to the timing statistics.", ""]
        for failure in report["setup_failures"]:
            lines.append(f"- `{failure['path']}`; SHA-256 `{failure['sha256']}`; "
                         f"error: {failure['error']}")
    if report["warnings"]:
        lines += ["", "## Data-quality notes", ""]
        lines += [f"- {warning}" for warning in report["warnings"]]
    return "\n".join(lines) + "\n"


def self_test():
    import tempfile

    # A known multiplicative effect remains exact despite a linear trend in
    # log latency, because ABBA and BAAB have equal mean sample positions.
    for order in ("ABBA", "BAAB"):
        values = [math.exp(.2 * position) * (2 if kind == "A" else 1)
                  for position, kind in enumerate(order)]
        effect = (statistics.fmean(math.log(values[i]) for i, kind in enumerate(order) if kind == "A")
                  - statistics.fmean(math.log(values[i]) for i, kind in enumerate(order) if kind == "B"))
        assert math.isclose(math.exp(effect), 2, rel_tol=1e-12)
    constant = estimate({"run": [math.log(1.08)] * 20}, draws=100, seed=17)
    assert math.isclose(constant["ratio"], 1.08, rel_tol=1e-12)
    assert all(math.isclose(value, 1.08, rel_tol=1e-12) for value in constant["ci95"])
    varied = {"a": [math.log(value) for value in (1, 1.01, 1.07, 1.02)], "b": [0.] * 4}
    assert estimate(varied, draws=100, seed=17) == estimate(varied, draws=100, seed=17)
    assert estimate({"one": [0.]}, draws=100, seed=17)["ci95"] is None
    assert math.isclose(estimate({"a": [math.log(4)] * 2, "b": [0.] * 20},
                                 draws=100, seed=17)["ratio"], 2, rel_tol=1e-12)
    # Exercise the actual raw schema, exact retention, identical-workload
    # grouping and output writers, rather than only the estimator in isolation.
    with tempfile.TemporaryDirectory(prefix="qwen-speed-analysis-") as directory:
        root = Path(directory)
        paths = []
        for process in range(2):
            report = dict(run_id=f"synthetic-{process}", context=2048, rounds=12, replays=2,
                          completed=True, prompt_sha256="synthetic", exact_logits_and_state=True,
                          parity_replays=[1, 2, 3], wrapper_parity=[dict(exact=True)] * 8, records=[])
            for block_id in range(12):
                for comparison, (reference, candidate) in COMPARISONS.items():
                    ratio = {"main": 1.08, "duplicate_control": 1., "build_control": 1.002}[comparison]
                    order = [reference, candidate, candidate, reference] if block_id % 2 == 0 else [candidate, reference, reference, candidate]
                    samples = []
                    used = defaultdict(int)
                    for position, variant in enumerate(order):
                        duration = 10 * math.exp(.01 * position + .03 * block_id + .1 * process)
                        duration /= ratio if variant == candidate else 1
                        samples.append(dict(variant=variant, position=position, slot=used[variant],
                                            gpu_ms=duration, gpu_ms_total=2 * duration))
                        used[variant] += 1
                    report["records"].append(dict(block_id=block_id, comparison=comparison,
                                                  order=order, samples=samples, enqueue_ns=1000))
            path = root / f"run{process}.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            paths.append(path)
        samples, pairs, blocks, sources, warnings = load_inputs(paths)
        assert (len(samples), len(pairs), len(blocks)) == (288, 144, 72)
        result = analyze(blocks, sources, warnings, draws=100, seed=17)
        assert not result["warnings"]
        assert len(result["repeated_workloads"]) == 1
        assert math.isclose(result["comparisons"]["main"]["ratio"], 1.08, rel_tol=1e-12)
        assert math.isclose(result["comparisons"]["duplicate_control"]["ratio"], 1., rel_tol=1e-12)
        write_csv(root / "samples.csv", samples)
        assert len(list(csv.DictReader((root / "samples.csv").open(encoding="utf-8")))) == 288
        assert "Identical-workload process repetitions" in markdown(result)
        try:
            load_inputs(paths + paths[:1])
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate runs must not be double-counted")
    print("CPU statistical and raw-schema self-checks passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--setup-failure", type=Path, action="append", default=[],
                        help="Document a failed setup with zero timing records; may be repeated")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        if not args.inputs:
            return
    if not args.inputs or args.output_dir is None:
        parser.error("Provide raw JSON files and --output-dir, or use --self-test")
    if args.bootstrap_draws < 1000:
        parser.error("Use at least 1,000 bootstrap draws")
    samples, pairs, blocks, sources, warnings = load_inputs(args.inputs)
    result = analyze(blocks, sources, warnings, draws=args.bootstrap_draws, seed=args.seed)
    analyzer_path = Path(__file__).resolve()
    result["analyzer"] = dict(path=str(analyzer_path), sha256=hashlib.sha256(analyzer_path.read_bytes()).hexdigest(),
                               python=sys.version.split()[0], argv=sys.argv)
    result["setup_failures"] = []
    for path in args.setup_failure:
        payload = path.read_bytes()
        failed = json.loads(payload)
        if failed.get("records"):
            raise ValueError(f"{path}: setup failure contains timing records; cannot exclude those observations")
        error = str(failed.get("error") or failed.get("errors") or failed.get("traceback") or "See original setup log")
        result["setup_failures"].append(dict(path=str(path.resolve()),
                                              sha256=hashlib.sha256(payload).hexdigest(),
                                              error=error.strip().splitlines()[-1]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("samples", samples), ("pairs", pairs), ("blocks", blocks)):
        write_csv(args.output_dir / f"{name}.csv", rows)
    (args.output_dir / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.output_dir / "report.md").write_text(markdown(result), encoding="utf-8")
    print(json.dumps(dict(report=str((args.output_dir / "report.md").resolve()),
                          samples=len(samples), quartets=len(blocks),
                          comparisons={name: {key: value for key, value in entry.items()
                                               if key in ("ratio", "ci95", "median_block_ratio")}
                                       for name, entry in result["comparisons"].items()},
                          warnings=warnings), indent=2))


if __name__ == "__main__":
    main()
