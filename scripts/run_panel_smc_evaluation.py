#!/usr/bin/env python3
"""Run SMC outcome-evaluation over a panel of resolved Polymarket markets.

This script is meant for the "is my model adequate?" empirical pass: run the
same resolved markets through a menu of model classes, save per-market outcome
scores and SMC diagnostics, and then inspect whether conclusions are stable
across neighboring model classes.

Example:

    conda run -n datasc_env python scripts/run_panel_smc_evaluation.py \
      --n-markets 1000 \
      --bucket-minutes 30 \
      --lookback 350 \
      --min-volume 10000 \
      --min-trades 200 \
      --models all \
      --n-particles 1500 \
      --mcmc-steps 2 \
      --record-traces \
      --store-arrays \
      --model-workers 3 \
      --output outputs/panel_smc_30m.pkl

The output pickle intentionally does not store final particle clouds; those are
too large for a 1000-market panel and are not needed for scoring.
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is in requirements, but keep robust.
    tqdm = None

from pml_market import (
    BaseModel,
    BasePrior,
    BurstMispricingLogARVolModel,
    BurstMispricingLogARVolPrior,
    GaussianVolModel,
    GaussianVolPrior,
    InverseProblem,
    LogARVolModel,
    LogARVolPrior,
    MispricingLogARVolModel,
    MispricingLogARVolPrior,
    ReversalMomentumBurstLogARVolModel,
    ReversalMomentumBurstLogARVolPrior,
    SMCInference,
)
from pml_market.data import (
    fetch_resolved_binary_markets,
    trajectory_to_arrays_raw,
    truncate_trajectory,
)
from pml_market.diagnostics import realized_information_gain


ModelFactory = Callable[[], tuple[Any, Any]]


MODEL_REGISTRY: dict[str, ModelFactory] = {
    "base": lambda: (BaseModel(), BasePrior()),
    "rw_volume": lambda: (GaussianVolModel(), GaussianVolPrior()),
    "log_ar_volume": lambda: (LogARVolModel(), LogARVolPrior()),
    "reversal": lambda: (MispricingLogARVolModel(), MispricingLogARVolPrior()),
    "reversal_burst": lambda: (BurstMispricingLogARVolModel(), BurstMispricingLogARVolPrior()),
    "reversal_momentum_burst": lambda: (
        ReversalMomentumBurstLogARVolModel(),
        ReversalMomentumBurstLogARVolPrior(),
    ),
}


def _clip_prob(p: float, eps: float = 1e-12) -> float:
    return min(max(float(p), eps), 1.0 - eps)


def _truth_probability(p_yes: float, y: int) -> float:
    p_yes = _clip_prob(p_yes)
    return p_yes if int(y) == 1 else 1.0 - p_yes


def _logit_safe(p: float) -> float:
    p = _clip_prob(p)
    return math.log(p / (1.0 - p))


def _posterior_entropy(p: float) -> float:
    p = _clip_prob(p)
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def _prices_yes_from_trajectory(trajectory: dict[str, Any]) -> np.ndarray:
    prices = np.asarray(trajectory["prices"], dtype=np.float64)
    winner_label = trajectory.get("winner_label")
    if winner_label == "Yes":
        return prices
    if winner_label == "No":
        return 1.0 - prices
    raise ValueError(f"trajectory winner_label must be 'Yes' or 'No', got {winner_label!r}")


def _market_metadata(trajectory: dict[str, Any], dx: np.ndarray, v: np.ndarray, y: int) -> dict[str, Any]:
    meta = dict(trajectory.get("metadata", {}))
    p_yes = _prices_yes_from_trajectory(trajectory)
    nonzero_volume = int(np.count_nonzero(v > 0))
    return {
        "slug": meta.get("slug"),
        "question": meta.get("question"),
        "condition_id": meta.get("condition_id"),
        "winner_label": trajectory.get("winner_label"),
        "y": int(y),
        "horizon": int(len(v)),
        "trade_count": int(meta.get("trade_count", 0) or 0),
        "total_bucket_volume": float(np.sum(v)),
        "mean_bucket_volume": float(np.mean(v)) if len(v) else 0.0,
        "max_bucket_volume": float(np.max(v)) if len(v) else 0.0,
        "nonzero_volume_buckets": nonzero_volume,
        "nonzero_volume_fraction": float(nonzero_volume / max(len(v), 1)),
        "mean_abs_dx": float(np.mean(np.abs(dx))) if len(dx) else 0.0,
        "max_abs_dx": float(np.max(np.abs(dx))) if len(dx) else 0.0,
        "initial_price_yes": float(p_yes[0]) if len(p_yes) else float("nan"),
        "final_price_yes": float(p_yes[-1]) if len(p_yes) else float("nan"),
        "pre_final_price_yes": float(p_yes[-2]) if len(p_yes) >= 2 else float("nan"),
    }


def _market_baseline_scores(p_yes: float, y: int) -> dict[str, Any]:
    p_yes = _clip_prob(p_yes)
    p_truth = _truth_probability(p_yes, y)
    return {
        "posterior": p_yes,
        "p_truth": p_truth,
        "log_loss": float(-math.log(p_truth)),
        "brier": float((p_yes - int(y)) ** 2),
        "correct_05": bool((p_yes >= 0.5) == bool(y)),
        "confidence": float(abs(p_yes - 0.5) * 2.0),
        "logit": float(_logit_safe(p_yes)),
    }


def _strip_result(
    result: dict[str, Any],
    *,
    y: int,
    pi0: float,
    elapsed_s: float,
    n_particles: int,
    ess_threshold: float,
    record_traces: bool,
) -> dict[str, Any]:
    posterior = _clip_prob(result["posterior"])
    p_truth = _truth_probability(posterior, y)
    out = {
        "posterior": posterior,
        "p_truth": p_truth,
        "log_loss": float(-math.log(p_truth)),
        "brier": float((posterior - int(y)) ** 2),
        "correct_05": bool((posterior >= 0.5) == bool(y)),
        "confidence": float(abs(posterior - 0.5) * 2.0),
        "posterior_entropy": float(_posterior_entropy(posterior)),
        "information_gain": float(realized_information_gain(posterior, pi0=pi0)),
        "log_BF": float(result["log_BF"]),
        "log_m0": float(result["log_m0"]),
        "log_m1": float(result["log_m1"]),
        "log_m_truth": float(result["log_m1"] if int(y) == 1 else result["log_m0"]),
        "elapsed_s": float(elapsed_s),
    }

    for label in ("smc0", "smc1"):
        ess = np.asarray(result[label]["ess_history"], dtype=np.float64)
        out[f"{label}_ess_min"] = float(np.min(ess)) if ess.size else float("nan")
        out[f"{label}_ess_mean"] = float(np.mean(ess)) if ess.size else float("nan")
        out[f"{label}_resample_count_proxy"] = int(
            np.sum(ess[:-1] < ess_threshold * n_particles)
        ) if ess.size > 1 else 0
        if record_traces:
            out[f"{label}_ess_history"] = ess
            if "log_inc" in result[label]:
                out[f"{label}_log_inc"] = np.asarray(result[label]["log_inc"], dtype=np.float64)

    if record_traces:
        out["pi_t"] = np.asarray(result["pi_t"], dtype=np.float64)
        out["log_BF_t"] = np.asarray(result["log_BF_t"], dtype=np.float64)

    return out


def _run_model_job(job: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Run one model on one market.

    Kept as a top-level function so ProcessPoolExecutor can import it under
    macOS' spawn-based multiprocessing start method.
    """
    model_name = str(job["model_name"])
    model_offset = int(job["model_offset"])
    market_idx = int(job["market_idx"])
    dx = np.asarray(job["dx"], dtype=np.float64)
    v = np.asarray(job["v"], dtype=np.float64)
    y = int(job["y"])

    model_seed = int(job["seed"]) + 100_000 * market_idx + 10_000 * model_offset
    model, prior = MODEL_REGISTRY[model_name]()
    problem = InverseProblem(model, prior)
    smc = SMCInference(
        n_particles=int(job["n_particles"]),
        ess_threshold=float(job["ess_threshold"]),
        mcmc_steps=int(job["mcmc_steps"]),
    )

    start = time.perf_counter()
    try:
        result = problem.infer(
            dx,
            v,
            smc,
            pi0=float(job["pi0"]),
            seed=model_seed,
            record_pi_t=bool(job["record_traces"]),
        )
        elapsed = time.perf_counter() - start
        return model_name, {
            "status": "ok",
            **_strip_result(
                result,
                y=y,
                pi0=float(job["pi0"]),
                elapsed_s=elapsed,
                n_particles=int(job["n_particles"]),
                ess_threshold=float(job["ess_threshold"]),
                record_traces=bool(job["record_traces"]),
            ),
        }
    except Exception as exc:  # keep the panel run alive.
        elapsed = time.perf_counter() - start
        return model_name, {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_s": float(elapsed),
        }


def _progress_write(message: str) -> None:
    if tqdm is not None:
        tqdm.write(message)
    else:
        print(message, flush=True)


def _calibration_bins(rows: list[dict[str, Any]], n_bins: int = 10) -> list[dict[str, Any]]:
    bins = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        vals = [
            (r["posterior"], r["y"])
            for r in rows
            if lo <= r["posterior"] < hi or (hi == 1.0 and r["posterior"] == 1.0)
        ]
        if vals:
            ps, ys = zip(*vals)
            bins.append({
                "lo": float(lo),
                "hi": float(hi),
                "n": len(vals),
                "mean_p": float(np.mean(ps)),
                "empirical_rate": float(np.mean(ys)),
            })
        else:
            bins.append({"lo": float(lo), "hi": float(hi), "n": 0})
    return bins


def summarize_runs(runs: list[dict[str, Any]], model_names: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {"models": {}, "pairwise": {}}
    for model_name in model_names:
        rows = []
        for run in runs:
            res = run["model_results"].get(model_name)
            if res and res.get("status") == "ok":
                rows.append({"y": run["market"]["y"], **res})
        if not rows:
            summary["models"][model_name] = {"n_success": 0}
            continue
        summary["models"][model_name] = {
            "n_success": len(rows),
            "mean_log_loss": float(np.mean([r["log_loss"] for r in rows])),
            "median_log_loss": float(np.median([r["log_loss"] for r in rows])),
            "mean_brier": float(np.mean([r["brier"] for r in rows])),
            "accuracy_05": float(np.mean([r["correct_05"] for r in rows])),
            "mean_p_truth": float(np.mean([r["p_truth"] for r in rows])),
            "mean_confidence": float(np.mean([r["confidence"] for r in rows])),
            "mean_information_gain": float(np.mean([r["information_gain"] for r in rows])),
            "overconfident_wrong_rate": float(
                np.mean([(not r["correct_05"]) and r["confidence"] >= 0.9 for r in rows])
            ),
            "mean_elapsed_s": float(np.mean([r["elapsed_s"] for r in rows])),
            "mean_resample_count_proxy": float(np.mean([
                r["smc0_resample_count_proxy"] + r["smc1_resample_count_proxy"]
                for r in rows
            ])),
            "calibration_bins": _calibration_bins(rows),
        }

    for i, a in enumerate(model_names):
        for b in model_names[i + 1:]:
            diffs = []
            disagreements = []
            for run in runs:
                ra = run["model_results"].get(a)
                rb = run["model_results"].get(b)
                if not ra or not rb or ra.get("status") != "ok" or rb.get("status") != "ok":
                    continue
                diffs.append(abs(ra["posterior"] - rb["posterior"]))
                disagreements.append((ra["posterior"] >= 0.5) != (rb["posterior"] >= 0.5))
            if diffs:
                summary["pairwise"][f"{a}__vs__{b}"] = {
                    "n": len(diffs),
                    "mean_abs_posterior_diff": float(np.mean(diffs)),
                    "median_abs_posterior_diff": float(np.median(diffs)),
                    "decision_disagreement_rate": float(np.mean(disagreements)),
                }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-markets", type=int, default=1000)
    parser.add_argument("--fetch-pool-size", type=int, default=None,
                        help=(
                            "Fetch/cache this many eligible trajectories, shuffle them, "
                            "then keep --n-markets after horizon filtering. Defaults to "
                            "ceil(n_markets * candidate_multiplier)."
                        ))
    parser.add_argument("--candidate-multiplier", type=float, default=2.0,
                        help=(
                            "Default fetch pool multiplier when --fetch-pool-size is not set. "
                            "Larger values make the final panel more random but cost more API time."
                        ))
    parser.add_argument("--bucket-minutes", type=int, default=30)
    parser.add_argument("--lookback", type=int, default=350,
                        help="Keep only the last N buckets; 0 disables truncation.")
    parser.add_argument("--min-volume", type=float, default=10_000.0)
    parser.add_argument("--min-trades", type=int, default=200)
    parser.add_argument("--min-horizon", type=int, default=30)
    parser.add_argument("--max-horizon", type=int, default=600)
    parser.add_argument("--cache-json", default=None,
                        help="Trajectory cache JSON. Defaults under outputs/.")
    parser.add_argument("--output", default=None,
                        help="Result pickle path. Defaults under outputs/.")
    parser.add_argument("--models", nargs="+", default=["all"],
                        choices=["all", *MODEL_REGISTRY.keys()])
    parser.add_argument("--n-particles", type=int, default=300)
    parser.add_argument("--mcmc-steps", type=int, default=2)
    parser.add_argument("--ess-threshold", type=float, default=0.5)
    parser.add_argument("--pi0", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--record-traces", action="store_true")
    parser.add_argument("--store-arrays", action="store_true",
                        help="Store exact truncated dx, v, times, and P(Yes) arrays in the pickle.")
    parser.add_argument("--store-raw-trajectory", action="store_true",
                        help="Store the truncated trajectory dict in each run for self-contained analysis.")
    parser.add_argument("--model-workers", type=int, default=1,
                        help=(
                            "Number of models to run in parallel within each market. "
                            "Use 1 for sequential execution."
                        ))
    parser.add_argument("--verbose-model-progress", action="store_true",
                        help="Print a line whenever an individual market/model fit finishes.")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--sleep-between", type=float, default=0.05)
    parser.add_argument("--fetch-workers", type=int, default=1,
                        help=(
                            "Threaded workers for fetching market histories when building "
                            "the JSON cache. Keep modest to avoid hammering the APIs."
                        ))
    parser.add_argument("--candidate-max-pages", type=int, default=50,
                        help=(
                            "Maximum Gamma market pages to scan before fetching histories. "
                            "Increase this when strict filters leave too few prepared markets."
                        ))
    parser.add_argument("--no-clob-history", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch/filter markets and write no inference results.")
    return parser.parse_args()


def main() -> None:
    run_start = time.perf_counter()
    args = parse_args()
    os.makedirs("outputs", exist_ok=True)
    model_names = list(MODEL_REGISTRY) if "all" in args.models else list(args.models)
    fetch_pool_size = args.fetch_pool_size
    if fetch_pool_size is None:
        fetch_pool_size = int(math.ceil(args.n_markets * max(args.candidate_multiplier, 1.0)))
    fetch_pool_size = max(fetch_pool_size, args.n_markets)
    cache_json = args.cache_json or (
        f"outputs/polymarket_panel_{args.bucket_minutes}m_"
        f"vol{int(args.min_volume)}_tr{args.min_trades}_pool{fetch_pool_size}.json"
    )
    output = args.output or (
        f"outputs/panel_smc_{args.bucket_minutes}m_"
        f"n{args.n_markets}_p{args.n_particles}_m{args.mcmc_steps}.pkl"
    )

    rng = random.Random(args.seed)
    fetch_start = time.perf_counter()
    trajectories = fetch_resolved_binary_markets(
        n=fetch_pool_size,
        bucket_minutes=args.bucket_minutes,
        min_volume=args.min_volume,
        min_trades=args.min_trades,
        cache_path=cache_json,
        use_clob_history=not args.no_clob_history,
        timeout=args.timeout,
        sleep_between=args.sleep_between,
        fetch_workers=args.fetch_workers,
        candidate_max_pages=args.candidate_max_pages,
        verbose=True,
    )
    fetch_elapsed_s = time.perf_counter() - fetch_start
    rng.shuffle(trajectories)

    prepare_start = time.perf_counter()
    prepared = []
    for traj in trajectories:
        if args.lookback and args.lookback > 0:
            traj = truncate_trajectory(traj, args.lookback)
        try:
            dx, v, y = trajectory_to_arrays_raw(traj)
        except Exception:
            continue
        if not (args.min_horizon <= len(v) <= args.max_horizon):
            continue
        prepared.append((traj, dx, v, y))
        if len(prepared) >= args.n_markets:
            break
    if len(prepared) < args.n_markets:
        print(
            f"warning: prepared only {len(prepared)} markets after filtering "
            f"from a fetched pool of {len(trajectories)}; consider increasing "
            "--fetch-pool-size or loosening horizon filters.",
            flush=True,
        )
    prepare_elapsed_s = time.perf_counter() - prepare_start

    payload: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {**vars(args), "effective_fetch_pool_size": fetch_pool_size},
        "model_names": model_names,
        "runs": [],
        "summary": {},
    }

    if args.dry_run:
        payload["prepared_markets"] = [
            _market_metadata(traj, dx, v, y)
            for traj, dx, v, y in prepared
        ]
        with open(output, "wb") as f:
            pickle.dump(payload, f)
        print(
            f"dry run prepared {len(prepared)} markets; "
            f"fetch={fetch_elapsed_s:.1f}s prepare={prepare_elapsed_s:.1f}s; "
            f"wrote {output}"
        )
        return

    iterator = enumerate(prepared)
    if tqdm is not None:
        iterator = tqdm(list(iterator), desc="markets")

    inference_start = time.perf_counter()
    for market_idx, (traj, dx, v, y) in iterator:
        market_meta = _market_metadata(traj, dx, v, y)
        p_yes = _prices_yes_from_trajectory(traj)
        run = {
            "market_index": int(market_idx),
            "market": market_meta,
            "market_baseline_final": _market_baseline_scores(market_meta["final_price_yes"], y),
            "market_baseline_pre_final": _market_baseline_scores(market_meta["pre_final_price_yes"], y),
            "model_results": {},
        }
        if args.store_arrays:
            run["arrays"] = {
                "dx": np.asarray(dx, dtype=np.float64),
                "v": np.asarray(v, dtype=np.float64),
                "prices_yes": np.asarray(p_yes, dtype=np.float64),
                "times": np.asarray(traj["times"], dtype=np.int64),
            }
        if args.store_raw_trajectory:
            run["trajectory"] = traj

        model_jobs = [
            {
                "market_idx": market_idx,
                "model_offset": model_offset,
                "model_name": model_name,
                "dx": dx,
                "v": v,
                "y": y,
                "seed": args.seed,
                "n_particles": args.n_particles,
                "ess_threshold": args.ess_threshold,
                "mcmc_steps": args.mcmc_steps,
                "pi0": args.pi0,
                "record_traces": args.record_traces,
            }
            for model_offset, model_name in enumerate(model_names)
        ]

        max_workers = min(max(int(args.model_workers), 1), len(model_jobs))
        if max_workers > 1:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_name = {
                    executor.submit(_run_model_job, job): job["model_name"]
                    for job in model_jobs
                }
                for future in as_completed(future_to_name):
                    try:
                        model_name, model_result = future.result()
                    except Exception as exc:
                        model_name = str(future_to_name[future])
                        model_result = {
                            "status": "error",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "elapsed_s": float("nan"),
                        }
                    run["model_results"][model_name] = model_result
                    if args.verbose_model_progress:
                        status = model_result.get("status", "unknown")
                        elapsed = float(model_result.get("elapsed_s", float("nan")))
                        _progress_write(
                            f"market {market_idx + 1}/{len(prepared)} "
                            f"{market_meta.get('slug')!r}: {model_name} "
                            f"{status} in {elapsed:.2f}s"
                        )
        else:
            for job in model_jobs:
                model_name, model_result = _run_model_job(job)
                run["model_results"][model_name] = model_result
                if args.verbose_model_progress:
                    status = model_result.get("status", "unknown")
                    elapsed = float(model_result.get("elapsed_s", float("nan")))
                    _progress_write(
                        f"market {market_idx + 1}/{len(prepared)} "
                        f"{market_meta.get('slug')!r}: {model_name} "
                        f"{status} in {elapsed:.2f}s"
                    )

        payload["runs"].append(run)
        if args.checkpoint_every > 0 and len(payload["runs"]) % args.checkpoint_every == 0:
            payload["summary"] = summarize_runs(payload["runs"], model_names)
            with open(output, "wb") as f:
                pickle.dump(payload, f)

    payload["summary"] = summarize_runs(payload["runs"], model_names)
    inference_elapsed_s = time.perf_counter() - inference_start
    total_elapsed_s = time.perf_counter() - run_start
    with open(output, "wb") as f:
        pickle.dump(payload, f)
    print(f"wrote {len(payload['runs'])} market runs to {output}")
    print(
        "timing: "
        f"fetch={fetch_elapsed_s:.1f}s "
        f"prepare={prepare_elapsed_s:.1f}s "
        f"inference={inference_elapsed_s:.1f}s "
        f"total={total_elapsed_s:.1f}s"
    )
    print("summary:")
    for model_name, summary in payload["summary"]["models"].items():
        if summary.get("n_success", 0):
            print(
                f"  {model_name:28s} n={summary['n_success']:4d} "
                f"logloss={summary['mean_log_loss']:.3f} "
                f"brier={summary['mean_brier']:.3f} "
                f"acc={summary['accuracy_05']:.3f} "
                f"time={summary['mean_elapsed_s']:.2f}s"
            )
        else:
            print(f"  {model_name:28s} n=0")


if __name__ == "__main__":
    main()
