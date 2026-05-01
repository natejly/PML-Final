#!/usr/bin/env python3
"""Post-process panel SMC evaluation pickles.

The panel runner saves a rich nested pickle so the expensive inference pass does
not have to be rerun for every table. This script flattens that pickle into
analysis-friendly CSV files and a short Markdown report.

Example:

    conda run --no-capture-output -n datasc_env python scripts/analyze_panel_results.py \
      --input outputs/panel_1000_30m_T500_p1500_m2_all.pkl \
      --outdir outputs/panel_1000_30m_T500_analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _clip_prob(p: float, eps: float = 1e-12) -> float:
    if not np.isfinite(p):
        return float("nan")
    return min(max(float(p), eps), 1.0 - eps)


def _safe_float(x: Any) -> float:
    try:
        out = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _safe_mean(values: list[float]) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    return float(np.mean(arr)) if arr.size else float("nan")


def _safe_median(values: list[float]) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    return float(np.median(arr)) if arr.size else float("nan")


def _safe_quantile(values: list[float], q: float) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    return float(np.quantile(arr, q)) if arr.size else float("nan")


def _truth_prob(p_yes: float, y: int) -> float:
    p_yes = _clip_prob(p_yes)
    return p_yes if int(y) == 1 else 1.0 - p_yes


def _log_loss_from_p(p_yes: float, y: int) -> float:
    p_truth = _truth_prob(p_yes, y)
    return float(-math.log(_clip_prob(p_truth)))


def _brier_from_p(p_yes: float, y: int) -> float:
    p_yes = _clip_prob(p_yes)
    return float((p_yes - int(y)) ** 2)


def _confidence(p_yes: float) -> float:
    p_yes = _clip_prob(p_yes)
    return float(abs(p_yes - 0.5) * 2.0)


def _bool_as_int(x: Any) -> int:
    return int(bool(x))


def _clean_for_json(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _clean_for_json(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean_for_json(v) for v in x]
    if isinstance(x, tuple):
        return [_clean_for_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_clean_for_json(v) for v in x.tolist()]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        value = float(x)
        return value if np.isfinite(value) else None
    if isinstance(x, float):
        return x if np.isfinite(x) else None
    return x


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(x: Any, digits: int = 4) -> str:
    try:
        value = float(x)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def _market_fields(run: dict[str, Any]) -> dict[str, Any]:
    market = run.get("market", {})
    return {
        "market_index": run.get("market_index"),
        "slug": market.get("slug"),
        "question": market.get("question"),
        "winner_label": market.get("winner_label"),
        "y": market.get("y"),
        "horizon": market.get("horizon"),
        "trade_count": market.get("trade_count"),
        "total_bucket_volume": market.get("total_bucket_volume"),
        "mean_bucket_volume": market.get("mean_bucket_volume"),
        "max_bucket_volume": market.get("max_bucket_volume"),
        "nonzero_volume_fraction": market.get("nonzero_volume_fraction"),
        "mean_abs_dx": market.get("mean_abs_dx"),
        "max_abs_dx": market.get("max_abs_dx"),
        "initial_price_yes": market.get("initial_price_yes"),
        "pre_final_price_yes": market.get("pre_final_price_yes"),
        "final_price_yes": market.get("final_price_yes"),
    }


def flatten_model_rows(payload: dict[str, Any], include_market_baselines: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model_names = list(payload.get("model_names", []))
    for run in payload.get("runs", []):
        market = _market_fields(run)
        y = int(market["y"])

        if include_market_baselines:
            for baseline_name, key in (
                ("market_pre_final", "market_baseline_pre_final"),
                ("market_final", "market_baseline_final"),
            ):
                base = run.get(key)
                if not base:
                    continue
                rows.append({
                    **market,
                    "model": baseline_name,
                    "status": "ok",
                    "posterior": base.get("posterior"),
                    "p_truth": base.get("p_truth"),
                    "log_loss": base.get("log_loss"),
                    "brier": base.get("brier"),
                    "correct_05": _bool_as_int(base.get("correct_05")),
                    "confidence": base.get("confidence"),
                    "elapsed_s": float("nan"),
                    "log_BF": float("nan"),
                    "log_m0": float("nan"),
                    "log_m1": float("nan"),
                    "log_m_truth": float("nan"),
                })

        for model_name in model_names:
            result = run.get("model_results", {}).get(model_name, {})
            status = result.get("status", "missing")
            row = {
                **market,
                "model": model_name,
                "status": status,
            }
            if status == "ok":
                row.update({
                    "posterior": result.get("posterior"),
                    "p_truth": result.get("p_truth"),
                    "log_loss": result.get("log_loss"),
                    "brier": result.get("brier"),
                    "correct_05": _bool_as_int(result.get("correct_05")),
                    "confidence": result.get("confidence"),
                    "posterior_entropy": result.get("posterior_entropy"),
                    "information_gain": result.get("information_gain"),
                    "log_BF": result.get("log_BF"),
                    "log_m0": result.get("log_m0"),
                    "log_m1": result.get("log_m1"),
                    "log_m_truth": result.get("log_m_truth"),
                    "elapsed_s": result.get("elapsed_s"),
                    "smc0_ess_min": result.get("smc0_ess_min"),
                    "smc1_ess_min": result.get("smc1_ess_min"),
                    "smc0_ess_mean": result.get("smc0_ess_mean"),
                    "smc1_ess_mean": result.get("smc1_ess_mean"),
                    "smc0_resample_count_proxy": result.get("smc0_resample_count_proxy"),
                    "smc1_resample_count_proxy": result.get("smc1_resample_count_proxy"),
                })
            else:
                row.update({
                    "error_type": result.get("error_type"),
                    "error": result.get("error"),
                    "elapsed_s": result.get("elapsed_s"),
                })
            # Recompute if older pickles lack a scoring field.
            if status == "ok" and "posterior" in row:
                p = _safe_float(row.get("posterior"))
                row.setdefault("p_truth", _truth_prob(p, y))
                row.setdefault("log_loss", _log_loss_from_p(p, y))
                row.setdefault("brier", _brier_from_p(p, y))
                row.setdefault("correct_05", int((p >= 0.5) == bool(y)))
                row.setdefault("confidence", _confidence(p))
            rows.append(row)
    return rows


def calibration_bins(rows: list[dict[str, Any]], n_bins: int) -> tuple[list[dict[str, Any]], float]:
    out: list[dict[str, Any]] = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0
    weighted_abs_error = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        vals = [
            (_safe_float(r.get("posterior")), int(r.get("y")))
            for r in rows
            if r.get("status") == "ok"
            and np.isfinite(_safe_float(r.get("posterior")))
            and (
                lo <= _safe_float(r.get("posterior")) < hi
                or (hi == 1.0 and _safe_float(r.get("posterior")) == 1.0)
            )
        ]
        if not vals:
            out.append({"lo": float(lo), "hi": float(hi), "n": 0})
            continue
        ps, ys = zip(*vals)
        mean_p = float(np.mean(ps))
        empirical_rate = float(np.mean(ys))
        count = len(vals)
        total += count
        weighted_abs_error += count * abs(mean_p - empirical_rate)
        out.append({
            "lo": float(lo),
            "hi": float(hi),
            "n": count,
            "mean_p": mean_p,
            "empirical_rate": empirical_rate,
            "abs_calibration_error": abs(mean_p - empirical_rate),
        })
    ece = weighted_abs_error / total if total else float("nan")
    return out, float(ece)


def summarize_by_model(rows: list[dict[str, Any]], n_bins: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("model"))].append(row)

    summary_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for model_name, model_rows in grouped.items():
        ok = [r for r in model_rows if r.get("status") == "ok"]
        errors = [r for r in model_rows if r.get("status") != "ok"]
        bins, ece = calibration_bins(ok, n_bins)
        for b in bins:
            calibration_rows.append({"model": model_name, **b})

        correct = [_safe_float(r.get("correct_05")) for r in ok]
        conf = [_safe_float(r.get("confidence")) for r in ok]
        wrong_high_conf_90 = [
            int((not bool(r.get("correct_05"))) and _safe_float(r.get("confidence")) >= 0.9)
            for r in ok
        ]
        wrong_high_conf_80 = [
            int((not bool(r.get("correct_05"))) and _safe_float(r.get("confidence")) >= 0.8)
            for r in ok
        ]
        summary_rows.append({
            "model": model_name,
            "n_success": len(ok),
            "n_error": len(errors),
            "mean_log_loss": _safe_mean([_safe_float(r.get("log_loss")) for r in ok]),
            "median_log_loss": _safe_median([_safe_float(r.get("log_loss")) for r in ok]),
            "p90_log_loss": _safe_quantile([_safe_float(r.get("log_loss")) for r in ok], 0.9),
            "mean_brier": _safe_mean([_safe_float(r.get("brier")) for r in ok]),
            "median_brier": _safe_median([_safe_float(r.get("brier")) for r in ok]),
            "accuracy_05": _safe_mean(correct),
            "mean_p_truth": _safe_mean([_safe_float(r.get("p_truth")) for r in ok]),
            "median_p_truth": _safe_median([_safe_float(r.get("p_truth")) for r in ok]),
            "mean_confidence": _safe_mean(conf),
            "median_confidence": _safe_median(conf),
            "overconfident_wrong_rate_90": _safe_mean(wrong_high_conf_90),
            "overconfident_wrong_rate_80": _safe_mean(wrong_high_conf_80),
            "ece": ece,
            "mean_information_gain": _safe_mean([_safe_float(r.get("information_gain")) for r in ok]),
            "mean_elapsed_s": _safe_mean([_safe_float(r.get("elapsed_s")) for r in ok]),
            "mean_smc_resample_count_proxy": _safe_mean([
                _safe_float(r.get("smc0_resample_count_proxy"))
                + _safe_float(r.get("smc1_resample_count_proxy"))
                for r in ok
                if np.isfinite(_safe_float(r.get("smc0_resample_count_proxy")))
                and np.isfinite(_safe_float(r.get("smc1_resample_count_proxy")))
            ]),
            "mean_min_ess": _safe_mean([
                min(_safe_float(r.get("smc0_ess_min")), _safe_float(r.get("smc1_ess_min")))
                for r in ok
                if np.isfinite(_safe_float(r.get("smc0_ess_min")))
                and np.isfinite(_safe_float(r.get("smc1_ess_min")))
            ]),
        })

    summary_rows.sort(key=lambda r: (np.inf if not np.isfinite(_safe_float(r["mean_log_loss"])) else r["mean_log_loss"]))
    return summary_rows, calibration_rows


def compare_to_reference(rows: list[dict[str, Any]], reference_model: str) -> list[dict[str, Any]]:
    by_market_model: dict[tuple[Any, str], dict[str, Any]] = {}
    models: set[str] = set()
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (row.get("market_index"), str(row.get("model")))
        by_market_model[key] = row
        models.add(str(row.get("model")))

    out: list[dict[str, Any]] = []
    for model in sorted(models):
        if model == reference_model:
            continue
        delta_log_loss: list[float] = []
        delta_brier: list[float] = []
        abs_posterior_diff: list[float] = []
        decision_disagree: list[int] = []
        logloss_win: list[int] = []
        brier_win: list[int] = []
        for market_index, candidate in by_market_model:
            if candidate != model:
                continue
            ref = by_market_model.get((market_index, reference_model))
            cur = by_market_model.get((market_index, model))
            if ref is None or cur is None:
                continue
            cur_ll = _safe_float(cur.get("log_loss"))
            ref_ll = _safe_float(ref.get("log_loss"))
            cur_br = _safe_float(cur.get("brier"))
            ref_br = _safe_float(ref.get("brier"))
            cur_p = _safe_float(cur.get("posterior"))
            ref_p = _safe_float(ref.get("posterior"))
            if np.isfinite(cur_ll) and np.isfinite(ref_ll):
                delta_log_loss.append(cur_ll - ref_ll)
                logloss_win.append(int(cur_ll < ref_ll))
            if np.isfinite(cur_br) and np.isfinite(ref_br):
                delta_brier.append(cur_br - ref_br)
                brier_win.append(int(cur_br < ref_br))
            if np.isfinite(cur_p) and np.isfinite(ref_p):
                abs_posterior_diff.append(abs(cur_p - ref_p))
                decision_disagree.append(int((cur_p >= 0.5) != (ref_p >= 0.5)))
        out.append({
            "reference_model": reference_model,
            "model": model,
            "n_shared": len(delta_log_loss),
            "mean_delta_log_loss": _safe_mean(delta_log_loss),
            "median_delta_log_loss": _safe_median(delta_log_loss),
            "log_loss_win_rate": _safe_mean(logloss_win),
            "mean_delta_brier": _safe_mean(delta_brier),
            "median_delta_brier": _safe_median(delta_brier),
            "brier_win_rate": _safe_mean(brier_win),
            "mean_abs_posterior_diff": _safe_mean(abs_posterior_diff),
            "median_abs_posterior_diff": _safe_median(abs_posterior_diff),
            "decision_disagreement_rate": _safe_mean(decision_disagree),
        })
    out.sort(key=lambda r: (np.inf if not np.isfinite(_safe_float(r["mean_delta_log_loss"])) else r["mean_delta_log_loss"]))
    return out


def pairwise_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_market_model: dict[tuple[Any, str], dict[str, Any]] = {}
    models: set[str] = set()
    for row in rows:
        if row.get("status") != "ok":
            continue
        model = str(row.get("model"))
        by_market_model[(row.get("market_index"), model)] = row
        models.add(model)

    out: list[dict[str, Any]] = []
    model_order = sorted(models)
    for i, a in enumerate(model_order):
        for b in model_order[i + 1:]:
            delta_log_loss: list[float] = []
            delta_brier: list[float] = []
            abs_posterior_diff: list[float] = []
            decision_disagree: list[int] = []
            for market_index, model in by_market_model:
                if model != a:
                    continue
                ra = by_market_model.get((market_index, a))
                rb = by_market_model.get((market_index, b))
                if ra is None or rb is None:
                    continue
                pa = _safe_float(ra.get("posterior"))
                pb = _safe_float(rb.get("posterior"))
                lla = _safe_float(ra.get("log_loss"))
                llb = _safe_float(rb.get("log_loss"))
                bra = _safe_float(ra.get("brier"))
                brb = _safe_float(rb.get("brier"))
                if np.isfinite(lla) and np.isfinite(llb):
                    delta_log_loss.append(lla - llb)
                if np.isfinite(bra) and np.isfinite(brb):
                    delta_brier.append(bra - brb)
                if np.isfinite(pa) and np.isfinite(pb):
                    abs_posterior_diff.append(abs(pa - pb))
                    decision_disagree.append(int((pa >= 0.5) != (pb >= 0.5)))
            out.append({
                "model_a": a,
                "model_b": b,
                "n_shared": len(abs_posterior_diff),
                "mean_log_loss_a_minus_b": _safe_mean(delta_log_loss),
                "mean_brier_a_minus_b": _safe_mean(delta_brier),
                "mean_abs_posterior_diff": _safe_mean(abs_posterior_diff),
                "median_abs_posterior_diff": _safe_median(abs_posterior_diff),
                "decision_disagreement_rate": _safe_mean(decision_disagree),
            })
    return out


def trace_features(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for run in payload.get("runs", []):
        market = _market_fields(run)
        for model, result in run.get("model_results", {}).items():
            if result.get("status") != "ok" or "pi_t" not in result:
                continue
            pi_t = np.asarray(result["pi_t"], dtype=np.float64)
            pi_t = pi_t[np.isfinite(pi_t)]
            if pi_t.size == 0:
                continue
            diffs = np.diff(pi_t)
            crosses = int(np.sum((pi_t[:-1] < 0.5) != (pi_t[1:] < 0.5))) if pi_t.size > 1 else 0
            conf = np.abs(pi_t - 0.5) * 2.0
            idx_90 = np.where(conf >= 0.9)[0]
            idx_95 = np.where(conf >= 0.95)[0]
            rows.append({
                "market_index": market["market_index"],
                "slug": market["slug"],
                "model": model,
                "trace_length": int(pi_t.size),
                "final_posterior": float(pi_t[-1]),
                "mean_abs_step_change": float(np.mean(np.abs(diffs))) if diffs.size else 0.0,
                "max_abs_step_change": float(np.max(np.abs(diffs))) if diffs.size else 0.0,
                "decision_flip_count_05": crosses,
                "first_t_confidence_90": int(idx_90[0]) if idx_90.size else "",
                "first_t_confidence_95": int(idx_95[0]) if idx_95.size else "",
                "fraction_time_confidence_90": float(np.mean(conf >= 0.9)),
                "fraction_time_confidence_95": float(np.mean(conf >= 0.95)),
            })

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)

    summary: list[dict[str, Any]] = []
    for model, model_rows in grouped.items():
        summary.append({
            "model": model,
            "n_traces": len(model_rows),
            "mean_trace_length": _safe_mean([_safe_float(r["trace_length"]) for r in model_rows]),
            "mean_abs_step_change": _safe_mean([_safe_float(r["mean_abs_step_change"]) for r in model_rows]),
            "median_abs_step_change": _safe_median([_safe_float(r["mean_abs_step_change"]) for r in model_rows]),
            "mean_max_abs_step_change": _safe_mean([_safe_float(r["max_abs_step_change"]) for r in model_rows]),
            "median_max_abs_step_change": _safe_median([_safe_float(r["max_abs_step_change"]) for r in model_rows]),
            "mean_decision_flip_count_05": _safe_mean([_safe_float(r["decision_flip_count_05"]) for r in model_rows]),
            "mean_fraction_time_confidence_90": _safe_mean([_safe_float(r["fraction_time_confidence_90"]) for r in model_rows]),
            "mean_fraction_time_confidence_95": _safe_mean([_safe_float(r["fraction_time_confidence_95"]) for r in model_rows]),
        })
    summary.sort(key=lambda r: r["model"])
    return rows, summary


def hardest_markets(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok" and not str(row.get("model")).startswith("market_"):
            grouped[row.get("market_index")].append(row)

    out: list[dict[str, Any]] = []
    for market_index, market_rows in grouped.items():
        if not market_rows:
            continue
        first = market_rows[0]
        out.append({
            "market_index": market_index,
            "slug": first.get("slug"),
            "question": first.get("question"),
            "winner_label": first.get("winner_label"),
            "y": first.get("y"),
            "horizon": first.get("horizon"),
            "mean_model_log_loss": _safe_mean([_safe_float(r.get("log_loss")) for r in market_rows]),
            "max_model_log_loss": _safe_quantile([_safe_float(r.get("log_loss")) for r in market_rows], 1.0),
            "mean_model_brier": _safe_mean([_safe_float(r.get("brier")) for r in market_rows]),
            "n_models_wrong": int(np.sum([not bool(r.get("correct_05")) for r in market_rows])),
            "mean_p_truth": _safe_mean([_safe_float(r.get("p_truth")) for r in market_rows]),
            "posterior_range": (
                _safe_quantile([_safe_float(r.get("posterior")) for r in market_rows], 1.0)
                - _safe_quantile([_safe_float(r.get("posterior")) for r in market_rows], 0.0)
            ),
        })
    out.sort(key=lambda r: (-_safe_float(r["n_models_wrong"]), -_safe_float(r["mean_model_log_loss"])))
    return out[:top_k]


def best_worst_by_model(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok" and not str(row.get("model")).startswith("market_"):
            grouped[str(row.get("model"))].append(row)
    for model, model_rows in grouped.items():
        sorted_rows = sorted(model_rows, key=lambda r: _safe_float(r.get("log_loss")), reverse=True)
        for rank, row in enumerate(sorted_rows[:top_k], start=1):
            out.append({
                "model": model,
                "rank_worst_log_loss": rank,
                "market_index": row.get("market_index"),
                "slug": row.get("slug"),
                "question": row.get("question"),
                "winner_label": row.get("winner_label"),
                "posterior": row.get("posterior"),
                "p_truth": row.get("p_truth"),
                "log_loss": row.get("log_loss"),
                "brier": row.get("brier"),
                "confidence": row.get("confidence"),
                "correct_05": row.get("correct_05"),
            })
    return out


def write_markdown_report(
    path: Path,
    *,
    input_path: Path,
    summary_rows: list[dict[str, Any]],
    vs_reference_rows: list[dict[str, Any]],
    pairwise_rows: list[dict[str, Any]],
    trace_summary_rows: list[dict[str, Any]],
    reference_model: str,
) -> None:
    lines: list[str] = []
    lines.append("# Panel SMC Post-Processing Report")
    lines.append("")
    lines.append(f"Input: `{input_path}`")
    lines.append("")
    lines.append("## Model Summary")
    lines.append("")
    lines.append("| model | n | mean log loss | median log loss | mean Brier | accuracy | mean P(true) | overconf wrong 0.9 | ECE | mean runtime s |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            "| {model} | {n} | {mean_ll} | {med_ll} | {brier} | {acc} | {ptruth} | {ow} | {ece} | {time} |".format(
                model=row["model"],
                n=row["n_success"],
                mean_ll=_fmt(row["mean_log_loss"], 3),
                med_ll=_fmt(row["median_log_loss"], 3),
                brier=_fmt(row["mean_brier"], 3),
                acc=_fmt(row["accuracy_05"], 3),
                ptruth=_fmt(row["mean_p_truth"], 3),
                ow=_fmt(row["overconfident_wrong_rate_90"], 3),
                ece=_fmt(row["ece"], 3),
                time=_fmt(row["mean_elapsed_s"], 2),
            )
        )
    lines.append("")
    lines.append("Lower log loss, lower Brier, lower ECE, and lower overconfident-wrong rate are better. Higher accuracy and P(true) are better.")
    lines.append("")

    if vs_reference_rows:
        lines.append(f"## Compared To `{reference_model}`")
        lines.append("")
        lines.append("| model | n | delta log loss | log loss win rate | delta Brier | Brier win rate | mean abs posterior diff | disagreement |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in vs_reference_rows:
            lines.append(
                "| {model} | {n} | {dll} | {llwin} | {dbr} | {brwin} | {pdiff} | {disagree} |".format(
                    model=row["model"],
                    n=row["n_shared"],
                    dll=_fmt(row["mean_delta_log_loss"], 3),
                    llwin=_fmt(row["log_loss_win_rate"], 3),
                    dbr=_fmt(row["mean_delta_brier"], 3),
                    brwin=_fmt(row["brier_win_rate"], 3),
                    pdiff=_fmt(row["mean_abs_posterior_diff"], 3),
                    disagree=_fmt(row["decision_disagreement_rate"], 3),
                )
            )
        lines.append("")
        lines.append("Negative deltas mean the candidate beat the reference on average.")
        lines.append("")

    if trace_summary_rows:
        lines.append("## Posterior Trace Diagnostics")
        lines.append("")
        lines.append("| model | n | mean max step change | mean decision flips | mean frac time conf >= 0.9 |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in trace_summary_rows:
            lines.append(
                "| {model} | {n} | {jump} | {flips} | {conf} |".format(
                    model=row["model"],
                    n=row["n_traces"],
                    jump=_fmt(row["mean_max_abs_step_change"], 3),
                    flips=_fmt(row["mean_decision_flip_count_05"], 3),
                    conf=_fmt(row["mean_fraction_time_confidence_90"], 3),
                )
            )
        lines.append("")

    if pairwise_rows:
        lines.append("## Output Files")
        lines.append("")
        lines.append("- `model_summary.csv`: aggregate scoring table.")
        lines.append("- `model_results_long.csv`: one row per market-model result.")
        lines.append("- `model_results_wide.csv`: one row per market with per-model columns.")
        lines.append("- `vs_reference.csv`: per-model deltas against the selected reference.")
        lines.append("- `pairwise_model_comparisons.csv`: posterior disagreement and metric deltas for all model pairs.")
        lines.append("- `calibration_bins.csv`: calibration-bin data for reliability plots.")
        lines.append("- `hardest_markets.csv`: markets where the model family struggled most.")
        lines.append("- `trace_features.csv` and `trace_summary.csv`: written only when posterior traces are present.")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def make_wide_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_market: dict[Any, dict[str, Any]] = {}
    for row in rows:
        market_index = row.get("market_index")
        if market_index not in by_market:
            by_market[market_index] = {
                key: row.get(key)
                for key in (
                    "market_index",
                    "slug",
                    "question",
                    "winner_label",
                    "y",
                    "horizon",
                    "trade_count",
                    "total_bucket_volume",
                    "mean_bucket_volume",
                    "max_bucket_volume",
                    "nonzero_volume_fraction",
                    "mean_abs_dx",
                    "max_abs_dx",
                    "initial_price_yes",
                    "pre_final_price_yes",
                    "final_price_yes",
                )
            }
        model = str(row.get("model"))
        prefix = model.replace(" ", "_")
        for metric in (
            "status",
            "posterior",
            "p_truth",
            "log_loss",
            "brier",
            "correct_05",
            "confidence",
            "elapsed_s",
            "log_BF",
            "log_m_truth",
        ):
            by_market[market_index][f"{prefix}_{metric}"] = row.get(metric)
    return list(by_market.values())


def make_plots(outdir: Path, summary_rows: list[dict[str, Any]], calibration_rows: list[dict[str, Any]]) -> None:
    mpl_cache = outdir / ".matplotlib"
    xdg_cache = outdir / ".cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots", file=sys.stderr)
        return

    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    models = [str(r["model"]) for r in summary_rows if not str(r["model"]).startswith("market_")]
    if models:
        metrics = [
            ("mean_log_loss", "Mean Log Loss", "lower is better"),
            ("median_log_loss", "Median Log Loss", "lower is better"),
            ("mean_brier", "Mean Brier", "lower is better"),
            ("accuracy_05", "Accuracy at 0.5", "higher is better"),
            ("overconfident_wrong_rate_90", "Overconfident Wrong Rate", "lower is better"),
            ("mean_elapsed_s", "Mean Runtime (s)", "lower is faster"),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        axes = axes.ravel()
        for ax, (key, title, subtitle) in zip(axes, metrics):
            vals = [_safe_float(next(r for r in summary_rows if r["model"] == m).get(key)) for m in models]
            ax.barh(models, vals)
            ax.set_title(f"{title}\n{subtitle}")
            ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / "model_metric_bars.png", dpi=180)
        plt.close(fig)

    cal_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in calibration_rows:
        if not str(row.get("model")).startswith("market_") and row.get("n", 0):
            cal_by_model[str(row["model"])].append(row)
    if cal_by_model:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, label="perfect calibration")
        for model, bins in sorted(cal_by_model.items()):
            xs = [_safe_float(b.get("mean_p")) for b in bins]
            ys = [_safe_float(b.get("empirical_rate")) for b in bins]
            ax.plot(xs, ys, marker="o", linewidth=1.5, label=model)
        ax.set_xlabel("Mean predicted P(Yes)")
        ax.set_ylabel("Empirical Yes frequency")
        ax.set_title("Calibration by Model")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / "calibration_overlay.png", dpi=180)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to panel pickle from run_panel_smc_evaluation.py")
    parser.add_argument("--outdir", default=None, help="Directory for CSV/report outputs")
    parser.add_argument("--reference-model", default="base", help="Reference for model-vs-reference deltas")
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=25, help="Number of hardest/worst markets to list")
    parser.add_argument("--no-market-baselines", action="store_true", help="Do not include market final/pre-final baselines in tables")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plot generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = input_path.with_suffix("")
        outdir = outdir.parent / f"{outdir.name}_analysis"
    outdir.mkdir(parents=True, exist_ok=True)

    with input_path.open("rb") as f:
        payload = pickle.load(f)

    rows = flatten_model_rows(payload, include_market_baselines=not args.no_market_baselines)
    summary_rows, calibration_rows = summarize_by_model(rows, args.calibration_bins)
    vs_reference_rows = compare_to_reference(rows, args.reference_model)
    pairwise_rows = pairwise_comparisons(rows)
    trace_rows, trace_summary_rows = trace_features(payload)
    wide_rows = make_wide_rows(rows)
    hard_rows = hardest_markets(rows, args.top_k)
    worst_rows = best_worst_by_model(rows, args.top_k)

    _write_csv(outdir / "model_results_long.csv", rows)
    _write_csv(outdir / "model_results_wide.csv", wide_rows)
    _write_csv(outdir / "model_summary.csv", summary_rows)
    _write_csv(outdir / "calibration_bins.csv", calibration_rows)
    _write_csv(outdir / "vs_reference.csv", vs_reference_rows)
    _write_csv(outdir / "pairwise_model_comparisons.csv", pairwise_rows)
    _write_csv(outdir / "hardest_markets.csv", hard_rows)
    _write_csv(outdir / "worst_markets_by_model.csv", worst_rows)
    if trace_rows:
        _write_csv(outdir / "trace_features.csv", trace_rows)
        _write_csv(outdir / "trace_summary.csv", trace_summary_rows)

    analysis_summary = {
        "input": str(input_path),
        "n_runs": len(payload.get("runs", [])),
        "model_names": payload.get("model_names", []),
        "reference_model": args.reference_model,
        "model_summary": summary_rows,
        "vs_reference": vs_reference_rows,
        "trace_summary": trace_summary_rows,
    }
    (outdir / "analysis_summary.json").write_text(
        json.dumps(_clean_for_json(analysis_summary), indent=2),
        encoding="utf-8",
    )

    write_markdown_report(
        outdir / "report.md",
        input_path=input_path,
        summary_rows=summary_rows,
        vs_reference_rows=vs_reference_rows,
        pairwise_rows=pairwise_rows,
        trace_summary_rows=trace_summary_rows,
        reference_model=args.reference_model,
    )

    if not args.no_plots:
        make_plots(outdir, summary_rows, calibration_rows)

    print(f"loaded {len(payload.get('runs', []))} markets")
    print(f"wrote analysis outputs to {outdir}")
    print("top model summary by mean log loss:")
    for row in summary_rows[:8]:
        print(
            f"  {row['model']:28s} n={row['n_success']:4d} "
            f"logloss={_fmt(row['mean_log_loss'], 3)} "
            f"brier={_fmt(row['mean_brier'], 3)} "
            f"acc={_fmt(row['accuracy_05'], 3)} "
            f"ow90={_fmt(row['overconfident_wrong_rate_90'], 3)}"
        )


if __name__ == "__main__":
    main()
