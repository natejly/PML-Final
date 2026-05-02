#!/usr/bin/env python
"""Post-process synthetic posterior concentration experiment outputs.

This script reads the pickle produced by
``scripts/run_synthetic_posterior_concentration.py`` and creates plots in the
style of the paper's posterior concentration experiment: a cloud of posterior
error traces, a median trace, a 10-90% band, and a KL-projection reference line.
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np


MODEL_LABELS = {
    "reversal": "reversal",
    "reversal_burst": "reversal + burst",
    "reversal_momentum_burst": "reversal + momentum + burst",
    "gated_reversal_momentum_burst": "gated reversal + momentum + burst",
}


def _clip_prob(x: np.ndarray, eps: float) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float64), eps, 1.0 - eps)


def _ok_runs(model_payload: dict[str, Any]) -> list[dict[str, Any]]:
    pc = model_payload.get("posterior_concentration", {})
    return [r for r in pc.get("runs", []) if r.get("status") == "ok" and "posterior_error_t" in r]


def _log_posterior_error_from_log_bf(
    log_bf_t: np.ndarray,
    *,
    pi0: float,
    y_true: int,
) -> np.ndarray:
    """Compute log posterior error without clipping saturated probabilities."""
    pi0 = float(np.clip(pi0, 1e-300, 1.0 - 1e-16))
    logit_pi0 = math.log(pi0) - math.log1p(-pi0)
    log_odds = logit_pi0 + np.asarray(log_bf_t, dtype=np.float64)
    if int(y_true) == 1:
        return -np.logaddexp(0.0, log_odds)
    return -np.logaddexp(0.0, -log_odds)


def _stack_log_errors(
    runs: list[dict[str, Any]],
    eps: float,
    *,
    pi0: float,
    y_true: int,
) -> np.ndarray:
    if not runs:
        return np.empty((0, 0), dtype=np.float64)
    lengths = [
        len(r["log_BF_t"]) if "log_BF_t" in r else len(r["posterior_error_t"])
        for r in runs
    ]
    T = min(lengths)
    out = np.empty((len(runs), T), dtype=np.float64)
    for i, run in enumerate(runs):
        if "log_BF_t" in run:
            out[i] = _log_posterior_error_from_log_bf(
                np.asarray(run["log_BF_t"][:T], dtype=np.float64),
                pi0=pi0,
                y_true=y_true,
            )
        else:
            err = _clip_prob(np.asarray(run["posterior_error_t"][:T], dtype=np.float64), eps)
            out[i] = np.log(err)
    return out


def _summary_stats(log_errors: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "q10": np.quantile(log_errors, 0.10, axis=0),
        "median": np.median(log_errors, axis=0),
        "q90": np.quantile(log_errors, 0.90, axis=0),
        "mean": np.mean(log_errors, axis=0),
    }


def _horizon_result(kl_payload: dict[str, Any], horizon: int) -> dict[str, Any] | None:
    results = kl_payload.get("horizon_results", {})
    return results.get(horizon) or results.get(str(horizon))


def _kl_rows(model_name: str, model_payload: dict[str, Any], horizons: list[int]) -> list[dict[str, Any]]:
    rows = []
    kl_payload = model_payload.get("kl_projection", {})
    for horizon in horizons:
        res = _horizon_result(kl_payload, horizon)
        if not res:
            continue
        rows.append({
            "model": model_name,
            "horizon": int(horizon),
            "true_expected_loglik_per_step": float(res.get("true_expected_loglik_per_step", math.nan)),
            "best_wrong_expected_loglik_per_step": float(
                res.get("best_wrong_expected_loglik_per_step", math.nan)
            ),
            "delta_hat": float(res.get("delta_hat", math.nan)),
            "optimizer_success": bool(res.get("optimizer_success", False)),
            "optimizer_message": str(res.get("optimizer_message", "")),
        })
    return rows


def _fit_late_slope(ts: np.ndarray, median_log_error: np.ndarray) -> tuple[float, float, float]:
    if ts.size < 3:
        return float("nan"), float("nan"), float("nan")
    late = ts >= max(50, int(ts[-1] // 4))
    if np.count_nonzero(late) < 3:
        late = np.ones_like(ts, dtype=bool)
    slope, intercept = np.polyfit(ts[late], median_log_error[late], 1)
    return float(slope), float(intercept), float(-slope)


def _plot_model_concentration(
    *,
    model_name: str,
    log_errors: np.ndarray,
    stats: dict[str, np.ndarray],
    kl_rows: list[dict[str, Any]],
    horizon_points: list[int],
    pi0: float,
    outdir: Path,
    cloud_max: int,
    cloud_alpha: float,
    show_cloud: bool,
    seed: int,
) -> None:
    import matplotlib.pyplot as plt

    T = log_errors.shape[1]
    ts = np.arange(1, T + 1)
    hs = np.asarray([h for h in horizon_points if 1 <= int(h) <= T], dtype=np.int64)
    if hs.size == 0:
        hs = ts
    h_idx = hs - 1
    h_median = stats["median"][h_idx]
    h_q10 = stats["q10"][h_idx]
    h_q90 = stats["q90"][h_idx]
    slope, intercept, posterior_slope_delta = _fit_late_slope(hs, h_median)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    rng = np.random.default_rng(seed)
    n_cloud = min(int(cloud_max), log_errors.shape[0]) if show_cloud else 0
    if n_cloud > 0:
        idx = rng.choice(log_errors.shape[0], size=n_cloud, replace=False)
        for row in log_errors[idx]:
            ax.plot(hs, row[h_idx], color="0.55", alpha=float(cloud_alpha), linewidth=0.55)

    ax.fill_between(
        hs,
        h_q10,
        h_q90,
        color="C0",
        alpha=0.20,
        linewidth=0,
        label="10-90% band",
    )
    ax.plot(
        hs,
        h_median,
        color="C0",
        marker="o",
        markersize=3.5,
        linewidth=2.2,
        label="median log posterior error",
    )

    if np.isfinite(slope) and np.isfinite(intercept):
        ax.plot(
            hs,
            intercept + slope * hs,
            color="C3",
            linestyle=":",
            linewidth=1.8,
            label=f"empirical rate = {posterior_slope_delta:.4f}",
        )

    finite_kl = [r for r in kl_rows if np.isfinite(r.get("delta_hat", math.nan))]
    if finite_kl:
        finite_kl = sorted(finite_kl, key=lambda r: int(r["horizon"]))
        kl_hs = np.asarray([int(r["horizon"]) for r in finite_kl], dtype=np.int64)
        deltas = np.asarray([float(r["delta_hat"]) for r in finite_kl], dtype=np.float64)
        kl_points = math.log(max(1.0 - float(pi0), 1e-12)) - kl_hs * deltas
        ax.plot(
            kl_hs,
            kl_points,
            color="black",
            linestyle="--",
            marker="o",
            markersize=5.5,
            linewidth=2.0,
            label="KL projection horizon estimates",
        )

    ax.set_xlabel("horizon T")
    ax.set_ylabel("log posterior error")
    ax.set_title(f"Posterior concentration: {MODEL_LABELS.get(model_name, model_name)}")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / f"{model_name}_posterior_concentration.png", dpi=180)
    plt.close(fig)


def _plot_delta_by_horizon(
    *,
    model_name: str,
    kl_rows: list[dict[str, Any]],
    outdir: Path,
) -> None:
    if not kl_rows:
        return

    import matplotlib.pyplot as plt

    rows = [r for r in kl_rows if np.isfinite(r.get("delta_hat", math.nan))]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: int(r["horizon"]))
    hs = [int(r["horizon"]) for r in rows]
    deltas = [float(r["delta_hat"]) for r in rows]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(hs, deltas, marker="o", linewidth=2.0)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    ax.set_xlabel("horizon T")
    ax.set_ylabel("estimated separation gap delta_hat_T")
    ax.set_title(f"KL projection gap: {MODEL_LABELS.get(model_name, model_name)}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / f"{model_name}_kl_gap_by_horizon.png", dpi=180)
    plt.close(fig)


def _plot_combined_medians(
    *,
    model_stats: dict[str, dict[str, np.ndarray]],
    model_log_errors: dict[str, np.ndarray],
    model_kl_rows: dict[str, list[dict[str, Any]]],
    model_horizon_points: dict[str, list[int]],
    pi0: float,
    outdir: Path,
) -> None:
    if not model_stats:
        return

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for model_name, stats in model_stats.items():
        T = model_log_errors[model_name].shape[1]
        hs = np.asarray(
            [h for h in model_horizon_points.get(model_name, []) if 1 <= int(h) <= T],
            dtype=np.int64,
        )
        if hs.size == 0:
            hs = np.arange(1, T + 1)
        h_idx = hs - 1
        ax.plot(
            hs,
            stats["median"][h_idx],
            marker="o",
            markersize=3.0,
            linewidth=2.0,
            label=f"{MODEL_LABELS.get(model_name, model_name)} median",
        )

        finite_kl = [
            r for r in model_kl_rows.get(model_name, [])
            if np.isfinite(r.get("delta_hat", math.nan))
        ]
        if finite_kl:
            finite_kl = sorted(finite_kl, key=lambda r: int(r["horizon"]))
            kl_hs = np.asarray([int(r["horizon"]) for r in finite_kl], dtype=np.int64)
            deltas = np.asarray([float(r["delta_hat"]) for r in finite_kl], dtype=np.float64)
            ax.plot(
                kl_hs,
                math.log(max(1.0 - float(pi0), 1e-12)) - kl_hs * deltas,
                linestyle="--",
                marker="o",
                markersize=3.0,
                linewidth=1.2,
                alpha=0.75,
                label=f"{model_name} KL ref",
            )

    ax.set_xlabel("horizon T")
    ax.set_ylabel("log posterior error")
    ax.set_title("Posterior concentration medians")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "combined_posterior_concentration_medians.png", dpi=180)
    plt.close(fig)


def _write_csvs(
    *,
    summary_rows: list[dict[str, Any]],
    kl_rows: list[dict[str, Any]],
    outdir: Path,
) -> None:
    if summary_rows:
        path = outdir / "posterior_concentration_summary.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    if kl_rows:
        path = outdir / "kl_projection_summary.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(kl_rows[0].keys()))
            writer.writeheader()
            writer.writerows(kl_rows)


def _write_markdown(
    *,
    summary_rows: list[dict[str, Any]],
    kl_rows: list[dict[str, Any]],
    outdir: Path,
) -> None:
    lines = ["# Synthetic Posterior Concentration Analysis", ""]
    lines.append("## Posterior Concentration")
    lines.append("")
    lines.append("| model | n | T | empirical slope delta | final median log error | final q10 | final q90 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    final_rows = [r for r in summary_rows if r.get("is_final_horizon")]
    for row in final_rows:
        lines.append(
            "| {model} | {n} | {T} | {delta:.4f} | {med:.3f} | {q10:.3f} | {q90:.3f} |".format(
                model=row["model"],
                n=int(row["n"]),
                T=int(row["horizon"]),
                delta=float(row["empirical_slope_delta"]),
                med=float(row["median_log_error"]),
                q10=float(row["q10_log_error"]),
                q90=float(row["q90_log_error"]),
            )
        )

    lines.extend(["", "## KL Projection", ""])
    lines.append("| model | horizon | delta_hat | true ll/T | best wrong ll/T | optimizer success |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for row in kl_rows:
        lines.append(
            "| {model} | {horizon} | {delta:.4f} | {true:.4f} | {wrong:.4f} | {success} |".format(
                model=row["model"],
                horizon=int(row["horizon"]),
                delta=float(row["delta_hat"]),
                true=float(row["true_expected_loglik_per_step"]),
                wrong=float(row["best_wrong_expected_loglik_per_step"]),
                success=str(bool(row["optimizer_success"])),
            )
        )

    (outdir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs/synthetic_posterior_concentration.pkl")
    parser.add_argument("--output-dir", default="outputs/synthetic_posterior_concentration_analysis")
    parser.add_argument("--models", nargs="+", default=None, help="Subset of model keys to analyze")
    parser.add_argument("--cloud-max", type=int, default=120, help="Max individual traces in each cloud plot")
    parser.add_argument("--cloud-alpha", type=float, default=0.025, help="Opacity for individual cloud traces")
    parser.add_argument("--no-cloud", action="store_true", help="Hide individual trace cloud lines")
    parser.add_argument("--seed", type=int, default=12345, help="Seed for subsampling cloud traces")
    parser.add_argument("--eps", type=float, default=1e-12, help="Probability clipping epsilon")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    with input_path.open("rb") as f:
        payload = pickle.load(f)

    config = payload.get("config", {})
    pi0 = float(config.get("pi0", 0.5))
    y_true = int(config.get("y_true", 1))
    horizons = [int(h) for h in config.get("horizons", [])]
    model_names = args.models if args.models is not None else list(payload.get("models", {}).keys())

    all_summary_rows: list[dict[str, Any]] = []
    all_kl_rows: list[dict[str, Any]] = []
    model_stats: dict[str, dict[str, np.ndarray]] = {}
    model_log_errors: dict[str, np.ndarray] = {}
    model_kl_rows: dict[str, list[dict[str, Any]]] = {}
    model_horizon_points: dict[str, list[int]] = {}

    for model_name in model_names:
        model_payload = payload.get("models", {}).get(model_name)
        if model_payload is None:
            print(f"warning: model {model_name!r} not found; skipping")
            continue

        runs = _ok_runs(model_payload)
        log_errors = _stack_log_errors(
            runs,
            float(args.eps),
            pi0=pi0,
            y_true=y_true,
        )
        if log_errors.size == 0:
            print(f"warning: no completed posterior traces for {model_name}; skipping plots")
            continue

        stats = _summary_stats(log_errors)
        model_stats[model_name] = stats
        model_log_errors[model_name] = log_errors

        T = log_errors.shape[1]
        model_horizons = horizons or [10, 25, 50, 100, 200, T]
        model_horizons = sorted({h for h in model_horizons if 1 <= h <= T})
        model_horizon_points[model_name] = model_horizons
        horizon_arr = np.asarray(model_horizons, dtype=np.int64)
        horizon_idx = horizon_arr - 1
        _, _, empirical_slope_delta = _fit_late_slope(
            horizon_arr,
            stats["median"][horizon_idx],
        )
        for horizon in model_horizons:
            idx = int(horizon) - 1
            all_summary_rows.append({
                "model": model_name,
                "n": int(log_errors.shape[0]),
                "horizon": int(horizon),
                "is_final_horizon": bool(horizon == T),
                "mean_log_error": float(stats["mean"][idx]),
                "median_log_error": float(stats["median"][idx]),
                "q10_log_error": float(stats["q10"][idx]),
                "q90_log_error": float(stats["q90"][idx]),
                "empirical_slope_delta": float(empirical_slope_delta),
            })

        rows = _kl_rows(model_name, model_payload, model_horizons)
        model_kl_rows[model_name] = rows
        all_kl_rows.extend(rows)

        _plot_model_concentration(
            model_name=model_name,
            log_errors=log_errors,
            stats=stats,
            kl_rows=rows,
            horizon_points=model_horizons,
            pi0=pi0,
            outdir=outdir,
            cloud_max=int(args.cloud_max),
            cloud_alpha=float(args.cloud_alpha),
            show_cloud=not bool(args.no_cloud),
            seed=int(args.seed),
        )
        _plot_delta_by_horizon(model_name=model_name, kl_rows=rows, outdir=outdir)

    _plot_combined_medians(
        model_stats=model_stats,
        model_log_errors=model_log_errors,
        model_kl_rows=model_kl_rows,
        model_horizon_points=model_horizon_points,
        pi0=pi0,
        outdir=outdir,
    )
    _write_csvs(summary_rows=all_summary_rows, kl_rows=all_kl_rows, outdir=outdir)
    _write_markdown(summary_rows=all_summary_rows, kl_rows=all_kl_rows, outdir=outdir)

    print(f"wrote analysis outputs to {outdir}")
    print("main plots:")
    for path in sorted(outdir.glob("*posterior_concentration*.png")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
