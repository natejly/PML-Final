#!/usr/bin/env python3
"""Synthetic posterior concentration and KL projection-gap experiment.

This script reproduces the paper-style posterior concentration experiment for
the jointly Markov volume models in this repository.  For each requested model
class it:

1. draws synthetic histories from the model's forward sampler;
2. runs SMC on each history with full posterior traces; and
3. estimates the outcome-separation KL projection gap by Monte Carlo
   likelihood optimization under the wrong outcome.

The script is intentionally checkpoint-friendly: pass ``--resume`` to reuse
completed model phases and completed KL horizons from an existing pickle.
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.optimize import Bounds, minimize

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is in requirements.
    tqdm = None

from pml_market import (
    BurstMispricingLogARVolModel,
    BurstMispricingLogARVolPrior,
    BurstMispricingLogARVolSampler,
    GatedReversalMomentumBurstLogARVolModel,
    GatedReversalMomentumBurstLogARVolPrior,
    GatedReversalMomentumBurstLogARVolSampler,
    InverseProblem,
    MispricingLogARVolModel,
    MispricingLogARVolPrior,
    MispricingLogARVolSampler,
    ReversalMomentumBurstLogARVolModel,
    ReversalMomentumBurstLogARVolPrior,
    ReversalMomentumBurstLogARVolSampler,
    SMCInference,
)


MODEL_ORDER = (
    "reversal",
    "reversal_burst",
    "reversal_momentum_burst",
    "gated_reversal_momentum_burst",
)
MODEL_OFFSETS = {
    "reversal": 11,
    "reversal_burst": 23,
    "reversal_momentum_burst": 37,
    "gated_reversal_momentum_burst": 41,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _progress(iterable, *, total: int, desc: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def _progress_write(message: str) -> None:
    if tqdm is not None:
        tqdm.write(message)
    else:
        print(message, flush=True)


def _run_jobs(worker_fn: Callable[[dict[str, Any]], Any],
              jobs: list[dict[str, Any]],
              *,
              workers: int,
              desc: str):
    """Run top-level worker jobs with tqdm in the parent process.

    ``ProcessPoolExecutor`` can be unavailable in restricted environments, so
    workers <= 1 deliberately uses a serial path.  Normal full runs should use
    workers > 1.
    """
    if int(workers) <= 1:
        for job in _progress(jobs, total=len(jobs), desc=desc):
            yield worker_fn(job)
        return

    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        futures = [pool.submit(worker_fn, job) for job in jobs]
        for fut in _progress(as_completed(futures), total=len(futures), desc=desc):
            yield fut.result()


def _model_prior_sampler(model_name: str):
    if model_name == "reversal":
        sampler = MispricingLogARVolSampler()
        return MispricingLogARVolModel(), MispricingLogARVolPrior(), sampler
    if model_name == "reversal_burst":
        sampler = BurstMispricingLogARVolSampler()
        return BurstMispricingLogARVolModel(), BurstMispricingLogARVolPrior(), sampler
    if model_name == "reversal_momentum_burst":
        sampler = ReversalMomentumBurstLogARVolSampler()
        return (
            ReversalMomentumBurstLogARVolModel(),
            ReversalMomentumBurstLogARVolPrior(),
            sampler,
        )
    if model_name == "gated_reversal_momentum_burst":
        sampler = GatedReversalMomentumBurstLogARVolSampler()
        return (
            GatedReversalMomentumBurstLogARVolModel(),
            GatedReversalMomentumBurstLogARVolPrior(),
            sampler,
        )
    raise ValueError(f"unknown model {model_name!r}; expected one of {MODEL_ORDER}")


def _default_theta(model_name: str) -> dict[str, np.ndarray]:
    _, _, sampler = _model_prior_sampler(model_name)
    return sampler.default_theta()


def _checkpoint(payload: dict[str, Any], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _clip_prob(p, eps: float = 1e-12):
    return np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)


def _posterior_error(pi_t: np.ndarray, y: int) -> np.ndarray:
    pi_t = _clip_prob(pi_t)
    if int(y) == 1:
        return 1.0 - pi_t
    return pi_t


def _summarize_posterior_runs(
    runs: list[dict[str, Any]],
    horizons: list[int],
) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    ok_runs = [r for r in runs if r.get("status") == "ok"]
    for horizon in horizons:
        idx = int(horizon) - 1
        errors = []
        log_errors = []
        pis = []
        for run in ok_runs:
            err_t = np.asarray(run["posterior_error_t"], dtype=np.float64)
            pi_t = np.asarray(run["pi_t"], dtype=np.float64)
            if idx < err_t.shape[0]:
                err = float(_clip_prob(err_t[idx]))
                errors.append(err)
                log_errors.append(float(math.log(err)))
                pis.append(float(pi_t[idx]))
        if not errors:
            out[int(horizon)] = {"n": 0}
            continue
        arr = np.asarray(errors, dtype=np.float64)
        log_arr = np.asarray(log_errors, dtype=np.float64)
        pi_arr = np.asarray(pis, dtype=np.float64)
        out[int(horizon)] = {
            "n": int(arr.size),
            "mean_error": float(np.mean(arr)),
            "median_error": float(np.median(arr)),
            "q10_error": float(np.quantile(arr, 0.10)),
            "q90_error": float(np.quantile(arr, 0.90)),
            "mean_log_error": float(np.mean(log_arr)),
            "median_log_error": float(np.median(log_arr)),
            "q10_log_error": float(np.quantile(log_arr, 0.10)),
            "q90_log_error": float(np.quantile(log_arr, 0.90)),
            "mean_pi_t": float(np.mean(pi_arr)),
            "median_pi_t": float(np.median(pi_arr)),
        }
    return out


def _sample_trajectory_job(job: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    model_name = str(job["model_name"])
    idx = int(job["index"])
    T = int(job["T"])
    y = int(job["y"])
    seed = int(job["seed"])

    _, _, sampler = _model_prior_sampler(model_name)
    theta = sampler.default_theta()
    dx, v, y_out = sampler.sample(T, y=y, theta=theta, seed=seed)
    return idx, {
        "dx": np.asarray(dx, dtype=np.float64),
        "v": np.asarray(v, dtype=np.float64),
        "y": int(y_out),
        "seed": seed,
    }


def _draw_trajectories(
    *,
    model_name: str,
    n: int,
    T: int,
    y: int,
    seed_base: int,
    workers: int,
    desc: str,
) -> dict[str, np.ndarray]:
    jobs = [
        {
            "model_name": model_name,
            "index": i,
            "T": T,
            "y": y,
            "seed": seed_base + i,
        }
        for i in range(int(n))
    ]
    dxs = np.empty((int(n), int(T)), dtype=np.float64)
    vs = np.empty((int(n), int(T)), dtype=np.float64)
    ys = np.empty(int(n), dtype=np.int64)
    seeds = np.empty(int(n), dtype=np.int64)

    for idx, traj in _run_jobs(_sample_trajectory_job, jobs, workers=int(workers), desc=desc):
        dxs[idx] = traj["dx"]
        vs[idx] = traj["v"]
        ys[idx] = traj["y"]
        seeds[idx] = traj["seed"]

    return {"dx": dxs, "v": vs, "y": ys, "seed": seeds}


def _inference_job(job: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    idx = int(job["index"])
    model_name = str(job["model_name"])
    dx = np.asarray(job["dx"], dtype=np.float64)
    v = np.asarray(job["v"], dtype=np.float64)
    y = int(job["y"])

    model, prior, _ = _model_prior_sampler(model_name)
    problem = InverseProblem(model, prior)
    smc = SMCInference(
        n_particles=int(job["n_particles"]),
        ess_threshold=float(job["ess_threshold"]),
        mcmc_steps=int(job["mcmc_steps"]),
        initial_step_size=float(job["initial_step_size"]),
    )

    start = time.perf_counter()
    try:
        result = problem.infer(
            dx,
            v,
            smc,
            pi0=float(job["pi0"]),
            seed=int(job["seed"]),
            record_pi_t=True,
        )
        elapsed = time.perf_counter() - start
        pi_t = np.asarray(result["pi_t"], dtype=np.float64)
        posterior_error_t = _posterior_error(pi_t, y)
        out = {
            "status": "ok",
            "seed": int(job["seed"]),
            "posterior": float(result["posterior"]),
            "log_BF": float(result["log_BF"]),
            "log_m0": float(result["log_m0"]),
            "log_m1": float(result["log_m1"]),
            "pi_t": pi_t,
            "log_BF_t": np.asarray(result["log_BF_t"], dtype=np.float64),
            "posterior_error_t": posterior_error_t,
            "log_posterior_error_t": np.log(_clip_prob(posterior_error_t)),
            "smc0_log_inc": np.asarray(result["smc0"]["log_inc"], dtype=np.float64),
            "smc1_log_inc": np.asarray(result["smc1"]["log_inc"], dtype=np.float64),
            "smc0_ess_history": np.asarray(result["smc0"]["ess_history"], dtype=np.float64),
            "smc1_ess_history": np.asarray(result["smc1"]["ess_history"], dtype=np.float64),
            "elapsed_s": float(elapsed),
        }
        return idx, out
    except Exception as exc:  # keep panel-style long runs alive.
        elapsed = time.perf_counter() - start
        return idx, {
            "status": "error",
            "seed": int(job["seed"]),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_s": float(elapsed),
        }


def _run_inference_phase(
    *,
    model_name: str,
    trajectories: dict[str, np.ndarray],
    args: argparse.Namespace,
    seed_base: int,
) -> list[dict[str, Any]]:
    dxs = np.asarray(trajectories["dx"], dtype=np.float64)
    vs = np.asarray(trajectories["v"], dtype=np.float64)
    ys = np.asarray(trajectories["y"], dtype=np.int64)
    jobs = []
    for i in range(dxs.shape[0]):
        jobs.append({
            "model_name": model_name,
            "index": i,
            "dx": dxs[i],
            "v": vs[i],
            "y": int(ys[i]),
            "seed": seed_base + i,
            "n_particles": int(args.n_particles),
            "mcmc_steps": int(args.mcmc_steps),
            "ess_threshold": float(args.ess_threshold),
            "initial_step_size": float(args.initial_step_size),
            "pi0": float(args.pi0),
        })

    runs: list[dict[str, Any] | None] = [None] * len(jobs)
    for idx, run in _run_jobs(
        _inference_job,
        jobs,
        workers=int(args.workers),
        desc=f"SMC inference: {model_name}",
    ):
        runs[idx] = run

    return [r if r is not None else {"status": "missing"} for r in runs]


def _true_loglik_job(job: dict[str, Any]) -> tuple[int, np.ndarray]:
    idx = int(job["index"])
    model_name = str(job["model_name"])
    horizons = [int(h) for h in job["horizons"]]
    dx = np.asarray(job["dx"], dtype=np.float64)
    v = np.asarray(job["v"], dtype=np.float64)
    y = int(job["y"])
    theta_true = job["theta_true"]
    model, _, _ = _model_prior_sampler(model_name)

    vals = np.empty(len(horizons), dtype=np.float64)
    for j, horizon in enumerate(horizons):
        vals[j] = float(model.loglik(dx[:horizon], v[:horizon], y, theta_true)) / horizon
    return idx, vals


def _compute_true_lambdas(
    *,
    model_name: str,
    trajectories: dict[str, np.ndarray],
    theta_true: dict[str, np.ndarray],
    horizons: list[int],
    workers: int,
) -> np.ndarray:
    dxs = np.asarray(trajectories["dx"], dtype=np.float64)
    vs = np.asarray(trajectories["v"], dtype=np.float64)
    ys = np.asarray(trajectories["y"], dtype=np.int64)
    vals = np.empty((dxs.shape[0], len(horizons)), dtype=np.float64)

    jobs = [
        {
            "index": i,
            "model_name": model_name,
            "dx": dxs[i],
            "v": vs[i],
            "y": int(ys[i]),
            "theta_true": theta_true,
            "horizons": horizons,
        }
        for i in range(dxs.shape[0])
    ]
    for idx, row in _run_jobs(
        _true_loglik_job,
        jobs,
        workers=int(workers),
        desc=f"true loglik: {model_name}",
    ):
        vals[idx] = row
    return vals.mean(axis=0)


def _candidate_score_chunk_job(job: dict[str, Any]) -> dict[str, Any]:
    model_name = str(job["model_name"])
    n = int(job["n"])
    seed = int(job["seed"])
    wrong_y = int(job["wrong_y"])
    horizons = [int(h) for h in job["horizons"]]
    z_bound = float(job["z_bound"])
    dxs = np.asarray(job["dx"], dtype=np.float64)
    vs = np.asarray(job["v"], dtype=np.float64)

    model, prior, _ = _model_prior_sampler(model_name)
    rng = np.random.default_rng(seed)
    theta_sample = prior.sample(rng, n)
    z = np.asarray(prior.to_unconstrained(theta_sample), dtype=np.float64)
    if np.isfinite(z_bound) and z_bound > 0:
        z = np.clip(z, -z_bound, z_bound)
    theta_batch, _ = prior.transform(z)

    scores = np.zeros((n, len(horizons)), dtype=np.float64)
    for i in range(dxs.shape[0]):
        dx = dxs[i]
        v = vs[i]
        for j, horizon in enumerate(horizons):
            scores[:, j] += (
                np.asarray(model.loglik(dx[:horizon], v[:horizon], wrong_y, theta_batch),
                           dtype=np.float64)
                / horizon
            )
    scores /= max(dxs.shape[0], 1)
    return {"z": z, "scores": scores, "seed": seed}


def _run_random_candidate_search(
    *,
    model_name: str,
    trajectories: dict[str, np.ndarray],
    horizons: list[int],
    wrong_y: int,
    args: argparse.Namespace,
    seed_base: int,
) -> dict[str, Any]:
    total = int(args.kl_random_candidates)
    chunk_size = int(args.kl_candidate_chunk_size)
    chunks = []
    start = 0
    chunk_idx = 0
    while start < total:
        n = min(chunk_size, total - start)
        chunks.append({
            "model_name": model_name,
            "n": n,
            "seed": seed_base + chunk_idx,
            "wrong_y": wrong_y,
            "horizons": horizons,
            "z_bound": float(args.kl_z_bound),
            "dx": trajectories["dx"],
            "v": trajectories["v"],
        })
        start += n
        chunk_idx += 1

    z_parts = []
    score_parts = []
    seeds = []
    for res in _run_jobs(
        _candidate_score_chunk_job,
        chunks,
        workers=int(args.kl_workers),
        desc=f"random KL candidates: {model_name}",
    ):
        z_parts.append(np.asarray(res["z"], dtype=np.float64))
        score_parts.append(np.asarray(res["scores"], dtype=np.float64))
        seeds.append(int(res["seed"]))

    z_all = np.concatenate(z_parts, axis=0) if z_parts else np.empty((0, 0))
    scores_all = np.concatenate(score_parts, axis=0) if score_parts else np.empty((0, len(horizons)))
    return {
        "z": z_all,
        "scores": scores_all,
        "horizons": list(map(int, horizons)),
        "wrong_y": int(wrong_y),
        "seeds": seeds,
        "candidate_count": int(z_all.shape[0]),
    }


def _avg_wrong_loglik_for_z(
    z: np.ndarray,
    *,
    model_name: str,
    dxs: np.ndarray,
    vs: np.ndarray,
    wrong_y: int,
    horizon: int,
) -> float:
    model, prior, _ = _model_prior_sampler(model_name)
    theta, _ = prior.transform(np.asarray(z, dtype=np.float64))
    vals = []
    for i in range(dxs.shape[0]):
        vals.append(float(model.loglik(dxs[i, :horizon], vs[i, :horizon], wrong_y, theta)) / horizon)
    return float(np.mean(vals))


def _local_opt_job(job: dict[str, Any]) -> dict[str, Any]:
    model_name = str(job["model_name"])
    horizon = int(job["horizon"])
    wrong_y = int(job["wrong_y"])
    z0 = np.asarray(job["z0"], dtype=np.float64)
    dxs = np.asarray(job["dx"], dtype=np.float64)
    vs = np.asarray(job["v"], dtype=np.float64)
    maxiter = int(job["maxiter"])
    z_bound = float(job["z_bound"])
    restart_index = int(job["restart_index"])

    if np.isfinite(z_bound) and z_bound > 0:
        z0 = np.clip(z0, -z_bound, z_bound)
        bounds = Bounds(
            np.full_like(z0, -z_bound, dtype=np.float64),
            np.full_like(z0, z_bound, dtype=np.float64),
        )
    else:
        bounds = None

    def objective(z_vec: np.ndarray) -> float:
        try:
            score = _avg_wrong_loglik_for_z(
                z_vec,
                model_name=model_name,
                dxs=dxs,
                vs=vs,
                wrong_y=wrong_y,
                horizon=horizon,
            )
        except Exception:
            return 1e100
        if not np.isfinite(score):
            return 1e100
        return -score

    start = time.perf_counter()
    try:
        initial_score = -objective(z0)
        res = minimize(
            objective,
            z0,
            method="Powell",
            bounds=bounds,
            options={
                "maxiter": maxiter,
                "xtol": float(job["xtol"]),
                "ftol": float(job["ftol"]),
                "disp": False,
            },
        )
        elapsed = time.perf_counter() - start
        score_hat = -float(res.fun) if np.isfinite(res.fun) else float("-inf")
        z_hat = np.asarray(res.x, dtype=np.float64)
        if np.isfinite(z_bound) and z_bound > 0:
            z_hat = np.clip(z_hat, -z_bound, z_bound)
        return {
            "status": "ok",
            "restart_index": restart_index,
            "horizon": horizon,
            "initial_score": float(initial_score),
            "score_hat": float(score_hat),
            "z_start": z0,
            "z_hat": z_hat,
            "success": bool(res.success),
            "message": str(res.message),
            "nfev": int(getattr(res, "nfev", -1)),
            "nit": int(getattr(res, "nit", -1)),
            "elapsed_s": float(elapsed),
        }
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return {
            "status": "error",
            "restart_index": restart_index,
            "horizon": horizon,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_s": float(elapsed),
            "z_start": z0,
        }


def _theta_from_z(model_name: str, z: np.ndarray) -> dict[str, Any]:
    _, prior, _ = _model_prior_sampler(model_name)
    theta, _ = prior.transform(np.asarray(z, dtype=np.float64))
    return {k: np.asarray(v, dtype=np.float64) for k, v in theta.items()}


def _run_local_optimization_for_horizon(
    *,
    model_name: str,
    trajectories: dict[str, np.ndarray],
    random_search: dict[str, Any],
    horizon: int,
    horizon_index: int,
    true_lambda: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    scores = np.asarray(random_search["scores"], dtype=np.float64)[:, horizon_index]
    z_candidates = np.asarray(random_search["z"], dtype=np.float64)
    n_restarts = min(int(args.kl_local_restarts), int(z_candidates.shape[0]))
    order = np.argsort(scores)[::-1]
    starts = z_candidates[order[:n_restarts]]
    random_best_score = float(scores[order[0]]) if scores.size else float("-inf")
    random_best_z = z_candidates[order[0]] if scores.size else None
    wrong_y = int(random_search["wrong_y"])

    jobs = [
        {
            "model_name": model_name,
            "horizon": int(horizon),
            "wrong_y": wrong_y,
            "z0": starts[i],
            "dx": trajectories["dx"],
            "v": trajectories["v"],
            "maxiter": int(args.kl_maxiter),
            "z_bound": float(args.kl_z_bound),
            "restart_index": i,
            "xtol": float(args.kl_xtol),
            "ftol": float(args.kl_ftol),
        }
        for i in range(n_restarts)
    ]

    local_results = list(_run_jobs(
        _local_opt_job,
        jobs,
        workers=int(args.kl_workers),
        desc=f"Powell KL T={horizon}: {model_name}",
    ))

    ok = [r for r in local_results if r.get("status") == "ok" and np.isfinite(r.get("score_hat", np.nan))]
    if ok:
        best = max(ok, key=lambda r: r["score_hat"])
        best_wrong = float(best["score_hat"])
        best_z = np.asarray(best["z_hat"], dtype=np.float64)
        best_theta = _theta_from_z(model_name, best_z)
        optimizer_status = "ok"
    else:
        best = None
        best_wrong = random_best_score
        best_z = np.asarray(random_best_z, dtype=np.float64) if random_best_z is not None else None
        best_theta = _theta_from_z(model_name, best_z) if best_z is not None else None
        optimizer_status = "random_only_fallback" if best_z is not None else "failed"

    delta_hat = float(true_lambda - best_wrong) if np.isfinite(best_wrong) else float("nan")
    return {
        "horizon": int(horizon),
        "status": optimizer_status,
        "true_expected_loglik_per_step": float(true_lambda),
        "best_wrong_expected_loglik_per_step": float(best_wrong),
        "delta_hat": delta_hat,
        "wrong_y": wrong_y,
        "random_best_score": random_best_score,
        "random_best_z": random_best_z,
        "best_z": best_z,
        "theta_hat": best_theta,
        "best_local_result": best,
        "local_results": local_results,
        "n_local_restarts": int(n_restarts),
    }


def _config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    cfg = vars(args).copy()
    cfg["models"] = list(args.models)
    cfg["horizons"] = [int(h) for h in args.horizons]
    return cfg


def _load_or_init_payload(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    if args.resume and output.exists():
        with output.open("rb") as f:
            payload = pickle.load(f)
        payload["resume_loaded_at"] = _utc_now()
        payload["config"] = _config_from_args(args)
        return payload
    return {
        "created_at": _utc_now(),
        "config": _config_from_args(args),
        "models": {},
        "timings": {},
    }


def _phase_done(model_payload: dict[str, Any], phase: str, expected_count: int | None = None) -> bool:
    if phase not in model_payload:
        return False
    if expected_count is None:
        return True
    phase_payload = model_payload[phase]
    if phase == "posterior_concentration":
        runs = phase_payload.get("runs", [])
        return len(runs) == expected_count and all(r.get("status") == "ok" for r in runs)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(MODEL_ORDER), choices=list(MODEL_ORDER))
    parser.add_argument("--n-trajectories", type=int, default=1000)
    parser.add_argument("--n-kl-trajectories", type=int, default=1000)
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--horizons", nargs="+", type=int, default=[10, 25, 50, 100, 200, 500])
    parser.add_argument("--n-particles", type=int, default=1500)
    parser.add_argument("--mcmc-steps", type=int, default=2)
    parser.add_argument("--ess-threshold", type=float, default=0.5)
    parser.add_argument("--initial-step-size", type=float, default=0.3)
    parser.add_argument("--pi0", type=float, default=0.5)
    parser.add_argument("--y-true", type=int, default=1, choices=[0, 1])
    parser.add_argument("--kl-random-candidates", type=int, default=2048)
    parser.add_argument("--kl-local-restarts", type=int, default=8)
    parser.add_argument("--kl-maxiter", type=int, default=100)
    parser.add_argument("--kl-candidate-chunk-size", type=int, default=128)
    parser.add_argument("--kl-z-bound", type=float, default=8.0)
    parser.add_argument("--kl-xtol", type=float, default=1e-3)
    parser.add_argument("--kl-ftol", type=float, default=1e-3)
    default_workers = max(1, min(8, (os.cpu_count() or 2) - 1))
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument("--kl-workers", type=int, default=default_workers)
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--output", default="outputs/synthetic_posterior_concentration.pkl")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    horizons = sorted({int(h) for h in args.horizons})
    if horizons[-1] > int(args.T):
        raise ValueError(f"max horizon {horizons[-1]} exceeds T={args.T}")
    args.horizons = horizons

    payload = _load_or_init_payload(args)
    total_start = time.perf_counter()

    for model_name in args.models:
        model_start = time.perf_counter()
        model_offset = MODEL_OFFSETS[model_name]
        model_payload = payload["models"].setdefault(model_name, {})
        theta_true = _default_theta(model_name)
        model_payload["theta_true"] = theta_true

        if args.resume and _phase_done(model_payload, "posterior_concentration", args.n_trajectories):
            _progress_write(f"skipping posterior concentration for {model_name}; already complete")
        else:
            _progress_write(f"drawing trajectories for inference: {model_name}")
            draw_start = time.perf_counter()
            trajectories = _draw_trajectories(
                model_name=model_name,
                n=int(args.n_trajectories),
                T=int(args.T),
                y=int(args.y_true),
                seed_base=int(args.seed) + 10_000_000 * model_offset,
                workers=int(args.workers),
                desc=f"draw inference paths: {model_name}",
            )
            draw_elapsed = time.perf_counter() - draw_start

            _progress_write(f"running SMC inference: {model_name}")
            infer_start = time.perf_counter()
            runs = _run_inference_phase(
                model_name=model_name,
                trajectories=trajectories,
                args=args,
                seed_base=int(args.seed) + 20_000_000 * model_offset,
            )
            infer_elapsed = time.perf_counter() - infer_start
            model_payload["posterior_concentration"] = {
                "trajectories": trajectories,
                "runs": runs,
                "summary_by_horizon": _summarize_posterior_runs(runs, horizons),
                "timings": {
                    "draw_s": float(draw_elapsed),
                    "inference_s": float(infer_elapsed),
                },
            }
            _checkpoint(payload, args.output)

        kl_payload = model_payload.setdefault("kl_projection", {})
        existing_horizons = {
            int(k) for k in kl_payload.get("horizon_results", {}).keys()
        }
        if args.resume and all(int(h) in existing_horizons for h in horizons):
            _progress_write(f"skipping KL projection for {model_name}; all horizons complete")
        else:
            if "trajectories" in kl_payload and int(kl_payload["trajectories"]["dx"].shape[0]) == int(args.n_kl_trajectories):
                kl_trajectories = kl_payload["trajectories"]
                _progress_write(f"reusing KL trajectories: {model_name}")
            else:
                _progress_write(f"drawing trajectories for outcome separation gap: {model_name}")
                kl_draw_start = time.perf_counter()
                kl_trajectories = _draw_trajectories(
                    model_name=model_name,
                    n=int(args.n_kl_trajectories),
                    T=int(args.T),
                    y=int(args.y_true),
                    seed_base=int(args.seed) + 30_000_000 * model_offset,
                    workers=int(args.workers),
                    desc=f"draw KL paths: {model_name}",
                )
                kl_payload["trajectories"] = kl_trajectories
                kl_payload.setdefault("timings", {})["draw_s"] = float(time.perf_counter() - kl_draw_start)
                _checkpoint(payload, args.output)

            if "true_expected_loglik_per_step" in kl_payload:
                true_lambdas = np.asarray(kl_payload["true_expected_loglik_per_step"], dtype=np.float64)
                _progress_write(f"reusing true expected log-likelihoods: {model_name}")
            else:
                true_start = time.perf_counter()
                true_lambdas = _compute_true_lambdas(
                    model_name=model_name,
                    trajectories=kl_trajectories,
                    theta_true=theta_true,
                    horizons=horizons,
                    workers=int(args.kl_workers),
                )
                kl_payload["true_expected_loglik_per_step"] = true_lambdas
                kl_payload.setdefault("timings", {})["true_loglik_s"] = float(time.perf_counter() - true_start)
                _checkpoint(payload, args.output)

            wrong_y = 1 - int(args.y_true)
            random_search = kl_payload.get("random_search")
            random_search_valid = (
                random_search is not None
                and int(random_search.get("candidate_count", 0)) == int(args.kl_random_candidates)
                and list(map(int, random_search.get("horizons", []))) == horizons
                and int(random_search.get("wrong_y", -1)) == wrong_y
            )
            if random_search_valid:
                _progress_write(f"reusing random KL candidates: {model_name}")
            else:
                random_start = time.perf_counter()
                random_search = _run_random_candidate_search(
                    model_name=model_name,
                    trajectories=kl_trajectories,
                    horizons=horizons,
                    wrong_y=wrong_y,
                    args=args,
                    seed_base=int(args.seed) + 40_000_000 * model_offset,
                )
                kl_payload["random_search"] = random_search
                kl_payload.setdefault("timings", {})["random_search_s"] = float(time.perf_counter() - random_start)
                _checkpoint(payload, args.output)

            horizon_results = kl_payload.setdefault("horizon_results", {})
            for horizon_index, horizon in enumerate(horizons):
                if args.resume and int(horizon) in {int(k) for k in horizon_results.keys()}:
                    _progress_write(f"skipping KL horizon T={horizon} for {model_name}; already complete")
                    continue
                horizon_start = time.perf_counter()
                result = _run_local_optimization_for_horizon(
                    model_name=model_name,
                    trajectories=kl_trajectories,
                    random_search=random_search,
                    horizon=int(horizon),
                    horizon_index=int(horizon_index),
                    true_lambda=float(true_lambdas[horizon_index]),
                    args=args,
                )
                result["elapsed_s"] = float(time.perf_counter() - horizon_start)
                horizon_results[int(horizon)] = result
                _checkpoint(payload, args.output)

        model_payload.setdefault("timings", {})["total_model_s"] = float(time.perf_counter() - model_start)
        _checkpoint(payload, args.output)

    payload["completed_at"] = _utc_now()
    payload["timings"]["total_s"] = float(time.perf_counter() - total_start)
    _checkpoint(payload, args.output)
    print(f"wrote synthetic posterior concentration results to {args.output}")
    print(f"total time: {payload['timings']['total_s']:.2f}s")


if __name__ == "__main__":
    main()
