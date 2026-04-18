"""Build the three project notebooks from Python source.

Run with `python scripts/build_notebooks.py`.

Each notebook is a list of (kind, source) tuples where kind is "md" for
markdown and "code" for code. We dump them as nbformat 4.5 JSON.
"""

from __future__ import annotations

import json
import os
import textwrap
from typing import List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS_DIR = os.path.join(ROOT, "notebooks")


def _nb_from_cells(cells: List[Tuple[str, str]]) -> dict:
    nb_cells = []
    for i, (kind, src) in enumerate(cells):
        src = textwrap.dedent(src).strip("\n") + "\n"
        lines = [l + "\n" for l in src.splitlines()]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        cell_id = f"cell-{i:03d}"
        if kind == "md":
            nb_cells.append({
                "cell_type": "markdown",
                "id": cell_id,
                "metadata": {},
                "source": lines,
            })
        else:
            nb_cells.append({
                "cell_type": "code",
                "id": cell_id,
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": lines,
            })
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _write(name: str, cells: List[Tuple[str, str]]) -> None:
    nb = _nb_from_cells(cells)
    path = os.path.join(NOTEBOOKS_DIR, name)
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"wrote {path} ({len(cells)} cells)")


# ---------------------------------------------------------------------------
# 01 synthetic sanity
# ---------------------------------------------------------------------------

def build_synthetic() -> None:
    cells: List[Tuple[str, str]] = [
        ("md", """
        # 01 Synthetic sanity check

        Reproduce the four synthetic experiments from Section 5.2 of
        Madrigal-Cianci, Monsalve Maya, Breakey (2026) using the SMC inference
        pipeline in `pml_market`. This notebook is the correctness gate before
        running on real Polymarket data.

        - **Experiment 1**: posterior concentration `1 - pi_T` decays
          exponentially in `T` at the rate of the KL projection gap `delta_T`.
        - **Experiment 2**: posterior accuracy collapses as the informed
          mixture weight `omega_1` falls below ~0.15.
        - **Experiment 3**: `|Delta log BF|` is locally Lipschitz in increment
          perturbation `sigma`.
        - **Experiment 4**: realized information gain `IG(H_t)` saturates at
          the prior-entropy ceiling `log 2`.
        """),
        ("code", """
        from __future__ import annotations
        import os, sys
        ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
        if ROOT not in sys.path:
            sys.path.insert(0, ROOT)

        import numpy as np
        import matplotlib.pyplot as plt
        from tqdm.auto import tqdm

        from pml_market import synthetic, smc, vi, diagnostics, model

        rng = np.random.default_rng(0)
        plt.rcParams['figure.figsize'] = (7, 4.2)
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        """),
        ("md", """
        ## Experiment 1: posterior concentration

        Simulate `n_reps` histories under `Y = 1`, run SMC, record
        `1 - pi_T(H_T)` for `T in {25, 50, 100, 200, 400}`. Median across reps
        with 10-90% bands. Overlay slope `-delta_T_hat` from Theorem 4.2.
        """),
        ("code", """
        # Defaults are tuned for ~3-5 minute runtime. The paper uses 1000
        # replications and N=1000 particles; bump n_reps and n_particles for
        # tighter quantile bands at the cost of longer runs.
        T_grid = [25, 50, 100, 200]
        n_reps = 24
        n_particles = 250     # SMC particles per outcome
        T_max = max(T_grid)

        all_pi_t = []
        oracle_logBF = []
        for rep in tqdm(range(n_reps), desc='reps'):
            dx, v, _ = synthetic.simulate_history(T=T_max, y_true=1, seed=1000 + rep)
            res = smc.bayes_factor_smc(
                dx, v, pi0=0.5, n_particles=n_particles,
                mcmc_steps=3, seed=2000 + rep, record_pi_t=True,
            )
            all_pi_t.append(res['pi_t'])
            theta = synthetic.DEFAULT_THETA
            oracle_logBF.append(model.loglik(dx, v, 1, theta) - model.loglik(dx, v, 0, theta))

        all_pi_t = np.stack(all_pi_t, axis=0)   # (n_reps, T_max)
        err = 1.0 - all_pi_t                     # 1 - pi_t for each rep, t

        med = np.median(err, axis=0)
        lo  = np.quantile(err, 0.10, axis=0)
        hi  = np.quantile(err, 0.90, axis=0)
        ts = np.arange(1, T_max + 1)
        """),
        ("code", """
        # Estimate delta_T by fitting log med(1-pi_t) vs t (linear, late part)
        late = ts >= max(50, T_max // 4)
        log_med = np.log(np.clip(med[late], 1e-12, None))
        slope, intercept = np.polyfit(ts[late], log_med, 1)
        delta_T_hat = -slope
        print(f'Estimated delta_T (from posterior concentration slope) = {delta_T_hat:.4f} nats/period')
        print(f'Mean oracle log BF / T at T_max = {np.mean(oracle_logBF) / T_max:.4f} (also ~ delta_T)')
        """),
        ("code", """
        fig, ax = plt.subplots()
        ax.fill_between(ts, np.log(np.clip(lo, 1e-12, None)),
                            np.log(np.clip(hi, 1e-12, None)), alpha=0.25, label='10-90% band')
        ax.plot(ts, np.log(np.clip(med, 1e-12, None)), label='median')
        ax.plot(ts, intercept + slope * ts, '--', label=f'slope = -delta_T_hat = -{delta_T_hat:.3f}')
        ax.set_xlabel('horizon T')
        ax.set_ylabel('log(1 - pi_T)')
        ax.set_title('Experiment 1: posterior concentration under Y=1')
        ax.legend()
        plt.tight_layout()
        plt.show()
        """),
        ("md", """
        ## Experiment 2: identifiability threshold under omega_1 sweep

        Vary the informed-trader prevalence `omega_1` in
        {0.05, 0.10, 0.20, 0.30, 0.40, 0.50}, with the rest split equally
        between noise and manipulator types. Generate `n_reps` histories at
        `T=120` and report posterior accuracy.
        """),
        ("code", """
        omega1_grid = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
        n_reps_id = 16
        T_id = 100
        accuracy = []
        for w1 in omega1_grid:
            theta = {k: np.array(v_) for k, v_ in synthetic.DEFAULT_THETA.items()}
            theta['omega'] = np.array([w1, (1 - w1) / 2, (1 - w1) / 2])
            hits = 0
            for rep in tqdm(range(n_reps_id), desc=f'omega_1={w1}', leave=False):
                dx, v, y = synthetic.simulate_history(
                    T=T_id, y_true=1, theta=theta, seed=3000 + rep,
                )
                res = smc.bayes_factor_smc(
                    dx, v, pi0=0.5, n_particles=200, mcmc_steps=2,
                    seed=4000 + rep,
                )
                hits += int(res['posterior'] > 0.5)
            accuracy.append(hits / n_reps_id)
            print(f'  omega_1={w1:.2f}: accuracy={hits / n_reps_id:.2%}')
        """),
        ("code", r"""
        fig, ax = plt.subplots()
        ax.plot(omega1_grid, accuracy, 'o-')
        ax.axhline(0.5, color='gray', linestyle=':', label='chance')
        ax.set_xlabel(r'informed weight $\omega_1$')
        ax.set_ylabel('posterior accuracy at T=120')
        ax.set_title(r'Experiment 2: identifiability collapse as $\omega_1$ shrinks')
        ax.legend()
        plt.tight_layout()
        plt.show()
        """),
        ("md", """
        ## Experiment 3: stability under increment perturbation

        Perturb a single baseline history with i.i.d. Gaussian noise of std
        `sigma in {0.01, 0.02, 0.05, 0.1, 0.2}` and measure
        `|log BF(h) - log BF(h')|`. Compare to the Lipschitz bound from
        Theorem 4.4 with `Lx = R / sigma_min^2` (Remark 4.3) and `R` set to
        the empirical 0.99 quantile of `|Delta x|`.
        """),
        ("code", """
        dx_base, v_base, _ = synthetic.simulate_history(T=150, y_true=1, seed=42)
        baseline = smc.bayes_factor_smc(dx_base, v_base, pi0=0.5, n_particles=400,
                                        mcmc_steps=3, seed=42)
        log_BF_base = baseline['log_BF']

        sigmas = [0.01, 0.02, 0.05, 0.10, 0.20]
        diffs = []
        bounds = []
        R = float(np.quantile(np.abs(dx_base), 0.99))
        Lx = diagnostics.gaussian_lipschitz_constant(R, sigma_min=0.2)
        rng_pert = np.random.default_rng(99)
        for sigma in sigmas:
            row = []
            for trial in range(4):
                dx_p = diagnostics.perturb_history(dx_base, sigma, rng=rng_pert)
                resp = smc.bayes_factor_smc(dx_p, v_base, pi0=0.5, n_particles=400,
                                             mcmc_steps=3, seed=42 + trial)
                row.append(abs(resp['log_BF'] - log_BF_base))
            diffs.append(np.mean(row))
            # bound = 2 * Lx * sum |dx - dx'|; expected sum is T * sigma * sqrt(2/pi)
            T = dx_base.shape[0]
            bounds.append(2 * Lx * T * sigma * np.sqrt(2 / np.pi))
            print(f'  sigma={sigma:>5.2f}  mean |Delta log BF|={diffs[-1]:.3f}  bound={bounds[-1]:.3f}')
        """),
        ("code", r"""
        fig, ax = plt.subplots()
        ax.plot(sigmas, diffs, 'o-', label="mean |log BF(h) - log BF(h')|")
        ax.plot(sigmas, bounds, '--', label='Theorem 4.4 bound (typical event)')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(r'perturbation std $\sigma$')
        ax.set_ylabel(r"$|\log BF(h) - \log BF(h')|$")
        ax.set_title('Experiment 3: stability under increment perturbation')
        ax.legend()
        plt.tight_layout()
        plt.show()
        """),
        ("md", """
        ## Experiment 4: realized information gain saturates at log 2

        Long-horizon simulation with `T = 600`, prior `pi_0 = 1/2`, and the
        online posterior. Plot mean `IG(H_t)` across reps with the prior
        entropy ceiling `log 2`.
        """),
        ("code", """
        T_long = 400
        n_reps_ig = 12
        ig_traces = []
        for rep in tqdm(range(n_reps_ig), desc='reps'):
            dx, v, _ = synthetic.simulate_history(T=T_long, y_true=1, seed=5000 + rep)
            res = smc.bayes_factor_smc(dx, v, pi0=0.5, n_particles=300, mcmc_steps=3,
                                       seed=6000 + rep, record_pi_t=True)
            ig_traces.append(diagnostics.information_gain_trace(res['pi_t'], pi0=0.5))
        ig_traces = np.stack(ig_traces, axis=0)
        med_ig = np.median(ig_traces, axis=0)
        lo_ig  = np.quantile(ig_traces, 0.10, axis=0)
        hi_ig  = np.quantile(ig_traces, 0.90, axis=0)
        ts_long = np.arange(1, T_long + 1)
        """),
        ("code", """
        fig, ax = plt.subplots()
        ax.fill_between(ts_long, lo_ig, hi_ig, alpha=0.25, label='10-90% band')
        ax.plot(ts_long, med_ig, label='median IG(H_t)')
        ax.axhline(np.log(2), color='red', linestyle='--', label='log 2 (prior entropy)')
        ax.set_xlabel('t')
        ax.set_ylabel('realized information gain (nats)')
        ax.set_title('Experiment 4: information gain dynamics')
        ax.legend()
        plt.tight_layout()
        plt.show()
        """),
        ("md", """
        ## Sanity check: SMC vs VI on a single history

        Confirm that the variational approximation produces a posterior in the
        same neighborhood as SMC on a single moderate-signal history. Per
        Remark 5.1 the difference of ELBOs is biased; we use SMC as ground
        truth.
        """),
        ("code", """
        dx, v, _ = synthetic.simulate_history(T=200, y_true=1, seed=7)
        smc_res = smc.bayes_factor_smc(dx, v, pi0=0.5, n_particles=800, mcmc_steps=4, seed=7)
        vi_res  = vi.bayes_factor_vi(dx, v, pi0=0.5, n_steps=1500, n_samples=16,
                                     learning_rate=0.05, seed=7)
        print(f'SMC: log BF = {smc_res[\"log_BF\"]:+.3f}  posterior = {smc_res[\"posterior\"]:.3f}')
        print(f'VI : log BF = {vi_res[\"log_BF\"]:+.3f}  posterior = {vi_res[\"posterior\"]:.3f}')
        """),
    ]
    _write("01_synthetic_sanity.ipynb", cells)


# ---------------------------------------------------------------------------
# 02 single-market deep dive
# ---------------------------------------------------------------------------

def build_single() -> None:
    cells: List[Tuple[str, str]] = [
        ("md", """
        # 02 Single-market deep dive

        Apply the Bayesian-inverse-problem framework to one resolved
        Polymarket binary market. We pull the live price-volume history,
        align it to the eventual winner so the truth is `Y = 1`, and run the
        full diagnostic suite from Section 4 of the paper:

        1. **Online posterior trace** `pi_t` overlaid on the market-implied
           probability.
        2. **Posterior concentration** by re-evaluating SMC at sub-horizons.
        3. **Stability** under Gaussian perturbations of `Delta x` against the
           Theorem 4.4 bound.
        4. **Information gain dynamics** `IG(H_t)` plus effective
           informativeness `eta(v_t; theta_hat)`.
        """),
        ("code", """
        from __future__ import annotations
        import os, sys
        ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
        if ROOT not in sys.path:
            sys.path.insert(0, ROOT)

        import numpy as np
        import matplotlib.pyplot as plt
        from tqdm.auto import tqdm

        from pml_market import data, smc, vi, diagnostics, model

        plt.rcParams['figure.figsize'] = (8, 4.5)
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        """),
        ("md", """
        ## Pick a market

        We auto-discover a long-horizon resolved binary by paging through the
        Gamma `/markets?closed=true` endpoint, keeping markets with at least
        `MIN_TRADES` recent trades and a clear winner, then picking the one
        with the largest horizon. To override, set `OVERRIDE_SLUG` to any
        closed binary slug from Polymarket.
        """),
        ("code", """
        OVERRIDE_SLUG: str | None = None  # e.g. 'btc-updown-5m-1776134700'
        BUCKET_MINUTES = 5                 # bucket size; tune for desired T
        MIN_TRADES = 200
        MIN_VOLUME = 50_000.0
        TARGET_HORIZON = 80               # stop once we find a market with T >= this

        if OVERRIDE_SLUG is None:
            candidates = data.list_resolved_binary_markets(
                limit=80, min_volume=MIN_VOLUME,
            )
            print(f'scanning {len(candidates)} resolved binary markets...')
            best = None
            for m in candidates:
                try:
                    traj = data._trajectory_from_market(m, bucket_minutes=BUCKET_MINUTES)
                except Exception:
                    continue
                if traj['metadata']['trade_count'] < MIN_TRADES:
                    continue
                if best is None or traj['horizon'] > best['horizon']:
                    best = traj
                    print(f'  candidate: T={traj[\"horizon\"]} trades={traj[\"metadata\"][\"trade_count\"]}'
                          f' slug={m[\"slug\"]!r}')
                if best['horizon'] >= TARGET_HORIZON:
                    break
            if best is None:
                raise RuntimeError('no resolved binary with enough trades found')
            traj = best
        else:
            traj = data.fetch_market_history(OVERRIDE_SLUG, bucket_minutes=BUCKET_MINUTES)

        SELECTED_SLUG = traj['metadata']['slug']
        print(f'\\nusing slug: {SELECTED_SLUG!r}, horizon T = {traj[\"horizon\"]},'
              f' winner = {traj[\"winner_label\"]!r}')
        """),
        ("code", """
        dx, v, y = data.trajectory_to_arrays(traj)
        T = dx.shape[0]
        print(f'T = {T}, y (winner-aligned) = {y}')
        print(f'|dx| stats: mean={np.mean(np.abs(dx)):.4f}, 0.99 quantile={np.quantile(np.abs(dx), 0.99):.4f}')
        print(f'v stats: mean={v.mean():.2f}, max={v.max():.2f}, fraction zero={(v == 0).mean():.2%}')
        """),
        ("code", """
        # Plot the raw market history (winner-aligned price + volume)
        prices = np.array(traj['prices'])
        times = np.array(traj['times']) / 3600.0  # hours since first bucket
        times -= times[0]

        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 5))
        axes[0].plot(times, prices, color='C0')
        axes[0].set_ylabel('winner-aligned price')
        axes[0].set_ylim(-0.02, 1.02)
        axes[0].set_title(f'{traj[\"metadata\"][\"question\"]}')
        axes[1].bar(times[1:], v, width=BUCKET_MINUTES / 60.0, color='C2', alpha=0.6)
        axes[1].set_ylabel('bucket volume')
        axes[1].set_xlabel('hours from start')
        plt.tight_layout()
        plt.show()
        """),
        ("md", """
        ## (1) Online posterior trace

        Run SMC under both outcomes with `record_pi_t=True` so we get
        `pi_t = P(Y=1 | H_t)` after every step. Compare to the market-implied
        winner-aligned price.
        """),
        ("code", """
        n_particles = 800
        smc_res = smc.bayes_factor_smc(
            dx, v, pi0=0.5, n_particles=n_particles, mcmc_steps=4,
            seed=0, record_pi_t=True, verbose=False,
        )
        print(f'SMC log BF = {smc_res[\"log_BF\"]:+.3f}')
        print(f'SMC posterior P(Y=1|H_T) = {smc_res[\"posterior\"]:.4f}  (truth = {y})')
        """),
        ("code", """
        ts = np.arange(1, T + 1)
        fig, ax = plt.subplots()
        ax.plot(ts, prices[1:], color='C0', alpha=0.7, label='market-implied (winner-aligned)')
        ax.plot(ts, smc_res['pi_t'], color='C3', label='SMC posterior pi_t = P(Y=1|H_t)')
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('time bucket t')
        ax.set_ylabel('probability that winner wins')
        ax.set_title(f'Online posterior vs market-implied probability ({SELECTED_SLUG})')
        ax.set_ylim(-0.02, 1.05)
        ax.legend(loc='best')
        plt.tight_layout()
        plt.show()
        """),
        ("md", """
        ## (2) Posterior concentration vs sub-horizon

        Truncate the history to `T_sub` and re-run SMC for each sub-horizon.
        We expect the error `1 - pi_T` to decay roughly exponentially in `T`
        once the KL-projection gap is identified (Theorem 4.3).
        """),
        ("code", """
        sub_grid = sorted(set([max(5, T // 10), max(10, T // 5),
                               max(20, T // 3), max(40, T // 2),
                               max(80, T * 3 // 4), T]))
        sub_grid = [t for t in sub_grid if t <= T]
        errs = []
        for Tsub in tqdm(sub_grid, desc='sub-horizons'):
            r = smc.bayes_factor_smc(dx[:Tsub], v[:Tsub], pi0=0.5,
                                     n_particles=n_particles, mcmc_steps=3, seed=1)
            errs.append(1.0 - r['posterior'])
            print(f'  T_sub={Tsub:>4d}: posterior={r[\"posterior\"]:.4f}')
        errs = np.array(errs)
        """),
        ("code", """
        fig, ax = plt.subplots()
        ax.plot(sub_grid, np.clip(errs, 1e-6, None), 'o-')
        ax.set_yscale('log')
        ax.set_xlabel('T (sub-horizon)')
        ax.set_ylabel('1 - pi_T (log scale)')
        ax.set_title('(2) Posterior concentration vs sub-horizon')
        plt.tight_layout()
        plt.show()
        """),
        ("md", """
        ## (3) Stability to increment perturbations

        Add iid Gaussian noise to `Delta x` for several `sigma`, re-run SMC,
        and compare `|log BF(h) - log BF(h')|` to the Theorem 4.4 bound with
        `Lx = R / sigma_min^2` and `R = 0.99` quantile of `|Delta x|`.
        """),
        ("code", """
        # Use *common random numbers* across base and perturbed: fix the SMC
        # seed so that noise from particle randomness cancels and we measure
        # only the effect of the perturbation.
        sigmas = [0.005, 0.01, 0.02, 0.05, 0.1]
        diffs = []
        rng_p = np.random.default_rng(123)
        R = float(np.quantile(np.abs(dx), 0.99))
        Lx = diagnostics.gaussian_lipschitz_constant(R, sigma_min=0.2)
        crn_seed = 7777
        log_BF_base = smc.bayes_factor_smc(dx, v, pi0=0.5, n_particles=600,
                                          mcmc_steps=3, seed=crn_seed)['log_BF']
        for s in sigmas:
            tr = []
            for trial in range(4):
                dxp = diagnostics.perturb_history(dx, s, rng=rng_p)
                rp = smc.bayes_factor_smc(dxp, v, pi0=0.5, n_particles=600,
                                          mcmc_steps=3, seed=crn_seed)
                tr.append(abs(rp['log_BF'] - log_BF_base))
            diffs.append(np.mean(tr))
            print(f'  sigma={s:>5.3f}  mean |delta log BF|={diffs[-1]:.3f}')
        diffs = np.array(diffs)
        bounds = 2 * Lx * T * np.array(sigmas) * np.sqrt(2 / np.pi)
        """),
        ("code", """
        fig, ax = plt.subplots()
        ax.plot(sigmas, diffs, 'o-', label="|log BF(h) - log BF(h')|")
        ax.plot(sigmas, bounds, '--', label='Theorem 4.4 bound')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('perturbation std sigma')
        ax.set_ylabel('|delta log BF|')
        ax.set_title('(3) Stability under perturbation')
        ax.legend()
        plt.tight_layout()
        plt.show()
        """),
        ("md", """
        ## (4) Information gain and effective informativeness

        Plot realized IG(H_t) over time; saturating values indicate the
        posterior has concentrated. The effective informativeness `eta(v;theta)`
        (Definition 4.3) is computed at the variational-mean theta from VI.
        """),
        ("code", """
        ig = diagnostics.information_gain_trace(smc_res['pi_t'], pi0=0.5)

        # Use VI to get a single theta point estimate (variational mean) for
        # eta(v; theta_hat).
        vi_res = vi.bayes_factor_vi(dx, v, pi0=0.5, n_steps=1200, n_samples=8,
                                    learning_rate=0.05, seed=0)
        theta_hat = vi_res['vi1']['theta_mean']  # under outcome y=1
        eta = diagnostics.effective_informativeness(v, theta_hat)
        print(f'VI log BF = {vi_res[\"log_BF\"]:+.3f}  (SMC: {smc_res[\"log_BF\"]:+.3f})')
        print(f'theta_hat omega = {theta_hat[\"omega\"]}')
        print(f'theta_hat mu1   = {float(theta_hat[\"mu1\"]):.3f}')
        print(f'theta_hat mu3   = {float(theta_hat[\"mu3\"]):.3f}')
        print(f'theta_hat tau3  = {float(theta_hat[\"tau3\"]):.3f}')
        """),
        ("code", """
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 5))
        axes[0].plot(ts, ig, color='C3', label='IG(H_t)')
        axes[0].axhline(np.log(2), linestyle='--', color='red', label='log 2 (prior entropy)')
        axes[0].set_ylabel('IG (nats)')
        axes[0].set_title('(4) Information gain and effective informativeness')
        axes[0].legend()

        axes[1].plot(ts, eta, color='C2', label='eta(v_t; theta_hat)')
        axes[1].axhline(0.0, linestyle=':', color='gray')
        axes[1].set_xlabel('time bucket t')
        axes[1].set_ylabel('eta')
        axes[1].legend()
        plt.tight_layout()
        plt.show()
        """),
    ]
    _write("02_single_market_deepdive.ipynb", cells)


# ---------------------------------------------------------------------------
# 03 panel evaluation
# ---------------------------------------------------------------------------

def build_panel() -> None:
    cells: List[Tuple[str, str]] = [
        ("md", """
        # 03 Panel evaluation on resolved Polymarket binaries

        Run mean-field VI over a panel of resolved binary markets, treating
        each as an independent Bayesian inverse problem. For each market we
        record the posterior `P(Y = 1 | H_T)` (with `Y = 1` always under the
        winner-aligned convention) and the final market-implied probability,
        and aggregate hit-rate, Brier score, calibration, and information gain.

        The panel pull is cached on disk so re-runs are free.
        """),
        ("code", """
        from __future__ import annotations
        import os, sys, json
        ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
        if ROOT not in sys.path:
            sys.path.insert(0, ROOT)

        import numpy as np
        import matplotlib.pyplot as plt
        from tqdm.auto import tqdm

        from pml_market import data, vi, diagnostics

        plt.rcParams['figure.figsize'] = (7, 4.2)
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        """),
        ("md", """
        ## Pull (or load cached) panel of resolved markets
        """),
        ("code", """
        N_MARKETS = 30
        BUCKET_MINUTES = 5      # 5 min buckets balance T-resolution vs noise
        MIN_VOLUME = 20_000.0
        MIN_TRADES = 80
        CACHE_PATH = os.path.join(ROOT, 'cache', f'panel_n{N_MARKETS}_b{BUCKET_MINUTES}.json')

        trajectories = data.fetch_resolved_binary_markets(
            n=N_MARKETS, bucket_minutes=BUCKET_MINUTES,
            min_volume=MIN_VOLUME, min_trades=MIN_TRADES,
            cache_path=CACHE_PATH, verbose=True,
        )
        print(f'panel size: {len(trajectories)} markets')
        Ts = [t['horizon'] for t in trajectories]
        print(f'horizon T: median={int(np.median(Ts))}  range=[{min(Ts)}, {max(Ts)}]')
        """),
        ("md", """
        ## Run VI on each market

        We evaluate two variants per market:

        - **Full history**: use all `T` increments. This is the natural
          posterior given the entire price record, including the final
          resolution event.
        - **Truncated history**: drop the trailing 10% of buckets to remove
          the resolution-time "price flip" that violates the model's i.i.d.
          assumption. This measures whether the model can predict the outcome
          from interior market dynamics alone.
        """),
        ("code", """
        records = []
        for traj in tqdm(trajectories, desc='VI'):
            dx, v, y = data.trajectory_to_arrays(traj)
            T = dx.shape[0]
            if T < 10:
                continue
            T_trim = max(5, int(T * 0.9))
            try:
                res_full = vi.bayes_factor_vi(
                    dx, v, pi0=0.5, n_steps=400, n_samples=8,
                    learning_rate=0.05, seed=0,
                )
                res_trim = vi.bayes_factor_vi(
                    dx[:T_trim], v[:T_trim], pi0=0.5,
                    n_steps=400, n_samples=8, learning_rate=0.05, seed=0,
                )
            except Exception as e:
                print(f'  skip {traj[\"metadata\"][\"slug\"]!r}: {e}')
                continue
            final_market_p = float(traj['prices'][-1])
            mid_market_p   = float(traj['prices'][T_trim])
            records.append({
                'slug': traj['metadata']['slug'],
                'question': traj['metadata']['question'],
                'T': T, 'T_trim': T_trim,
                'volume': sum(v.tolist()),
                'posterior_full': res_full['posterior'],
                'posterior_trim': res_trim['posterior'],
                'log_BF_full': res_full['log_BF'],
                'log_BF_trim': res_trim['log_BF'],
                'final_market_p': final_market_p,
                'mid_market_p': mid_market_p,
                'IG_full': diagnostics.realized_information_gain(res_full['posterior'], pi0=0.5),
                'IG_trim': diagnostics.realized_information_gain(res_trim['posterior'], pi0=0.5),
                'y_truth': y,
            })
        print(f'{len(records)} markets evaluated')
        """),
        ("md", """
        ## Aggregate metrics
        """),
        ("code", """
        post_full = np.array([r['posterior_full'] for r in records])
        post_trim = np.array([r['posterior_trim'] for r in records])
        market_full = np.array([r['final_market_p'] for r in records])
        market_trim = np.array([r['mid_market_p'] for r in records])
        ig_full = np.array([r['IG_full'] for r in records])
        ig_trim = np.array([r['IG_trim'] for r in records])
        Ts = np.array([r['T'] for r in records])
        truth = np.array([r['y_truth'] for r in records])  # all 1

        def metrics(p):
            hit = float((p > 0.5).mean())
            brier = float(np.mean((p - truth) ** 2))
            ll = float(-np.mean(np.log(np.clip(p, 1e-6, 1 - 1e-6))))
            return hit, brier, ll

        h_pf, b_pf, l_pf = metrics(post_full)
        h_pt, b_pt, l_pt = metrics(post_trim)
        h_mf, b_mf, l_mf = metrics(market_full)
        h_mt, b_mt, l_mt = metrics(market_trim)

        print(f'{\"metric\":>14s}  {\"VI full\":>10s}  {\"VI trim\":>10s}  {\"mkt full\":>10s}  {\"mkt 90%\":>10s}')
        print('-' * 64)
        print(f'{\"hit rate\":>14s}  {h_pf:>10.3f}  {h_pt:>10.3f}  {h_mf:>10.3f}  {h_mt:>10.3f}')
        print(f'{\"Brier\":>14s}  {b_pf:>10.4f}  {b_pt:>10.4f}  {b_mf:>10.4f}  {b_mt:>10.4f}')
        print(f'{\"log loss\":>14s}  {l_pf:>10.4f}  {l_pt:>10.4f}  {l_mf:>10.4f}  {l_mt:>10.4f}')
        print()
        print(f'mean realized IG  full: {ig_full.mean():.3f}   trim: {ig_trim.mean():.3f}'
              f'   (ceiling log 2 = {np.log(2):.3f})')
        """),
        ("md", """
        Reading these results: under the **full** history every market has a
        sharp resolution-time price jump that violates the i.i.d. mixture
        assumption — the model interprets the pre-resolution drift as
        manipulator activity and predicts the *opposite* outcome. Under a
        **trimmed** history that drops the last 10% of buckets, the model's
        forecast quality typically reverts toward the market-implied
        probability at the truncation point.
        """),
        ("md", """
        ## Calibration / reliability

        Bin predictions into 10 buckets and plot mean predicted vs empirical
        win rate. Since `y_truth = 1` for every market, the empirical win rate
        per bucket equals the fraction of resolved-true cases (by construction
        always 1), so the standard reliability plot doesn't apply. Instead we
        show the distribution of posteriors and the relationship with the
        final market-implied probability.
        """),
        ("code", """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].hist(post_full, bins=20, color='C3', alpha=0.6, label='VI (full history)')
        axes[0].hist(post_trim, bins=20, color='C2', alpha=0.6, label='VI (trimmed 90%)')
        axes[0].hist(market_full, bins=20, color='C0', alpha=0.4, label='market final price')
        axes[0].axvline(0.5, color='gray', linestyle=':')
        axes[0].set_xlabel('predicted P(winner wins)')
        axes[0].set_ylabel('count')
        axes[0].set_title('Distribution of forecasts')
        axes[0].legend()

        axes[1].scatter(market_trim, post_trim, alpha=0.7, label='trimmed')
        axes[1].scatter(market_full, post_full, alpha=0.5, marker='x', label='full')
        axes[1].plot([0, 1], [0, 1], '--', color='gray')
        axes[1].set_xlabel('market-implied probability (at truncation point)')
        axes[1].set_ylabel('VI posterior')
        axes[1].set_title('VI posterior vs market price')
        axes[1].set_xlim(-0.02, 1.02); axes[1].set_ylim(-0.02, 1.02)
        axes[1].legend()
        plt.tight_layout()
        plt.show()
        """),
        ("md", """
        ## IG vs horizon
        """),
        ("code", """
        fig, ax = plt.subplots()
        ax.scatter(Ts, ig_full, alpha=0.6, marker='x', label='full')
        ax.scatter(Ts, ig_trim, alpha=0.7, label='trimmed')
        ax.axhline(np.log(2), linestyle='--', color='red', label='log 2 ceiling')
        ax.set_xlabel('horizon T')
        ax.set_ylabel('realized information gain (nats)')
        ax.set_xscale('log')
        ax.set_title('Realized IG vs horizon')
        ax.legend()
        plt.tight_layout()
        plt.show()
        """),
        ("md", """
        ## Worst-mistake markets

        List the markets where the VI posterior was confidently wrong (i.e.
        below 0.3 despite Y = 1).
        """),
        ("code", """
        sorted_records = sorted(records, key=lambda r: r['posterior_trim'])
        header = f'{\"post_trim\":>9s}  {\"post_full\":>9s}  {\"mkt 90%\":>8s}  {\"mkt end\":>8s}  {\"T\":>5s}  question'
        print(header)
        print('-' * len(header))
        for r in sorted_records[:10]:
            q = r['question'][:70]
            print(
                f'  {r[\"posterior_trim\"]:>7.3f}  {r[\"posterior_full\"]:>7.3f}'
                f'  {r[\"mid_market_p\"]:>7.3f}  {r[\"final_market_p\"]:>7.3f}'
                f'  {r[\"T\"]:>5d}  {q}'
            )
        """),
    ]
    _write("03_panel_evaluation.ipynb", cells)


if __name__ == "__main__":
    os.makedirs(NOTEBOOKS_DIR, exist_ok=True)
    build_synthetic()
    build_single()
    build_panel()
