"""Run the panel evaluation logic directly (mirror of notebook 03)."""
from __future__ import annotations

import os, sys, json
import numpy as np
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pml_market import data, vi, diagnostics  # noqa: E402

N_MARKETS = 30
BUCKET_MINUTES = 5
MIN_VOLUME = 20_000.0
MIN_TRADES = 80
CACHE_PATH = os.path.join(ROOT, 'cache', f'panel_n{N_MARKETS}_b{BUCKET_MINUTES}.json')

print(f"loading panel from {CACHE_PATH} ...")
trajectories = data.fetch_resolved_binary_markets(
    n=N_MARKETS, bucket_minutes=BUCKET_MINUTES,
    min_volume=MIN_VOLUME, min_trades=MIN_TRADES,
    cache_path=CACHE_PATH, verbose=True,
)
print(f'panel size: {len(trajectories)} markets')
Ts = [t['horizon'] for t in trajectories]
print(f'horizon T: median={int(np.median(Ts))}  range=[{min(Ts)}, {max(Ts)}]')

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
        print(f'  skip {traj["metadata"]["slug"]!r}: {e}')
        continue
    final_market_p = float(traj['prices'][-1])
    mid_market_p   = float(traj['prices'][T_trim])
    records.append({
        'slug': traj['metadata']['slug'],
        'question': traj['metadata']['question'],
        'T': T, 'T_trim': T_trim,
        'volume': float(np.sum(v)),
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

print(f'\n{len(records)} markets evaluated')

post_full = np.array([r['posterior_full'] for r in records])
post_trim = np.array([r['posterior_trim'] for r in records])
market_full = np.array([r['final_market_p'] for r in records])
market_trim = np.array([r['mid_market_p'] for r in records])
ig_full = np.array([r['IG_full'] for r in records])
ig_trim = np.array([r['IG_trim'] for r in records])
truth = np.array([r['y_truth'] for r in records])


def metrics(p):
    hit = float((p > 0.5).mean())
    brier = float(np.mean((p - truth) ** 2))
    ll = float(-np.mean(np.log(np.clip(p, 1e-6, 1 - 1e-6))))
    return hit, brier, ll


h_pf, b_pf, l_pf = metrics(post_full)
h_pt, b_pt, l_pt = metrics(post_trim)
h_mf, b_mf, l_mf = metrics(market_full)
h_mt, b_mt, l_mt = metrics(market_trim)

print()
print(f'{"metric":>14s}  {"VI full":>10s}  {"VI trim":>10s}  {"mkt full":>10s}  {"mkt 90%":>10s}')
print('-' * 64)
print(f'{"hit rate":>14s}  {h_pf:>10.3f}  {h_pt:>10.3f}  {h_mf:>10.3f}  {h_mt:>10.3f}')
print(f'{"Brier":>14s}  {b_pf:>10.4f}  {b_pt:>10.4f}  {b_mf:>10.4f}  {b_mt:>10.4f}')
print(f'{"log loss":>14s}  {l_pf:>10.4f}  {l_pt:>10.4f}  {l_mf:>10.4f}  {l_mt:>10.4f}')
print()
print(f'mean realized IG  full: {ig_full.mean():.3f}   trim: {ig_trim.mean():.3f}'
      f'   (ceiling log 2 = {np.log(2):.3f})')

OUT = os.path.join(ROOT, 'cache', 'panel_results.json')
with open(OUT, 'w') as f:
    json.dump(records, f, indent=2)
print(f'\nwrote {OUT}')
