"""Data adapters for Polymarket histories.

Two main entry points:

  - `trajectory_to_arrays(trajectory)`: converts the dict produced by
    `notebooks/polymarket_data_pull.ipynb` into the (Delta x, v, y) tuple the
    model consumes.

  - `fetch_resolved_binary_markets(...)`: pulls a panel of resolved binary
    Polymarket markets via the public Gamma/Data/CLOB APIs (stdlib only) and
    caches them on disk so the notebooks don't have to refetch.

The fetch helpers re-implement the small stdlib HTTP wrappers from the
notebook (`fetch_json`, `parse_json_list`, etc.) inside the package so the
package is usable without first running the notebook.
"""

from __future__ import annotations

import json
import os
import time
import math
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np


GAMMA_API = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"


# ---------------------------------------------------------------------------
# Trajectory -> (dx, v, y) adapter
# ---------------------------------------------------------------------------

def truncate_trajectory(trajectory: Dict[str, Any],
                        lookback: int) -> Dict[str, Any]:
    """Return a copy of *trajectory* keeping only the last *lookback* buckets.

    Markets are often open for months; the price action relevant to resolution
    is concentrated near the end.  Slicing the last *lookback* buckets gives a
    horizon of min(T, lookback) steps ending at the resolution bucket.

    Parameters
    ----------
    trajectory : dict produced by build_trajectory / fetch_market_history.
    lookback   : number of buckets to keep (counting back from resolution).
                 If lookback >= T the trajectory is returned unchanged.
    """
    prices  = list(trajectory["prices"])   # length T+1
    volumes = list(trajectory["volumes"])  # length T
    times   = list(trajectory["times"])    # length T+1
    T = len(volumes)
    if lookback >= T:
        return trajectory
    prices  = prices[-(lookback + 1):]
    volumes = volumes[-lookback:]
    times   = times[-(lookback + 1):]
    return {
        **trajectory,
        "prices":  prices,
        "volumes": volumes,
        "times":   times,
        "horizon": lookback,
    }

def _logit(p, eps: float = 1e-6) -> float:
    p = float(p)
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def trajectory_to_arrays(trajectory: Dict[str, Any]
                         ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Convert the notebook trajectory dict to (dx, v, y).

    The trajectory dict is the output of `build_binary_trajectory(...)` in
    `notebooks/polymarket_data_pull.ipynb` and contains:
      - "prices":  length T+1 list, winner-aligned market-implied probabilities,
      - "volumes": length T list of bucket volumes,
      - "winner_index": index of the resolved outcome.

    Because prices are aligned to the eventual winner, the truth is always
    Y = 1 in our convention, regardless of `winner_index`.

    Returns
    -------
    dx : (T,) array of log-odds increments dx_t = logit(p_t) - logit(p_{t-1}).
    v  : (T,) array of volumes.
    y  : 1 (by construction, since prices are winner-aligned).
    """
    prices = list(trajectory["prices"])
    volumes = list(trajectory["volumes"])
    if len(prices) != len(volumes) + 1:
        raise ValueError(
            f"trajectory has prices length {len(prices)} but volumes length "
            f"{len(volumes)}; expected len(prices) = len(volumes) + 1"
        )
    logits = np.array([_logit(p) for p in prices])
    dx = np.diff(logits)
    v = np.asarray(volumes, dtype=np.float64)
    return dx, v, 1


# ---------------------------------------------------------------------------
# Stdlib HTTP helpers (mirrors of the notebook)
# ---------------------------------------------------------------------------

def _fetch_json(base_url: str, path: str, params: Optional[dict] = None,
                timeout: int = 20) -> Any:
    query = urlencode(params or {}, doseq=True)
    url = f"{base_url}{path}"
    if query:
        url = f"{url}?{query}"
    req = Request(url, headers={
        "User-Agent": "pml-market/1.0 (research)",
        "Accept": "application/json",
    })
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _parse_json_list(value):
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return json.loads(value)
    raise TypeError(f"cannot parse JSON list from {type(value)!r}")


def _normalize_market(market: dict) -> dict:
    outcomes = [str(x) for x in _parse_json_list(market.get("outcomes"))]
    outcome_prices = [float(x) for x in _parse_json_list(market.get("outcomePrices"))]
    token_ids = [str(x) for x in _parse_json_list(market.get("clobTokenIds"))]
    return {
        "question": str(market.get("question", "")),
        "slug": str(market.get("slug", "")),
        "condition_id": str(market.get("conditionId", "")),
        "outcomes": outcomes,
        "outcome_prices": outcome_prices,
        "token_ids": token_ids,
        "closed": bool(market.get("closed", False)),
        "active": bool(market.get("active", False)),
        "volume": float(market.get("volumeNum", market.get("volume", 0.0)) or 0.0),
    }


def get_market_by_slug(slug: str, timeout: int = 20) -> dict:
    markets = _fetch_json(GAMMA_API, "/markets", {"slug": slug, "limit": 5},
                          timeout=timeout)
    if markets:
        return _normalize_market(markets[0])
    events = _fetch_json(GAMMA_API, "/events", {"slug": slug, "limit": 5},
                         timeout=timeout)
    if not events:
        raise ValueError(f"no Polymarket market found for slug {slug!r}")
    nested = events[0].get("markets") or []
    if not nested:
        raise ValueError(f"event {slug!r} did not contain any markets")
    return _normalize_market(nested[0])


def get_trades(condition_id: str, page_size: int = 500,
               max_pages: int = 200, timeout: int = 20) -> List[dict]:
    trades, seen = [], set()
    for page in range(max_pages):
        try:
            batch = _fetch_json(DATA_API, "/trades", {
                "market": condition_id,
                "limit": page_size,
                "offset": page * page_size,
            }, timeout=timeout)
        except HTTPError as exc:
            if exc.code == 400 and page > 0:
                break
            raise
        if not batch:
            break
        for tr in batch:
            key = (tr.get("transactionHash"), tr.get("asset"),
                   tr.get("timestamp"), tr.get("price"), tr.get("size"))
            if key not in seen:
                seen.add(key)
                trades.append(tr)
        if len(batch) < page_size:
            break
    trades.sort(key=lambda r: (int(r["timestamp"]), str(r.get("transactionHash", ""))))
    return trades


def get_price_history(token_id: str, fidelity_minutes: int = 1,
                      interval: str = "max", timeout: int = 20) -> List[dict]:
    payload = _fetch_json(CLOB_API, "/prices-history", {
        "market": token_id,
        "interval": interval,
        "fidelity": fidelity_minutes,
    }, timeout=timeout)
    return list(payload.get("history", []))


# ---------------------------------------------------------------------------
# Trajectory builder (mirror of the notebook, more tolerant)
# ---------------------------------------------------------------------------

def _clip(p: float, eps: float = 1e-6) -> float:
    p = float(p)
    if p < eps:
        return eps
    if p > 1.0 - eps:
        return 1.0 - eps
    return p


def _resolved_winner_index(market: dict, threshold: float = 0.99) -> Optional[int]:
    if len(market["outcomes"]) != 2:
        return None
    for i, p in enumerate(market["outcome_prices"]):
        if float(p) >= threshold:
            return i
    return None


def build_trajectory(market: dict, trades: List[dict],
                     bucket_minutes: int = 1, resolve_threshold: float = 0.99,
                     price_history: Optional[List[dict]] = None) -> Optional[dict]:
    """Bucketize trades into (price, volume) sequence aligned to the winner.

    Returns None if the market is not a binary that can be aligned.
    """
    winner = _resolved_winner_index(market, resolve_threshold)
    if winner is None or len(market["token_ids"]) != 2:
        return None
    if not trades:
        return None

    bucket_seconds = int(bucket_minutes * 60)
    buckets: Dict[int, dict] = {}
    for tr in sorted(trades, key=lambda r: int(r["timestamp"])):
        ts = int(tr["timestamp"])
        price = float(tr["price"])
        size = float(tr["size"])
        outcome_index = int(tr["outcomeIndex"])
        implied = price if outcome_index == winner else 1.0 - price
        bucket = (ts // bucket_seconds) * bucket_seconds + bucket_seconds
        entry = buckets.setdefault(bucket,
                                   {"price": None, "volume": 0.0, "trade_count": 0})
        entry["price"] = _clip(implied)
        entry["volume"] += size
        entry["trade_count"] += 1

    history_by_bucket: Dict[int, float] = {}
    if price_history:
        winner_token_history = price_history
        for pt in winner_token_history:
            ts = int(pt["t"])
            bucket = (ts // bucket_seconds) * bucket_seconds + bucket_seconds
            history_by_bucket[bucket] = _clip(pt["p"])

    if not buckets:
        return None
    bucket_keys = list(range(min(buckets), max(buckets) + bucket_seconds, bucket_seconds))
    rows = []
    for bk in bucket_keys:
        row = buckets.get(bk, {"price": None, "volume": 0.0, "trade_count": 0})
        if row["price"] is None and bk in history_by_bucket:
            row["price"] = history_by_bucket[bk]
        rows.append({"bucket": bk, **row})

    known = [r["price"] for r in rows if r["price"] is not None]
    if not known:
        return None
    cur = known[0]
    for r in rows:
        if r["price"] is None:
            r["price"] = cur
        else:
            cur = r["price"]

    prices = [rows[0]["price"]] + [r["price"] for r in rows]
    volumes = [r["volume"] for r in rows]
    times = [rows[0]["bucket"] - bucket_seconds] + [r["bucket"] for r in rows]

    return {
        "winner_index": winner,
        "winner_label": market["outcomes"][winner],
        "prices": prices,
        "volumes": volumes,
        "times": times,
        "horizon": len(volumes),
        "metadata": {
            "source": "polymarket",
            "question": market["question"],
            "slug": market["slug"],
            "condition_id": market["condition_id"],
            "outcomes": market["outcomes"],
            "trade_count": len(trades),
        },
    }


# ---------------------------------------------------------------------------
# Single-market convenience
# ---------------------------------------------------------------------------

def fetch_market_history(slug: str, bucket_minutes: int = 1,
                         use_clob_history: bool = True,
                         timeout: int = 20) -> dict:
    """Pull a single resolved binary market by slug and return its trajectory."""
    market = get_market_by_slug(slug, timeout=timeout)
    return _trajectory_from_market(market, bucket_minutes=bucket_minutes,
                                   use_clob_history=use_clob_history,
                                   timeout=timeout)


def _trajectory_from_market(market: dict, bucket_minutes: int = 1,
                            use_clob_history: bool = True,
                            timeout: int = 20) -> dict:
    """Build the trajectory for a market dict already obtained from Gamma."""
    trades = get_trades(market["condition_id"], timeout=timeout)
    history: List[dict] = []
    if use_clob_history and market["token_ids"]:
        winner = _resolved_winner_index(market) or 0
        try:
            history = get_price_history(market["token_ids"][winner],
                                        fidelity_minutes=bucket_minutes,
                                        timeout=timeout)
        except (HTTPError, URLError):
            history = []
    traj = build_trajectory(market, trades, bucket_minutes=bucket_minutes,
                            price_history=history)
    if traj is None:
        raise ValueError(f"could not build trajectory for slug {market['slug']!r}")
    return traj


# ---------------------------------------------------------------------------
# Panel fetcher with caching
# ---------------------------------------------------------------------------

def list_resolved_binary_markets(limit: int = 200, min_volume: float = 5_000.0,
                                 page_size: int = 100, max_pages: int = 50,
                                 timeout: int = 20,
                                 recent_first: bool = True,
                                 keyword_filter: Optional[List[str]] = None) -> List[dict]:
    """Page through Gamma `/markets?closed=true` and keep binary markets with a
    clear winner and at least `min_volume` total notional volume.

    With `recent_first=True` (default) the Gamma API is sorted by `endDate`
    descending so we get markets the Polymarket trades API still has indexed.
    The /trades endpoint typically only retains data for the most recently
    resolved few months of markets.

    If `keyword_filter` is a list of strings, only markets whose question
    contains at least one keyword (case-insensitive substring match) are kept.
    """
    out: List[dict] = []
    base_params = {"closed": "true", "limit": page_size}
    if recent_first:
        base_params["order"] = "endDate"
        base_params["ascending"] = "false"
    for page in range(max_pages):
        params = dict(base_params)
        params["offset"] = page * page_size
        try:
            batch = _fetch_json(GAMMA_API, "/markets", params, timeout=timeout)
        except HTTPError as exc:
            if exc.code == 400 and page > 0:
                break
            raise
        if not batch:
            break
        for m in batch:
            try:
                mn = _normalize_market(m)
            except (ValueError, TypeError, KeyError):
                continue
            if (mn["closed"] and len(mn["outcomes"]) == 2
                    and len(mn["token_ids"]) == 2
                    and _resolved_winner_index(mn) is not None
                    and mn["volume"] >= min_volume):
                if keyword_filter is not None:
                    q = mn["question"].lower()
                    if not any(kw.lower() in q for kw in keyword_filter):
                        continue
                out.append(mn)
                if len(out) >= limit:
                    return out
        if len(batch) < page_size:
            break
    return out


def fetch_resolved_binary_markets(
    n: int = 100,
    bucket_minutes: int = 5,
    min_volume: float = 5_000.0,
    min_trades: int = 100,
    cache_path: Optional[str] = None,
    use_clob_history: bool = True,
    timeout: int = 20,
    sleep_between: float = 0.05,
    verbose: bool = True,
    keyword_filter: Optional[List[str]] = None,
) -> List[dict]:
    """Build a panel of resolved binary market trajectories.

    Caches the resulting list of trajectories (and metadata) at `cache_path` so
    re-runs are free. Skips markets with fewer than `min_trades` trades or
    horizon < 5.
    """
    if cache_path is not None and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            payload = json.load(f)
        if verbose:
            print(f"loaded {len(payload['trajectories'])} cached trajectories"
                  f" from {cache_path}")
        return payload["trajectories"]

    # When keyword filtering we need to scan many more pages to find enough matches.
    search_limit = int(n * (20 if keyword_filter else 4))
    candidates = list_resolved_binary_markets(
        limit=search_limit, min_volume=min_volume, timeout=timeout,
        keyword_filter=keyword_filter,
    )
    if verbose:
        print(f"found {len(candidates)} candidate resolved markets;"
              f" pulling histories until we have {n}")

    trajectories: List[dict] = []
    for i, market in enumerate(candidates):
        if len(trajectories) >= n:
            break
        try:
            trades = get_trades(market["condition_id"], timeout=timeout)
        except (HTTPError, URLError, TimeoutError):
            continue
        if len(trades) < min_trades:
            continue
        history: List[dict] = []
        if use_clob_history:
            winner = _resolved_winner_index(market) or 0
            try:
                history = get_price_history(market["token_ids"][winner],
                                            fidelity_minutes=bucket_minutes,
                                            timeout=timeout)
            except (HTTPError, URLError, TimeoutError):
                history = []
        traj = build_trajectory(market, trades, bucket_minutes=bucket_minutes,
                                price_history=history)
        if traj is None or traj["horizon"] < 5:
            continue
        trajectories.append(traj)
        if verbose and len(trajectories) % 5 == 0:
            print(f"  [{len(trajectories)}/{n}] kept slug={market['slug']!r} "
                  f"horizon={traj['horizon']} trades={len(trades)}")
        time.sleep(sleep_between)

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "n": len(trajectories),
                "bucket_minutes": bucket_minutes,
                "trajectories": trajectories,
            }, f)
        if verbose:
            print(f"cached {len(trajectories)} trajectories to {cache_path}")

    return trajectories
