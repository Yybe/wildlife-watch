"""Predictive conflict risk: forecast a qualifying-species visit at a camera site.

The retrospective gate in `alert.py` measures persistence that a plain frame count already captures, so it
ties a counting baseline at equal alert volume. This module replaces the question "has a big cat been seen
today" with "will a big cat appear in the next H hours", which a frame count cannot imitate because the
label is in the future.

Everything is deliberately site-agnostic so it can be evaluated on held-out camera sites: features come
only from the site's own history up to the decision time, and the climatology prior is estimated on the
training split.

Usage (self-check):  .venv/Scripts/python.exe DL_PBL/src/sequence/forecast.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

HORIZON_HOURS = 6
WINDOW_HOURS = 24
STEP_HOURS = 1
FEATURE_NAMES = ["visits_window", "high_window", "mid_window", "low_window",
                 "hours_since_high", "night_now", "hour_sin", "hour_cos",
                 "mean_conf_window", "site_qualifying_prior", "visits_today_so_far"]


@dataclass
class ForecastConfig:
    horizon_hours: int = HORIZON_HOURS
    window_hours: int = WINDOW_HOURS
    step_hours: int = STEP_HOURS
    tier_min: int = 3


def visits(df: pd.DataFrame, species_of: dict[int, str], tiers: dict[str, int],
           default_tier: int = 1, conf: np.ndarray | None = None,
           predicted: np.ndarray | None = None) -> pd.DataFrame:
    """Collapse frame rows to one event per visit, carrying tier and confidence."""
    d = df.copy()
    d["datetime"] = pd.to_datetime(d["datetime"])
    lab = predicted if predicted is not None else d["label"]
    out = (d.assign(used=lab, c=conf if conf is not None else 1.0)
             .groupby(["location", "seq_id"], as_index=False)
             .agg(start=("datetime", "min"), used=("used", lambda s: int(s.mode().iloc[0])),
                  c=("c", "max"), frames=("used", "size")))
    out["species"] = out.used.map(lambda l: species_of.get(int(l), "?"))
    out["tier"] = out.species.map(lambda s: tiers.get(s, default_tier))
    return out.sort_values(["location", "start"]).reset_index(drop=True)


def site_day_windows(ev: pd.DataFrame, cfg: ForecastConfig,
                     prior: dict[int, float], global_prior: float) -> pd.DataFrame:
    """One row per (site, day, hour) decision point, with history-only features and a future label.

    A day is only kept if the site has activity that day, because a ranger queue is made of days with
    data; a day with no camera activity cannot be alerted on.
    """
    ev = ev.copy()
    ev["day"] = ev.start.dt.strftime("%Y-%m-%d")
    rows = []
    for (site, day), grp in ev.groupby(["location", "day"]):
        base = pd.Timestamp(day)
        hi = grp[grp.tier >= cfg.tier_min]
        prior_rate = prior.get(int(site), global_prior)
        for h in range(0, 24, cfg.step_hours):
            t = base + pd.Timedelta(hours=h)
            past = grp[grp.start <= t]
            if past.empty:
                continue
            win = past[past.start > t - pd.Timedelta(hours=cfg.window_hours)]
            future = grp[(grp.start > t) & (grp.start <= t + pd.Timedelta(hours=cfg.horizon_hours))]
            last_high = hi[hi.start <= t]
            hours_since = ((t - last_high.start.max()).total_seconds() / 3600.0
                           if len(last_high) else cfg.window_hours * 4)
            rows.append({
                "site": int(site), "day": day, "t": t,
                "visits_window": len(win), "high_window": int((win.tier >= cfg.tier_min).sum()),
                "mid_window": int(((win.tier == 2)).sum()), "low_window": int(((win.tier <= 1)).sum()),
                "hours_since_high": min(hours_since, cfg.window_hours * 4),
                "night_now": int(t.hour >= 18 or t.hour < 6),
                "hour_sin": math.sin(2 * math.pi * t.hour / 24), "hour_cos": math.cos(2 * math.pi * t.hour / 24),
                "mean_conf_window": float(win.c.mean()) if len(win) else 0.0,
                "site_qualifying_prior": prior_rate,
                "visits_today_so_far": len(past),
                "label": int((future.tier >= cfg.tier_min).any()),
                "qualifying_today": int((grp.tier >= cfg.tier_min).any()),
                "frames_today_so_far": int(past.frames.sum()),
            })
    return pd.DataFrame(rows)


def decision_points(evidence: pd.DataFrame, truth: pd.DataFrame, cfg: ForecastConfig,
                    prior: dict[int, float], global_prior: float) -> pd.DataFrame:
    """Calendar and labels from the truth stream, evidence features from the model's own detections.

    Building the grid once instead of merging two independently generated point frames keeps the
    label column unambiguous and lets a day the classifier saw nothing appear with zero evidence
    rather than vanishing from the evaluation.
    """
    pts = site_day_windows(truth, cfg, prior, global_prior)
    feats = site_day_windows(evidence, cfg, prior, global_prior).drop(
        columns=["label", "qualifying_today", "frames_today_so_far"], errors="ignore")
    out = pts.merge(feats, on=["site", "day", "t"], how="left", suffixes=("", "_model"))
    out["label"] = out["label"].astype(int)
    for c in FEATURE_NAMES:
        if c in out.columns:
            out[c] = out[c].fillna(0.0)
    return out


def priors(ev: pd.DataFrame, cfg: ForecastConfig) -> tuple[dict[int, float], float]:
    """Per-site rate of qualifying days, estimated on the training events only (smoothed toward the
    global rate so unseen sites get a sane prior instead of zero)."""
    per_day = ev.groupby(["location", ev.start.dt.strftime("%Y-%m-%d")]).tier.max()
    rate = (per_day >= cfg.tier_min).groupby(level="location").mean()
    g = float((per_day >= cfg.tier_min).mean())
    smoothed = ((rate * 10 + g * 5) / 15).to_dict()
    return {int(k): float(v) for k, v in smoothed.items()}, g


def budget_rank(points: pd.DataFrame, score_col: str, k: int) -> dict:
    """Precision / recall of alerting the top-k decision points by one score."""
    order = points.sort_values([score_col, "t"], ascending=[False, True]).head(k)
    tp = int(order.label.sum())
    total = int(points.label.sum())
    return {"rule": score_col, "budget": k, "true_alerts": tp, "false_alerts": k - tp,
            "precision": round(tp / max(k, 1), 4), "recall": round(tp / max(total, 1), 4),
            "alerted_days": int(order.assign(key=order.site.astype(str) + "@" + order.day).key.nunique())}


def day_level_budget(points: pd.DataFrame, score_cols: list[str], k_days: int,
                     seed: int = 13) -> pd.DataFrame:
    """Rank camera-days by each rule's best hour and take the top k_days: the unit a ranger sees.

    Hour-point budgets make every rule look like chance because a day contributes up to 24 correlated
    points; alerts are per site-day, so that is the denominator the pre-registered bar must use.

    Ties are broken with a seeded draw, not by date. A constant score is possible - a site-level prior is
    the same value for every day at an unseen site - and breaking that tie by calendar order quietly turns
    a degenerate rule into a number that looks measured. `tied_at_cut` reports how many days share the
    score at the budget boundary, so a rule whose ranking is decided by ties cannot be quoted as a result.
    """
    cols = list(dict.fromkeys(score_cols + ["label"]))        # label must survive the groupby
    days = points.groupby(["site", "day"], as_index=False)[cols].max()
    total = int((days.label > 0).sum())
    days = days.assign(_tie=np.random.default_rng(seed).random(len(days)))
    rows = []
    for col in score_cols:
        order = days.sort_values([col, "_tie"], ascending=[False, True]).head(k_days)
        tp = int((order.label > 0).sum())
        cut = float(days[col].iloc[min(k_days, len(days)) - 1])
        rows.append({"rule": col, "budget_days": k_days, "true_days": tp, "false_days": k_days - tp,
                     "precision": round(tp / max(k_days, 1), 4),
                     "recall": round(tp / max(total, 1), 4),
                     "tied_at_cut": int((days[col] == cut).sum())})
    chance = round(k_days * total / max(len(days), 1) / max(k_days, 1), 4)
    rows.append({"rule": "chance", "budget_days": k_days, "true_days": round(k_days * total / max(len(days), 1), 1),
                 "false_days": round(k_days - k_days * total / max(len(days), 1), 1),
                 "precision": chance, "recall": round(k_days * total / max(len(days), 1) / max(total, 1), 4),
                 "tied_at_cut": 0})
    return pd.DataFrame(rows)


def _self_check() -> None:
    rng = np.random.default_rng(13)
    cfg = ForecastConfig()
    # Synthetic site with a strict nocturnal big-cat pattern: visits at 20:00-23:00 every day.
    rows = []
    for d in range(40):
        day = pd.Timestamp("2015-01-01") + pd.Timedelta(days=d)
        for h in (7, 20, 21):
            cats = d % 3 != 2                       # every third day is deer-only: nothing to forecast
            tier = 4 if (h >= 20 and cats) else 1
            rows.append({"location": 1, "seq_id": f"{d}-{h}", "label": 1 if tier == 4 else 2,
                         "datetime": day + pd.Timedelta(hours=h, minutes=5 * h),
                         "filename": f"{d}-{h}.jpg"})
    df = pd.DataFrame(rows)
    species_of = {1: "panthera onca", 2: "mazama temama"}
    tiers = {"panthera onca": 4, "mazama temama": 1}
    ev = visits(df, species_of, tiers)
    assert len(ev) == 120 and set(ev.tier) == {1, 4}, ev.tier.value_counts().to_dict()
    prior, g = priors(ev, cfg)
    pts = site_day_windows(ev, cfg, prior, g)
    assert pts.label.isin([0, 1]).all() and len(pts) > 100, len(pts)
    # The label must be in the future: a 07:00 point (deer seen, nothing ahead for 6 h) is 0, while a
    # 19:00 point (big cat due at 20:05) is 1.
    early = pts[(pts.day == pts.day.iloc[3]) & (pts.t.dt.hour == 19)]
    assert len(early) == 1 and int(early.label.iloc[0]) == 1, early
    dawn = pts[(pts.t.dt.hour.isin([8, 9, 10]))]  # after the deer visit, before the cats
    assert len(dawn) and (dawn.label == 0).all(), dawn
    # The design claim, made testable: retrospective evidence ("we have already seen N big cats") is a
    # worse forecast than a forward-looking prior. In this synthetic site the big cats arrive at 20:05 and
    # 21:05, so the hours with the largest high_window are 22:00-23:00 - after the animals left, with no
    # qualifying animal in the next 6 hours. Counting the past scores zero; the night prior scores above it.
    r_high = budget_rank(pts, "high_window", 30)
    r_night = budget_rank(pts, "night_now", 30)
    assert r_night["true_alerts"] > r_high["true_alerts"], (r_night, r_high)
    assert r_night["precision"] > r_high["precision"], (r_night, r_high)
    assert r_night["recall"] > r_high["recall"], (r_night, r_high)
    dl = day_level_budget(pts, ["night_now", "high_window", "label"], 10).set_index("rule")
    # Structural checks only. Which evidence source wins is a property of the corpus, not something to
    # bake into a test: on this clustered-night synthetic case the retrospective count is a good same-day
    # predictor, so asserting the forecast must win here would be fitting the test to a hoped-for result.
    assert dl.index.tolist() == ["night_now", "high_window", "label", "chance"], dl
    assert ((dl.true_days + dl.false_days) == 10).all(), dl                    # budget is exactly spent
    assert int(dl.loc["label", "true_days"]) == 10, dl                          # oracle is the ceiling
    assert dl.loc["label", "true_days"] >= max(dl.loc[["night_now", "high_window"], "true_days"]), dl
    assert abs(float(dl.loc["chance", "true_days"]) - 10 * 27 / 40) < 0.6, dl   # chance matches its formula
    single = day_level_budget(pts, ["night_now"], 3)
    assert len(single) == 2 and int(single.iloc[0].true_days) <= 3, single
    assert (single.true_days + single.false_days == 3).all(), single

    # Tie-break attack. Every day scores the same, so the only signal in the top-k is *how ties are broken*.
    # Positives live only in the last 11 days: a calendar tie-break would alert the 10 earliest days and
    # report 0 true days - a fabricated zero that looks like a measured result. The seeded draw must not.
    flat = pd.DataFrame({"site": 1, "day": [f"d{i:02d}" for i in range(40)],
                         "label": [1 if i >= 29 else 0 for i in range(40)], "const": 0.0})
    tie = day_level_budget(flat, ["const"], 10).iloc[0]
    assert int(tie.tied_at_cut) == 40, tie                     # the tie is reported, not hidden
    assert int(tie.true_days) > 0, tie                         # fails if ties break by date again
    assert int(tie.true_days) + int(tie.false_days) == 10, tie
    again = day_level_budget(flat, ["const"], 10).iloc[0]
    assert (again.true_days, again.recall) == (tie.true_days, tie.recall), (again, tie)  # seed is fixed

    print("forecast.py self-check passed:", {"events": len(ev), "decision_points": len(pts),
                                             "positive_rate": round(float(pts.label.mean()), 4),
                                             "prior_site1": round(prior[1], 3), "global": round(g, 3)})


if __name__ == "__main__":
    _self_check()
