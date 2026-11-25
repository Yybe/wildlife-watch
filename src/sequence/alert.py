"""Conflict-alert rule: turn species detections at a camera site into a risk decision.

No camera-trap corpus carries human-wildlife-conflict ground truth, so the risk score is a declared
function that must be justified, not a learned predictor. What is measurable is its behaviour:
alerts per camera-day, alerts fired when no qualifying species was present, and alerts missed.

The point of the module is the comparison it enables: alerting per detection (the naive product)
versus alerting on the event sequence (this module's gate). Same evidence, different rule, counted
difference.

Usage (self-check):
    .venv/Scripts/python.exe DL_PBL/src/sequence/alert.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Danger tier 1 = nuisance or prey species, 2 = livestock depredation or injury possible,
# 3 = serious threat to people and livestock, 4 = lethal to people and habituated to
# settlements. Assigned per species and cited in the report; `empty` is tier 0.
DANGER_TIER = {
    "empty": 0,
    "dasyprocta punctata": 1, "aepyceros melampus": 1, "cephalophus nigrifrons": 1,
    "meleagris ocellata": 1, "crax rubra": 1, "mazama temama": 1, "tayassu pecari": 2,
    "urocyon cinereoargenteus": 2, "tapirus bairdii": 2,
    "leopardus pardalis": 3, "puma concolor": 3, "panthera onca": 4,
}
DEFAULT_TIER = 1  # an unlisted species is treated as low risk, never silently ignored


@dataclass
class AlertConfig:
    tier_min: int = 3                 # only big cats and equivalents can raise an alert
    half_life_hours: float = 6.0      # recency weighting of the last qualifying visit
    night_weight: float = 1.5         # 18:00-05:59 raises the score, crepuscular predators
    min_visits: int = 2               # a lone sighting is not an incident
    score_threshold: float = 0.5


@dataclass
class SiteDay:
    site: int
    day: str
    visits: list = field(default_factory=list)
    score: float = 0.0
    alert: bool = False
    qualifying: int = 0


def visits_from_manifest(df: pd.DataFrame, confidence: np.ndarray | None = None,
                       predicted: np.ndarray | None = None) -> pd.DataFrame:
    """Collapse per-frame rows to one row per (site, seq_id).

    `predicted` (model output) is what production feeds; without it the ground-truth `label` is used,
    which is honest for measuring the alert rule alone but must not be reported as model accuracy.
    """
    d = df.copy()
    d["datetime"] = pd.to_datetime(d["datetime"])
    d["label_used"] = predicted if predicted is not None else d["label"]
    d["conf"] = confidence if confidence is not None else 1.0
    out = (d.groupby(["location", "seq_id"], as_index=False)
             .agg(start=("datetime", "min"), end=("datetime", "max"), frames=("label_used", "size"),
                  label_used=("label_used", lambda s: int(s.mode().iloc[0])),
                  conf=("conf", "max")))
    return out


def score_site_day(visits: list[tuple[pd.Timestamp, int, float]], at: pd.Timestamp,
                   cfg: AlertConfig) -> float:
    """Risk score in [0, 1) for a camera-day: tier-weighted, recency-decayed, night-biased evidence.

    `visits` items are (start time, tier, confidence). Exponential recency means a sighting one
    half-life ago counts half as much as one now. Evidence is combined with a noisy-OR
    (`1 - exp(-sum / min_visits)`) rather than a clamped sum, because a clamped sum saturates at 1.0 and
    then a day with one big-cat sighting is indistinguishable from a day with ten.

    The score is deliberately continuous even below `min_visits`: a threshold rule whose sub-threshold
    days all score exactly 0 cannot be compared at an equal alert budget, because the ranking below the
    cut is an arbitrary tie block. `min_visits` decides whether a day alerts, not what it is worth.
    """
    q = [v for v in visits if v[1] >= cfg.tier_min]
    if not q:
        return 0.0
    lam = math.log(2) / cfg.half_life_hours
    total = 0.0
    for t_start, tier, conf in q:
        age_h = max((at - t_start).total_seconds() / 3600.0, 0.0)
        night = cfg.night_weight if (t_start.hour >= 18 or t_start.hour < 6) else 1.0
        total += (tier / 4.0) * conf * night * math.exp(-lam * age_h)
    return float(1.0 - math.exp(-total / max(1, cfg.min_visits)))


def build_site_days(visits: pd.DataFrame, species_of: dict[int, str], cfg: AlertConfig,
                    step_hours: float = 1.0) -> list[SiteDay]:
    """Score every camera-day on a fixed hourly grid; the last grid point above threshold is the alert."""
    visits = visits.copy()
    visits["day"] = pd.to_datetime(visits["start"]).dt.strftime("%Y-%m-%d")
    results: list[SiteDay] = []
    for (site, day), grp in visits.groupby(["location", "day"]):
        rows = [(pd.Timestamp(r.start), DANGER_TIER.get(species_of.get(int(r.label_used), "?"),
                                                         DEFAULT_TIER), float(r.conf))
                for r in grp.itertuples()]
        qualifying = sum(1 for _, tier, _ in rows if tier >= cfg.tier_min)
        day0 = pd.Timestamp(day)
        grid = [day0 + pd.Timedelta(hours=h) for h in
                range(0, 24, max(1, int(step_hours)))]
        scores = [score_site_day([r for r in rows if r[0] <= g], g, cfg) for g in grid]
        best = max(scores or [0.0])
        results.append(SiteDay(site=int(site), day=day, visits=len(rows), score=best,
                               alert=bool(best >= cfg.score_threshold and qualifying >= cfg.min_visits),
                               qualifying=qualifying))
    return results


def day_universe(*tables: pd.DataFrame) -> set[tuple[int, str]]:
    out = set()
    for v in tables:
        d = v[["location", "day"]].drop_duplicates()
        out |= set(map(tuple, d.itertuples(index=False, name=None)))
    return out


def qualified_days(truth: pd.DataFrame, species_of: dict[int, str], tier_min: int) -> set:
    """Camera-days where a qualifying species is REALLY present - the ground truth of this metric."""
    t = truth.copy()
    t["tier"] = t.label_used.map(lambda l: DANGER_TIER.get(species_of.get(int(l), "?"), DEFAULT_TIER))
    q = t[t.tier >= tier_min]
    return set(map(tuple, q[["location", "day"]].itertuples(index=False, name=None)))


def _arm_counts(alert_days: set, universe: set, qualified: set) -> dict:
    tp = len(alert_days & qualified)
    fp = len(alert_days - qualified)
    fn = len(qualified - alert_days)
    return {
        "alerts": len(alert_days),
        "alerts_per_camera_day": round(len(alert_days) / max(len(universe), 1), 4),
        "true_alerts": tp,
        "false_alerts": fp,
        "missed_alerts": fn,
        "precision": round(tp / max(len(alert_days), 1), 4),
        "recall_of_qualifying_days": round(tp / max(len(qualified), 1), 4),
    }


def evaluate_per_frame_vs_gated(evidence: pd.DataFrame, truth: pd.DataFrame,
                                species_of: dict[int, str], cfg: AlertConfig) -> dict:
    """Score both alerting rules against ground-truth qualifying days.

    `evidence` carries the model's predicted labels and confidences; `truth` carries the real labels of
    the same frames. Judging an alert stream against the predictions that produced it is circular - the
    gate demands two qualifying visits, so a "false alert" defined on predictions could never occur. The
    denominator here is the truth table instead.
    """
    ev, tr = evidence.copy(), truth.copy()
    for t in (ev, tr):
        t["day"] = pd.to_datetime(t["start"]).dt.strftime("%Y-%m-%d")
        t["tier"] = t.label_used.map(lambda l: DANGER_TIER.get(species_of.get(int(l), "?"), DEFAULT_TIER))

    universe = day_universe(ev, tr)
    qualified = qualified_days(tr, species_of, cfg.tier_min)

    gated_days = {(s.site, s.day) for s in build_site_days(ev, species_of, cfg) if s.alert}
    per_frame_days = set(map(tuple, ev[ev.tier >= cfg.tier_min][["location", "day"]]
                             .itertuples(index=False, name=None)))

    gated, naive = _arm_counts(gated_days, universe, qualified), _arm_counts(per_frame_days, universe, qualified)
    empty_only_days = {(site, day) for (site, day) in universe
                       if not (tr[(tr.location == site) & (tr.day == day)].tier > 0).any()}
    return {
        "camera_days": len(universe),
        "qualifying_camera_days": len(qualified),
        "days_with_no_animal": len(empty_only_days),
        "gated": gated,
        "per_frame": naive,
        "alerts_reduced_by": round(1 - gated["alerts"] / max(naive["alerts"], 1), 4),
    }


def day_rankings(evidence: pd.DataFrame, species_of: dict[int, str], cfg: AlertConfig) -> pd.DataFrame:
    """Rank every camera-day two ways: by the gate's score, and by raw qualifying-frame volume.

    The volume ranking is the budget-matched baseline. Comparing a gate that fires on 74 days against a
    per-frame rule that fires on 173 is not a fair fight - patrol capacity is finite either way, so both
    arms must be judged at the same number of alerts.
    """
    ev = evidence.copy()
    ev["day"] = pd.to_datetime(ev["start"]).dt.strftime("%Y-%m-%d")
    ev["tier"] = ev.label_used.map(lambda l: DANGER_TIER.get(species_of.get(int(l), "?"), DEFAULT_TIER))
    scores = {(s.site, s.day): s.score for s in build_site_days(ev, species_of, cfg, step_hours=1.0)}
    volume = (ev.assign(q=np.where(ev.tier >= cfg.tier_min, ev.frames, 0))
                .groupby(["location", "day"]).q.sum())
    days = sorted(set(scores) | set(map(tuple, volume.index)))
    return pd.DataFrame([{"site": d[0], "day": d[1], "score": scores.get(d, 0.0),
                          "qualifying_frames": float(volume.get(d, 0.0))} for d in days])


def budget_curve(evidence: pd.DataFrame, truth: pd.DataFrame, species_of: dict[int, str],
                 cfg: AlertConfig, budgets: tuple[int, ...] = (25, 50, 75, 100, 125, 150)) -> pd.DataFrame:
    """Precision and recall of each ranking rule at equal alert volume."""
    ranked = day_rankings(evidence, species_of, cfg)
    tr = truth.copy()
    tr["day"] = pd.to_datetime(tr["start"]).dt.strftime("%Y-%m-%d")
    qualified = qualified_days(tr, species_of, cfg.tier_min)
    rows = []
    for key, label in (("score", "sequence_gate"), ("qualifying_frames", "volume_ranked")):
        order = ranked.sort_values([key, "day"], ascending=[False, True])
        for k in budgets:
            picked = set(map(tuple, order.head(k)[["site", "day"]].itertuples(index=False, name=None)))
            tp = len(picked & qualified)
            rows.append({"rule": label, "budget": k, "true_alerts": tp,
                         "false_alerts": k - tp, "precision": round(tp / max(k, 1), 4),
                         "recall": round(tp / max(len(qualified), 1), 4)})
    return pd.DataFrame(rows)


def _self_check() -> None:
    cfg = AlertConfig()
    t0 = pd.Timestamp("2014-06-01 20:00")
    # A single sighting is soft evidence and must score above zero (so a budget ranking can use it) but
    # below two sightings, and must not alert while min_visits=2.
    lone = score_site_day([(t0, 4, 1.0)], t0 + pd.Timedelta(hours=1), cfg)
    assert 0.0 < lone < 1.0, lone
    solo = AlertConfig(min_visits=1)
    one = score_site_day([(t0, 4, 1.0)], t0 + pd.Timedelta(hours=1), solo)
    two = score_site_day([(t0, 4, 1.0), (t0 + pd.Timedelta(minutes=20), 4, 1.0)],
                         t0 + pd.Timedelta(hours=1), solo)
    assert two > one > 0, (one, two)
    # Recency decays: the same evidence scored later must be lower.
    late = score_site_day([(t0, 4, 1.0)], t0 + pd.Timedelta(hours=12), solo)
    assert late < one, (late, one)
    # The score must stay below 1 so persistence keeps being visible: ten visits > two visits > one.
    many = score_site_day([(t0 + pd.Timedelta(minutes=i), 4, 1.0) for i in range(10)],
                          t0 + pd.Timedelta(hours=1), solo)
    assert one < two < many < 1.0, (one, two, many)
    # A low-tier species can never reach the threshold, however often it appears.
    deer = score_site_day([(t0 + pd.Timedelta(minutes=i), 1, 1.0) for i in range(20)], t0, cfg)
    assert deer == 0.0, deer
    # Tier ordering is respected.
    assert DANGER_TIER["panthera onca"] > DANGER_TIER["puma concolor"] > DANGER_TIER["mazama temama"]

    # The metric must be judged against ground truth, not against the predictions that produced the
    # alert. Four camera-days: site1 the model hallucinates two jaguar visits (false alert), site2 it
    # misses two real ones (missed), site3 both agree, site4 is a real lone jaguar visit that
    # min_visits=2 suppresses - the cost a ranger would never accept.
    species_of = {0: "empty", 1: "panthera onca", 2: "mazama temama"}
    JAG, DEER = 1, 2

    def visit(site, seq, hour, label):
        return {"location": site, "seq_id": seq, "start": pd.Timestamp(f"2014-06-01 {hour:02d}:00"),
                "end": pd.Timestamp(f"2014-06-01 {hour:02d}:10"), "frames": 5,
                "label_used": label, "conf": 1.0, "day": "2014-06-01"}

    ev = pd.DataFrame([visit(1, "a", 20, JAG), visit(1, "b", 20, JAG),
                       visit(2, "c", 21, DEER), visit(2, "d", 21, DEER),
                       visit(3, "e", 22, JAG), visit(3, "f", 22, JAG),
                       visit(4, "g", 23, JAG)])
    tr = pd.DataFrame([visit(1, "a", 20, DEER), visit(1, "b", 20, DEER),
                       visit(2, "c", 21, JAG), visit(2, "d", 21, JAG),
                       visit(3, "e", 22, JAG), visit(3, "f", 22, JAG),
                       visit(4, "g", 23, JAG)])
    res = evaluate_per_frame_vs_gated(ev, tr, species_of, cfg)
    assert res["camera_days"] == 4, res
    assert res["qualifying_camera_days"] == 3, res
    assert (res["gated"]["alerts"], res["gated"]["true_alerts"], res["gated"]["false_alerts"],
            res["gated"]["missed_alerts"]) == (2, 1, 1, 2), res["gated"]
    assert (res["per_frame"]["alerts"], res["per_frame"]["true_alerts"],
            res["per_frame"]["missed_alerts"]) == (3, 2, 1), res["per_frame"]

    # Budget-matched ranking must be internally consistent and able to reach every qualifying day.
    curve = budget_curve(ev, tr, species_of, cfg, budgets=(1, 2, 3, 4))
    assert len(curve) == 8 and set(curve.rule) == {"sequence_gate", "volume_ranked"}, curve
    assert ((curve.true_alerts + curve.false_alerts) == curve.budget).all(), curve
    assert curve[curve.budget == 3].true_alerts.eq(2).all(), curve
    assert curve[curve.budget == 4].recall.eq(1.0).all(), curve

    print("alert.py self-check passed:",
          {"lone_visit": round(lone, 4), "two_visits": round(two, 4),
           "twelve_hours_later": round(late, 4), "suppressed_lone_day": True})


if __name__ == "__main__":
    _self_check()
