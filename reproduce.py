"""Reproduce the paper's flood-reference scores, reservoir errors and integration fits.

Install: python -m pip install -r requirements.txt
Run:     python reproduce.py
Test:    python -m unittest discover -s tests

These checks reproduce the retained study data, not field or operational accuracy.
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import csv
import json
import math

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"


def close(actual, expected):
    if not math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-12):
        raise AssertionError((actual, expected))


def detection():
    with (DATA / "detection_labels.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    counts, weights = Counter(), defaultdict(float)
    for row in rows:
        label, prediction = row["user_label"], int(row["detector_class"])
        category = "uncertain" if label == "uncertain" else (
            "TP" if label == "flood" and prediction else
            "FN" if label == "flood" else "FP" if prediction else "TN")
        if category != "uncertain":
            assert category == row["diagnostic_category"]
        weight = float(row["inverse_probability_weight"])
        close(weight, 1 / float(row["inclusion_probability"]))
        counts[category] += 1
        weights[category] += weight

    def scores(totals):
        tp, fp, fn = (totals[key] for key in ("TP", "FP", "FN"))
        return {"precision": tp/(tp+fp), "recall": tp/(tp+fn),
                "f1": 2*tp/(2*tp+fp+fn), "iou": tp/(tp+fp+fn)}

    return {"locations": len(rows), "counts": dict(counts),
            "weighted_counts": dict(weights), "raw": scores(counts),
            "weighted": scores(weights)}


def _level_cases():
    frame = pd.read_csv(DATA / "reservoir_levels.csv")
    frame = frame.loc[frame.dam.isin(("Bhakra", "Pong"))].copy()
    assert frame.level_unit.eq("m").all()
    frame["date"] = pd.to_datetime(frame.date, format="%Y-%m-%d", errors="raise")
    frame["level"] = pd.to_numeric(frame.level_value, errors="coerce")
    gates = {"Bhakra": (412.06, 522.06), "Pong": (323.67, 433.67)}
    for dam, (low, high) in gates.items():
        bad = frame.dam.eq(dam) & ~frame.level.between(low, high)
        frame.loc[bad, "level"] = np.nan
    for _, group in frame.groupby(["dam", "date"]):
        if group.level.nunique(dropna=False) > 1:
            frame.loc[group.index, "level"] = np.nan
    rows, accepted = [], {}
    for dam in gates:
        series = frame.loc[frame.dam.eq(dam)].drop_duplicates("date").set_index("date").level
        accepted[dam] = int(series.notna().sum())
        for issue in series.index.sort_values():
            known = issue - pd.DateOffset(days=1)
            history = [series.get(known-pd.DateOffset(days=i), np.nan) for i in range(8)]
            if not np.isfinite(history).all():
                continue
            for horizon in range(1, 6):
                target = issue + pd.DateOffset(days=horizon)
                observed = series.get(target, np.nan)
                if not np.isfinite(observed):
                    continue
                row = {"dam": dam, "issue": issue.date().isoformat(),
                       "target": target.date().isoformat(), "horizon": horizon,
                       "lead": horizon+1, "known": history[0], "observed": observed}
                for days in (1, 3, 7):
                    row[f"trend{days}"] = (history[0]-history[days])/days
                rows.append(row)
    return pd.DataFrame(rows), accepted


def _predict(frame, days, alpha):
    if not days:
        return frame.known.to_numpy()
    return (frame.known + alpha * frame[f"trend{days}"] * frame.lead).to_numpy()


def reservoir():
    cases, accepted = _level_cases()
    training = cases[cases.target.le("2018-12-31")]
    selection = cases[cases.issue.gt("2018-12-31") & cases.target.le("2020-12-31")]
    candidates, chosen = [], {}
    for dam in ("Bhakra", "Pong"):
        for horizon in range(1, 6):
            train = training[training.dam.eq(dam) & training.horizon.eq(horizon)]
            select = selection[selection.dam.eq(dam) & selection.horizon.eq(horizon)]
            choices = []
            for days in (0, 1, 3, 7):
                alpha = 0.0
                if days:
                    x = (train[f"trend{days}"] * train.lead).to_numpy()
                    y = (train.observed-train.known).to_numpy()
                    energy = float(x @ x)
                    alpha = float(np.clip((x @ y)/energy, 0, 1)) if energy else 0.0
                mae = float(np.abs(_predict(select, days, alpha)-select.observed.to_numpy()).mean())
                row = {"dam": dam, "horizon_days": horizon, "lookback_days": days,
                       "alpha": alpha, "training_cases": len(train),
                       "calibration_cases": len(select), "calibration_mae_m": mae}
                candidates.append(row)
                choices.append(row)
            chosen[dam, horizon] = min(choices, key=lambda row: (row["calibration_mae_m"], row["lookback_days"]))
    expected_candidates = pd.read_csv(DATA / "reservoir_candidates.csv").set_index(
        ["dam", "horizon_days", "lookback_days"])
    for row in candidates:
        expected = expected_candidates.loc[row["dam"], row["horizon_days"], row["lookback_days"]]
        for column in ("alpha", "training_cases", "calibration_cases", "calibration_mae_m"):
            close(row[column], expected[column])
    later = cases[cases.issue.gt("2020-12-31") & cases.issue.str[:4].eq(cases.target.str[:4])].copy()
    later["period"] = np.where(later.target.le("2024-12-31"), "primary_2021_2024", "supplement_2025")
    expected_metrics = pd.read_csv(DATA / "reservoir_metrics.csv").set_index(
        ["period", "dam", "horizon_days"])
    metrics = []
    for (period, dam, horizon), group in later.groupby(["period", "dam", "horizon"]):
        choice = chosen[dam, horizon]
        errors = _predict(group, choice["lookback_days"], choice["alpha"])-group.observed.to_numpy()
        persistence = group.known.to_numpy()-group.observed.to_numpy()
        undamped = _predict(group, choice["lookback_days"], 1.0)-group.observed.to_numpy()
        row = {"period": period, "dam": dam, "horizon_days": int(horizon), "n": len(group),
               "mae_m": float(np.abs(errors).mean()), "persistence_mae_m": float(np.abs(persistence).mean()),
               "paired_mae_difference_m": float((np.abs(errors)-np.abs(persistence)).mean()),
               "undamped_mae_m": float(np.abs(undamped).mean()),
               "rmse_m": float(np.sqrt(np.square(errors).mean())), "bias_m": float(errors.mean())}
        expected = expected_metrics.loc[period, dam, horizon]
        for column in ("n", "mae_m", "persistence_mae_m", "paired_mae_difference_m"):
            close(row[column], expected[column])
        metrics.append(row)
    assert len(candidates) == 40 and len(metrics) == len(expected_metrics) == 15
    return {"accepted_records": accepted, "candidate_fits_checked": 40,
            "selected": list(chosen.values()), "metrics": metrics}


def integration():
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    frame = pd.read_csv(DATA / "integration_audit.csv")
    original = pd.read_csv(DATA / "integration_original.csv")
    expected = pd.read_csv(DATA / "integration_predictions.csv")
    shared = ["location_x_km", "location_y_km"]
    satellite = ["past_last_known_class", "past_state_age_days", "past_neighbour_flood_fraction", "past_neighbour_known_fraction"]
    weather = [f"forecast_{basin}_{h}h_mm" for basin in
               ("bhakra", "pong", "sutlej_local", "beas_local", "harike_local") for h in (24, 72)]
    reservoir_fields = [f"reported_{dam}_{field}" for dam in ("bhakra", "pong")
                        for field in ("level_ft", "outflow_cusecs", "measurement_age_days")]
    features = {"satellite_signals": shared+satellite,
                "river_watch_informed_signals": shared+weather+reservoir_fields,
                "combined_signals": shared+satellite+weather+reservoir_fields}
    test = frame[frame.round2_common_eligible & frame.split.eq("test")]
    truth_by_id = test.set_index("sample_id").round2_outcome.astype(int)
    scenarios = {
        "original_training_labels": original[original.common_comparison_eligible & original.split.eq("train")],
        "round2_user_training_labels": frame[frame.round2_common_eligible & frame.split.eq("train")],
        "round2_excluding_provisional_concerns": frame[
            frame.round2_common_eligible & frame.split.eq("train") & ~frame.sample_id.eq("H23A_L016")],
    }
    checked, maximum, results = 0, 0.0, []
    for scenario, training in scenarios.items():
        assert training.target_utc.max() < test.issue_utc.min()
        assert set(training.sample_id).isdisjoint(test.sample_id)
        y = training.outcome if scenario == "original_training_labels" else training.round2_outcome
        for arm, columns in features.items():
            model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                LogisticRegression(C=.1, class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=0))
            model.fit(training[columns], y.astype(int))
            scores = model.predict_proba(test[columns])[:, 1]
            saved = expected[expected.scenario.eq(scenario) & expected.arm.eq(arm)].set_index("sample_id")
            counts = Counter()
            for sample_id, score in zip(test.sample_id, scores):
                reference = saved.loc[sample_id]
                close(score, reference.score)
                predicted, truth = int(score >= .5), int(truth_by_id.loc[sample_id])
                assert truth == int(reference.reference_outcome)
                assert predicted == int(reference.prediction)
                counts["TP" if truth and predicted else "FN" if truth else "FP" if predicted else "TN"] += 1
                maximum = max(maximum, abs(float(score)-float(reference.score)))
                checked += 1
            results.append({"scenario": scenario, "arm": arm, "training_cases": len(training),
                            "test_cases": len(test), "counts": dict(counts)})
    assert checked == len(expected) == 180
    return {"reconstructed_scores": checked, "maximum_score_difference": maximum, "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", choices=("all", "detection", "reservoir", "integration"), default="all")
    args = parser.parse_args()
    experiments = {"detection": detection, "reservoir": reservoir, "integration": integration}
    selected = experiments if args.experiment == "all" else {args.experiment: experiments[args.experiment]}
    print(json.dumps({name: function() for name, function in selected.items()}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
