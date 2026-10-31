"""Train and apply a regression model for Lightroom adjustments."""
from __future__ import annotations

import math
from pathlib import Path

from .catalog import TARGETS, LIMITS

MODEL_VERSION = "rawstyle-raw-sensor-v2"
MIN_MATCHED = 20


def _clean_targets(records):
    """Use actual observed adjustments; do not impute Lightroom edits with zero."""
    import numpy as np
    targets = []
    def has_number(value):
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False
    # Require each target to be observed on >= 90% of the matched images.
    for target in TARGETS:
        available = [row.get(target) for row in records]
        present = sum(has_number(v) for v in available)
        if present >= len(records) * .9:
            targets.append(target)
    if "Temperature" not in targets or "Tint" not in targets:
        raise ValueError("Matched Lightroom examples must have numeric Temperature and Tint settings.")
    y = np.array([
        [row.get(key, np.nan) for key in targets]
        for row in records
    ], dtype="float64")
    # For rare absent optional values, impute a numeric per-target median only for
    # TRAINING completeness; never use this as predicted output or as a WB rule.
    for j in range(len(targets)):
        column = y[:, j]
        valid = np.isfinite(column)
        median = float(np.median(column[valid]))
        column[~valid] = median
    return targets, y


def train(features, labels):
    import numpy as np
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error

    if len(features) < MIN_MATCHED:
        raise ValueError(f"At least {MIN_MATCHED} matched, decoded RAW/edit pairs are required to train.")
    x = np.asarray(features, dtype="float32")
    if x.ndim != 2 or not np.all(np.isfinite(x)):
        raise ValueError("RAW feature vectors must be finite and equal length.")
    keys, y = _clean_targets(labels)
    means = y.mean(axis=0)
    scales = np.maximum(y.std(axis=0), .0001)
    y_scaled = (y - means) / scales
    config = dict(n_estimators=140, min_samples_leaf=2, max_features=.85,
                  n_jobs=1, random_state=162)
    scores = {}
    if len(x) >= 40:
        x_train, x_hold, y_train, y_hold = train_test_split(
            x, y_scaled, test_size=.2, random_state=41)
        valid_model = ExtraTreesRegressor(**config)
        valid_model.fit(x_train, y_train)
        predicted = valid_model.predict(x_hold) * scales + means
        actual = y_hold * scales + means
        err = mean_absolute_error(actual, predicted, multioutput="raw_values")
        scores = {name: round(float(e), 1) for name, e in zip(keys, err)}
    final = ExtraTreesRegressor(**config)
    final.fit(x, y_scaled)
    return {
        "format": MODEL_VERSION,
        "model": final,
        "targets": keys,
        "means": means,
        "scales": scales,
        "feature_count": x.shape[1],
        "training_count": len(x),
        "validation_mae": scores,
    }


def predict(model_bundle, features) -> list[dict]:
    import numpy as np
    if model_bundle.get("format") != MODEL_VERSION:
        raise ValueError("Saved model is from an incompatible version.")
    features = np.asarray(features, dtype="float32")
    if features.ndim != 2 or features.shape[1] != model_bundle["feature_count"]:
        raise ValueError("RAW features do not match the trained model.")
    output = model_bundle["model"].predict(features)
    output = output * model_bundle["scales"] + model_bundle["means"]
    records = []
    for row in output:
        settings = {}
        for key, value in zip(model_bundle["targets"], row):
            lo, hi = LIMITS.get(key, (-100, 100))
            val = max(lo, min(hi, float(value)))
            settings[key] = round(val) if key == "Temperature" else round(val, 2)
        # Lightroom must be in Custom mode to actually use the predicted numbers.
        # This IS NOT a prediction of the original camera's WB mode.
        if "Temperature" in settings and "Tint" in settings:
            settings["WhiteBalance"] = "Custom"
        records.append(settings)
    return records


def save_model(bundle: dict, where: str | Path):
    import joblib
    path = Path(where)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    joblib.dump(bundle, temp)
    temp.replace(path)


def open_model(where: str | Path):
    import joblib
    path = Path(where)
    if not path.exists():
        return None
    return joblib.load(path)
