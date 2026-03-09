#!/usr/bin/env python3
"""Offline preference calibration from clip feedback labels."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
except Exception as exc:  # noqa: BLE001
    raise SystemExit(
        "scikit-learn is required for calibration. Install it with: pip install scikit-learn"
    ) from exc


FEEDBACK_PATH = Path.home() / ".fastcap" / "clip_feedback.jsonl"
PROFILE_PATH = Path.home() / ".fastcap" / "preference_profile.json"
MIN_ROWS = 50
MIN_CONFIDENCE_THRESHOLD = 0.55
CLIP_TYPES = ("Teaching", "Conviction", "Declaration", "Encouragement")


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return float(default)


def _load_feedback_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"Feedback dataset not found: {path}")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _feature_names() -> list[str]:
    names = [
        "editorial_editor",
        "editorial_hook",
        "editorial_cadence",
        "editorial_standalone",
        "editorial_emotion",
        "feature_overall_candidate",
        "feature_cadence",
        "feature_energy",
        "feature_contrast",
        "hook_score_llm",
        "duration_sec",
        "assessment_confidence_score",
    ]
    names.extend([f"clip_type_{name.lower()}" for name in CLIP_TYPES])
    return names


def _extract_row_features(row: dict) -> list[float]:
    editorial = row.get("editorial_scores", {}) if isinstance(row.get("editorial_scores", {}), dict) else {}
    feature_scores = row.get("feature_scores", {}) if isinstance(row.get("feature_scores", {}), dict) else {}
    feature_signals = row.get("feature_signals", {}) if isinstance(row.get("feature_signals", {}), dict) else {}
    confidence = (
        row.get("assessment_confidence", {})
        if isinstance(row.get("assessment_confidence", {}), dict)
        else {}
    )
    clip_type = str(row.get("clip_type", "")).strip().lower()

    vector = [
        _safe_float(editorial.get("editor", 0.0)),
        _safe_float(editorial.get("hook", 0.0)),
        _safe_float(editorial.get("cadence", 0.0)),
        _safe_float(editorial.get("standalone", 0.0)),
        _safe_float(editorial.get("emotion", 0.0)),
        _safe_float(feature_scores.get("overall_candidate", 0.0)),
        _safe_float(feature_scores.get("cadence", 0.0)),
        _safe_float(feature_scores.get("energy", 0.0)),
        _safe_float(feature_scores.get("contrast", 0.0)),
        _safe_float(feature_signals.get("hook_score_llm", 0.0)),
        _safe_float(row.get("duration_sec", 0.0)),
        _safe_float(confidence.get("score", 0.0)),
    ]
    for known in CLIP_TYPES:
        vector.append(1.0 if clip_type == known.lower() else 0.0)
    return vector


def _build_dataset(rows: list[dict]) -> tuple[list[list[float]], list[int], int, int]:
    x_data: list[list[float]] = []
    y_data: list[int] = []
    up_count = 0
    down_count = 0
    for row in rows:
        label = str(row.get("label", "")).strip().lower()
        if label not in {"up", "down"}:
            continue
        x_data.append(_extract_row_features(row))
        y_value = 1 if label == "up" else 0
        y_data.append(y_value)
        if y_value == 1:
            up_count += 1
        else:
            down_count += 1
    return x_data, y_data, up_count, down_count


def _cross_validated_accuracy(model: LogisticRegression, x_data: list[list[float]], y_data: list[int]) -> float:
    if len(x_data) < 10:
        return 0.0
    min_class = min(sum(y_data), len(y_data) - sum(y_data))
    if min_class < 2:
        return 0.0
    folds = max(2, min(5, min_class))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, x_data, y_data, cv=splitter, scoring="accuracy")
    return float(scores.mean()) if len(scores) else 0.0


def _next_profile_version(path: Path) -> int:
    if not path.is_file():
        return 1
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
        return int(existing.get("version", 0)) + 1
    except Exception:  # noqa: BLE001
        return 1


def main() -> None:
    rows = _load_feedback_rows(FEEDBACK_PATH)
    x_data, y_data, up_count, down_count = _build_dataset(rows)
    sample_count = len(y_data)

    if sample_count < MIN_ROWS:
        raise SystemExit(
            f"Need at least {MIN_ROWS} labeled rows; found {sample_count}. Keep reviewing clips first."
        )
    if up_count == 0 or down_count == 0:
        raise SystemExit("Need both 'up' and 'down' labels to calibrate preferences.")

    model = LogisticRegression(max_iter=2000, solver="lbfgs")
    model.fit(x_data, y_data)
    accuracy_cv = _cross_validated_accuracy(model, x_data, y_data)

    feature_names = _feature_names()
    weights = model.coef_[0]
    feature_weights = {
        feature_names[idx]: float(weights[idx]) for idx in range(min(len(feature_names), len(weights)))
    }

    profile = {
        "version": _next_profile_version(PROFILE_PATH),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": int(sample_count),
        "up_count": int(up_count),
        "down_count": int(down_count),
        "feature_weights": feature_weights,
        "intercept": float(model.intercept_[0]),
        "accuracy_cv": float(accuracy_cv),
        "min_confidence_threshold": float(MIN_CONFIDENCE_THRESHOLD),
    }

    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"Wrote preference profile: {PROFILE_PATH}")
    print(
        f"samples={sample_count} up={up_count} down={down_count} "
        f"accuracy_cv={accuracy_cv:.3f} version={profile['version']}"
    )


if __name__ == "__main__":
    main()
