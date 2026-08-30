"""Analyze descriptive education-testing disparities and candidate segments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SCORE_COLUMNS = ["math_score", "language_score", "science_score", "social_studies_score"]
REQUIRED_COLUMNS = {
    "candidate_id",
    "region",
    "gender",
    "age",
    "distance_to_center_miles",
    "prep_attempts",
    "days_between_tests",
    *SCORE_COLUMNS,
}


def prepare_candidates(data: pd.DataFrame, pass_score: int = 145) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing:
        raise ValueError(f"candidate data is missing required columns: {missing}")

    clean = data.copy()
    clean.loc[~clean["age"].between(16, 90), "age"] = np.nan
    clean["average_score"] = clean[SCORE_COLUMNS].mean(axis=1)
    clean["subjects_passed"] = clean[SCORE_COLUMNS].ge(pass_score).sum(axis=1)
    clean["credential_ready"] = clean["subjects_passed"].eq(len(SCORE_COLUMNS))
    clean["log_prep_attempts"] = np.log1p(clean["prep_attempts"].clip(lower=0))
    return clean


def disparity_summary(clean: pd.DataFrame, group: str) -> pd.DataFrame:
    if group not in clean.columns:
        raise ValueError(f"unknown grouping column: {group}")
    summary = (
        clean.groupby(group, dropna=False)
        .agg(
            candidates=("candidate_id", "nunique"),
            credential_ready_rate=("credential_ready", "mean"),
            average_score=("average_score", "mean"),
            average_distance_miles=("distance_to_center_miles", "mean"),
            average_prep_attempts=("prep_attempts", "mean"),
        )
        .reset_index()
    )
    summary["credential_ready_rate"] = summary["credential_ready_rate"].mul(100)
    return summary.round(2).sort_values("credential_ready_rate")


def subject_summary(clean: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject in SCORE_COLUMNS:
        rows.append(
            {
                "subject": subject.removesuffix("_score").replace("_", " "),
                "average_score": clean[subject].mean(),
                "pass_rate": clean[subject].ge(145).mean() * 100,
            }
        )
    return pd.DataFrame(rows).round(2).sort_values("pass_rate")


def segment_candidates(clean: pd.DataFrame, clusters: int = 4, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = [
        "age",
        "distance_to_center_miles",
        "log_prep_attempts",
        "days_between_tests",
        "average_score",
    ]
    model_frame = clean.dropna(subset=features).copy()
    scaled = StandardScaler().fit_transform(model_frame[features])
    model_frame["segment"] = KMeans(n_clusters=clusters, random_state=seed, n_init=20).fit_predict(scaled)
    profile = (
        model_frame.groupby("segment")
        .agg(
            candidates=("candidate_id", "count"),
            credential_ready_rate=("credential_ready", "mean"),
            average_score=("average_score", "mean"),
            average_distance_miles=("distance_to_center_miles", "mean"),
            average_prep_attempts=("prep_attempts", "mean"),
            average_days_between_tests=("days_between_tests", "mean"),
        )
        .reset_index()
    )
    profile["credential_ready_rate"] = profile["credential_ready_rate"].mul(100)
    return model_frame, profile.round(2).sort_values("credential_ready_rate")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "synthetic_candidates.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()

    clean = prepare_candidates(pd.read_csv(args.input))
    segmented, profiles = segment_candidates(clean)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    disparity_summary(clean, "region").to_csv(args.output_dir / "regional_disparities.csv", index=False)
    disparity_summary(clean, "gender").to_csv(args.output_dir / "gender_disparities.csv", index=False)
    subject_summary(clean).to_csv(args.output_dir / "subject_summary.csv", index=False)
    profiles.to_csv(args.output_dir / "segment_profiles.csv", index=False)
    segmented[["candidate_id", "segment"]].to_csv(args.output_dir / "candidate_segments.csv", index=False)
    print(profiles.to_string(index=False))


if __name__ == "__main__":
    main()
