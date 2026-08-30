"""Create deterministic synthetic education-testing data."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def generate(seed: int = 42, rows: int = 1800) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    regions = np.array(["metro", "regional", "rural", "remote"])
    region = rng.choice(regions, size=rows, p=[0.36, 0.30, 0.24, 0.10])
    gender = rng.choice(["woman", "man", "nonbinary_or_undisclosed"], size=rows, p=[0.50, 0.46, 0.04])
    age = np.clip(rng.normal(29, 9, rows).round(), 17, 70).astype(int)

    distance_base = {"metro": 4, "regional": 12, "rural": 27, "remote": 48}
    distance = np.array([distance_base[r] for r in region]) + rng.gamma(2, 3, rows)
    prep_attempts = np.maximum(0, rng.poisson(4.5, rows) - (region == "remote").astype(int))
    days_between_tests = np.clip(rng.gamma(2.2, 18, rows) + distance * 0.35, 1, 240)

    access_penalty = np.select(
        [region == "regional", region == "rural", region == "remote"],
        [1.0, 2.6, 4.2],
        default=0.0,
    )
    preparation_effect = np.log1p(prep_attempts) * 3.0
    base = 145 + preparation_effect - access_penalty - days_between_tests * 0.025

    frame = pd.DataFrame(
        {
            "candidate_id": [f"C{i:05d}" for i in range(1, rows + 1)],
            "region": region,
            "gender": gender,
            "age": age,
            "distance_to_center_miles": distance.round(1),
            "prep_attempts": prep_attempts,
            "days_between_tests": days_between_tests.round(1),
            "math_score": (base - 2.0 + rng.normal(0, 6.5, rows)).round(0),
            "language_score": (base + 0.5 + rng.normal(0, 6.0, rows)).round(0),
            "science_score": (base + 1.5 + rng.normal(0, 6.0, rows)).round(0),
            "social_studies_score": (base + 1.0 + rng.normal(0, 6.0, rows)).round(0),
        }
    )
    score_columns = [c for c in frame.columns if c.endswith("_score")]
    frame[score_columns] = frame[score_columns].clip(100, 200)
    return frame


def main() -> None:
    output = ROOT / "data" / "synthetic_candidates.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    data = generate()
    data.to_csv(output, index=False)
    print(f"Wrote {len(data):,} synthetic candidate records to {output}")


if __name__ == "__main__":
    main()
