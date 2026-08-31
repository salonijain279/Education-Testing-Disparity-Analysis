# Identifying Education Testing Disparities

I built this Python workflow to examine where testing outcomes differ across candidate groups and regions—and to separate a measurable disparity from a causal conclusion I could not support with the available features.

## Business question

Where do credential candidates experience the largest outcome gaps, and which observable access or preparation factors should program leaders investigate first?

## What I did

1. Validate candidate-level records and plausible age/score ranges.
2. Compare completion and pass rates across regions and demographic groups.
3. Measure subject-level performance gaps.
4. Segment candidates with K-Means using preparation, access, timing, and score features.
5. Export decision-ready disparity and segment summaries.

## Data and confidentiality

I rebuilt the analysis with deterministic synthetic data for this public repository. I did not include learner-level records, client files, geographic identifiers, course instructions, or private outputs from the original academic engagement.

## Run

```bash
python src/generate_sample_data.py
python src/analyze_disparities.py
python -m unittest discover -s tests
```

Outputs are written to `outputs/`.

## Methods

`Python` · `pandas` · exploratory data analysis · cohort comparisons · K-Means · interpretable segmentation

## Interpretation boundary

Group differences are descriptive. They are not causal estimates and should guide further investigation rather than automatic decisions about individuals.
