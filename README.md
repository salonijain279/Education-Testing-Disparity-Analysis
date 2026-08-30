# Education Testing Disparity Analysis

An exploratory analytics workflow for identifying performance disparities across candidate groups and geographic regions.

## Business question

Where do credential candidates experience the largest outcome gaps, and which observable access or preparation factors should program leaders investigate first?

## Analysis workflow

1. Validate candidate-level records and plausible age/score ranges.
2. Compare completion and pass rates across regions and demographic groups.
3. Measure subject-level performance gaps.
4. Segment candidates with K-Means using preparation, access, timing, and score features.
5. Export decision-ready disparity and segment summaries.

## Data and confidentiality

This repository is a **public portfolio reconstruction using deterministic synthetic data**. No learner-level records, client files, geographic identifiers, course instructions, or private outputs from the original academic engagement are included.

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
