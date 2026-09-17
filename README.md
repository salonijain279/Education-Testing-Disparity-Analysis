# Education Outcomes & Achievement Gap Segmentation

A Python workflow examining where testing outcomes differ across candidate groups and regions —
and drawing a clear line between a measurable disparity and a causal claim the available
features can't support.

## Business question

Where do credential candidates experience the largest outcome gaps, and which observable access
or preparation factors should program leaders investigate first?

## Approach

1. Validate candidate-level records and plausible age/score ranges.
2. Compare completion and pass rates across regions and demographic groups.
3. Measure subject-level performance gaps.
4. Segment candidates with K-Means using preparation, access, timing, and score features.
5. Export decision-ready disparity and segment summaries.

## Data and confidentiality

The public implementation uses deterministic synthetic data and contains no learner-level
records, client files, geographic identifiers, or private engagement outputs.

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

Group differences are descriptive, not causal estimates — they're meant to guide further
investigation, not automatic decisions about individuals.
