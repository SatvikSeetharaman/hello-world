# Passing the Pressure

**Quantifying passer→receiver pressure transfer and short-horizon possession risk in elite women's soccer**

## Research question
A completed pass can relieve the passer's pressure while leaving the receiver in an even worse situation. We define **pressure transfer** as:

`ΔP = P_receiver - P_passer`

where local pressure is `sum(exp(-distance_to_opponent / 5))` over visible opponents in the StatsBomb 360 freeze frame. Negative values indicate pressure relief; positive values mean the receiver inherits more pressure.

## Data
Primary sample: **all 31 matches of UEFA Women's Euro 2025** in the public Hudl StatsBomb Open Data repository, using event data plus StatsBomb 360 freeze frames.

Source: https://github.com/hudl/open-data

Exact match IDs are in `data/match_manifest_2025.csv`.

Eligible passes are completed open-play passes with a named recipient and 360 freeze frames at both release and the linked `Ball Receipt*` event. Set pieces (corners, free kicks, throw-ins, kick-offs, goal kicks) are excluded.

This yielded **16,908 passes**.

## Outcome
The primary outcome is **5-second possession retention** after receipt.

## Adjusted model
Covariates:
- pressure transfer
- starting passer pressure
- goal-distance progression
- pass length
- starting x
- receiving centrality

Predictors are standardized within each fold. The tournament is divided into eight non-overlapping folds; fold-specific pressure-transfer coefficients are combined with inverse-variance fixed-effect pooling.

## Primary result
A one-standard-deviation increase in pressure transferred to the receiver was associated with lower odds of retaining possession for five seconds:

- pooled log-odds coefficient: **-0.355**
- odds ratio: **0.701**
- 95% CI: **0.647–0.760**
- p-value: **4.7 × 10^-18**

Strongly pressure-relieving passes (`ΔP < -0.5`) retained possession for five seconds **93.7%** of the time versus **85.5%** for strongly pressure-transferring passes (`ΔP ≥ 0.5`). High-transfer passes were also substantially more progressive, motivating joint evaluation of progression and pressure.

## Preliminary external check
The same pipeline on 12 successfully processed UEFA Women's Euro 2022 matches produced **6,591 eligible passes** and a pooled OR of **0.720** (95% CI **0.639–0.811**). This subset is preliminary and is not the basis of the primary abstract claim.

## Reproduce
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python analysis.py --season 2025
```

## Files
- `analysis.py` — reproducible pipeline
- `data/match_manifest_2025.csv`
- `data/match_manifest_2022_validation.csv`
- `results/primary_results.csv`
- `results/pressure_transfer_bins.csv`
- `results/figure_pressure_transfer.svg`
- `abstract.md`
- `METHODS.md`
- `requirements.txt`

## Novelty positioning
This project does **not** claim defensive pressure or pressure change itself is new. Prior work has modeled pressure, pass difficulty, and pressure changes. The narrower contribution is to operationalize **passer-to-receiver pressure transfer as a pass-level risk quantity** and quantify its downstream possession cost while controlling for the progression-pressure tradeoff in elite women's soccer.

## Limitations
StatsBomb 360 is freeze-frame rather than continuous tracking; the pressure function is intentionally simple; the analysis is observational; five-second retention is a short-horizon outcome; and a full manuscript should add multilevel match/team effects and alternative pressure definitions.

## Attribution
Source football data are governed by Hudl StatsBomb's open-data terms. Analysis based on these data should attribute StatsBomb/Hudl.
