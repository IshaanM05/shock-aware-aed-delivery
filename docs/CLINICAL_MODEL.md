# Clinical model (Track B): survival, EMS delay, economics, decision

Code: `src/aedrover/clinical/` (`survival.py`, `ems_delay.py`, `economics.py`, `decision.py`) and
`src/aedrover/analysis/stats.py`. Units: clinical times in **minutes from collapse**, distances
in metres, speeds in metres per second, survival as a probability in [0, 1]. Economics is
strictly dimensionless (ratios, hours, years only). Every constant taken from the literature
cites a key in `docs/references.json`; everything else is flagged `ASSUMPTION` below.

## 1. Survival model

### Larsen et al. 1993 (`larsen1993`, PMID 8214853)

Published equation (multiple linear regression, 1,667 King County patients with heart disease,
ventricular fibrillation and collapse before EMS arrival):

    S = 0.67 - 0.023 * t_cpr - 0.011 * t_defib - 0.021 * t_acls        (clipped to [0, 0.67])

`t_cpr`, `t_defib`, `t_acls` are the minutes from collapse to CPR, to the first shock and to
ACLS. 0.67 is the survival if all three happen at collapse; with no treatment survival falls by
0.023 + 0.011 + 0.021 = 0.055 per minute. The cohort is a high-survival-likelihood subgroup, so
absolute levels are optimistic for all-comers OHCA; the project uses the model mainly for
differences between delivery modes.

**Correction of the student draft.** An earlier draft used 0.046 per minute for defibrillation.
That is wrong: the published slope is **0.011** per minute (0.046 is not a coefficient of this
model). The tests pin the published values.

### Rule of thumb (`rule_of_thumb`, `_low` = 0.07, `_high` = 0.10; source `kim2026`)

The 2025 Korean CPR guidelines (Part 4, adult ALS) state that without CPR survival after
defibrillation falls by approximately 7% to 10% per minute (3% to 4% with bystander CPR). This is
a guideline statement, not a fitted model. Implemented as
`S = S0 - rate * t_defib` (percentage points per minute, clipped to [0, S0]); CPR and ACLS times
are ignored. Used only for model-form sensitivity.

### Valenzuela et al. 1997 (`valenzuela1997`): not implemented

The logistic-regression model needs published coefficients. The PubMed abstract (PMID 9396421)
lists the predictors (collapse-to-CPR, collapse-to-defibrillation, bystander CPR, age and
interactions) but no coefficients; the full text is closed access (Unpaywall: closed; the
publisher returns HTTP 403 to scripted access). Coefficients are never cited from memory here,
so the model is omitted. The reference stays in `references.json` as context.

## 2. Time-to-first-shock decomposition

All times from collapse. `c2c` = collapse to call, `alert` = call-taker triage plus dispatch
latency until the device is alerted, `R` = sampled ambulance response time (call to arrival).

| Mode | First-shock time |
| --- | --- |
| Ambulance | `t_scene = c2c + R (+ busy delay)`; `t_defib = t_scene + ems_arrival_to_shock` |
| Rover | `c2c + alert + radius * route_factor / speed / 60 + handoff` |
| Drone | `c2c + alert + launch + radius / speed / 60 + handoff` |

`R` is lognormal with (median, p90) taken from Naess et al. (Section 3); the ambulance term
already runs from the call, so `alert` is not added to it. **The rover changes only the
defibrillation time.** CPR time is `min(bystander_cpr_delay, t_scene)` (or `t_scene` if no
bystander CPR) and ACLS time is `t_scene`, identical for every mode, so a delivery device
influences survival solely through the `t_defib` term.

Two comparisons, never to be mixed up:

* **Parallel** (`parallel=True`, default in `mode_times`): the device is dispatched with the
  ambulance and the first shock is `min(t_device, t_ambulance)`. Gain over the ambulance alone is
  the marginal benefit of adding the device and is never negative.
* **Standalone** (`parallel=False`, used by `breakeven_radius`): device alone versus ambulance
  alone. The straight-line radius where expected survivals are equal is the break-even radius
  (`scipy.optimize.brentq` on a deterministic quantile grid of the response time, so the
  objective is smooth). Flags: `crossing`, `rover_never_better` (0), `rover_always_better` (inf).

Ambulance and device samples are paired per simulated arrest (shared `R`), and bootstrap CIs
resample arrests jointly. Simulated rover travel times plug in via `rover_travel_min`.

## 3. Source-verified numbers

| Quantity | Value | Key |
| --- | --- | --- |
| Larsen intercept and slopes | 0.67; 0.023 (CPR), 0.011 (defib), 0.021 (ACLS) per min | `larsen1993` |
| Response time, all incidents (Central Norway 2013-2022, n = 216,787) | mean 15.8 (SD 12.6), median 12.2 (IQR 10.4), p90 29.1 min | `naess2024` Table 1 |
| Response time, rural (n = 134,116) | mean 18.3 (SD 14.0), median 14.8 (IQR 13.2), p90 33.3 min | `naess2024` Table 1 |
| Response time, urban (n = 82,671) | mean 11.6 (SD 8.5), median 10.0 (IQR 5.6), p90 17.7 min | `naess2024` Table 1 |
| Busy-ambulance delay per +10 percentage points of busy probability | 0.60 (95% CI 0.58-0.62) all; 0.81 rural; 0.30 urban min | `naess2024` |
| Mean extra response time due to busy ambulances | 1.61 all; 1.76 rural; 1.04 urban min | `naess2024` |
| Rule-of-thumb decline | 7% to 10% per minute (3% to 4% with bystander CPR) | `kim2026` |
| Drone AED arrived before ambulance | 37 of 55 (67%); median benefit 3 min 14 s (validation target only) | `schierbeck2023` |

Response time is defined as the first call to the emergency communication centre until the first
ambulance arrives. The default ambulance distribution is the **urban** row (a rover is an urban
device); the lognormal fitted to (median, p90) reproduces the reported means within 6% and IQRs
within 12% (checked in `tests/test_survival.py`).

## 4. Assumptions table

Every value below is an ASSUMPTION (not verified against a primary source). "Range" is what the
sensitivity analysis (`tornado`, `breakeven_surface`) uses.

| Parameter | Default | Range | Why |
| --- | --- | --- | --- |
| `collapse_to_call_min` | 1.0 min | 0.5 to 3.0 | Recognition and dialling delay; common to all modes, so it mainly moves absolute survival, not gains |
| `call_to_alert_min` | 1.5 min | 0.5 to 3.0 | Call-taker triage plus dispatch until the rover/drone is alerted; not added to ambulances |
| `handoff_min` | 1.0 min | 0.5 to 2.5 | Bystander fetches the AED from the rover and applies pads |
| `bystander_cpr_delay_min` | 3.0 min (or `None`) | 1.0 to 6.0 | Dispatcher-assisted CPR start; `None` means CPR starts at ambulance arrival |
| `ems_arrival_to_shock_min` | 0.0 min | 0.0 to 3.0 | Conservative for the rover comparison: the crew shocks on arrival |
| `radius_m` | 1000 m | 250 to 2000 | Scenario choice, not a claim about any city |
| `rover_speed_mps` | 2.0 m/s | 1.0 to 3.0 | Placeholder until the MuJoCo simulation supplies measured travel times |
| `route_factor` | 1.3 | 1.1 to 1.6 | Sidewalk path length over straight-line distance |
| `drone_speed_mps` | 15 m/s | 10 to 25 | Placeholder cruise speed (drone comparator only) |
| `drone_launch_min` | 0.5 min | 0.25 to 2.0 (not in default tornado) | Drone preparation and take-off |
| `busy_increase` | 0.0 | 0.0 to 0.20 | Busier-than-usual system; slope itself is sourced (`naess2024`), the size of the increase is a scenario |
| Response-time family | lognormal | mean/IQR check within 6%/12% | Only median and p90 are used; shape is a modelling choice |
| Busy delay form | constant shift on the response time | n/a | Linear use of a sourced slope; reference level is the regional average busyness |
| `t_acls=None` | `t_acls = t_defib` | n/a | Convention when ACLS time is not supplied; the decision code always supplies `t_scene` |
| Rule-of-thumb midpoint rate | 0.085 per min | 0.07 to 0.10 | Guideline gives only the range |
| Rule-of-thumb initial survival | 0.67 | n/a | Shares Larsen's starting level so only slope and structure differ; read as percentage points per minute |
| `energy_ratio` | 0.04 | 0.02 to 0.08 | Annual energy burden as a fraction of annual reclaimed-labour value |
| `maintenance_ratio` | 0.12 | 0.05 to 0.25 | As above, maintenance |
| `infrastructure_ratio` | 0.06 | 0.02 to 0.15 | As above, docking and connectivity |
| `capex_parity_factor` | 0.60 years | 0.3 to 1.5 | Up-front capital over annual reclaimed-labour value |
| `annual_reclaimed_task_hours` | 1040 h | 500 to 2080 | Half of one full-time equivalent |

Course conventions (specified by the course model, not measured): `kappa <= 0.25` target and
2080 h per full-time equivalent (40 h x 52 weeks). `qaly_gain_per_encounter` has no defaults:
life expectancy and quality of life must be supplied by the caller, undiscounted.

Illustrative outputs at the defaults (Larsen, urban response times; each depends on the
ASSUMPTION inputs above): ambulance expected survival 0.230; rover marginal gain (parallel)
0.053, 0.029, 0.004 at 250, 500, 1000 m; rover-alone break-even radius about 701 m (924 m with
the rural median 14.8 min); rule-of-thumb break-even about 378 m. Economics defaults give
`kappa = 0.22` (meets target), payback 9.2 months, 0.5 FTE. Because Larsen's defibrillation
slope is only 0.011 per minute, the model attributes modest survival value to a shock-time
saving; the rule of thumb is far steeper, which is why both are reported.

## 5. Statistics layer (`analysis/stats.py`)

Welch and paired t tests return n, mean difference, df, t, exact two-sided p, Cohen's d (pooled)
or d_z (paired) with the exact non-central-t CI, and the t-based CI of the mean difference.
Also: seeded BCa bootstrap (`scipy.stats.bootstrap`; percentile fallback for n above 4000),
Wilson interval, Holm adjustment, Mann-Whitney with rank-biserial effect size, exact
`n_per_group_for_d`, and `compare_groups`. For paired designs `compare_groups` pairs the t test
with the Wilcoxon signed-rank test instead of Mann-Whitney, because Mann-Whitney assumes
independent samples.

## 6. Limitations

* The survival and response-time inputs come from Western cohorts (King County, Central
  Norway). No claim is made about Mumbai or any other Indian setting; transferring the results
  needs local response-time and outcome data.
* Larsen's model is linear, fitted on a favourable subgroup (witnessed VF with heart disease),
  and clips to zero beyond about 12 minutes of untreated delay; it is not valid for non-shockable
  rhythms, unwitnessed arrests or long delays. The rule of thumb is a heuristic.
* Survival, delays and travel times are treated as independent; real response times correlate
  with distance, traffic and time of day. Busy-ambulance effects are association estimates.
* Rover reliability (failed deliveries, blocked sidewalks), bystander willingness and AED misuse
  are not modelled; every device is assumed to arrive if its computed time is shorter.
* Economics is a course template: undiscounted, single-period, with placeholder ratios.
