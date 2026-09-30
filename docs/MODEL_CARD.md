# Model card: what this study supports and what it does not

## Intended use

A research and teaching artefact for studying, in simulation, whether a sidewalk robot can deliver an AED
faster than an ambulance and how vehicle design, control and dispatch policy change that answer. It is not
a product, not a safety case, and not medical advice.

## What the evidence supports

* The simulator reproduces closed-form suspension mechanics (`VALIDATION.md`), so relative comparisons of
  designs and controllers inside it are meaningful.
* Within the simulated scenario families, the reported orderings of controllers and vehicle designs, with
  paired statistics and confidence intervals (`RESULTS.md`).
* The clinical numbers are conditional on the stated models and assumptions: they show how the answer moves
  with response radius, crossing density, drone availability and ambulance delay, not what any city will observe.

## What it does not support

* **No sim-to-real claim.** No hardware was built or measured. Rigid cylindrical wheels with calibrated contact
  softness are a simplification of a pneumatic tyre; real kerbs have rounded edges, dirt, water and gradients.
* **Pedestrians are a model.** Social-force streams with assumed strengths are not recorded human behaviour, and
  the tracker is a noisy abstraction of a perception stack.
* **The 3 g payload budget is a design requirement inherited from the course brief.** It is not a certified shock
  limit for any specific AED.
* **Survival models come from Western cohorts.** Larsen (1993) is 30 years old; the rule-of-thumb model is a
  sensitivity check, not a fit.
* **Ambulance delays come from a Norwegian study** (Naess et al. 2024), not from Mumbai. Local response times
  are likely worse, which raises the value of any delivery device; the tornado analysis shows the size of that effect.
* **OpenStreetMap tags almost none of the crossings in the study area** (98 percent of road length has no sidewalk
  tag), so route crossing density is swept, not measured. Route factors come from the network graph and are real.
* **Economics are illustrative.** The course's dimensionless model is evaluated with assumed ratios; nothing here
  estimates a real cost.
* **Drone comparator**: a simplified quadrotor with no rotor aerodynamics beyond drag, no battery voltage sag and
  no airspace regulation. It is not tuned to lose; parameters and sensitivity ranges are published.

## Known limitations of the evaluation

* Episodes are 36 m segments; kilometre routes are composed by bootstrap under an independence assumption.
* A route fails if any composed segment fails, which is conservative because a stalled rover might eventually proceed.
* Single PPO training seeds are reported with their variance across two seeds only.
* Success and shock thresholds are fixed; they are reported as safe-delivery rates so the reader can judge the trade-off.

## Ethical note

Autonomous vehicles sharing space with people carry real risk. Nothing in this repository should be read as evidence
that such a system is safe to deploy; the point of the safety filter, the pedestrian-clearance metrics and the
out-of-distribution study is to make failure modes visible in simulation first.
