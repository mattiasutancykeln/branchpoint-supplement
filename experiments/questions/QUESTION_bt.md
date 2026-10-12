# Research question

To what extent is the growth of nestling blue tits influenced by competition with siblings?

## Data summary (the experiment behind `data.csv`)

These data come from a brood-size manipulation experiment on wild blue
tits nesting in boxes at Wytham Wood (Oxfordshire, UK), 2001–2003
breeding seasons. At each nest, researchers recorded first-egg date,
clutch size, and hatching date. Brood sizes were manipulated by
CROSS-FOSTERING: broods were matched for hatching date and brood size,
and chicks were moved between paired nests one or two days after
hatching, enlarging or reducing manipulated broods by approximately one
fourth (net manipulations from ±1 chick in broods of 4–5 up to ±4 in
broods of 12–16). To control for the effect of being moved, each
reduced brood had part of its brood replaced by chicks from the paired
enlarged brood, and vice versa. Roughly one third of broods were left
unmanipulated (not systematically matched to manipulated broods in
clutch size or laying date). Because of cross-fostering, each chick has
BOTH a hatch (natal) nest and a rearing nest, with identities of both
sets of parents recorded. The dataset contains one row for each of 3,720
chicks alive and measured at day 14, from 167 experimentally enlarged,
165 experimentally reduced, and 120 unmanipulated rearing broods. Day-14
mass and tarsus length are complete in the supplied file. Missing values
in several identity, parentage, manipulation, and later-survival fields
are coded as `.`.

## Column guide

- `chick_ring_number` — individual chick ID.
- `hatch_year` — breeding season (2001–2003).
- Hatch (natal) nest: `hatch_nest_breed_ID` (nest ID), `hatch_Area`,
  `hatch_Box` (location), `hatch_mom_Ring` / `hatch_nest_dad_Ring`
  (social parents), `hatch_nest_LD` (lay date), `hatch_nest_CS`
  (clutch size), `hatch_nest_OH` (hatch date),
  `d0_hatch_nest_brood_size` / `d14_hatch_nest_brood_size` (natal-nest
  brood sizes at day 0 / day 14).
- Paternity: `Extra-pair_paternity` (coded 1/2, "." unknown),
  `Extra-pair_dad_ring`, `genetic_dad_ring_(WP_or_EP)`.
- Rearing nest: `rear_nest_breed_ID` (nest ID), `rear_area`,
  `rear_Box`, `rear_mom_Ring` / `rear_dad_Ring` (foster parents),
  `rear_nest_trt` (assigned treatment category of the rearing nest,
  coded 5/6/7 — decode the categories by cross-referencing
  `net_rearing_manipulation`: positive = enlarged, negative = reduced,
  0 = unmanipulated), `home_or_away` (coded 1/2: reared in natal nest
  vs cross-fostered),
  `rear_nest_LD`, `rear_nest_CS`, `rear_nest_OH` (rearing-nest lay
  date, clutch size, hatch date).
- Manipulation bookkeeping: `rear_Cs_out` / `rear_Cs_in` (chicks moved
  out of / into the rearing nest), `net_rearing_manipulation` (net
  chicks added, negative for reductions),
  `rear_Cs_at_start_of_rearing` (brood size of the rearing nest right
  after the manipulation), `rear_d0_rear_nest_brood_size` /
  `d14_rear_nest_brood_size` (rearing-nest brood size at day 0 /
  day 14), `number_chicks_fledged_from_rear_nest`.
- Outcomes and covariates: `Date_of_day14` (measurement date),
  `day_14_tarsus_length`, `day_14_weight`, `day14_measurer`,
  `chick_sex_molec` (molecular sex, coded 1/2, "." unknown),
  `chick_survival_to_first_breed_season`.

## Declared estimand

effect of sibling competition on nestling growth

## Frame policy

the blue-tit population, sibling-competition exposures, and nestling-growth
outcomes represented in the provided dataset

Analyze the data in `data.csv`. Report the fitted analysis's focal estimate and
uncertainty in its native model scale.
Work in small, checkable
Python scripts (any installed library is fine, but prefer the standard
library and pandas/numpy/scipy/statsmodels if you need them) rather than
one large script. Report a note describing exactly what model/variables/
exclusions you used, so the choice is auditable.

Also report, in the `predictor` field, the exact quantity whose effect you
are reporting, as you constructed it; and in the `orientation` field,
whether larger values of that quantity correspond to MORE of what this
question's declared estimand is oriented on (`as_declared`) or to LESS of
it (`reversed`), or `not_applicable` if no direction applies. Either
construction is legitimate: report the estimate signed for the quantity
exactly as you built it and declare which case you are in, rather than
silently flipping the sign.
