# Research question

How does grass cover influence Eucalyptus spp. seedling recruitment?

## Data summary (the survey behind `data.csv`)

These data come from a field survey of Eucalyptus seedling recruitment in
remnant woodland patches on grazing properties in south-eastern Australia.

DESIGN AND LEVELS OF MEASUREMENT. The survey is nested within 18 named
PROPERTIES, which were visited in three survey rounds (Winter 2006,
Spring 2006, Autumn 2007). In every round, researchers randomly allocated
a DIFFERENT set of 15 x 15 m quadrats within 60 m of existing tree
canopies. One row is one physical quadrat sampled in one round. Properties
recur, but physical quadrats do not; `Quadrat no` labels are reused across
rounds and are not persistent quadrat identifiers. The supplied file has
351 rows and 38 columns.

What was recorded, and where:
- Measured afresh at each quadrat visit: the percent ground cover
  estimates for each vegetation and substrate class, the eucalypt seedling
  counts in three height classes, distance to the nearest eucalypt canopy,
  and overhead eucalypt canopy cover.
- Recorded spatial descriptors: position (easting, northing), aspect and
  landscape position.
- Attached per site from spatial layers: the climate, solar-radiation and
  soil covariates.

Cover was estimated visually as a percentage of the quadrat; seedlings
were counted, not estimated. Some cover and covariate values are missing.

## Column guide

- Identifiers and design: `SurveyID` (row id), `Date`, `Season` (survey
  round: Winter 2006 / Spring 2006 / Autumn 2007), `Property` (18 named
  properties — the natural spatial cluster), `Quadrat no` (quadrat
  number within a property), `Easting`, `Northing` (coordinates),
  `Aspect`, `Landscape position` (categorical site descriptors).
- Ground-cover predictors, each a PERCENT COVER of the quadrat (larger
  value = more of that cover class present): `ExoticAnnualGrass_cover`,
  `ExoticPerennialGrass_cover`, `ExoticAnnualHerb_cover`,
  `ExoticPerennialHerb_cover`, `ExoticShrub_cover`,
  `NativePerennialGrass_cover`, `NativePerennialGraminoid_cover`,
  `NativePerennialHerb_cover`, `NativePerennialFern_cover`,
  `NativeShrub_cover`, `BareGround_cover`, `Litter_cover`,
  `MossLichen_cover`, `Rock_cover`. The GRASS-COVER classes are the
  exotic and native perennial/annual grass columns; which of them (or
  which sum of them) operationalizes "grass cover" is an analytical
  choice, not a given.
- Eucalypt context: `Euc_canopy_cover` (percent overhead eucalypt canopy
  above the quadrat), `Distance_to_Eucalypt_canopy(m)` (metres to the
  nearest eucalypt canopy; larger = further from a potential seed
  source).
- Recruitment outcomes, all COUNTS of eucalypt seedlings in the quadrat
  (larger = more recruitment): `euc_sdlgs0_50cm` (0–50 cm),
  `euc_sdlgs50cm-2m` (50 cm–2 m), and `euc_sdlgs>2m` (over 2 m).
  Totals, occurrence indicators, and size-weighted indices are not
  precomputed; constructing one is an analytical choice.
- Environmental covariates: `annual_precipitation`,
  `precipitation_warmest_quarter`, `precipitation_coldest_quarter`,
  `PET` (potential evapotranspiration), `MrVBF` (multi-resolution valley
  bottom flatness), `K_perc`, `Th_ppm`, `U_ppm` (radiometric soil
  measures), `SRad_Jan`, `SRad_Jul` (solar radiation).

## Declared estimand

effect of grass cover on Eucalyptus seedling recruitment

## Frame policy

the Eucalyptus population, grass-cover exposures, and seedling-recruitment
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
