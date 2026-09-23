# Proposed rect_coeffs "database" layout (draft, not yet populated)

Craig wants a git-tracked database of saved `xtrans_rect`/`ytrans_rect` transforms (the
`rect_coeffs_*.dat` files rectifyProcess already writes and can read back via
`rect_coeffs_file`), one per instrument/mode, for every case where the trace is stable across
exposures - reused the same way `verified/` and `superFATBOY/data/templates/` are already used
this session.

## Proposed location and layout

```
superFATBOY/data/rect_transforms/
  README.md                          - explains the convention below
  OSIRIS_R500B_longslit_S1.dat        - one file per (instrument, grism/grating, mode, amp)
  OSIRIS_R500B_longslit_S2.dat
  OSIRIS_R2500U_longslit_S1.dat
  FLAMINGOS1_MOS_r3c1m2z.dat          - MOS masks are per-mask-design, not reusable across masks
  MIRADAS_SOL_<config>.dat            - not yet safe to add, see caveat below
```

Key = **instrument + grism/grating + mode + amp/CCD-side** (not just instrument+mode) - the
real files inspected tonight (`rect_coeffs_sdss_j120937S1.dat` vs. `sarik_osiris`'s own, both
OSIRIS longslit but different grisms R2500U vs R500B) have very different coefficients because
the dispersion/curvature genuinely differs by grating. For MOS mode, curvature is tied to the
*physical slit-mask design*, not just the instrument - specBench's `r3c1m2z_jhjh` mask isn't
interchangeable with a different MOS mask on the same instrument, so the key needs the mask
name too, not just instrument+mode.

A `verified_transforms.md` index (same spirit as `verified/verified_configs.md`) should map
each file to: instrument, grism/mode/mask key, which XML config's `rect_coeffs_file` produced
it, and the git commit/date it was captured, so it's traceable if the instrument's optics ever
get re-aligned and the transform needs regenerating.

## Format

Already exists, don't invent a new one for the polynomial case - `rect_coeffs_*.dat`'s current
`poly <order>` header + coefficient lists (see `calcLongslitRectification`, ~line 1218) is fine
to check in as-is. If the spline option (see the fit-function report) ever gets wired in for
real, it'll need a second header tag (e.g. `spline <k> <n_knots>`) alongside `poly` so old and
new formats coexist in the same directory without ambiguity - noted for whoever does that work,
not solved here.

## Caveat: don't populate this yet for MIRADAS

Per Craig's note tonight on the collapseSpaxels work: MIRADAS is simulated data (instrument not
live), and he's not confident individual-exposure geometry is even trace-to-trace comparable,
let alone stable enough to bless as a checked-in calibration artifact. Only add a MIRADAS entry
if/when real (or at least trusted-simulated) data confirms the trace doesn't move between
exposures - same logic as why `verified/` only holds configs that have actually been confirmed,
not ones that plausibly should work.

## What's safe to add now

Once tonight's `sarik_osiris.xml`/`avrajit-osiris.xml` reruns (both real, already-verified
OSIRIS longslit data) finish, their `rect_coeffs_*.dat` outputs are reasonable first entries -
real instrument, real grism, already confirmed stable/working. Left unpopulated tonight since
Craig should see the format proposal first.
