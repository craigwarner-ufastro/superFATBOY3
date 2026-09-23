# rectifyProcess.py review (overnight, 2026-09-22)

## 1. Current approach & concrete weaknesses

**Trace-finding** (`calculateMOSContinuaTrans` for MOS/IFU, `calcLongslitContinuaRectification`
for longslit; mirrored by `calculateMOSSkylineTrans`/`calcLongslitSkylineRectification` for
skylines/arclines): both trace by cross-correlating a narrow 1-D cut (a few columns/rows wide)
against a reference cut every few pixels along the dispersion axis, then subpixel-refining the
correlation peak with a 3-point parabola. This is the same "step-and-correlate" idea Craig
described being unhappy with in findSlitletProcess (sibling task) — same failure mode here:
low S/N regions (slit edges, faint continua) give a noisy or flat correlation peak with no
signal that the step should be down-weighted or rejected outright before it pollutes the fit.

**Transformation fit**: plain 2-D power-series polynomials (`surfaceFunction`, order `fit_order`)
per slit/segment (`independent_slitlets` mode), or a 3-variable power series adding slit position
as a third fit dimension (`surface3dFunction`, `use_slitpos` mode) so all slits share one global
fit. `whole_chip` mode is a single 2-D fit ignoring slit identity entirely.

Concrete problems with plain power-series polynomials here:
- **Extrapolation blowup.** Ran a synthetic sanity check (`/tmp/rectify_poly_vs_spline_demo.py`,
  not part of the repo): a smooth trace + realistic centroid noise, fit with polynomials order
  3/5/7 and evaluated just slightly outside the sampled x-range — order 7 misses the true edge
  value by ~50 pixels' worth of curvature error, and error *grows* with fit order, which is
  backwards from what you'd want (higher order should mean better fit, not worse extrapolation).
  A smoothing spline evaluated over the same range tracks the true curve to ~1 pixel. This is
  exactly the failure `rectify_max_transform_factor` (added 2026-09-11) guards against
  symptomatically — it catches the blowup after the fact rather than the fit family not
  producing one in the first place.
- **Global-only flexibility.** A single polynomial can't be *locally* stiff where the trace is
  well-sampled and *locally* loose where it isn't — one noisy segment can visibly warp the curve
  everywhere else via the fit's global coefficients (Runge-phenomenon-adjacent behavior, worse
  at the higher orders `mos_fit_order`/`fit_order` already default to in some configs).

**Recommendation**: swap the transform functions for **smoothing B-splines**
(`scipy.interpolate.UnivariateSpline`/`LSQUnivariateSpline` for longslit's 1-D-in-x trace;
`scipy.interpolate.SmoothBivariateSpline` for the MOS 3-variable `use_slitpos` case) with the
smoothing factor `s` set from the trace's own centroiding-noise estimate (already computed as
`residstddev` in the existing sigma-clip loop — reuse it directly, e.g. `s = len(x)*residstddev**2`,
the standard scipy convention). Splines are local (a bad region only warps its own neighborhood),
don't need `checkFitSanity`'s runaway guard nearly as much (though I'd keep it as a second line of
defense, not remove it), and `UnivariateSpline` is a drop-in for `surfaceFunction`'s 1-D use.
`SmoothBivariateSpline` is a bigger lift for `use_slitpos` (3-variable) — I'd stage that second,
after the simpler 1-D swap is validated on real data. Not recommending Chebyshev/Legendre
polynomials instead — they fix numerical conditioning at high order but *not* the extrapolation
behavior, which is the actual problem here.

## 2. Partial continuum coverage ("some slits have no bright star")

Found it exactly: `calculateMOSContinuaTrans`, `independent_slitlets` branch, ~line 1697
(`if (b.sum() == 0): ... ytransData[...] = yind[...]`) — a slit with zero continuum datapoints
falls back to a pure **identity** transform (no rectification at all), logged as ERROR and
counted in `n_slits_not_rectified` (already escalated from a quiet WARNING on 2026-09-11), but
never actually fixed. Two options, not mutually exclusive:
- **Use `use_slitpos` mode when any slits lack continuum** — it already solves this in
  principle: slit position is a *fit variable*, not a fit boundary, so a slit with no data of
  its own is naturally interpolated from the shared 3-variable surface fit to every OTHER slit's
  data. This isn't a new feature, just a mode switch — worth trying on `specBench.xml` (which has
  exactly this shape: one bright standard-star slit, several faint MOS science slits) before
  writing new code.
- **In `independent_slitlets` mode**, when a slit has `b.sum()==0`, borrow the nearest
  data-bearing slit's fit coefficients (by `slitx` distance) instead of identity — strictly
  better than identity if slit curvature varies smoothly across the detector (true for
  grating/grism spectrographs), strictly no worse if it doesn't (still degrades to "some
  transform" rather than "no transform"). Small, bounded change if you want it; I did not write
  it since it touches tested production code — flagging for your call.

## 3. Caching (`xtrans_rect`/`ytrans_rect`/`rect_coeffs_file`) — already exists

This already does almost exactly what you described wanting. `getCalibs` (~line 2601-2682)
checks, in order: FDU property → in-memory master-calib cache keyed by `rct_id` (so repeat
frames in one run reuse a computed fit for free) → an on-disk file (`rect_coeffs_file` for
longslit, `xtrans_rect_file`/`ytrans_rect_file` for MOS) if you set that XML option, read via
`readCoeffsFile`/loaded as a `fatboySpecCalib` FITS file. It also always **writes** these to
`<outputdir>/rectified/{xtrans,ytrans}_rect_<id>.fits` (MOS) or `rect_coeffs_<id>.dat` (longslit)
regardless of whether you set the file-reuse option. Confirmed real files from tonight's run:
`avrajitOsiris-refactor-gpu/rectified/rect_coeffs_sdss_j120937S1.dat` is a human-readable
`poly <order>` + two 6-coefficient blocks (x and y 2-D polynomial coeffs) — a coefficient
representation, not literally a per-pixel forward map, for longslit; the MOS `xtrans_rect`/
`ytrans_rect` *are* literal per-pixel forward maps (one FITS float array, same shape as the
detector, value at pixel = the output x or y coordinate). So: the feature you want already
exists, just isn't wired into any of the currently-verified configs' XML — the missing piece is
just setting `rect_coeffs_file`/`xtrans_rect_file`/`ytrans_rect_file` to a saved path from a
previous run.

## 4. `newRectifyProcess.py` — not usable as-is

`_calculateImprovedMOSMaps` and `_calculateImprovedLongslitCoeffs` are literal `pass` stubs, so
`execute()` on longslit or MOS data hits its own `ERROR: Coeffs/Maps not found` branch and
disables the FDU on first real use — it cannot currently run end-to-end regardless of dataset.
The tracing methods that *do* have real code (`traceMOSContinuaRectification`,
`traceMOSSkylineRectification`) use the same cross-correlate-every-few-pixels approach as the
original despite the docstring claiming "robust iterative tracing" — `use_bivariate_splines` is
checked but then the branch is a literal `pass` with a comment admitting it "is complex for a
prototype." One real idea worth keeping: separately configurable step sizes for continuum vs.
sky tracing (`mos_continuum_step_size`/`mos_sky_step_size`) — trivial to backport into the real
`rectifyProcess.py` independent of anything else here. Bottom line: treat this file as
throwaway scaffolding, not a rewrite to build on.

## Open questions for Craig
1. OK to try `mos_mode="use_slitpos"` on `specBench.xml` as a first, code-free test of the
   partial-continuum-coverage question, before I write the nearest-slit-fallback code?
2. Want the `UnivariateSpline` swap prototyped as a real (feature-flagged, opt-in) option in
   `rectifyProcess.py`, or as a standalone script first validated against a real dataset's
   saved trace coordinates before touching the process file at all?
3. Is `newRectifyProcess.py` meant to be finished, or should it be deleted/archived now that
   it's confirmed non-functional? (Not deleting it myself since Gemini-authored `new*` files
   are explicitly out of scope per earlier project convention.)
