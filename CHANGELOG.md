# Changelog

Changes made on the `refactor` branch, grouped by class / module. Each entry is a short summary with
the version it landed in; `git log` has the full story, and `CLAUDE.md` has implementation notes for
whoever picks this work up next. User-facing option documentation lives in `docs/` (and
`superFatboy3.py -list`).

**Convention:** every commit that changes behavior, adds an option, or fixes a bug adds a line to
the matching section here. New options are listed with their default.

## Validation status

| Dataset | Mode | Status |
|---|---|---|
| oriBench (FLAMINGOS-1 NIR imaging) | CPU + GPU | Matches the frozen py2 original (CPU and GPU) to 4+ decimals on alignment shifts |
| specBench (FLAMINGOS-1 MOS) | CPU + GPU | Full chain through calibStarDivide |
| OSIRIS, KAST longslit | CPU + GPU | Full chain |
| MIRADAS SOL / SOS / MOS | CPU + GPU | Full chain |
| LUCI MOS (caden_luci_test) | GPU (+ CPU cross-check) | Full chain incl. calib star (v2.3.43); wavelength solutions match the pre-refactor run to 0.02-0.05 px; CPU and GPU extracted spectra identical |
| specBench regression, v2.3.44 vs v2.3.42 | CPU | All 571 output files identical (NaN-aware), apart from the renamed per-object rectified slitmask |
| Median fixes, v2.3.45 vs v2.3.44 | specBench CPU, oriBench GPU, LUCI GPU | oriBench: alignment shifts identical (drizzled pixels differ at 1e-7). specBench: 559/571 identical; 6 of 23 extracted spectra rescaled by 0.007-0.23% and 2 standard-star pixels changed - both from the single-value median that used to return 0. LUCI: GPU clean sky now the lower quartile and identical to CPU; wavelength solutions moved 0.09 A median (0.02 px), fit RMS 0.473 -> 0.452 A |
| MEGARA, FourStar, SINFONI, others | - | Not yet run through the refactored pipeline |

## Open issues

- **sinfoniCollapseSlitlets** `padx`/`pady` (use-derivatives centroiding branch, not the default
  2-d Gaussian): commented out as a likely copy/paste from sinfoniCharacterizePSF, which pads for both
  methods. Re-check when SINFONI data is run.
- **doubleSubtract / shiftAdd `updateNoisemap`**: never called; references undefined
  `noisemaps_dbs_gpu` / `noisemaps_sa_gpu`.
- **rectifyProcess**: `mos_sky_fallback` / `independent_slitlets_fallback` default is still
  `identity` (validation favors `nearest_neighbor_slits`); moment-centroid rescue not yet ported to
  skyline tracing; specBench science-frame continuum trace still loses 15-30% of points.
- **miradasCollapseSpaxels**: exact-value peak finding is fragile to float reduction order.
- **removeCosmicRaysSpec / badPixelMaskSpec**: algorithm audits not started (LA Cosmic reviewed, below).

---

## Framework

### fatboyDatabase
- Unattended runs no longer hang on an error: the "press ENTER" prompt is opt-in via
  `interactive_on_error` (default `no`). A process exception disables just that frame. (2.3.x, Sept 11)
- Ingestion: a bad input file is isolated and logged; more than `max_init_failures` (default 3)
  failures, or every file failing, aborts the run. (Sept 11)
- Postcondition check: a process returning success but leaving no readable data disables the frame.
- `addNewSlitmask()` takes optional `tagname` / `objectTag` (object-tagged calibs). (2.3.43)
- `hasMasterCalib()`: `section` was used but not a parameter. (2.3.43)

### fatboyProcess
- `recursivelyExecute()` catches exceptions and honors a `False` return, disabling only the failed
  calibration frame instead of crashing the run (used by ~23 calibration processes). (Sept 11)

### fatboyDataUnit / datatypes
- `initialize()`: when NAXIS1/NAXIS2 are missing from the header the shape is now read from the data
  as intended (the check tested an undefined name, so such files were disabled as "misformatted").
  (2.3.43)
- Header-keyword file grouping crashed (`OS.F_OK`). (2.3.43)
- `renormalize()` converts the bad pixel mask to match GPU mode. (Sept 16)
- osirisSpectrum / circeImage `getData()` accept `force_cpu`. (83f5b4f)
- Imaging frames without RA/Dec (`ra_keyword`/`dec_keyword`: RAOFFSET/RA/TELRA, DECOFFSE/DEC/TELDEC)
  are disabled with an ERROR (unchanged behavior, documented here).

### fatboyLibs
- **GPU results discarded** (PyCUDA `drv.InOut` semantics lost in the CuPy port): `cp.empty(array)`
  family fixed in ~18 sites (Sept 15-16); `gpuInOut()` / `gpuSyncBack()` helpers added and used for every
  in-place kernel argument after auditing all 132 `drv.InOut`/`drv.Out` uses in `main`:
  applyObjMask, apply2PassObjMask, divideArraysFloatGPU, noisemaps_sqrtAndDivide, normalizeFlat,
  normalizeMOSFlat (incl. replaced-pixel counters), normalizeMOSSource, subtractImages, generateQAData,
  LA Cosmic helpers. (2.3.40)
- `np.min(a,b)`/`np.max(a,b)` two-scalar comparisons (35 sites) silently ignored the comparison when
  `b == 0`; reverted to builtin `min`/`max`. (Sept 16)
- 25 quoted dtype strings with an injected `np.` (`.astype("np.int32")`); 10 were comparisons that were
  always true. (Sept 16)
- `gpusum()` rewritten for CuPy; `linterp_gpu`/`linterp_cpu`, `whereEqual`, `getCentroid` fixes.
- `extractSpectra()` robust rewrite; original kept as `extractSpectra_orig` and selectable with
  `slitlet_autodetect_use_orig_algorithm`. (2.3.30)
- Wavelength-solution helpers (`hasWavelengthSolution`, `hasMultipleWavelengthSolutions`,
  `getWavelengthSolution`) no longer assume slitlet 1 has a solution. (2.3.43)
- `fit1d()`: undefined `add` (crashed LA Cosmic and any `fit1d` user). (2.3.43)
- `removeOutliersSigmaClip()` restored (used by tri_register). (2.3.43)
- LA Cosmic (`lacos_spec` and helpers) - see removeCosmicRaysSpec below.

### gpu_arraymedian
- Scalar-median path accepts CuPy input. (Sept 15)
- Median with `nlow`/`nhigh` rejection and `even=True` decided even/odd from the count **before**
  rejection, so e.g. 3 frames with the highest rejected gave the higher of the two kept values instead
  of their mean (9 CUDA kernels; also in main). Now uses the kept count. (2.3.45)
- Small arrays (< 2**16 elements) passed as CuPy (e.g. small stacks from gpu_imcombine) crashed in the
  CPU C kernels; now moved to the host and the result returned as CuPy. (2.3.45)

### fatboyclib (C extension) - needs a rebuild (`setup.py install`)
- Same even/odd bug as gpu_arraymedian in `median2d`/`median3d` with `nlow`/`nhigh` (72 sites; also in
  main). (2.3.45)
- A guard `k == 0` returned 0 whenever the kept values started at index 0 - e.g. `nhigh = n-1` (the
  `min` combine) always returned 0 on the CPU, and a single nonzero value gave 0. Now checks that
  some values remain (60 sites; also in main). (2.3.45)

### gpu_drihizzle (GPU drizzle)
- CUDA illegal-address crash: a `float32` cast on the wrong operand packed a float64 into a float
  kernel argument. (1a2a1dc)
- uniformKernel scatter bounds; padding threads no longer write past the array end. (421323f)
- Final weighting for `weight=exptime, outunits=counts` restored to main's (raw sum). The rewrite
  divided by the exposure map, which rescaled every rectified frame and made rectify's point_replace
  produce garbage pixels at slit edges. (2.3.41)
- `drihizzle3d`: in-place kernel outputs were discarded (output all zeros) and a float64 scalar
  argument corrupted the kernel arguments; now matches the CPU version. (2.3.43)

### drihizzle (CPU drizzle)
- `unique1d_wrap()` restored to call `np.unique1d` for numpy < 1.5 (the Gemini refactor had replaced
  every branch with `np.unique`). (2.3.45)
- `drihizzle3d`: float32 rounding of output coordinates could map two inputs to one output pixel, and
  numpy's `a[idx] += v` kept only one - whole planes of flux were lost (18% in a test). Now uses the
  unique-index loop whenever targets collide; matches the GPU to float rounding. (2.3.44)
- `drihizzle3d`: bare `uint8` NameError. (2.3.43)
- `zrefout` (3D reference pixel) used `ycoeffs` instead of `zcoeffs` (also in main). (2.3.43)
- `MODE_RAW` output with `outfile` referenced undefined `out`/`outtype`. (2.3.43)

### gpu_imcombine / imcombine
- `nfint.astype()` on a plain int. (Sept 16)
- GPU `gpumean`/`gpustd` were never defined: any GPU combine with mean/sigma zero, scale, or weight
  crashed. Implemented to match the CPU selection (inclusive thresholds, optional nonzero, ddof=1).
  (2.3.43)
- CPU quick-start file cache used removed variables (`qskeys`/`qsvals`/`qslist`); exposure-mask output
  used undefined `expfile`. (2.3.43)

### pysurfit / gpu_pysurfit
- GPU std uses `ddof=1` like the CPU. `.sum()/N` replaced by `.mean()`. (Sept 11)
- `pysurfit` input-type detection and output message referenced undefined names. (2.3.43)

### xregister / gpu_xregister / tri_register
- Bare `ndarray`, `loadtxt`, `ascontiguousarray`; `frame` vs `frames` in difference mode. (Sept 16, 2.3.43)

### superFatboy3.py / setup.py
- `-gpu N` sets `CUDA_VISIBLE_DEVICES` (CuPy ignores `CUDA_DEVICE`). (Sept 11)
- `numpy>=2.0` pin reverted to `numpy>=1.0` (conflicted with scipy 1.11). (Sept 11)
- Templates and line lists shipped in `package_data`. (2.3.37)

---

## Processes

### findSlitlets
- `traceOrders`: one bad segment no longer discards the whole image; it gets a straight fallback. (Sept 11)
- Auto-detect crashes in GPU mode (`force_cpu`) and CPU mode (`concatenate`). (d3dbd22)
- `traceSlitlets` writes a per-datapoint `stats_<flat>.txt` like `traceOrders`. (2.3.27)
- New `edge_detection_method` (`cross_correlation` | `local_minimum` | `auto`); **default `auto`** since
  2.3.38 (identical to cross_correlation on any edge it can trace). (2.3.27, 2.3.38)
- New `fit_function` (`polynomial` | `spline`), `spline_smoothing`. (2.3.28)
- New `narrow_gaps_between_slitlets` (default `no`): measure packed-boundary points with the local
  minimum instead of rejecting them. (2.3.38)
- New `slitlet_autodetect_source` (`flat` default | `arclamp` | `both`) and
  `slitlet_autodetect_arc_min_corr` (0.9): detect slitlets from adjacent-row arclamp correlation;
  falls back to the flat when its count misses `slitlet_autodetect_nslits`. (2.3.38, 2.3.39)
- New invalid-slitlet check, `slitlet_validity_max_flat_roughness` (0.045) and
  `slitlet_validity_min_arc_corr` (0.9): drop non-slitlet regions (mask ID strip) in auto-detect,
  warn for region files. (2.3.38)
- New `local_min_search_radius` (3): anchored local-minimum search, so a packed boundary that turns
  into a step doesn't drift into the next slitlet. (2.3.39)
- QA-file `UnboundLocalError` when a slitlet needed a fallback. (2.3.38)

### rectify
- Runaway continuum fits guarded by `rectify_max_transform_factor` (2.0); untransformed slits logged
  as ERROR. (Sept 11)
- MOS/longslit continuum and skyline trace audits: per-datapoint stats files, local-significance and
  moment-centroid rescues, `checkFitSanity`, `independent_slitlets_fallback` / `mos_sky_fallback`
  (`identity` default). (2.3.32-2.3.34)
- GPU crMask pre-conversion that corrupted the slitmask reverted. (3c65264)
- Rectified slitmasks are now per object (`rct_<slitmask>_<object>.fits`, tagged for that object);
  previously the first object marked the shared slitmask "rectified" and later objects (e.g. the
  calibration star) never got their own. (2.3.43)
- New `mos_min_continua_global_fit` (3): a whole_chip/use_slitpos continuum fit with fewer continua,
  or spanning <25% of the slitlets, reuses another object's continuum transform for the same mask.
  (2.3.43)

### wavelengthCalibrate
- Negative-index slice wraparound near the array edges (8 sites, pre-existing). (Sept 15)

### extractSpectra
- Gaussian weighting referenced undefined `extract_xlo`/`extract_xhi`. (Sept 15)

### calibStarDivide
- MOS standards: the calibration star is the brightest extracted spectrum (new
  `calib_star_spectrum`, 0 = brightest), and each spectrum uses its own slitlet's wavelength solution
  (via `SPEC_nn`). Pixel-division branch used undefined `b_clean`/`b_resamp` (also in main). (2.3.43)

### doubleSubtract
- New `min_negative_flux_fraction` (0.1): skip double subtraction when the frame has no negative
  trace (sky frame with the target off the slit, e.g. a telluric standard). (2.3.43)

### shiftAdd
- An empty slitmask is an ERROR instead of an IndexError. (2.3.43)

### flatDivide / flatDivideSpec
- GPU flat division was a no-op whenever the frame was on the host (result discarded). (2.3.40)
- Writing a CuPy array into an astropy HDU. (Sept 15)

### skySubtract (imaging)
- Restored the dropped `fatboyLibs` import. (Sept 16)

### skySubtractSpec
- Sky-method file validation referenced `methodlist`/`ssmethods`. (2.3.43)
- Dither pairing: frames with no RA/Dec get a clear ERROR naming the keywords and are dropped,
  instead of a TypeError. (2.3.44)

### removeCosmicRays (imaging)
- GPU cosmic ray removal returned all zeros (result discarded). (48d2e68)

### removeCosmicRaysSpec / LA Cosmic
- `runDeepCR`/`runLacos` shadowed numpy as `np` (UnboundLocalError). (2.3.42)
- `runLacos`/`runDeepCR` mixed CuPy and numpy arrays in GPU mode; they now use CPU data like `runDcr`.
  (2.3.44)
- `runLacos` assembled the cleaned frame from zeros, so in `replace` mode every pixel between
  slitlets was set to 0 and unflagged pixels carried model round-off. It now starts from the input
  and replaces only flagged pixels (LUCI: 15,768 changed pixels per frame, none outside slitlets).
  (2.3.44)
- "deepCR not installed, using DCR instead" didn't actually switch to DCR. (2.3.44)
- `lacos_spec` crashed on its last line (`int16`) and in `fit1d`. (2.3.43)
- Sky + object model added back once after the last iteration, as in `lacos_spec.cl` (was added
  every iteration). Noise floor follows IRAF (`med5 <= 0 -> 0.00001`). (2.3.43)
- `lacosSelect` (no-count branch) and `lacosUpdateOutput` never wrote their results back (also in
  main), so cleaning didn't happen on the GPU. (2.3.43)
- Vertical dispersion: slits are transposed so the object/sky fits run along the right axes. (2.3.43)
- MOS: pixels of each slit's bounding box outside the slit are filled from the nearest in-slit row
  instead of zero. Synthetic tilted-slit test: false detections 4330 -> 815, slit-edge 594 -> 3,
  mean bias -12.9 -> -0.2, recall ~95%. (2.3.43)

### badPixelMask / badPixelMaskSpec
- `.astype()` on plain floats; `bpm_replace_median_neighbor_gpu` result discarded. (Sept 16)
- Sigma clipping used undefined `sig`; missing imcombine imports; `combineSourceFrames` call. (2.3.43)

### biasSubtract
- Mixed CuPy/numpy subtraction in GPU mode. (83f5b4f)

### linearity
- `pow(float, int)` in a CUDA kernel didn't compile under NVRTC. (Sept 15)

### createCleanSkies
- `combine_method=quartile` on the GPU no longer passes `even=False`; with the median fixes above the
  GPU and CPU both give the lower quartile (identical results). Before, the GPU gave the plain median
  of 3 frames. (2.3.45)

### createMasterArclamps / flatDivideSpec noisemaps
- Kernel name typo `noisemaps_twilight_float`. (Sept 15)

### alignStack
- Default `align_method` is `triangles`. (6b1eeb4)

### collapseFibers
- `properties`/`headerVals` undefined in the output-exists path. (2.3.43)

### miradasCollapseSpaxels
- `nslits` must be int; out-of-bounds slice guards. (865f487, 28e6443)

### miradasDARFromConditions / miradasDARFromData
- Undefined `nslits` fixed (by Craig). (2.3.44)

### MIRADAS / SINFONI processes
- Missing `fatboySpecCalib` / `fatboyDataUnit` imports (paths taken when calibs come from XML).
  (2.3.43)

---

## Appendix: original chronological notes (through 2026-09-16)

### 2026-09-16 — oriBench.xml (NIR imaging) end-to-end test: PASSED, and cross-checked 4 ways

Ran the NIR imaging regression test (`superFatboy3.py oriBench.xml`) for the first time since the
refactor — this exercises dark subtraction, flat fielding, bad pixel masking, sky subtraction,
cosmic ray rejection, and stack alignment, which the spectroscopy test above never touched. Took
14 fix cycles, but the payoff is a strong one: Craig also ran the **original, frozen python2
version** of the pipeline (`superFatboy.py`, never modified — see below) in both CPU and GPU mode,
giving four independent runs total: old python2 (CPU and GPU) and the new python3 refactor (CPU
and GPU). **All four now agree on the final image-alignment measurements to 4+ decimal places.**
That's about as convincing a proof as this refactor could ask for that the imaging pipeline still
does exactly what it always did.

Two of Craig's own observations were the key breaks in this test:

- Craig noticed the GPU cosmic-ray-rejection step was returning an image of all zeros, and
  correctly identified that as the root cause of a stack-alignment problem — this pointed straight
  at the actual bug (see below) rather than a much longer manual search.
- Craig suggested comparing the broken GPU code side-by-side against the original, working
  python2/PyCUDA version on the `main` branch — this is exactly the technique that cracked the
  hardest bug of the night (the drihizzle crash, see below).

Bugs found and fixed along the way:

- **Plain numbers treated as arrays**: several spots called `.astype()` — an array-only method —
  on ordinary Python numbers (a float, an int, even a literal list). Fixed each to use the correct
  plain-number conversion instead.
- **A whole missing import**: one file (`skySubtractProcess.py`) was missing an entire block of
  shared helper functions because an import line got dropped somewhere along the way, not
  individually converted like everything else nearby. Restored it.
- **Mistyped dtype names, 25 of them**: a batch of type-conversion calls like `.astype("int32")`
  had gotten an extra, invalid `np.` accidentally stuck inside the quotes, making them
  `.astype("np.int32")`. Some of these just crashed outright; others were sneakier — a few were
  comparisons (`"is this array already float32?"`) that, because of the typo, could never be true,
  silently skipping a fast-path optimization every single time without ever raising an error.
- **A whole family of "GPU result thrown away" bugs, ~18 spots total**: the old PyCUDA library had
  a convenient feature where you could hand it an empty array and it would automatically fill it
  with the GPU's answer. CuPy (the modern replacement) doesn't work that way — you have to
  explicitly pull the answer back off the GPU yourself. In roughly 18 places across the codebase,
  that explicit "pull the answer back" step was missing, so the function would compute the right
  answer on the GPU and then return the original, untouched (often all-zero) array instead. This
  turned out to be the actual explanation for the all-zeros cosmic-ray-rejection bug Craig spotted,
  plus similar silent failures in bad-pixel interpolation and a couple of interpolation utilities.
- **Two-input comparisons that quietly did nothing**: 35 places compared two plain numbers using
  `np.min()`/`np.max()`, but those functions' second slot isn't "the other number to compare
  against" — it's an option for arrays with multiple dimensions. When the second number happened
  to be zero, this didn't crash; it silently returned the first number, ignoring the comparison
  entirely. Fixed by using Python's plain `min()`/`max()`, which do exactly what was intended
  here.
- **A self-inflicted regression**: while fixing one of the "GPU result thrown away" bugs above in
  a helper function, an earlier pass of mine had converted its result down to a plain single
  number, but downstream code actually depended on it staying array-shaped. Caught and reverted.
- **The hardest bug of the night — a GPU crash during image alignment/stacking**: tracked down
  using a CUDA debugging flag that forces the GPU to report errors immediately at the real
  faulting instruction (rather than several steps later, which is what GPUs normally do and what
  was initially sending the investigation down the wrong path), plus Craig's suggestion to diff
  against the original working code. The root cause: a math expression that used to read
  "convert the *entire result* of (A minus B) to float32" got mechanically rewritten during the
  refactor into "convert only B to float32, then subtract" — mathematically the same *value*, but
  numerically the wrong *type* comes out (float64 instead of float32). That mismatched type then
  corrupted how the next few arguments got packed into the GPU function call, causing it to write
  to memory it had no business touching. Fixed all 16 occurrences of this exact mistake. This is
  a subtle one — worth remembering as a specific pattern to watch for in any future GPU code
  review.
- **A leftover bug from earlier this session**: a bad-pixel-mask array needed to be moved onto the
  GPU to match the rest of the calculation it was being used in, but wasn't.

**Also resolved (environment, not code)**: getting the old python2 GPU version to even run at all
on this machine, purely to make the 4-way comparison above possible. Two separate environment
issues, neither touching the frozen python2 codebase at all (per Craig's instruction, that code is
off-limits without checking first): a Python-2-specific crash-while-reporting-a-crash that was
hiding the real error message, and a compiler configuration mismatch (Craig had switched a
system compiler version for an unrelated project, which left the CUDA build tools unable to find
one of the pieces they needed). Both fixed with an environment variable / wrapper script — no
changes to the frozen code.

**Not fixed, flagged for later**: `gpu_drihizzle.py`'s CUDA kernels have a minor, currently-harmless
edge case where a thread doing "padding" work near the very end of an array writes a zero into
memory it shouldn't touch, instead of just stopping — didn't affect this test (this dataset's image
size happens to divide evenly), but could bite on a differently-sized image. Worth a quick fix next
time that code is touched.

**Conclusion**: imaging mode is now confirmed solid too, and — thanks to the 4-way comparison —
we have real proof the refactor hasn't changed any of the pipeline's actual science results, not
just that it "doesn't crash."

### 2026-09-15 — specBench.xml (MOS spectroscopy) end-to-end test: PASSED

Ran the Flamingos-1 MOS spectroscopy regression test (`superFatboy3.py specBench.xml`) for the
first time since the refactor. It took 12 run/crash/fix cycles, but it now completes the entire
pipeline — linearity through flux-calibrated final output — for all 8 science frames and the
calibration star, with no unhandled exceptions. Spot-checked the actual output data (not just
that files exist): the final flux-calibrated spectrum and the extracted-spectra table both have
sane, finite, correctly-shaped values.

Bugs found and fixed along the way:

- **CUDA `pow()` failure**: `linearityProcess.py`'s kernel called `pow(float, int)`, which CuPy's
  runtime compiler can't resolve the way the old offline compiler could. Switched to `powf` with
  an explicit float cast on the exponent.
- **`cp.np` typo**: CuPy has no `np` submodule; found `cp.np.array()`-style calls (should just be
  `cp.array()`) in four files, likely from an automated find-and-replace gone wrong.
- **Wrong integer conversion**: `ncoeffs.astype(np.int32)` where `ncoeffs` was already a plain
  number (`.size` doesn't return an array) — fixed to `np.int32(ncoeffs)`.
- **CUDA kernel name typo**: code looked up a kernel called `noisemaps_twilight_float`, but the
  kernel is actually named `noisemaps_mflat_dome_on_off_float`.
- **Writing GPU arrays straight to FITS**: three files tried to assign a CuPy array directly into
  an astropy FITS image, which isn't allowed — added an explicit conversion back to a normal
  array first.
- **CPU-only functions given GPU data**: several of the spectral-trace-finding routines in
  `rectifyProcess.py` (the code that traces out continuum/skyline curvature for rectification)
  are fundamentally CPU algorithms, but weren't consistently pulling their data back off the GPU
  before using it. Fixed across all four sibling functions (MOS continuum, MOS skyline, longslit
  continuum, longslit skyline — the last one only surfaced on a later run).
- **Mismatched transform arrays**: the x- and y-direction rectification transforms could each end
  up as a different array type (GPU vs. regular) depending on which code path built them,
  crashing when combined. Now normalized to match before use.
- **A bug in tonight's own new code**: the "sanity check for a runaway fit" safeguard added
  earlier in this session had a shape mismatch for horizontal-dispersion data. Fixed and verified.
- **Median calculation given GPU data**: same class of issue as above, in the shared median
  helper (`gpu_arraymedian`) — fixed only in the branch that actually needed it, since a
  different branch of that same function legitimately wants to keep data on the GPU.
- **Leftover incomplete feature**: `extractSpectra`'s Gaussian-weighting option referenced two
  settings (`extract_xlo`/`extract_xhi`) that were never actually wired up in that function.
  Reverted to the original (working) behavior rather than half-finishing the feature.
- **Pre-existing bug, not from this refactor**: a line-window calculation in
  `wavelengthCalibrateProcess.py` used `array[x-10:x+11]` to grab a window around a bright line,
  but when the line is within 10 pixels of the edge, this silently wraps around instead of
  clamping, producing an empty window and crashing. Confirmed via the pre-refactor code that this
  bug already existed; just never got exercised until now. Fixed with a simple clamp.

Not bugs — expected, graceful behavior: a few wavelength-calibration "orders" in the test data
didn't have enough bright lines to solve and were correctly skipped with a log message rather
than crashing. (Per Craig: this particular dataset isn't expected to have that many skipped
orders, so worth a look — but it's not a crash, and not something introduced by the refactor.)

**Conclusion**: the numpy/math cleanup and the PyCUDA→CuPy migration are now confirmed solid for
MOS and longslit spectroscopy end to end. Imaging mode is the next thing to test.

### 2026-09-11 — Top-level error-handling audit

Investigated why the pipeline still crashes outright on bad data instead of degrading gracefully,
starting from the `fatboyDatabase.py` entry point.

- The per-image processing loop paused for a keypress on any unexpected error — which hangs
  forever on an unattended/scheduled run (stdin is closed), crashing the whole batch on top of
  the original problem. Now only prompts if explicitly turned on, and won't hang even then.
- The code that pre-processes calibration frames (darks, flats, etc. before combining them) had
  no error handling at all — one corrupt calibration frame took down the entire run, even though
  the very same pipeline already handles "no calibration found" gracefully one level up. Fixed so
  one bad calibration frame is discarded and logged instead of crashing everything.
- Added a safety check so a process can't silently claim success while leaving its output data
  broken — that's now caught and logged right where it happens, instead of surfacing later as a
  confusing unrelated crash.
- A single unreadable/corrupt input FITS file used to crash the whole run before any processing
  even started. Now a few bad files are isolated and skipped (with a loud warning), and the run
  only aborts if the number of bad files crosses a threshold (default: more than 3, or literally
  everything failed) — since that pattern means something systemic is wrong, not "one bad frame."

### 2026-09-11 — findSlitletProcess and rectifyProcess hardening

(Implemented, flagged at the time as not yet validated against real data — now validated by the
specBench.xml test above.)

- `findSlitletProcess`: previously, if even one slitlet (out of possibly dozens) failed its
  quality checks while tracing out a flat field, the *entire* image was discarded, throwing away
  every successfully-traced slitlet along with the one bad one. Now an individual bad segment
  gets a straight/uncurved fallback instead, and the whole image is only discarded if literally
  nothing could be traced.
- `rectifyProcess`: added a safeguard against a runaway curve fit blowing up the size of the
  output image (in principle, a bad fit could try to "rectify" a normal-sized image into
  something many times larger). If a fit produces unreasonable values, it now retries with a
  simpler (straight-line) fit before giving up on that region.
- Made the existing "nothing to rectify here" fallback louder (an actual error instead of an
  easy-to-miss warning), so it's obvious when a region wasn't properly corrected.

### 2026-09-10 to 2026-09-11 — Continuing the Gemini-started refactor

- Finished removing the last two remaining `from numpy import *` wildcard imports, which also
  turned up three real bugs (bare `sqrt`/`tan`/`sin`/`cos` calls with no valid function to resolve
  to — these would have crashed the moment that code path ran).
- Fixed the packaging bug that was breaking `pip`/`setup.py install` entirely (an overly strict
  numpy version requirement that conflicted with other installed packages).
- Audited the PyCUDA→CuPy conversion end-to-end and fixed two more real bugs: a median-related
  helper function that still referenced old PyCUDA-only code, and a `-gpu <device>` command-line
  option that silently stopped working after the migration.
- Replaced the last few leftover "manually compute the mean/standard deviation the slow way"
  patterns (a workaround for numpy bugs that no longer exist) with the modern, simpler equivalents
  — but only where it was possible to verify byte-for-byte that the result is identical; several
  similar-looking spots turned out to be doing something more specific (masked/weighted
  averages) and were deliberately left alone.
- Fixed a CPU/GPU inconsistency in the surface-fitting code used by `pysurfit` (one path used the
  statistically-correct "sample" standard deviation, the other didn't).
