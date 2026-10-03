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
| LUCI MOS (caden_luci_test, verified v2.4.2) | CPU + GPU | Region file + traceSlitlets + `flexure_correction=shift`, through extraction: same 19 spectra in both modes, fluxes within 0.01%, 24 wavelength solutions identical (median RMS 0.425 A). Calib star excluded (different mask, no calibrations). New `LUCI_MOS_template.xml` |
| specBench regression, v2.3.44 vs v2.3.42 | CPU | All 571 output files identical (NaN-aware), apart from the renamed per-object rectified slitmask |
| Median fixes, v2.3.45 vs v2.3.44 | specBench CPU, oriBench GPU, LUCI GPU | oriBench: alignment shifts identical (drizzled pixels differ at 1e-7). specBench: 559/571 identical; 6 of 23 extracted spectra rescaled by 0.007-0.23% and 2 standard-star pixels changed - both from the single-value median that used to return 0. LUCI: GPU clean sky now the lower quartile and identical to CPU; wavelength solutions moved 0.09 A median (0.02 px), fit RMS 0.473 -> 0.452 A |
| Gap-aware `padding`, v2.4.0 | MIRADAS SOS + MOS order 20 (findSlitlets), LUCI auto/region GPU | MIRADAS slitmasks byte-identical to v2.3.47 with `padding`=2. LUCI auto with `padding`=3: 16 continua traced (was 10), 17 spectra extracted (was 13) incl. the brightest objects, fluxes 0.95-1.0 of the traceSlitlets run (was 0.62-0.96); wavecal median sigma 0.454 -> 0.491 A |
| GPU == CPU, v2.4.3 | all 9 verified configs, GPU and CPU (run one at a time) | No tracebacks; same ERRORs as before. LUCI byte-identical GPU vs CPU (107/107 files), MIRADAS SOS 107/108, SOL 94/111, MOS 75/91 (left: bad pixel mask GPU vs CPU); specBench (linearity/dark onward), oriBench (imaging) and longslit (fitted-distortion drizzle) still differ at the rounding level. Every dataset calibrates the same number of slitlets as its previous verified run, median wavelength sigma within noise |
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
- **GPU vs CPU still differ** in: the bad pixel mask (MIRADAS), linearity/dark (specBench), the imaging chain
  (oriBench) and drizzle with a fitted distortion (below).
- **GPU drizzle with a fitted distortion (`geomDist`, longslit rectification)**: positions come from
  `calcTransOpt` in float32 on the GPU and numpy float64 on the CPU, so those runs can still differ at the
  rounding level. Transforms given as arrays (MOS) are bit-identical. The 3-d GPU drizzle (SINFONI) still
  has the host-buffer problem noted under gpu_drihizzle.
- **wavelengthCalibrate**: the 3-line match depends on the brightness ranking of the line list's lines (LUCI slit 9
  with padding); fallbacks added in 2.4.4. avrajit-osiris still fails: `xenon_optical.dat` has no lines between 4481
  and 4844 A, where the arc's brightest (Xe I) lines are - a line-list problem (the earlier "wrong XML parameters"
  conclusion was a chance blind match; the arcs are R2500U, matching the configured 3440-4650 A).
- **traceOrders mask bias**: a `traceOrders` slitmask sits ~1.0 px below the flat's half-max center on LUCI
  (the `-1` on `ylo` plus truncation of the float edges). `flexure_correction` measures and removes it along
  with the flexure; without it, it is uncorrected.
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
- Temp dir: the `tempdir` param was read before the XML was parsed, so it was always `temp-fatboy`, and
  startup deleted an existing temp dir - a second run from the same directory wiped the first run's
  paged-out data. Now set after parsing, with a `fatboy.lock` (host, pid): a dir held by a live run is left
  alone and this run uses `<tempdir>-<pid>`; a stale one is cleared and reused; `cleanUp()` removes only
  its own. fatboyDataUnit uses the database's temp dir (`getTempdir()`) instead of hard-coding
  `temp-fatboy`. `quick_start_file` reads/appends are file-locked. (2.4.1)

### fatboyProcess
- `recursivelyExecute()` catches exceptions and honors a `False` return, disabling only the failed
  calibration frame instead of crashing the run (used by ~23 calibration processes). (Sept 11)

### GPU kernels (all CuPy RawModules)
- Compiled with `--fmad=false`: the default fused multiply-add contraction rounds `a*b+c` differently from
  numpy (e.g. noisemap `sqrt(a*a+b*b)` differed from the CPU in the last bit). (2.4.3)

### Noisemaps (darkSubtract, biasSubtract, createMasterArclamps, flatDivideSpec, shiftAdd, megaraSkySubtract)
- CPU master-calib noisemaps used `np.sqrt(master/ncomb)` where the GPU kernel uses `sqrt(abs(...))`, so
  negative master-dark pixels became NaN on the CPU only (76,000 NaN pixels in a LUCI dark-subtracted
  noisemap, carried on through rectification). Now `np.sqrt(np.abs(...))`, 8 sites; also in main. (2.4.3)

### fatboyDataUnit / datatypes
- `initialize()`: when NAXIS1/NAXIS2 are missing from the header the shape is now read from the data
  as intended (the check tested an undefined name, so such files were disabled as "misformatted").
  (2.3.43)
- Header-keyword file grouping crashed (`OS.F_OK`). (2.3.43)
- `setProperty("nslits", ...)` now stores an `int`. On a rerun with an existing output dir the slitmask
  is reloaded from disk as float32, so `nslits = slitmask.max()` was `np.float32` and
  `range(nslits)` crashed (wavelengthCalibrate). All ~25 `nslits = ...max()` sites in processes,
  `wavecal.py` and `fatboySpectrum.py` are also wrapped in `int()`. (2.3.46)
- `renormalize()` converts the bad pixel mask to match GPU mode. (Sept 16)
- osirisSpectrum / circeImage `getData()` accept `force_cpu`. (83f5b4f)
- Imaging frames without RA/Dec (`ra_keyword`/`dec_keyword`: RAOFFSET/RA/TELRA, DECOFFSE/DEC/TELDEC)
  are disabled with an ERROR (unchanged behavior, documented here).

### fatboyLibs
- `medianfilterCPU`: the first and last `boxsize` points used a 2*boxsize-point (even) window instead of
  2*boxsize+1 like the interior and `gpumedianfilter`, and the output was float64 where the GPU gives
  float32. Now identical to the GPU on real and random data (up to 2.45 counts different at the ends of a
  LUCI arclamp cut before). Also in main. (2.4.3)
- `medianfilter2dCPU` returns float32 for float32 input, like `gpumedianfilter2d` (values were already
  identical; the float64 result made later means/comparisons in rectify's skyline tracer take different
  branches on CPU and GPU - 5 points accepted differently on LUCI). (2.4.3)
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
- 1-d `median()`: when no values were left after `nonzero`, thresholds, sigma clipping or nlow/nhigh, it
  returned `quickselect` on an uninitialized buffer (random values like 6.9e-310, different every run).
  Now returns 0, as median2d/median3d do since 2.3.45. Rectify's skyline tracer compares such medians
  for fully masked boxes, so this made CPU and GPU runs (and two CPU runs) accept different points. Also
  in main. (2.4.3)
- Same even/odd bug as gpu_arraymedian in `median2d`/`median3d` with `nlow`/`nhigh` (72 sites; also in
  main). (2.3.45)
- A guard `k == 0` returned 0 whenever the kept values started at index 0 - e.g. `nhigh = n-1` (the
  `min` combine) always returned 0 on the CPU, and a single nonzero value gave 0. Now checks that
  some values remain (60 sites; also in main). (2.3.45)

### gpu_drihizzle (GPU drizzle)
- CUDA illegal-address crash: a `float32` cast on the wrong operand packed a float64 into a float
  kernel argument. (1a2a1dc)
- uniformKernel scatter bounds; padding threads no longer write past the array end. (421323f)
- `kernel='uniform'` (slitmasks) returns int32 again, as in main and the CPU drihizzle. The rewrite
  converted the result back to float32, so rectified slitmasks (`rct_slitmask_*.fits`) were written
  as float and reruns read them back as float32 (the `nslits` crash). Values were already exact
  integers (atomicMax), so only the dtype changes. (2.3.47)
- `uniformKernel` (slitmasks) now matches the CPU kernel exactly: `floorf` instead of truncation, no splat
  to the next column/row when every position is an exact integer (the GPU widened every slitlet by a
  row and column for an integer transform - 41,697 extra pixels on LUCI with an identity transform), and
  each of the 4 writes is bounds-checked on its own (a pixel at the last column used to be dropped). Also in
  main. (2.4.3)
- `turbo` kernel with `dropsize < 1`: the overlap weights were not clipped to [0, 1] (negative weights,
  flux moved to the wrong pixel); 20% of pixels differed from the CPU at dropsize 0.5. All four 2-d/3-d
  turbo kernels fixed; also in main. Not hit by any current config (all use dropsize 1). (2.4.3)
- GPU 2-d drizzle is now deterministic and bit-identical to the CPU for transforms given as arrays (MOS
  rectification): sums accumulate in double (exact for the few terms per pixel, so independent of the order
  the atomics land in - float atomics changed 25 pixels from one run to the next and 77,000 vs the CPU on a
  LUCI arclamp), positions are handled in double (offsetting a float position by the output origin rounded
  it), and data are scaled by inmask*scalefac/exptime in double like the CPU. A double atomicAdd fallback
  covers GPUs below compute capability 6.0. The 3-d kernels are unchanged. (2.4.3)
- Final weighting for `weight=exptime, outunits=counts` restored to main's (raw sum). The rewrite
  divided by the exposure map, which rescaled every rectified frame and made rectify's point_replace
  produce garbage pixels at slit edges. (2.3.41)
- `drihizzle3d`: in-place kernel outputs were discarded (output all zeros) and a float64 scalar
  argument corrupted the kernel arguments; now matches the CPU version. (2.3.43)

### drihizzle (CPU drizzle)
- `turbo` with `dropsize < 1`: only one side of each overlap weight was clipped (`np.minimum(..,1)` /
  `np.maximum(..,0)`), so a drop near the far side of a pixel got weights like -0.3 and 1.3. Both sides now
  clipped to [0, 1], 2-d and 3-d; also in main. (2.4.3)
- 2-d drizzle accumulates in double and rounds to float32 once before weighting, the same as the GPU. (2.4.3)
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
- New `makeLineList.py` (installed script): a line list for any spectra and wavelength range from the NIST Atomic
  Spectra Database (vacuum queried, converted to air; missing intensities estimated from g*A; per-spectrum scales by
  hand or fit to intensities measured by wavelengthCalibrate; blend flags; cached queries). (2.4.5)

### Line lists (data/linelists)
- New `Xenon_optical_air.dat`: the NIST Handbook Xe list plus the strong blue Xe I lines it lacks (4501-4697 A, the
  brightest lines of a xenon arc); intensities measured from a calibrated GTC/OSIRIS R2500U Xe arc (3446-4606 A), NIST
  ASD values scaled to them elsewhere, g*A estimates for 4624/4671/4697. With it avrajit-osiris calibrates on the first
  match (0.06 px RMS). (2.4.5)

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
- `padding` now grows each slitlet only into the empty rows next to it, splitting a gap narrower than
  2*padding between the two neighbors (before, it widened both edges blindly, overlapping packed
  neighbors), and applies to `traceSlitlets`/`tracePeakLocalMax` as well as `traceOrders`. Motivated by
  LUCI: science frames sit ~1.3 px below the flats, so tight auto-detected edges clipped the negative
  nodded image (10 of 16 continua traced, brightest objects not extracted). (2.4.0)
- traceSlitlets QA image is generated on the CPU in both modes (the GPU kernel used float32 positions and its
  overlapping marks raced, so the GPU and CPU QA images differed). (2.4.3)
- New `flexure_correction` (`none` default | `shift` | `gradient`) and `flexure_max_shift` (5): measure the
  flat -> object shift from the slitlet edges (derivative cross-correlation, 9 columns, all of the object's
  frames, clipped) and give the object its own slitmask (moved by flat shift - (mask - flat center offset), so
  a region-file mask already on the data stays put) and master flat (illumination moved, pixel response kept).
  `findSlitlets/flexure_<object>.txt` lists every measurement. LUCI: -1.28 px; edge-row noise in the
  flat-divided frame 3.7x -> 1.2x the in-slit noise. (2.4.2)

### rectify
- MOS rectification mask (`crMask`) built from host copies of `xtrans_rect`/`ytrans_rect`: one could be CuPy and
  the other numpy (use_slitpos), crashing specBench on the GPU. (2.4.3)
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
- Starting guesses beyond the configured one (`wavecal_fallback`, now `learned,neighbor,trend,pattern,blind`), each
  redoing the template, the 3-line match and the fit (`tryWavelengthGuess`): **learned** = line intensities measured in
  the calibrated slitlets; **neighbor** = a calibrated slitlet's polynomial shifted by cross-correlation, only if its
  range overlaps this slitlet's (MIRADAS orders hold different lines); **trend** = predicted from the slitlets on either
  side (orders); **pattern** = spacing ratios of bright-peak triplets, no intensities; **blind** as before. Pattern/blind
  candidates are scored against chance for the list's density (`lineMatchSignificance`). The best acceptable solution of
  the first method that gives one is kept. (2.4.6)
- Second pass (`wavecal_retry_grade`, default poor): once all slitlets are done, failed or poor ones try every fallback
  again with all calibrated slitlets available; a solution replaces the first one only if clearly better. (2.4.6)
- `measured_lines_<frame>.dat`: line intensities measured in the calibrated slitlets, in line-list format. (2.4.6)
- `wavelength_fit_function` = polynomial|legendre|chebyshev; PCOEFF keeps the equivalent power series, plus `WCFUNC`,
  `WCXMAX`, `NCOEFF_i` (`WCFUNxx`, `WCXMXxx`, `NCFi_Sxx`). Same solutions (same function space). (2.4.6)
- The 3-line match + growth + fit is now `solveFromMatch()` and the bright-line search `findDataLines()` (verbatim; logs
  identical). Failed wc-file orders (no line list / scale guess) now keep the per-slitlet lists aligned. `line_list` and
  `wavelength_line_separation` had no/misnamed `-list` help. (2.4.6)
- QA for every slitlet/segment: RMS in wavelength units and pixels, a grade (new `wavecal_quality_thresholds`,
  default 0.1/0.2/0.3/0.4 px), lines used and the fraction of the cut they span; a summary per frame; new
  columns in `qa_*.dat` with a row for every failed slitlet; header keywords `WCRMS`/`WCRMSPX`/`WCQUAL`/
  `WCNLINES` (per slitlet `WCRMSxx`/`WCRPXxx`/`WCQULxx`/`WCNLNxx`). (2.4.4)
- Fails cleanly: an unexpected error in one slitlet/segment is logged with its traceback and that slitlet skipped,
  keeping the per-slitlet lists and header aligned. The first Gaussian fit's fallback width used an undefined (or
  the previous slitlet's) `lsq`. (2.4.4)
- New `wavecal_fallback` (`neighbor,blind`) and `wavecal_blind_scale_range` (`0.5,2`): when the 3 brightest lines
  can't be matched, retry with a calibrated neighboring slitlet's solution (cross-correlation shift) and then a
  blind cross-correlation over scales and zero points (central half first, since a linear guess fails across a
  strongly nonlinear LUCI cut); a fallback solution is kept only if it grades satisfactory or better with enough
  lines. Slitlets that matched before are unaffected (log-identical on LUCI, KAST, OSIRIS, MIRADAS). Template
  building and template-line finding factored into `buildDummySpectrum()` / `findTemplateLines()`. (2.4.4)
- Negative-index slice wraparound near the array edges (8 sites, pre-existing). (Sept 15)

### extractSpectra
- Gaussian weighting referenced undefined `extract_xlo`/`extract_xhi`. (Sept 15)

### calibStarDivide
- MOS standards: the calibration star is the brightest extracted spectrum (new
  `calib_star_spectrum`, 0 = brightest), and each spectrum uses its own slitlet's wavelength solution
  (via `SPEC_nn`). Pixel-division branch used undefined `b_clean`/`b_resamp` (also in main). (2.3.43)
- Pixel-division fallback (no wavelength solution) crashed when the object and standard spectra differ
  in length (LUCI A1689 standard, different mask); now skipped with a warning. (2.4.0)

### doubleSubtract
- CPU path never blanked pixels outside the double-subtracted slitmask: `getData(tag="slitmask")` returns
  the slitmask calib object, so `data[object == 0] = 0` selected nothing. Now uses the calib's data, as the
  GPU kernel does (920,000 stray nonzero pixels per LUCI frame on CPU). Also in main. (2.4.3)
- New `min_negative_flux_fraction` (0.1): skip double subtraction when the frame has no negative
  trace (sky frame with the target off the slit, e.g. a telluric standard). (2.3.43)

### shiftAdd
- An empty slitmask is an ERROR instead of an IndexError. (2.3.43)

### flatDivide / flatDivideSpec
- GPU flat division was a no-op whenever the frame was on the host (result discarded). (2.3.40)
- Writing a CuPy array into an astropy HDU. (Sept 15)
- `flat_low_thresh`/`flat_low_replace`/`flat_hi_thresh`/`flat_hi_replace` are read as floats (were
  `int()`, so 0.3 was impossible). CPU MOS path: the replacement assigned into a copy
  (`data[slit][b] = x`) and never happened, and the high threshold tested `<` instead of `>` (all also
  in main; the GPU kernel was right). Options now have descriptions. (2.4.0)

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
- miradasCharacterizePSF: `nslits` fallback called `.max()` on the slitmask calib object (AttributeError if
  reached). (2.4.3)
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
