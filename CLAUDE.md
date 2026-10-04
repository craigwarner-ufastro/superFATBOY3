# Project: superFATBOY3

## Context

This refactor was started by Gemini CLI (see `GEMINI.md` for the original brief) and is being
continued here. Read `GEMINI.md` first — it lays out the four refactor goals (drop
`from numpy import *`, drop numpy-bug-era tricks like `.sum()/N`, migrate PyCUDA to CuPy,
improve error handling) and the specific algorithms flagged for later improvement
(findSlitletProcess, removeCosmicRaysSpecProcess, rectifyProcess, wavelengthCalibrateProcess).
Work happens on the `refactor` branch, one commit per meaningful change.

## User documentation now exists (docs/, added v2.3.36)

The documentation GEMINI.md asked for is written, as Markdown (GitHub renders it natively; raw `.html` in a repo is
shown as source) under `docs/`: `quickstart.md`, `instruments.md` (validated-instrument table + template list),
`xml-guide.md`, `processes/` (index, imaging, spectroscopy, instrument-specific), `miradas.md`, `api.md`, and
`options-reference.md`. The last one is **generated** - after adding or changing any process option, rerun
`python3 docs/gen_options_reference.py` (it shells out to `superFatboy3.py -list`) and, if the option matters to users,
mention it in the matching `docs/processes/*.md` page. When a new instrument gets a both-modes-passing config, add a row to
the validated table in `docs/instruments.md` and `docs/quickstart.md`'s template table. The old HTML docs live in an
untracked, git-ignored `html/` folder (copied in by the user for reference; never commit it). The code examples in
`docs/api.md` marked *(tested)* were actually run (custom process via `processdir`, custom datatype via `datatypedir`,
Python-API `fatboyDatabase(...).execute()`); rerun them if the fatboyProcess/fatboyDatabase API changes.
(The EMIR/OSIRIS `overwite_files` typo, the MIRADAS `wc_*_new.xml` template names, and `package_data` missing templates/linelists were all fixed in v2.3.37.)
Since v2.4.5-2.4.7: line lists and `makeLineList.py` are documented in `docs/instruments.md`, the standalone `wavecal`
module in `docs/api.md` (its example is *(tested)* on the LUCI arc), and the wavelength-calibration fallbacks, second
pass and fit functions in `docs/processes/spectroscopy.md`.

## Orientation for anyone writing documentation on superFATBOY (read this first)

This section is a summary for a Claude session that has **not** been doing this refactor work and
needs to get oriented fast — e.g. to write the documentation GEMINI.md itself asked for (see
below). Everything below this is distilled from ~3 weeks of session logs (the rest of this file,
in chronological order) and the project's auto-memory; if you need the full story behind any
claim here, search this file or the memory files (`project_algorithm_audit_list`,
`project_verified_configs_folder`, `project_miradas_collapse_spaxels_investigation`) for the topic.
As always, verify anything specific (a line number, a file's current contents) against the live
code before asserting it as fact — this is a point-in-time summary, not living state.

**What superFATBOY is.** A general-purpose astronomical data-reduction pipeline framework with
"processes" (one class per reduction step — dark subtraction, flat fielding, cosmic ray removal,
slit tracing, wavelength calibration, etc.) chained together via an XML config per dataset,
supporting a dozen+ imaging and spectroscopic instruments (FLAMINGOS-1, OSIRIS, KAST, MIRADAS,
EMIR, MEGARA, LUCI, SINFONI, and others — see `superFATBOY/fatboyProcesses/` for the ~80 process
files and `superFATBOY/data/templates/` for per-instrument XML templates). It was originally
written in Python 2 with PyCUDA for GPU acceleration.

**What this refactor is.** Started by Gemini CLI (`GEMINI.md` is its original brief — read it, it's
short) and continued here on the `refactor` branch (one commit per meaningful change, version
bumped in `setup.py`/`superFATBOY/__init__.py` every commit — see
`feedback_version_bump_per_commit` memory). Four original goals: (1) remove `from numpy import *`/
`from math import *` and disambiguate the resulting bare names, (2) remove numarray-era `.sum()/N`-
instead-of-`.mean()` tricks (but keep `arraymedian`/`gpu_arraymedian`'s own hand-rolled quickselect,
that's intentional), (3) migrate PyCUDA → CuPy, (4) improve error handling. GEMINI.md also flagged
four specific algorithms as needing improvement beyond straight translation: `findSlitletProcess`,
`removeCosmicRaysSpecProcess`, `rectifyProcess`, `wavelengthCalibrateProcess`. **GEMINI.md's own
"Documentation" section is almost certainly why you're reading this** — it asks for exactly this
kind of writeup: a main doc (install/run/XML style guide), a processes doc (or split into
imaging.md/spectroscopy.md), and a miradas.md. It also points to now-dead original HTML docs at
`/home/cwarner/FATBOY/superFATBOY/` and `/home/cwarner/FATBOY/miradas/` as source material, and
notes `superFatboy3.py -list` prints every process and its options live from the current code —
probably the fastest way to get an accurate, current options reference for a processes doc.

**Current maturity — what's actually been run and confirmed working.** All four goals are done for
the translation/migration part of the work (goals 1-3 complete tree-wide; goal 4's top-level
framework pass is done, see the error-handling section below). Beyond translation, real end-to-end
runs (not just "compiles") have validated:
- **Imaging**: `oriBench.xml` (FLAMINGOS-1 NIR imaging) — cross-validated 4 ways (py2 CPU/GPU, py3
  CPU/GPU) agreeing to 4+ decimal places on final alignment shifts. Strongest validation this
  refactor has.
- **MOS spectroscopy**: `specBench.xml` (FLAMINGOS-1) — full chain linearity→...→calibStarDivide,
  both CPU and GPU.
- **Longslit spectroscopy**: OSIRIS (`sarik_osiris.xml`/`avrajit-osiris.xml`) and KAST
  (`sarik_quack1.xml`/`sarik_quack3.xml`), both CPU and GPU.
- **MIRADAS (IFU-fed-by-MOS-slits)**: all three modes — SOL, SOS, MOS — both CPU and GPU. This one
  had the longest bug tail (see `project_miradas_collapse_spaxels_investigation` memory for the
  full bisection story) before landing clean.
- **LUCI MOS** (`caden_luci_test.xml`, v2.4.2): region file + traceSlitlets + flexure correction,
  through extraction; GPU and CPU byte-identical since v2.4.3. `LUCI_MOS_template.xml`.
- **Not yet run through the refactored pipeline at all**: MEGARA (fiber-fed spectrograph — some
  process-level review happened, see the algorithm-audit-list memory, but no real end-to-end
  pipeline run), FourStar, SINFONI, and whatever other instruments have templates/data but no
  session log entry here. Don't assume these work; there's no evidence either way yet.
- `/home/cwarner/work/xml/verified/verified_configs.md` is the authoritative list of
  known-both-modes-passing configs, each with a matching instrument template (also mirrored into
  `superFATBOY/data/templates/`). If a dataset/instrument isn't in that table, treat it as untested.

**Codebase orientation.**
- Entry point: `superFatboy3.py` → `fatboyDatabase.py` (`initializeAll()` ingests raw FITS into
  `fatboyDataUnit` (FDU) objects, `executeProcesses()` runs the configured process chain per FDU).
- Each reduction step is a `fatboyProcess` subclass in `superFATBOY/fatboyProcesses/`; options are
  read via `self.getOption(name, tag)` with defaults registered in `setDefaultOptions()`
  (`self._options.setdefault(...)` + a paired `self._optioninfo.setdefault(...)` human-readable
  description — this is what `-list` prints).
- CPU/GPU dual-mode throughout: `fdb.getGPUMode()` gates numpy (`np`) vs CuPy (`cp`) code paths;
  many core array ops live in `fatboyLibs.py`/`gpu_*.py` sibling pairs (e.g. `gpu_arraymedian.py`).
  `force_cpu=True` on `fdu.getData(...)` forces a CPU-mode fetch even in GPU-mode runs, used by the
  several trace/fit functions that are inherently CPU-bound (small-array medians, `np.correlate`,
  etc. — see item 6 of the specBench bug list below for the exact function list).
  - **Mechanical-refactor bug shapes to watch for in code you haven't audited yet**, all real and
    found this session (details in the specBench/oriBench bug lists below): `cp.empty(existing_array)`
    meant-to-be-a-device-copy (CuPy has no PyCUDA-style `drv.Out()` auto-copyback, this silently
    returns an unmodified/zero host array instead of erroring); `np.min(a, b)`/`np.max(a, b)` used
    to compare two scalars (the 2nd positional arg is `axis`, not a 2nd value — silently wrong, not
    a crash, when that arg happens to be `0`); a cast landing on the wrong operand in a mechanical
    `float32(a - b)` → `np.float32(a - b)` rewrite (`a - np.float32(b)` instead) — silently corrupts
    a CuPy `RawKernel`'s packed argument buffer rather than raising a Python-level error.
- `from superFATBOY.fatboyLibs import *` (and similar local wildcard imports, ~70 files) is this
  project's own namespace convention, not a numpy/math collision — intentionally untouched by goal
  1's cleanup. `arraymedian`/`gpu_arraymedian` (hand-rolled CPU/CUDA quickselect) are deliberately
  kept per GEMINI.md, not replaced with `np.median`.
- `new*Process.py` files (`newFindSlitletProcess.py`, `newRectifyProcess.py`) are Gemini's
  experimental rewrites, explicitly out of scope — left as untracked reference, not maintained, not
  wired into any real config. Don't assume they work; `newRectifyProcess.py`'s coefficient-building
  methods are literal `pass` stubs.
- The **frozen Python 2/PyCUDA original** lives at `/home/cwarner/work/superFATBOY/superFATBOY`
  (installed as `superFatboy.py` on PATH) — **never edit this, ask the user first**. It's the
  ground-truth oracle for cross-validation (see the oriBench 4-way comparison above).
- Several `.py` files embed CUDA C kernel source as string literals (compiled via `cp.RawModule`) —
  their C-level `sqrt`/`exp`/`abs`/etc. calls are not part of the Python numpy/math cleanup.

**The algorithm-audit program.** Six processes were flagged (2026-09-18) as the most
brittle/heuristic-heavy in the codebase and queued for a deeper robustness pass once translation
bug-hunting was done: `findSlitletProcess`, `rectifyProcess`, `wavelengthCalibrateProcess`,
`miradasCollapseSpaxelsProcess`, `removeCosmicRaysSpecProcess`, `badPixelMaskSpecProcess`. Status
as of v2.4.7 (2026-10-04):
- **`findSlitletProcess`** (`traceOrders`/`traceSlitlets`) — full Q1-5 audit done, shipped
  (`fc1e581`/`c474a5e`, v2.3.27/2.3.28): `edge_detection_method=auto` (local-minimum rescue for weak
  packed-slit boundaries cross-correlation misses), a `stats_<flatid>.txt` per-datapoint diagnostic
  file added to `traceSlitlets` (previously only `traceOrders` had one — a real asymmetry, worth
  checking for elsewhere), `fit_function=spline` option (numerically identical to polynomial at
  this function's typical fit_order 2-4, so a safe no-op default). Real MEGARA fiber-tracing
  concerns from round 1 turned out to be a double-overscan-trim data bug, not an algorithm problem
  — `findSlitlets` itself is now considered solid.
- **`rectifyProcess`** — the biggest single piece of audit work, four trace/transform function
  pairs all brought to the same fix set (MOS continuum, MOS skyline, longslit continuum, longslit
  skyline — v2.3.32/2.3.33/2.3.34): per-datapoint `stats_<fduid>[.txt|-skylines.txt]` diagnostics
  everywhere; a local-significance rescue for MOS continuum's "too faint vs. global first-point
  peak" bug (real MIRADAS failure, 16.6%→86% coverage on the worst slit); a flux-weighted-moment
  centroid rescue (`centroidMoment()`) for when a free-width Gaussian fit's FWHM comes back
  implausible — validated as measurably better than the Gaussian at low S/N and tied at high S/N,
  real head-to-head comparison in `project_algorithm_audit_list` memory; a `checkFitSanity()`
  runaway-high-order-fit guard (`rectify_max_transform_factor`) added everywhere it was missing;
  `independent_slitlets_fallback`/`mos_sky_fallback` substitute-fit options
  (`identity`/`pooled_good_slits`/`nearest_neighbor_slits`) for a slit with no usable trace — real
  leave-one-out validation on MIRADAS shows `nearest_neighbor_slits` is consistently best, **but
  the default is still `identity`** pending the user's sign-off on a production-affecting default
  change. Several real bugs found and fixed along the way (a shape bug in the fallback's grid
  slicing, a tuple-unpacking arity mismatch and missing-comma column bugs in a dead
  coords-as-filename code path, an unguarded `xcenters[0]` crash-on-zero-skylines). **Still open**:
  specBench's science-frame continuum trace has a more diverse failure profile than MIRADAS
  (`fwhm_range`/`sanity_far_reject` still ~15-30% of points there even after the fixes above) that
  isn't fully root-caused; the moment-centroid rescue was validated as real (6.1% of points on real
  EMIR skyline data) but not yet ported to skyline tracing; `rectifyMOS` is missing a
  slitmask-renumbering block `main` has (closes index gaps from discarded guide-star boxes) — not
  yet hit by any real dataset but a real gap; `gpu_drihizzle.py`'s padding-thread kernels still
  write one element past the real array bounds in some conditions (harmless for power-of-2 image
  sizes, would bite a non-power-of-2 one).
- **`miradasCollapseSpaxelsProcess`** — root cause of its peak-finder's CPU/GPU brittleness is
  understood (exact-value `np.where(z==max)` peak-picking has zero tolerance for ordinary float
  reduction-order differences) and a design for `scipy.signal.find_peaks`-based replacement was
  explored, but synthetic validation didn't clearly beat exact-max (prominence alone doesn't
  distinguish a narrow spurious spike from the true broad gap) — **not shipped, not further
  designed**. Separately, a real GPU-vs-CPU rectify bug that was corrupting its input slitmask *is*
  fixed (see `project_miradas_collapse_spaxels_investigation` memory for the full bisection) — that
  was blocking MIRADAS GPU mode entirely and is unrelated to the peak-finder brittleness itself.
- **`wavelengthCalibrateProcess`** — three audit rounds, done (v2.4.4-2.4.7; details in the
  "wavelengthCalibrate round 3" and "audit round 2" sections below): per-slitlet QA grades (RMS in px)
  in log/qa file/header; fail-clean per segment; fallbacks `learned,neighbor,trend,pattern,blind`
  behind a quality gate; a second pass for poor/failed slitlets (fixed MIRADAS SOS order 2's wrong match
  and LUCI arc slit 10); measured line intensities written per frame; legendre/chebyshev fits (same
  solutions, native coefficients in the header). The avrajit failure was a **line-list gap**, fixed by the
  shipped `Xenon_optical_air.dat`; `makeLineList.py` builds lists from NIST. `wavecal.py` (standalone, one
  cut) now subclasses the process. Earlier: negative-index slice wraparound fixed (all 8 sites). The draft
  tools `tools_linelist_builder_draft.py` / `tools_linelist_intensity_check_draft.py` (untracked) are
  superseded by `makeLineList.py` and the measured_lines files - ask before deleting them.
- **`removeCosmicRaysSpecProcess` / `badPixelMaskSpecProcess`** — **not yet started** at all beyond
  whatever bugs were hit incidentally during translation bug-hunting.

**Top-level error handling (goal 4).** Framework-level gaps fixed at the `fatboyDatabase.py`
entry point: an unconditional `input()` prompt on error that would hang any unattended/scripted run
(now opt-in via `interactive_on_error`); calibration-building processes
(`fatboyProcess.recursivelyExecute()`, used by ~23 processes) had zero exception handling and
discarded failure return values, so one bad calibration frame crashed the whole run instead of
degrading to "disable this one science frame, keep going"; a postcondition check so a process lying
about `success=True` while leaving unreadable data doesn't silently propagate. Ingestion
(`initializeAll`) deliberately does *not* just "disable and continue" on a bad input file — a
single bad file is isolated and logged loudly, but more than `max_init_failures` (default 3) or
every file failing aborts the whole run, on the reasoning that a malformed input FITS file is
usually a config/setup error worth surfacing immediately, not something to paper over. This was a
framework-wide pass, not algorithm-specific — the per-algorithm error-handling hardening is part of
the audit-list work above.

**Practical gotchas for running/testing this pipeline** (see `project_verified_configs_folder`
memory for the full list): always prefix `PYTHONPATH=/home/cwarner/work/superFATBOY3` (the
installed console script/egg can be stale and won't pick up source changes, or invoke the source
`.py` directly); **never** leave `debug_mode="yes"` on an unattended run (pops an interactive
`plt.show()` on the user's actual screen — use `write_plots="yes"` instead to still get QA PNGs);
the pipeline auto-creates *sub*directories under `outputdir` but never the top-level dir itself; only
ever delete output directories you created yourself, never the user's own; a run can take
15-25 minutes, and can also hang on a flaky NFS mount unrelated to any code issue (check `ps -o stat`
for kernel `D` state before assuming a bug).

## Documentation convention (from Craig, 2026-10-02) - follow on every change

- **`CHANGELOG.md` is grouped by class/module** (Framework: fatboyDatabase, fatboyLibs, gpu_drihizzle, ...;
  Processes: findSlitlets, rectify, ...), short summary bullets with the version. Every commit that changes
  behavior, adds an option, or fixes a bug adds a line to the matching section. Keep the Validation status and
  Open issues tables at the top current. (Old chronological notes are in its appendix.)
- **Every new option gets an `_optioninfo` entry** next to its `_options.setdefault` (that's what `-list`
  prints), then rerun `python3 docs/gen_options_reference.py` and describe it in the matching
  `docs/processes/*.md` page. Algorithmic changes also get a note in that page.

## wavelengthCalibrate round 3: fallbacks, second pass, line lists (2026-10-03, v2.4.5-2.4.6)

- **Offline bench** (scratchpad `wcbench.py` pattern): every finished run's `spec_*.dat` (lambda, 1-d cut) + `resid_*.dat`
  (lines used) is a truth set; build the process with `W('wavelengthCalibrate')` + `setDefaultOptions()` + `parseXML`
  of the XML's process node (run from `xml/` so wc files resolve), then call `tryWavelengthGuess` per guess method.
  Compare solutions only over the span of the truth's lines - outside it both extrapolate (OSIRIS "16 px wrong" was
  that). 10 datasets, ~180 cuts: no fallback accepted a wrong solution (gate: satisfactory+ and >= max(2(order+1),8) lines).
- What worked / didn't: neighbor must pass the neighbor's **polynomial** shifted (a linear guess is 28 px off mid-cut on
  LUCI: 10/24 -> 20/24). Pattern match: triplets + 1.5 px + unweighted votes + linear verification scored at 1.5 and
  4 px against chance is the best variant; a fitted-quadratic verification overfits chance coincidences (accepted a
  10 px-wrong OSIRIS solution), quadruplets / 1/N vote weights / per-scale contrast all did worse. Pattern/blind find
  nothing on dense lists (LUCI NeArXe, MIRADAS UArNe) - learned/neighbor/trend cover those.
- **Poor solutions must not feed trend/neighbor/learned** (MIRADAS SOS wrong order 2 poisoned order 1's trend); and try
  each guess with measured intensities, then the list's.
- Pipeline results: SOS orders 1-2 and LUCI arc slit 10 fixed by the second pass; everything else log-identical
  (first pass byte-identical; second-pass lines appended).
- `wavecal.py` (v2.4.7) subclasses wavelengthCalibrateProcess; its old helper copies had `np.np.correlate` (crashed
  every match). Check: loop the 24 LUCI arc slitlets through it passing `solvedCuts`/`lineMeasures` and compare with
  the pipeline's `qa_mlamp-clear-lamp.dat` (scratchpad `wavecal_test/run.py` pattern).
- NIST ASD query (makeLineList.py): `format=1`, `show_av=3` = vacuum; needs a User-Agent (403 otherwise); columns
  differ by spectrum (parse by header). Strong blue Xe I lines have no NIST intensity; Handbook omits them.


- **QA**: every fit prints RMS in wavelength units and px (residual / local dispersion from the polynomial
  derivative), a grade (`wavecal_quality_thresholds`), lines used, coverage of the cut; per-frame summary line;
  `qa_*.dat` columns + failure rows; header `WCRMS/WCRMSPX/WCQUAL/WCNLINES` (MOS `WCRMSxx/WCRPXxx/WCQULxx/WCNLNxx`).
- **Fail clean**: the per-segment body of `wavelengthCalibrate` is inside try/except (log + traceback, skip that
  segment, keep fitParams/min/maxLambdaList aligned). Most of the 2000-line diff is that re-indent.
- **Fallbacks** (`wavecal_fallback=neighbor,blind`) run only after the original match fails, so slitlets that
  matched before are untouched: WC-log-identical (apart from the QA lines) on LUCI, KAST, OSIRIS, MIRADAS SOS/SOL.
  Blind = FFT xcorr of the cut vs a template with intensities^0.25 over 600 log-spaced scales, verified by how
  many of the 15 brightest peaks land on lines; central half of the cut first (a linear guess fails across
  LUCI's nonlinear cut). Gate: a fallback solution must grade satisfactory+ with >= max(2(order+1),8) lines -
  without the gate, aliases slip through.
- **avrajit-osiris**: arcs are R2500U, XML is right (earlier "XML wrong" was a chance match). `xenon_optical.dat`
  lacks the 4481-4844 A Xe I lines -> line-list problem; now fails cleanly.
- **MIRADAS grades** cluster 0.2-0.5 px with ~50 lines over 90% of the cut; order 2 seg 1 (3.35 px) is a wrong
  primary match (c2 4x the neighbouring orders') that the old code also produced - the grade now flags it.
- WC-only rerun technique: `cp -al` a finished output dir, delete `wavelengthCalibrated/` and later dirs, rerun
  with `overwrite_files=no`, diff the WC log lines (scratchpad `wccmp.py` pattern).

## LUCI four-run comparison, gap-aware padding (2026-10-03, v2.4.0)

Compared auto (autodetect + traceOrders), region (region file + traceOrders), test (`trace_slitlets_individually=no`,
traceSlitlets) and main on LUCI. Everything traced back to the slitmasks: auto traced 10 of 16 continua and
extracted 13 spectra (missing the brightest) because **LUCI science frames sit 1.1-1.5 px below the flats**
(flexure, measured by cross-correlating d/dy of flat vs science at 5 columns) and auto's half-max edges then clip
the lower wing where the negative nodded image sits; the rectify continuum finder and extractSpectra both reject a
cut that starts at its peak. Proved by swapping slitmasks between runs offline (`contswap.py` pattern: replay
rectify's per-slit finder on each run's own clean frame - matched the logs slit for slit). Fix: `padding` is now
gap-aware (`padSlitletEdges`: grow into empty rows, split gaps < 2*padding at the midpoint, never overlap) and applies
in traceOrders/traceSlitlets/tracePeakLocalMax; `slitmaskFromEdges` is the CPU twin of `createSlitmask`. Padded rows
of a flat-divided frame are amplified (normalized flat ~0.1): `flat_low_thresh` (now float, CPU replacement fixed)
can stop that but costs 10-20% flux. Rectification straightness and wavelength solutions were equivalent across all
four runs; `new` == `main` (findSlitlets byte-identical, spectra within 1-3%). Matching spectra between runs needs
content correlation - slit labels shift. Wavecal uses the (non-flat-divided) clean sky, so flat options don't affect it.

**GPU == CPU (v2.4.3).** LUCI now gives byte-identical output in both modes (107/107 files). What it took, in the
order found - use the same method (rerun both modes, diff every FITS, then bisect with offline replays of the step):
(1) gpu_drihizzle uniform kernel ignored the CPU's integer-shift case and dropped edge pixels; (2) turbo `dropsize<1`
weights unclipped (GPU) / half-clipped (CPU); (3) **float atomics made the GPU drizzle non-deterministic** - now double
accumulation on both sides, positions in double, data scaling as the CPU; (4) `medianfilterCPU` used a 50-point window
at the ends and float64 output, `medianfilter2dCPU` float64 output (dtype alone changed later comparisons); (5) the C
1-d `median()` returned uninitialized memory when nothing was left after `nonzero`/thresholds (random per run);
(6) CPU noisemaps `np.sqrt(master)` without abs -> NaN; (7) CPU doubleSubtract never blanked outside the slitmask
(`getData(tag="slitmask")` returns the calib object); (8) all RawModules compiled with `--fmad=false`. Most were also in
main. Still not covered: drizzle with a fitted `geomDist` (calcTransOpt float32 on GPU), the 3-d GPU drizzle.

**Flexure correction (v2.4.2)** - `findSlitlets` option `flexure_correction` (`none`|`shift`|`gradient`, `linear`=`shift`):
per object, `measureFlexure` cross-correlates d/dy of the master flat vs each of the object's frames slit by slit at 9
columns (sky-lit edges), MAD-clips, and `maskFlatCenterOffsets` gives (mask - flat center) on isolated slits; the object
gets an object-tagged slitmask moved by flat shift - mask offset (region-file masks drawn on the data stay put) and an
object-tagged master flat (created under the flat's own process name so flatDivideSpec's getTaggedMasterCalib finds it)
with illumination (31-px median along dispersion) shifted and pixel response kept. Measuring mask->science with a binary
mask profile does NOT work (width mismatch, MAD 0.6-3 px) - use centers. traceOrders masks sit ~1 px below the flat
(`-1` on ylo + truncation). LUCI is verified (region file + traceSlitlets + shift, GPU == CPU spectra to 0.01%) and has
`LUCI_MOS_template.xml`. Rectified GPU vs CPU still differ (drizzle kernel) without changing the spectra.

**Temp dir (v2.4.1)**: two runs from the same directory used to share (and delete) `temp-fatboy`; now locked per
run (`setupTempdir`, `fatboy.lock`), the second run gets `temp-fatboy-<pid>`. Tested with two concurrent findSlitlets
runs forced to page data out (`memory_image_limit`=5): identical outputs, both dirs cleaned. Parallel test runs from
one directory no longer need staggering. **Flexure vs bleeding** was checked by measuring lower and upper half-max
edges separately (rigid shift = both move together; bleeding = width grows, worse in bright-line columns).

## Calib star, LA Cosmic, 3-d drizzle, undefined names (2026-10-02, v2.3.43-44)

- **Calib star (LUCI A1689)** needed `<calib type="standard">` in the XML (it was an `<object>`), plus:
  per-object rectified slitmasks (rectify), `mos_min_continua_global_fit` (one continuum can't constrain
  a whole_chip fit - it gave y-scale 1.986), `min_negative_flux_fraction` (doubleSubtract; sky frame had
  the star off the slit), MOS standards in calibStarDivide (`calib_star_spectrum`, SPEC_nn slit mapping).
- **LA Cosmic** reviewed against `lacos_spec.cl`: sky model now added back once after the loop; IRAF noise
  floor; vertical dispersion transposed; MOS slit bounding boxes filled from the nearest in-slit row (masking
  them out of the fits instead made a tilted slit's partial rows extrapolate wildly - tested); frame assembled
  from the input, only flagged pixels replaced; runLacos/runDeepCR use `force_cpu=True` like runDcr.
  Synthetic test scripts were in the session scratchpad (`lacos_test_tilt.py` pattern: tilted slits, injected
  CRs, recall / false positives / edge false positives / bias).
- **CuPy RawKernel scalar packing**: int64 scalars are converted correctly, but **float64 scalars into a
  `float` slot read as 0** (confirmed with a 1-line RawKernel test). Any `python_float op np.float32(x)`
  kernel argument is suspect; ints are fine.
- **CPU `drihizzle3d`** lost whole planes of flux: float32 rounding mapped two inputs to one output and
  numpy `a[idx] += v` keeps one duplicate. The no-distortion shortcut now falls back to the unique-index
  loop when targets collide (2-d was unaffected and left unchanged).
- **Undefined-name sweep that sees through star imports**: copy each file, replace `from X import *` with
  the module's actual exported names, run pyflakes. Found 133 latent NameErrors; 7 remain (see CHANGELOG
  open issues). Rerun this after large edits.
- **Regression method used**: specBench on current code vs the previous commit (git worktree, copy the
  untracked `.so` files in), compare every output FITS with `np.array_equal(..., equal_nan=True)` - plain
  array_equal reports NaN-containing files as different. Result for v2.3.44: 571/571 identical.
- **Medians with nlow/nhigh (v2.3.45)**: the GPU kernels and the fatboyclib C extension decided even/odd from
  the count before rejection, and the C code returned 0 when `k == 0` (one kept value: `min` combine, a single
  nonzero value). Both fixed (also in main); createCleanSkies `quartile` now drops `even=False` so GPU == CPU ==
  lower quartile. Test matrix (depth 2-7 x nlow x nhigh x even x nonzero, small and large arrays) in the
  session: 424/424 match numpy. **fatboyclib is a C extension**: rebuild it (`setup.py install`, or
  `build_ext --build-temp <tmp> --build-lib <tmp>` and copy the .so in - `build/` is root-owned after a sudo
  install) or a test silently uses the old .so. For a baseline worktree of an older commit, copy in the OLD .so.
- `unique1d_wrap` (drihizzle.py) keeps Craig's numpy-version branches (`np.unique1d` for numpy < 1.5).
- **Open**: SINFONI `padx`/`pady` commented out by Craig - recheck on SINFONI data.

## Full LUCI MOS run (caden_luci_test.xml) and the InOut / drihizzle bugs it exposed (2026-10-02, v2.3.40-42)

`xml/caden_luci_claude_auto.xml` (autodetect, `slitlet_autodetect_source=both`, nslits=24) -> `superFATBOYdata/cadenLUCI-py3-gpu`
and `xml/caden_luci_claude_region.xml` (the user's region file) -> `cadenLUCI-claude-region`. Reference
`superFATBOYdata/cadenLUCI` (2026-04-03) **predates the Gemini refactor** (5b4bbbd, 2026-06-18), so it is effectively a
pre-refactor oracle - but it only got through shiftAdd + wavelengthCalibrate slits 1-5 (no extraction).
- **InOut family (v2.3.40)**: CuPy kernel calls passing `cp.asarray(host_array)` inline for an argument the kernel writes
  silently discard the result (PyCUDA's `drv.InOut` copied back). Hit for real: `flatDivideImage` returned frames
  **undivided** whenever the data was on the host (MIRADAS verified runs happened to have it on device - checked).
  Fixed with `gpuInOut(x)`/`gpuSyncBack(x, x_gpu)` in fatboyLibs. The audit method: list every `drv.InOut`/`drv.Out` in
  `main` (132), match to the current call by kernel name + order, classify the argument (inline asarray = suspect; named
  device buffer = check it's used after the call). Script pattern was in this session's scratchpad (`inout_audit.py`).
- **gpu_drihizzle final weighting (v2.3.41)**: the CuPy rewrite divided `weight=exptime, outunits=counts` output by the
  expmap and multiplied by totexp; main returns the raw sum. rectify uses counts + point_replace (which divides by
  expmap itself) -> double division -> ~70 garbage px/frame (to 9e8) at slit edges -> distorted clean frame -> wrong
  doubleSubtract shift (-16/-22 vs true -12) -> 4-10 rows lost per slit -> 0-5 of 24 spectra extracted. Diagnosed by
  checking flux conservation stage by stage (`pos/neg` sums of clean frames before/after rectify).
- **removeCosmicRaysSpec (v2.3.42)**: `runDeepCR`/`runLacos` assigned a local `np` -> UnboundLocalError on every np call.
- After fixes: no tracebacks; wavelength solutions match April slits 1-5 to 0.06-0.23 A median (0.02-0.05 px, 4.35 A/px),
  all 24 slits calibrate at 0.38-0.64 A RMS; extraction finds spectra in 14/24 slits (both runs).
- **CPU vs GPU cross-check** (`xml/caden_luci_claude_region_cpu.xml` -> `cadenLUCI-claude-region-cpu`, same code, CPU
  drihizzle always matched main): byte-identical through skySubtracted; rectified/shiftAdded agree to ~1e-5; **extracted
  spectra identical** (14x2050, median diff 0); wavelength solutions within 1.87 A max. Remaining GPU/CPU differences:
  `rct_cleanSky` (turbo kernel, counts) GPU ~2% higher median (p99 1.57x) - not yet investigated, probable cause of the
  1.87 A; `dbs_` outside the slitmask GPU writes 0, CPU leaves values (cosmetic).
- **Known open**: calib star A1689 - shiftAdd "Could not find slitmask associated with A1689.0123" (no
  `slitmask_dbs_A1689` written) and doubleSubtract shift -25 vs rectify trace -11 (guess +/-16) -> calibStarDivide has no
  standard. `drihizzle3d` (SINFONI only) has the InOut problem throughout (np.empty buffers filled via inline asarray) - needs
  the same device-allocation rewrite the 2D version got. LA Cosmic GPU: `lacosSelect` no-count branch and
  `lacosUpdateOutput` were already no-ops on main (astype copies never returned) - LA Cosmic GPU never fully worked.

## findSlitletProcess: packed slitlets, arclamp autodetect, invalid slitlets (2026-10-02, v2.3.38)

Driven by `caden_luci_fs_test.xml` (LUCI MOS, 25 region-file slitlets, several packed with 1-2px boundaries).
- **Region-file edges finding 0 datapoints** were all rejection code 7: `traceOrders`'s `cmax/q1 < cut1d_max_threshold`
  check assumes one side of an edge is dark background, so a packed boundary (flux on both sides) is rejected before
  cross-correlation even runs. `edge_detection_method` default is now **`auto`** (byte-identical to `cross_correlation` on
  any edge that method can trace, by construction; verified on specBench + MIRADAS SOS). New `narrow_gaps_between_slitlets`
  (default `no`): when `yes`, a point failing the q1 check is measured with local_minimum *for that point* instead of rejected.
  Simply skipping the check was tried first and is worse (cross-correlation on a packed edge gives sigma 1.7-2.9 vs 0.09-0.24,
  and the garbage nonzero point count blocks `auto`'s rescue). Phase-2 outlier rejection now applies the `maxcors` criterion
  per edge-method group (cc peak vs dip depth are different scales).
- **`slitlet_autodetect_source = flat|arclamp|both`** (default `flat`): correlation between adjacent rows of the
  high-pass-filtered master arclamp is ~1 inside a slitlet, ~0 in background, and notches sharply at a packed boundary (LUCI
  y=1035: flat dips 3%, arc corr 0.998->0.70). `both` = arc segmentation + flat half-max outer edges: 24/24 LUCI slitlets within
  3px of the hand-made region file (flat autodetect merges two pairs). Needs the full dispersion range (±128-256px windows lose
  boundaries), so not suited to strongly tilted slitlets. Master arclamp comes from getTaggedMasterCalib/getMasterCalib, falling
  back to createMasterArclampProcess.getCalibs (same pattern as flatDivideSpec for the master flat).
- **Invalid slitlets** (LUCI mask-ID "digits" strip): `findInvalidSlitlets()` - flat row-to-row MAD roughness over a 201-column
  median (`slitlet_validity_max_flat_roughness`=0.045; real LUCI/MIRADAS slits 0.001-0.024, ID strip 0.083-0.09) and, with an
  arclamp, mean arc row correlation (`slitlet_validity_min_arc_corr`=0.9). Dropped with ERROR in autodetect, WARNING-only for a
  region file.
- Fixed a crash from the 2026-09-11 hardening: the degraded-slitlet QA write did `del qaData`, then the normal slitmask QA write
  used it -> `UnboundLocalError` whenever any slitlet fell back and `write_calib_output=yes`.
- Gotcha (fixed in v2.4.1): launching several runs from the same directory used to race on `temp-fatboy/` - each run now locks its own temp dir.
- v2.3.39 follow-ups: (a) `local_minimum` searched the whole 21px cut, so where LUCI's 587 packed boundary turns into a
  plain step between two lit slits (x<440) argmin landed on random points of the fainter plateau and the trace drifted
  3-18px into the next slit. Once a point is accepted since the last reset ("anchored"), it now searches only
  ±`local_min_search_radius` (3) px of currY and rejects an edge-of-window minimum (new stats code **9**); before anchoring
  it still searches the whole cut, because the region-file y can be ~3px off the true dip (LUCI edge 970 -> 973.4). 587
  coverage 64%->76%, no drift. (b) Arc-mode nslits fallback: if `slitlet_autodetect_nslits` is set and the arc's valid
  count misses it while the flat's (judged by flat criteria only - MIRADAS arc rows don't correlate within a slit) matches,
  use the flat and stop using the lamp for validation. All 3 MIRADAS configs with `both` now fall back and trace cleanly.
- The installed `superFatboy3.py` (egg) does not pick up source changes - the user must `sudo python setup.py install`
  after each commit before running XMLs themselves.

## Algorithm-audit methodology, distilled from findSlitletProcess (2026-09-28)

`findSlitletProcess` (`traceOrders`/`traceSlitlets`) just went through a full five-question robustness
audit (root-cause a class of failures, propose alternatives, ship them as new options) — commits
`fc1e581`/`c474a5e`, versions 2.3.27/2.3.28. `rectifyProcess` and `wavelengthCalibrateProcess` are next
on the deferred audit list (see the algorithm-audit-list memory for what's already been found on both).
Before starting either, apply these lessons — they're about *how* to run this kind of audit well, not
what was found this time:

- **Pull the real 1-d cuts / raw arrays directly from the pipeline's own intermediate FITS files
  (master flats, etc.) before hypothesizing a fix.** The LUCI "0 datapoints" bug looked like it could
  be a dozen different things from the log alone; reading the actual flux profile at the failing
  y-position immediately showed it was a weak local-minimum dip, not a step edge — a completely
  different failure mode than a first guess (bad threshold, off-by-one, etc.) would suggest.
- **A new detection/fit mode must be validated against the datasets that already work, not just the
  one that's failing.** `edge_detection_method=local_minimum` looked like a strict win on LUCI's
  weak-dip edges until it was run on LUCI's own *good* step edges too — where it regressed catastrophically
  (383→14 datapoints on one edge). The fix was `auto`: try the existing method first, only fall back
  per-edge when that method demonstrably finds nothing. **Prefer this "try-old-then-fall-back"
  pattern over a global mode switch** unless you've proven the new mode safe across every regime the
  old one already covers.
- **Test across datasets with deliberately different characteristics, not just the one that surfaced
  the bug.** Craig explicitly asked for this ("the slits can look very different — right next to each
  other or well spaced out") and it mattered: LUCI (packed slits) hit the weak-dip failure,
  MIRADAS order 20 (~30px real gaps) never did — confirming `auto` mode's fallback correctly never
  fires there (byte-identical rejection-code histogram to baseline). One dataset's result generalizes
  badly in a codebase supporting a dozen+ instruments.
- **Don't port a numeric conclusion from a sibling algorithm's audit without re-validating it at the
  target function's actual regime.** The earlier `rectifyProcess` audit found splines beat polynomials
  ~4x on RMS — but that was at fit order ~7, where polynomial extrapolation (Runge's phenomenon) is a
  real problem. Naively assuming "splines are just better" and porting that into `findSlitletProcess`
  would have been wrong: at its conventional fit order (2-4), a smoothing spline finds 0 interior knots
  and becomes numerically identical to the polynomial — confirmed both on real LUCI data and
  synthetically by sweeping fit order until the two actually diverged. The lesson generalizes both
  ways: when auditing `wavelengthCalibrateProcess` next, check what regime *it* actually operates in
  before assuming a fix that worked for `findSlitletProcess` or `rectifyProcess` transfers.
- **Check sibling code paths for diagnostic-capability parity before concluding "there's no way to see
  why this was rejected."** `traceOrders` had a per-datapoint `stats_<flatid>.txt` rejection-code log;
  its sibling `traceSlitlets` (same file, same algorithm family, `trace_slitlets_individually=no`) had
  none, which meant a real EMIR dataset's failure had to be diagnosed by hand-reimplementing the whole
  loop in a scratchpad script. Adding the same stats file to `traceSlitlets` fixed this permanently and
  took one pass. Check for this kind of asymmetry early in `rectifyProcess`/`wavelengthCalibrateProcess`
  too — they likely have similar near-duplicate function pairs.
- **Prove a "no-op" default with a real full-log diff, not just "it still passes."** Any refactor that
  touches a function on the default code path (e.g. factoring the polynomial fit into a shared
  `fitTraceCurve()` helper used by both new and old options) should be validated by diffing the complete
  log of a real run before vs. after against the *exact* same dataset/config — not just checking the
  new run also succeeds. That's how the `fit_function` refactor was confirmed behavior-preserving
  (byte-identical `found N datapoints`/`rejecting outliers`/`Sigma` lines across ~50 fits).
- Per [[feedback_scrub_debug_mode_before_runs]] (auto-memory): never enable `debug_mode` for
  verification runs even when it would show the exact plot you want — write the equivalent data to
  disk (a stats file, an ad-hoc `np.save`, a synthetic reproduction of the same math) instead.

## Second real end-to-end test: oriBench.xml (NIR imaging) PASSED, all 4 combos cross-validated (2026-09-16)

`cd /home/cwarner/work/xml && superFatboy3.py oriBench.xml` — a NIR imaging dataset (dark/flat/
badPixelMask/skySubtract/cosmicRays/alignStack, 9-frame dither), the first **imaging**-mode test
since the refactor (specBench.xml above was spectroscopy only). Took 14 bug-fix commits, but now
passes clean. More importantly: the user also ran the **frozen python2/PyCUDA original**
(`/home/cwarner/work/superFATBOY/superFATBOY`, installed as `superFatboy.py` — **never edit this,
ask first**, per explicit user instruction) in both CPU and GPU mode, giving a genuine 4-way
cross-check: py2 CPU / py2 GPU / py3 CPU / py3 GPU. **All four now agree on the final alignment
shifts to 4+ decimal places** (e.g. shift to frame 7: py2 CPU -49.878698, py2 GPU -49.878660,
py3 CPU -49.878698, py3 GPU -49.878660 — the tiny CPU-vs-GPU gap is ordinary float rounding,
consistent within each backend across py2/py3). This is about as strong a validation as this
refactor is going to get for imaging mode.

**Test configs**: `oriBench.xml` (GPU) and `oriBench-cpu.xml` (CPU, hyphen not underscore —
already existed, gpumode=no is the only diff) live in `/home/cwarner/work/xml/`. For the 4-way
comparison, created `oriBench-py2-gpu.xml`, `oriBench-py2-cpu.xml`, `oriBench-py3-cpu.xml` there
too — same content, each with its own `outputdir` so all 4 runs can be inspected side by side
without clobbering each other (the user confirmed this is fine; they'd been blowing away the same
dir between manual runs). Per the user: **only ever delete `superFATBOYdata/oriBench-test/`** (or
whichever of these dedicated dirs you created) between your own iterations — never touch dirs you
didn't create (e.g. `oriBench-py3`/`oriBench-new` are the user's own unrelated 2020-era artifacts).

**Running the old py2 original**: `superFatboy.py <config>.xml` is on PATH (installed egg,
version 2.2.0). CPU mode just works. GPU mode needs two things fixed via environment only, never
via editing that frozen codebase:
1. Python 2's `print` chokes with `UnicodeEncodeError` trying to `str()` an exception message
   containing a smart-quote character, which **masks whatever the real underlying error was**.
   `PYTHONIOENCODING=utf-8` does NOT fix this (that only affects stdout encoding, not `str()`
   coercion of a `unicode` object, which Python 2 always does via strict ASCII). The real fix:
   `sys.setdefaultencoding('utf-8')` — deleted from `sys` after site init, restore via `reload(sys)`
   in a tiny wrapper script (see `/tmp/.../scratchpad/run_py2_utf8.py` pattern: `reload(sys);
   sys.setdefaultencoding('utf-8'); sys.argv = [...]; execfile('/usr/local/bin/superFatboy.py')`).
2. Once real errors are visible: PyCUDA compiles kernels via `nvcc`, which shells out to the
   system `gcc` for C++ preprocessing. This machine's default `gcc` (`update-alternatives`) points
   at gcc-12, but only the `gcc-12` package was ever installed, not `g++-12` — so gcc-12 has no
   `cc1plus` binary and nvcc fails with `cannot execute 'cc1plus'`. **Do not** `apt install g++-12`
   or touch `update-alternatives` — the user deliberately pinned `gcc`→12 for an unrelated project
   and plain `g++` is intentionally left at 9. Fix scope-per-invocation instead:
   `NVCC_PREPEND_FLAGS='-ccbin=/usr/bin/g++-9' superFatboy.py <config>.xml` (now in the user's
   `~/.bashrc`). This is a standard CUDA-toolkit-supported env var, not a PyCUDA-specific hack —
   confirmed directly with a standalone `nvcc --cubin` test before trusting it on the real run.

**Bugs found and fixed in superFATBOY3 (this branch) this session, in the order hit** — every one
confirmed against `main` before fixing, and (unlike the specBench.xml pass) most were verified
with a *direct* synthetic/real-GPU repro of the actual failure mechanism, not just "no longer
crashes":
1. `badPixelMaskProcess.py`: `lo.astype(np.float32)` etc. on plain Python floats/ints (no
   `.astype`) — same `.astype()`-on-a-scalar mistake as `ncoeffs.astype()` from the specBench pass.
2. `fatboyDataUnit.renormalize()`: the bad-pixel-mask argument needed `cp.asarray()`-ing to match
   the FDU's own GPU mode before `self.getData()*(1-bpm)` — callers build `bpm` as plain numpy.
3. `gpu_imcombine.py`: `nfint.astype(np.int32)` (6 sites) — `nfint` is `int(nfiles)`, a plain int.
   Proactively swept the same "scalar `.astype()`" pattern afterward and fixed 3 more files
   (`fatboyLibs.py`'s `idx = [-1].astype(...)`, `findSlitletProcess.py`'s
   `ycoords[i]+.5.astype(...)` — precedence bug, `.astype` binds to the bare literal `.5` — and
   `gpu_drihizzle.py`'s 3-d function, 5 sites) rather than waiting to hit each one via another
   15-25-minute test cycle.
4. `skySubtractProcess.py`: missing `from superFATBOY.fatboyLibs import *` entirely (not converted
   to explicit imports like the file's other imports — just dropped), taking `applyObjMask()` down
   with it. This is the project's own local-namespace wildcard import, unrelated to the
   numpy/math cleanup that was actually in scope — confirmed via a full main-vs-current diff that
   no other file has the same gap.
5. `fatboyLibs.py` + `gpu_drihizzle.py` + 6 more files: **25 occurrences** of a quoted numpy dtype
   string that got an erroneous `"np."` prefix injected inside the quotes — `.astype("int32")` (a
   valid string dtype descriptor, unrelated to the `from numpy import *` cleanup) became the
   invalid `.astype("np.int32")`. 10 of these were in `fatboyLibs.py`'s `array.dtype != 'np.float32'`
   comparisons, which don't crash at all — they just always evaluate `True` (no real dtype ever
   equals that string), silently skipping whatever fast-path the check gated. **This class of bug
   is silent, not just crash-prone — worth a special note for future review.**
6. `fatboyLibs.py::linterp_gpu()`: the first of what turned out to be a *whole family* of
   `cp.empty(existing_np_array)` mistakes. PyCUDA's `drv.Out(x)`/`drv.InOut(x)` auto-copied device
   results back into the host array `x` after a kernel call; the refactor's mechanical translation
   kept passing the *host* array inline (`cp.empty(output)`), but `cp.empty()` doesn't accept an
   array as "make a device copy of this" — it treats the array's *current (uninitialized) values*
   as a **shape** argument, allocates a throwaway device buffer of that bogus shape, and the real
   kernel result is written there and discarded. The function then returns the original,
   never-updated `np.empty()`/`np.zeros()` array. Since fresh OS pages are typically zero, this
   often manifests as "function silently returns all zeros" rather than a crash.
7. Same bug, swept proactively: **15 more sites**, all in `fatboyLibs.py` — `rawToFlatDivided`'s
   main kernel + its cosmic-ray loop, `blkavg`, `blkrep`, both `convolve2d` variants,
   `fwhm2d_cube_gpu`, `getCentroid_cube_gpu`, and the 3 LA Cosmic helpers (one of which had it on
   *two* outputs at once). Fix pattern throughout: allocate a real `cp.empty_like()` device buffer,
   pass *that* to the kernel, `.get()` the result back to the host variable afterward.
8. `gpu_xregister.py` + 5 more files (`pysurfit.py`, `tri_register.py`, `xregister.py`,
   `gpu_drihizzle.py`, `fatboyDataUnit.py`): **11 occurrences** of bare `isinstance(x, ndarray)`
   with no valid name to resolve to — missed by the original numpy-wildcard-import sweep because
   these specific `isinstance` type-dispatch branches aren't exercised by every process. Same bug
   class as the `sqrt`/`tan`/`shape()` fixes from the specBench pass; fixed to `np.ndarray`.
9. **35 occurrences**, 5 files (`gpu_xregister.py`, `xregister.py`, `fatboyLibs.py`, `wavecal.py`,
   `miradasStitchOrdersProcess.py`): `np.min(a, b)`/`np.max(a, b)` used to compare two *scalars* —
   but `np.min`/`np.max`'s second positional argument is `axis`, not a second value. **This one is
   dangerous, not just crash-prone**: when the second argument is literally `0`, `np.max(a, 0)`
   "succeeds" (axis=0 on a 0-d array is a harmless no-op) and silently returns `a` **unchanged,
   ignoring the comparison entirely** — only a *nonzero* second argument raises `AxisError`. main
   used Python's builtin `min(a,b)`/`max(a,b)` (available unqualified via `from numpy import *`)
   at every one of these sites; fixed by reverting to the builtin, not `np.min`/`np.max`.
10. `fatboyLibs.py::whereEqual()`: a *regression in my own fix from #6/#7 above* — I'd correctly
    changed `idx = [-1].astype(...)` to `idx = np.int32([-1])`, but then converted the *result*
    back to a plain Python `int` (`idx = int(idx_gpu[0])`) before the array-shaped tuple math below
    it (`idx//data.shape[1], idx%data.shape[1]`). main's PyCUDA version left `idx` as a genuine
    1-element numpy array through this same arithmetic (floor-div/mod on a 1-element array yields
    another 1-element array), so the function has *always* returned a tuple of 1-element **arrays**
    (matching `numpy.where()`'s convention) — callers like `gpu_xregister.py`'s `p[1] = b[1][0]`
    need `b[1]` to be indexable. Converting to a scalar broke that. **Lesson: when fixing a
    `.astype()`-on-non-array bug, check whether downstream code relies on the *array-ness* of the
    result, not just its numeric value — don't reflexively convert to a Python scalar.**
11. **The big one** — `gpu_drihizzle.py`'s CUDA `CUDA_ERROR_ILLEGAL_ADDRESS` crash in
    `alignStack`'s drihizzle step. Root-caused via `CUDA_LAUNCH_BLOCKING=1` (CUDA kernel launches
    are async — the reported error line is often *not* the actual faulting call; this env var
    forces synchronous launches so the error appears at the real culprit) plus direct device-side
    instrumentation (temporarily added `print()`s computing `intx`/`inty`/`idx` bounds, checking
    for NaN/Inf, and printing argument `type()`s and `.flags['C_CONTIGUOUS']` — all came back
    provably fine except one thing). The actual bug: `xsh[j] - np.float32(xshmin)` — the refactor's
    mechanical `float32(x-y)` → `np.float32(...)` rewrite put the cast around only the *second*
    operand instead of the whole expression (main: `float32(xsh[j]-xshmin)`, wrapping the entire
    subtraction). Subtracting a `numpy.float32` from a plain Python int/float promotes the *result*
    to `numpy.float64` (confirmed: `type(0 - np.float32(x))` is `numpy.float64`) — and **CuPy's
    `RawKernel` packs a scalar argument's bytes according to the Python object's own dtype, not the
    kernel's declared C signature**. An 8-byte float64 landed in a slot the compiled kernel reads
    as a 4-byte `float`, shifting every subsequent argument (`ysh`, `xsize`, `size`) by 4 bytes in
    the packed buffer — turning the array-write index the kernel computes into garbage. Reproduced
    this exact mechanism standalone with a 4-line CuPy `RawKernel` test (see commit `1a2a1dc`):
    passing the broken value returns `-3.689349e+19` from every thread; the fixed value returns the
    correct number. **This is the single most important lesson from tonight**: a `float32(a - b)`
    →`np.float32(a - b)` mechanical rewrite is easy to get right; `a - float32(b)` →
    `a - np.float32(b)` (cast landing on the wrong sub-expression) is subtly, silently wrong in a
    way that only surfaces as GPU memory corruption, not a Python-level type error — **if you ever
    see another `X[j] - np.float32(Y)`-shaped expression anywhere in this codebase, check it
    against `main` immediately, don't assume it's fine.** Fixed all 16 sites (12 in 2D drihizzle, 4
    in the 3D sibling) by wrapping the whole subtraction. Swept the rest of the tree for the same
    "indexed-variable arithmetic with a cast landing on one operand" shape; found nothing else.
12. `fatboyLibs.py::linterp_cpu()`: bare `shape(data)[0]` — the CPU sibling of `linterp_gpu`
    (fix #6), never hit until the py3-CPU-mode run since every prior test this session was GPU
    mode. Same bug class as the specBench pass's `sqrt`/`tan`/`ndarray` fixes.
13. `badPixelMaskSpecProcess.py::bpm_replace_median_neighbor_gpu()`: same `cp.asarray(output)`-
    as-kernel-buffer mistake as #6, found by proactively grepping for the pattern rather than
    waiting to hit it — **not yet exercised by any test run** (spectroscopy bad-pixel
    interpolation; oriBench.xml is imaging, specBench.xml didn't hit this specific method).
    Flagged for validation whenever a spectroscopy dataset next exercises this method.

**Known follow-up, not done**: `gpu_drihizzle.py`'s CUDA kernels (`calcXYin`, `calcTransOpt`, and
others) have a real "write past the array bounds for padding threads" bug — `if (i >= size) {
out[i] = 0; return; }` writes to `out[i]` *before* returning, and `i` can exceed the array's real
size whenever `blocks*block_size` isn't an exact multiple of the true element count (harmless for
this session's 2048×2048 test images, since 2048²=4194304 divides evenly by 512, but would bite on
a non-power-of-2 image size). Should be `return;` with no write. Not fixed tonight — ran out of
scope once the actual crash (#11 above) turned out to be unrelated to this; worth fixing
opportunistically if touching these kernels again.

## First real end-to-end test: specBench.xml PASSED (2026-09-15)

`cd /home/cwarner/work/xml && superFatboy3.py specBench.xml` — a classic Flamingos-1 MOS
spectroscopy dataset, the user's standard regression test — now runs **all the way through**
(linearity → noisemap → darkSubtract → createCleanSkies → createMasterArclamps → findSlitlets →
flatDivideSpec → skySubtractSpec → rectify → wavelengthCalibrate → extractSpectra →
calibStarDivide, for all 8 science frames plus the hd32008 calibration star) with zero unhandled
exceptions, and produces real, sane, finite, non-degenerate final flux-calibrated output
(spot-checked pixel values and table structure directly, not just file existence). Command run
with `PYTHONPATH=/home/cwarner/work/superFATBOY3` prepended so it picks up this working tree
instead of whatever is `sudo python setup.py install`-ed system-wide — do that (or reinstall)
before testing again. **Always `rm -rf` only the `specBench_test` subfolder of
`../superFATBOYdata/`** before rerunning, per the user's instruction — never the parent dir.

It took 12 iterations (run→crash→diagnose→fix→rerun) to get there. Each run takes ~15-25 minutes
of real processing, so the fix loop was genuinely slow — don't expect faster than that rerunning
this same test. One run also hung for ~17 minutes on an unresponsive NFS mount
(`/net/bolt/home/warner/FATBOY/specBench/`) unrelated to any code issue — confirmed via `ps`
showing the process in kernel `D` state and a plain `ls` on the same path timing out; it resumed
on its own once the mount recovered. If a run seems stuck, check `ps -o stat` on the
`superFatboy3.py` process and try `ls` on the dataset's NFS path before assuming a code bug.

**Bugs found and fixed, in the order they were hit** (all real, all confirmed either by direct
reproduction with cupy/numpy or by diffing against `main`):
1. `pow((float)x, n)` with an int exponent in linearityProcess.py's CUDA kernel — NVRTC (which
   `cp.RawModule` uses) can't resolve it the way offline `nvcc` could. Fixed with `powf` + an
   explicit `(float)` cast on the exponent.
2. `cp.np.X` (CuPy has no `np` submodule, ever) in 4 files — likely a mistaken find/replace that
   prefixed numpy calls with `cp.` instead of replacing them. `cp.np.` → `cp.` throughout.
3. `ncoeffs.astype(np.int32)` where `ncoeffs = arr.size` — `.size` is a plain Python `int`, no
   `.astype`. Reverted to `np.int32(ncoeffs)` (`main` had this right before the refactor).
4. A CUDA kernel name typo: looked up `"noisemaps_twilight_float"`, the real kernel is
   `noisemaps_mflat_dome_on_off_float`.
5. Three files (`flatDivideSpecProcess`, `createMasterArclampProcess`, `flatDivideProcess`)
   assigning a possibly-CuPy array straight into an astropy HDU's `.data` — astropy blocks
   CuPy's implicit `__array__`. Fixed with `cp.asnumpy(data)` at the write.
6. rectify's trace-finding functions (`traceMOSContinuaRectification`,
   `traceMOSSkylineRectification`, `calcLongslitContinuaRectification`, and — missed on the
   first pass, caught by a *later* run — `calcLongslitSkylineRectification`) are inherently
   CPU-bound throughout (`medianfilterCPU`, `smooth1dCPU`, `np.correlate`, small-array
   `gpu_arraymedian`, which itself falls back to a CPU kernel under 2**16 elements). Several
   `fdu.getData(tag="cleanFrame")`/`skyFDU.getData()` calls in these functions omitted
   `force_cpu=True` even though a sibling function (skyline MOS) already had it right — added it
   everywhere in this family. **If you touch any of these four functions again, grep the whole
   function for every `.getData(` call and check force_cpu, not just the ones near your change.**
7. `xtrans_rect`/`ytrans_rect` in `rectifyMOS` can be built by different code paths that don't
   agree on numpy vs cupy (continuum trace vs. skyline trace vs. identity fallback vs. a calib
   cached from an earlier frame) — normalized both explicitly before combining rather than
   assuming they match. (An earlier attempt at this same run's crash converted the *mask* side up
   to cupy instead — wrong direction, reverted; see item 6, the mask was headed into CPU-only
   consumers.)
8. A shape-broadcast bug in *this session's own* `use_slitpos` runaway-fit sanity check (1D
   `xind` vs 2D `yind`/`z` for horizontal dispersion) — not a refactor artifact, a bug in code
   added earlier this session. Fixed by broadcasting `xind` to `yind`'s shape first.
9. `gpu_arraymedian()`'s `axis=="both"` branch handed CuPy input straight to
   `fatboyclib.median`/`fatboycudalib.gpumedian`/`cp_select.cpmedian`, all of which are C
   extensions needing genuine numpy (they handle GPU dispatch internally themselves; sibling
   `gpumedianS()` already converted defensively, this one didn't). Fixed *only* inside that
   branch — the sibling `axis!="both"` branch legitimately uses GPU-native `kernel2d`/`kernel3d`
   with `gputranspose`, converting there too would have wrongly forced it onto CPU.
10. `extractSpectra()`'s `'gaussian'` weighting branch referenced `extract_xlo`/`extract_xhi`,
    never defined anywhere in that function's scope (only in sibling `findSpectra()`). Confirmed
    via `main` this slicing never existed here pre-refactor at all — reverted to the original
    unsliced `np.sum(slit, 1)`/`np.sum(slit, 0)` rather than inventing the wiring for a feature
    that was never actually implemented.
11. A **pre-existing** (confirmed via `main`, predates this refactor) negative-index slice bug in
    `wavelengthCalibrateProcess.py`: `oned[blref-10:blref+11]` wraps around instead of clamping
    when the brightest line lands within 10px of an edge. Fixed all 8 occurrences with
    `max(blref-10,0)`. Flagged as algorithm-adjacent (this file is one of the four named for later
    review) but the fix itself is a minimal safety clamp, not a change to the algorithm.

**Not bugs, expected behavior:** a handful of `wavelengthCalibrateProcess` orders (2, 3, 25 for
`r3c1m2z_jhjh.0001`) logged `ERROR: Could not match 3 brightest lines ... Skipping order!` and
were gracefully skipped — this is the algorithm correctly giving up on genuinely faint/lineless
orders, not a crash. A few downstream `KeyError`-shaped prints (`'PCF0_S02'` etc. from
`getWavelengthSolution`) are the direct, harmless consequence of those same skipped orders having
no stored wavelength solution — cosmetic, not fatal, don't chase these.

**Validated by this run:** the numpy/math disambiguation pass and the PyCUDA→CuPy migration are
now confirmed sound end-to-end for MOS + longslit spectroscopy (imaging modes still untested).
Item 11 above is the only bug that predates this refactor; everything else was introduced by it.

## Algorithm-specific failure hardening: findSlitletProcess + rectifyProcess (2026-09-11)

**Status: implemented, NOT YET validated against real data.** The user is checking both against
real data next session — if either produces a visibly wrong slitmask/rectified frame, look here
first before assuming it's an unrelated bug. `new...Process.py` files (Gemini's experimental
designs) are explicitly out of scope for this and were not touched.

**`findSlitletProcess.traceOrders()`** used one shared `is_error` flag across its entire
per-slitlet loop — one bad segment out of possibly dozens (out of `nslits` slitlets) discarded
the *whole* image (`fdu.disable()`), throwing away every successfully-traced slitlet too. Fixed:
a segment that fails its fit (exception) or quality checks (coverage fraction, residual sigma)
now gets the same straight/uncurved fallback the code already used for "insufficient data"
(`np.zeros(xstride)` spliced into `z1`) instead of propagating a bad curve or aborting
everything. `z1` feeds both the CPU inline slitmask write and the GPU `yloMask`/`yhiMask` arrays
`createSlitmask` uses afterward, so one code path covers both — confirmed by reading
`createSlitmask`'s CUDA source (`fatboyLibs.py:317`) and `findRegions`' CPU path
(`fatboyLibs.py:2053`): the latter does `np.where(data==slitidx+1).min()`, which would crash on
a *fully empty* slit, which is exactly why the fallback keeps each slit's region present-but-
straight rather than empty. Only discards the whole image now if literally every slitlet needed
a fallback. QA file is now written whenever any slitlet was degraded, not only when discarding,
so partial failures are visible for review.

**`rectifyProcess.calculateMOSContinuaTrans()`** has three `mos_mode` branches
(`use_slitpos` — the default, `whole_chip`, `independent_slitlets`), each fitting a surface to
continuum trace points and writing the result into `ytransData`/`xtransData`, which `drihizzle`
sizes its output array from. None checked the fit's range before writing — a runaway high-order
extrapolation could turn a 2048x2048 image into something like 9216x12288. Added
`rectify_max_transform_factor` (default 2.0): after each fit, the transform values actually
about to be written (already masked to the relevant region) are checked against
`maxTransformFactor*ysize`; out-of-range triggers one retry at linear order; still-bad falls back
per-branch — `independent_slitlets` uses an identity transform for just that slit/segment (same
treatment as "no data"), `whole_chip`/`use_slitpos` discard the image (single global fit, no
per-slit region to fall back around). Added a shared `checkFitSanity()` method for the
`surfaceFunction`-based branches; `use_slitpos` uses `surface3dFunction`/`surface3dResiduals` (a
3-variable fit, different term-counting convention, GPU-accelerated eval via `calcTrans3d`) so it
has its own inline version rather than a forced/mismatched abstraction. Also escalated the
existing "no continuum data at all" identity-fallback (the original "out=in" report) from a lone
`WARNING` to `ERROR` + a per-call summary count (`n_slits_not_rectified`), so it can't quietly
scroll by.

**Verification performed:** compiles, pyflakes-clean (no new undefined names), and — critically —
the actual fit/sanity-check math was exercised directly with real `leastsq`/`surfaceFunction`/
`surface3dFunction` calls on synthetic data (not mocks): a well-behaved fit passes through
unchanged, an injected runaway high-order coefficient is detected and replaced by a linear
refit, and an impossibly tight bound correctly reports failure, for both the 2-variable and
3-variable fit paths. `traceOrders()` was *not* exercised end-to-end (needs real flat-field FITS
data and a full `fatboyDatabase`/XML setup) — verified by careful manual control-flow trace
instead. Neither has been run against a real instrument dataset yet.

**Known follow-up, not done tonight:** `calculateMOSSkylineTrans` has the identical "no data ->
identity, logged as WARNING" pattern at `rectifyProcess.py:2030` (skyline-based rectification,
sibling to the continuum-based function above) and was not audited for the same runaway-fit risk
— same class of issue, different function, deferred.

## Top-level error-handling audit (goal #4, 2026-09-11)

Per the user's request, audited the framework's failure design as a whole (not yet individual
algorithms — that's a separate later pass the user will drive with specific bad-data examples),
starting from the `fatboyDatabase.py` entry point. The question was: why does an otherwise-mature
pipeline (used on a dozen+ instruments) still crash outright on "unexpected bad data" instead of
degrading gracefully? Found three real gaps and fixed them:

1. **`fatboyDatabase.executeProcesses()`** (the per-image process loop): its exception handler
   called `input("Press ENTER to continue")` *unconditionally*. In any unattended/scripted run
   (cron, cluster, a shell loop over many nights) stdin is closed, so `input()` raises `EOFError`
   immediately — uncaught, since it's outside the `try` that caught the original error — crashing
   the whole batch on top of whatever originally failed. Now gated behind a new `interactive_on_error`
   param (default `'no'`, matching the existing `prompt_for_missing_dark`-style opt-in pattern) and
   wrapped in its own `try/except EOFError` even when opted in. Also now calls `image.disable()` on
   an unexpected exception, since the FDU's data state after a partial crash mid-process is unknown
   and shouldn't be trusted for later steps.
2. **`fatboyProcess.recursivelyExecute()`**: the method ~23 calibration-building processes
   (darkSubtract, flatDivide, biasSubtract, etc.) use to pre-process raw calibration frames (e.g.
   linearity-correcting individual darks before combining into a master dark). Had *zero* exception
   handling and discarded the process's success/failure return value entirely — so one corrupt
   calibration frame crashed the whole run, bypassing the graceful "no master dark found → disable
   this one science frame, keep going" fallback that callers like `darkSubtractProcess.execute()`
   already implement correctly one level up. Now catches exceptions and honors a `False` return,
   disabling and discarding just that one calibration frame either way.
3. **Both of the above, plus `executeProcesses()`**, now add a lightweight postcondition check: a
   process reporting `success=True` is no longer trusted blindly — if it left the FDU with no
   readable data, that's caught and disabled right where it broke, instead of letting corrupted
   state silently propagate through the rest of the process chain until something unrelated crashes
   far from the root cause. All three fixes verified with synthetic tests exercising the actual code
   paths (partial vs. all-failed for `initializeAll`; raise/`False`/lied-about-success/genuine-success
   for `recursivelyExecute`), not just import-time checks.

**Deliberately different from a "just disable and continue" approach — ingestion (`initializeAll`,
which calls `fdu.readHeader()`/`initialize()`/`reformatData()` before any processing starts):** the
user pushed back that a malformed input FITS file is usually a *user/config* error worth knowing
about immediately, not something to quietly paper over. Landed on a hybrid: a single bad file (or a
few) is isolated — logged loudly (`ERROR` with filename + exception), that one FDU disabled, run
continues — but if **more than `max_init_failures` files fail (default 3), or literally every file
fails** (covers small batches, e.g. 2 of 2, that wouldn't trip a `>3` threshold), that's treated as
systemic (wrong directory, wrong instrument, bad file list) and the run aborts with `sys.exit(-1)`
rather than silently limping along on whatever's left. `max_init_failures` is a normal XML-overridable
param. This went through two iterations — first "abort only if literally everything failed," which
the user correctly pointed out was too narrow (15 bad files out of 20 wouldn't have tripped it) —
so if this threshold needs tuning again, that's expected; it's a judgment call, not a derived constant.

**Not in scope for this pass, noted but not touched:** `newFindSlitletProcess.py`'s bare `except:`
around a spline fallback (low severity, and that file — along with any other `new...Process.py` —
is explicitly out of scope: the user said these were Gemini's experimental designs, not to worry
about them for now). Also not touched: the ~6 processes with option-gated `input()` prompts for
picking a calib file (`prompt_for_missing_dark` etc.) — those are already opt-in, not a default
hazard. Also not touched: the broader "no enforced success/failure contract across ~80 process
files" observation (e.g. `removeCosmicRaysProcess.execute()` always returns `True` regardless of
what happened inside) — the postcondition check above is a generic safety net for this, but making
every individual process honest about its own success is algorithm-specific work for the later pass.

## Status of the numpy/math disambiguation pass (goal #1)

As of 2026-09-10, `from numpy import *` and `from math import *` no longer appear anywhere in
`superFATBOY/` — the last two (`removeCosmicRaysProcess.py`, `drihizzle.py`) were removed in this
session. Fixing them surfaced three latent `NameError` bugs where a bare `sqrt`/`tan`/`sin`/`cos`/`pi`
call had no valid import path once the wildcard import was gone — these were real crashes waiting to
happen, not just style. Fixed in `miradasDARFromConditionsProcess.py`, `rectifyProcess.py`, and
`drihizzle.py`.

**Disambiguation rule used:** if the argument is a scalar (a single float pulled from a header
keyword, a fit coefficient, a loop-scalar), use `math.foo`. If the argument is a numpy array or
array slice, use `np.foo`. `abs()`/`pow()`/`round()`/`min()`/`max()`/`sum()` are Python builtins
that dispatch correctly on both scalars and numpy arrays via `__abs__`/etc — leaving them bare is
not a bug and not worth a mechanical sweep. `math.sqrt` vs `np.sqrt` matters because `math.sqrt`
throws on arrays and negative numbers, while `np.sqrt` returns `nan` and is vectorized — pick based
on what the call site actually receives.

**How this was verified**, and how to re-verify after further changes:
1. `grep -rn "^from numpy import \*\|^from math import \*" --include="*.py" superFATBOY` should
   return nothing.
2. `python3 -m pyflakes $(find superFATBOY -name "*.py" -not -path "*/build/*")` — genuine
   `undefined name` errors are the signal to act on. The `'from X import *' used; unable to detect
   undefined names` lines are expected noise from this project's *local* wildcard imports (e.g.
   `from superFATBOY.fatboyLibs import *`) — those export the project's own helper functions
   (including the intentionally-kept `arraymedian`/`gpu_arraymedian`, see GEMINI.md item 2) and are
   not part of this cleanup; pyflakes just can't see through them.
3. A number of `.py` files still contain embedded CUDA C kernel source as string literals (compiled
   via cupy `RawKernel`/`fatboy_mod`). Those bodies use C's `sqrt`/`exp`/`abs` etc. legitimately —
   don't "fix" them, and account for them when grepping (they produce false positives for any
   Python-level import scan).

**Not yet swept:** the local wildcard imports (`from superFATBOY.fatboyLibs import *` and similar,
~70 files) are unrelated to the numpy/math ambiguity and were left alone — they're the project's own
namespace, not stdlib/numpy collisions. If a future pass wants to clean those up too for the same
"proper python syntax" reasons, treat it as a separate task from this one since pyflakes can't help
verify it (see point 2 above) and it needs its own review per file.

## Build failure fixed (2026-09-11): setup.py's numpy>=2.0 pin

`sudo python setup.py install` was failing at the `pkg_resources.require()` step (the deprecated
easy_install console-script wrapper) with a numpy/scipy version conflict. Root cause: the gemini-cli
refactor commit (5b4bbbd) bumped `install_requires` from `numpy>=1.0` to `numpy>=2.0` in `setup.py`
with no numpy-2.0-only API actually used anywhere (checked — nothing uses `np.trapezoid`,
`numpy.exceptions`, `np.astype()`, etc.). That pin conflicted with the environment's scipy 1.11.3,
which is ABI-locked to `numpy<1.28`. Reverted to `numpy>=1.0`. If a real numpy>=2.0 dependency gets
introduced later, scipy needs upgrading past 1.11.3 in the same change, or this will break again the
same way.

## CuPy migration audit (2026-09-11)

Grepped the whole tree for PyCUDA-only APIs (`gpuarray`, `to_gpu`, `mem_alloc`, `ElementwiseKernel`,
`pycuda.driver`, `.autoinit`, `cumath`) that should have been converted to CuPy equivalents. Found
and fixed two real bugs, both reachable at runtime (not just style):

- `fatboyLibs.py`'s `gpusum()` still called `gpuarray.to_gpu()`, `gpuarray.sum()`, and a PyCUDA-style
  `ReductionKernel` — none of which exist without `import pycuda`, so calling it (it's used from
  `gpu_pysurfit.py`) threw `NameError`. Rewrote with `cp.asarray`/`cp.sum` and `cupy.ReductionKernel`
  — note CuPy's `map_expr` addresses the element directly as the param name (e.g. `x`), not as a
  pointer (`x[i]`) like PyCUDA's did. Verified against a real GPU in this environment (thresholded
  and nonzero cases matched the original semantics exactly).
- `superFatboy3.py`'s `-gpu N` flag set `CUDA_DEVICE` (PyCUDA's `autoinit` device-selection variable),
  which CuPy never reads — so the flag silently did nothing post-migration. Fixed to set
  `CUDA_VISIBLE_DEVICES` instead (verified CuPy actually honors it) and must still be set before any
  transitive `import cupy` happens later in the script.

Everything else that matched a PyCUDA-shaped grep turned out to be either a stale comment (harmless)
or already-correct CuPy (`cp.RawModule`, `cp.fft`, etc.) — see the commit for the full list checked.

**Pre-existing CPU/GPU inconsistency — fixed 2026-09-11:** `pysurfit.py`'s CPU path computed std with
`ddof=1` (sample variance / Bessel's correction — the right choice here, since the residuals are a
sample used to *estimate* the fit's underlying noise, not the whole population); `gpu_pysurfit.py`'s
GPU path used CuPy's default `ddof=0`, biasing its std estimate low. Predated the refactor (same on
`main`), so it wasn't part of the original migration bug sweep, but once flagged it was a one-line
fix: `residb_gpu.std()` → `residb_gpu.std(ddof=1)`. Verified CuPy's `ddof=1` agrees with numpy's to
float precision on a real GPU.

## `.sum()/N` sweep (goal #2, 2026-09-11)

Audited for the numarray-era `x.sum()/N` (instead of `x.mean()`) and manually-expanded variance
formulas per GEMINI.md item 2. Fixed three call sites where the divisor is provably the full size of
the array being summed (verified numerically that the manual formula and `.mean()`/`.std(ddof=1)`
agree to float precision): `pysurfit.py`'s `tempmean`/`tempstddev`, `imcombine.py`'s unmasked
per-image std branch, and `badPixelMaskSpecProcess.py`'s `np.sum(pts)/npts` on a plain list.

**Deliberately left alone** — these look like the same pattern but aren't:
- `imcombine.py`'s *masked* mean/std branches (`immean = ((data+0.)*b).sum()/nb`, threshold/nonzero
  cases) — `nb` is a threshold-mask count, not the array's full size, so this is a masked mean/std
  that `.mean()`/`.std()` can't reproduce without first materializing `data[b]`.
- `imcombine.py`'s per-pixel frame-stack combine (`reduce(np.add, inp*inp) ... avg*avg*n/nm1`,
  ~15 call sites across the file) — `n`/`nm1`/`tmask` are **per-pixel** valid-frame counts that vary
  spatially because different input frames can mask different pixels. This is the pipeline's core
  co-add statistics engine; a plain axis-wise `.mean()`/`.std()` cannot reproduce per-pixel ragged
  masking, and getting this wrong would corrupt science reduction. Don't touch it without dedicated
  test data and the user's sign-off — this belongs in the algorithmic-improvement phase, not cleanup.
- Anywhere a `.sum()` is counting a boolean mask (e.g. `badpix[:51].sum()//2`) or computing a ratio
  of two sums (`cut1d.sum()/islit.sum()`) — not a mean at all, skip.
- `arraymedian.py`/`gpu_arraymedian.py` and their quickselect implementations — explicitly excluded
  per GEMINI.md item 2, keep as-is.

## Untracked files present at session start (not yet triaged)

`GEMINI.md`, `test`, `test.py`, `superFATBOY/data/config/wc_miradas_sos_rev*.xml`,
`superFATBOY/datatypeExtensions/fourStarImage.py`,
`superFATBOY/fatboyProcesses/mosaicFourStarProcess.py`, `superFATBOY/superFATBOYPlot/sfbPlot.jar`
and `sfbPlot/`. These look like in-progress work (a FourStar imaging mode, a triangle-matching
registration prototype in `test.py`) from outside this refactor's scope — left as-is. Ask the user
before adding, deleting, or committing any of them.

## Coding style (carried over from GEMINI.md)

- 4-space indents, standard Python style.
- No multi-statement lines (`x, y, z = True, False, True` or semicolon-chained statements).
- New functions/classes get a short comment; don't add multi-paragraph docstrings to old code just
  because you're touching it nearby.
- Commit after every meaningful change, on the `refactor` branch, with a message that explains why.
