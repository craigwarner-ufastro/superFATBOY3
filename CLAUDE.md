# Project: superFATBOY3

## Context

This refactor was started by Gemini CLI (see `GEMINI.md` for the original brief) and is being
continued here. Read `GEMINI.md` first — it lays out the four refactor goals (drop
`from numpy import *`, drop numpy-bug-era tricks like `.sum()/N`, migrate PyCUDA to CuPy,
improve error handling) and the specific algorithms flagged for later improvement
(findSlitletProcess, removeCosmicRaysSpecProcess, rectifyProcess, wavelengthCalibrateProcess).
Work happens on the `refactor` branch, one commit per meaningful change.

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
