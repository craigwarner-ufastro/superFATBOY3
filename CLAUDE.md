# Project: superFATBOY3

## Context

This refactor was started by Gemini CLI (see `GEMINI.md` for the original brief) and is being
continued here. Read `GEMINI.md` first — it lays out the four refactor goals (drop
`from numpy import *`, drop numpy-bug-era tricks like `.sum()/N`, migrate PyCUDA to CuPy,
improve error handling) and the specific algorithms flagged for later improvement
(findSlitletProcess, removeCosmicRaysSpecProcess, rectifyProcess, wavelengthCalibrateProcess).
Work happens on the `refactor` branch, one commit per meaningful change.

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

**Known pre-existing inconsistency, not touched:** `pysurfit.py`'s CPU path computes std with
`ddof=1`; `gpu_pysurfit.py`'s GPU path uses CuPy's default `ddof=0`. This predates the refactor
(same on `main`) — it's an algorithmic discrepancy between the two backends, not something the CuPy
migration introduced, so it's deferred to the later algorithmic-improvement pass rather than fixed
here.

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
