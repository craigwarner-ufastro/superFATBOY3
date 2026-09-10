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
