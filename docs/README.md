<p align="center">
  <img src="images/superFATBOY.png" alt="superFATBOY mascot" width="220">
</p>

# superFATBOY documentation

**superFATBOY** (*Florida Analysis Tool Born Of Yearning for high quality scientific data*) is a general-purpose,
extensible data-reduction pipeline for astronomical imaging and spectroscopy. It was written to reduce near-IR data
but handles mid-IR and optical data too. You describe a reduction in an XML file (which frames you have, which steps to run,
and with what settings), and superFATBOY runs the steps in order, on CPU or on an NVIDIA GPU.

This is the Python 3 / CuPy version. It descends from a Python 2 / PyCUDA pipeline (superFATBOY 2.x), which itself
descends from the original FATBOY, the pipeline for the Flamingos-2 near-IR imager and spectrograph.

## Where to start

| If you want to... | Read |
|---|---|
| Install superFATBOY and reduce your first dataset | [Quick start](quickstart.md) |
| Know whether your instrument is supported, and find its template XML | [Instruments and templates](instruments.md) |
| Write or edit an XML configuration file | [XML style guide](xml-guide.md) |
| Understand what each reduction step does and which options matter | [Process guide](processes/README.md) |
| Look up every option and its default | [Options reference](options-reference.md) |
| Reduce MIRADAS data (SOL, SOS or MOS mode) | [MIRADAS guide](miradas.md) |
| Find or build a line list for wavelength calibration | [Line lists and `makeLineList.py`](instruments.md#line-lists-and-wavelength-calibration-files-shipped-with-superfatboy) |
| Write your own process or datatype, or script the pipeline from Python | [API guide](api.md) |

## The idea in thirty seconds

```
  XML file                     superFATBOY                          output directory
 ┌─────────────┐      ┌──────────────────────────────┐      ┌──────────────────────────┐
 │ <queries>   │ ───▶ │ find and read the FITS files │      │ darkSubtracted/ds_*.fits │
 │ <processes> │      │ for each process, in order:  │ ───▶ │ flatDivided/fd_*.fits    │
 │ <parameters>│      │   find the calibrations it   │      │ rectified/rct_*.fits     │
 └─────────────┘      │   needs, build them if       │      │ extractedSpectra/es_*    │
                      │   necessary, apply them      │      │ ... plus logs (flogs/)   │
                      └──────────────────────────────┘      └──────────────────────────┘
```

- **`<queries>`** describes your data: where it lives, which files are science frames, darks, flats, arclamps and so on.
- **`<processes>`** lists the reduction steps to run, in order, each with its options.
- **`<parameters>`** holds global settings: output directory, GPU on/off, FITS keyword names.

Each step is a *process*, a small Python class (`darkSubtract`, `flatDivide`, `rectify`, `wavelengthCalibrate`, ...).
A process does not need to be told where its calibration frames are. It asks the database for darks that match the
science frame's exposure time, builds a master dark if none exists yet, and moves on. The same machinery lets one
process quietly run an earlier step on a calibration frame, for example linearizing the raw darks before combining them.

## Conventions used in these docs

- `code font` is something you type or see in an XML file, a FITS header or the terminal.
- "GPU mode" means `gpumode` is `yes` and CuPy found a CUDA device; "CPU mode" means `gpumode` is `no`.
  Results from the two modes agree to floating-point rounding (see [validated instruments](instruments.md)).
- Options are always written as `option_name = default`, and the default is what you get if you omit the option.

## Other files in this repository

- [`README.md`](../README.md): the short project README with install commands for HPC environments.
- [`CHANGELOG.md`](../CHANGELOG.md): a running, human-readable list of changes made during the Python 3 refactor.
- [`CLAUDE.md`](../CLAUDE.md) and [`GEMINI.md`](../GEMINI.md): working notes for the AI assistants that carried out the
  refactor. They are dense, but they hold the full story behind many design decisions, including why some
  algorithm options exist and what has and has not been validated.

## Contact

Craig Warner, University of Florida: cwarner at astro.ufl.edu.
