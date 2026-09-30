<p align="center"><img src="images/superFATBOY.png" alt="superFATBOY" width="110"></p>

# Quick start

*[Docs home](README.md)*

This page takes you from a fresh checkout to a finished reduction. It assumes a Linux machine; superFATBOY has
been developed and tested on Linux only.

## 1. Requirements

| Needed | For | Notes |
|---|---|---|
| Python 3 (3.8 or newer) | everything | |
| `numpy`, `scipy`, `astropy` | everything | FITS I/O is through `astropy.io.fits` |
| a C++ compiler, Python headers | everything | one small C++ extension (`fatboyclib`) is built during install |
| `matplotlib` | optional | QA plots and `debug_mode` |
| `sep` (preferred) or `sextractor` | optional | object detection, mainly for imaging sky subtraction and the `triangles` and `sep_*` alignment methods |
| CUDA toolkit, `cupy` | optional | GPU mode. Pick the CuPy build that matches your CUDA version (for example `cupy-cuda12x`) |
| `deepCR` | optional | only for `cosmic_ray_algorithm = deepcr` |

No GPU? That is fine. Set `<param name="gpumode" value="no"/>` and everything runs on the CPU, just more slowly.

## 2. Install

```bash
# (Optional, GPU users) build the CUDA helper libraries first.
# CUDA_HOME must point at your CUDA install; the Makefile guesses the rest.
export CUDA_HOME=/usr/local/cuda
cd superFATBOY3/superFATBOY
make gpu3

# Install the package and the superFatboy3.py command (system-wide)
cd ..
sudo python3 setup.py install
```

If you cannot use `sudo`, use `python3 setup.py install --prefix=/path/to/install` and add that prefix's `bin` and
`lib/python3.x/site-packages` to your `PATH` and `PYTHONPATH`.

**Running straight from the source tree** also works, and is the safest way to be sure you are testing the code
you just edited. An installed copy can be stale:

```bash
PYTHONPATH=/path/to/superFATBOY3 python3 /path/to/superFATBOY3/superFATBOY/superFatboy3.py my_data.xml
```

**HPC clusters** vary a lot (compatible GCC and CUDA versions, module names). The [main README](../README.md) has a
worked recipe for HiperGator using `conda` and `module load`.

Check that it works:

```bash
superFatboy3.py -h
superFatboy3.py -list | head -40
```

## 3. The command line

| Command | What it does |
|---|---|
| `superFatboy3.py my_data.xml` | Run the reduction described by `my_data.xml` |
| `superFatboy3.py -list` | Print every global parameter, every process, and every option with its default |
| `superFatboy3.py -list miradas` | The same, limited to the processes of one mode tag: `imaging`, `spectroscopy`, `miradas`, `circe`, `megara`, `sinfoni` or `emir` |
| `superFatboy3.py -config` | List the configuration files shipped inside the package (line lists, wavelength-calibration XML files) |
| `superFatboy3.py my_data.xml -gpu 1` | Run on GPU number 1 (sets `CUDA_VISIBLE_DEVICES` before CuPy loads) |
| `superFatboy3.py -h` | Short usage message |

`-list` is always correct for the version you have installed, because it is generated from the code. A snapshot of it is
in the [options reference](options-reference.md).

## 4. Reduce your first dataset

### Step 1: copy the template for your instrument

Template XML files live in [`superFATBOY/data/templates/`](../superFATBOY/data/templates/). Each one is a complete,
commented, working configuration with placeholder paths. The [instruments page](instruments.md) explains which to pick
and what has been validated.

| Template | Instrument and mode | Notes |
|---|---|---|
| [`FLAMINGOS1_imaging_template.xml`](../superFATBOY/data/templates/FLAMINGOS1_imaging_template.xml) | FLAMINGOS-1, near-IR imaging | The best-validated template. Good starting point for any near-IR imager. |
| [`EMIR_2024_imaging_template.xml`](../superFATBOY/data/templates/EMIR_2024_imaging_template.xml) | EMIR (GTC), imaging | Calibrations in sub-directories; uses sky flats |
| [`FLAMINGOS1_MOS_template.xml`](../superFATBOY/data/templates/FLAMINGOS1_MOS_template.xml) | FLAMINGOS-1, multi-object spectroscopy | Includes the standard-star calibration chain |
| [`OSIRIS_2019_longslit_template.xml`](../superFATBOY/data/templates/OSIRIS_2019_longslit_template.xml) | OSIRIS (GTC), longslit | `datatype="osirisSpectrum"`, calibrations in sub-directories |
| [`KAST_longslit_template.xml`](../superFATBOY/data/templates/KAST_longslit_template.xml) | KAST (Lick/Shane), dual-arm longslit | Two datasets, tagged `blue` and `red` |
| [`MIRADAS_SOL_template.xml`](../superFATBOY/data/templates/MIRADAS_SOL_template.xml) | MIRADAS, single-object long (SOL) | 12 slitlets |
| [`MIRADAS_SOS_template.xml`](../superFATBOY/data/templates/MIRADAS_SOS_template.xml) | MIRADAS, single-object short (SOS) | 13 slitlets |
| [`MIRADAS_MOS_template.xml`](../superFATBOY/data/templates/MIRADAS_MOS_template.xml) | MIRADAS, multi-object (MOS) | 12 slitlets, one per probe arm |

```bash
cp superFATBOY/data/templates/FLAMINGOS1_imaging_template.xml ~/reductions/my_data.xml
```

> The templates are more than examples: each one records the option values that worked for a real dataset, with
> comments explaining the choices. Reading the one closest to your instrument is the fastest way to learn what
> matters.

### Step 2: edit `<queries>` to describe your data

Point `dir` at the top-level directory of your data, then tell superFATBOY which files are which:

```xml
<dataset dir="/data/2025-03-10/" tag="mydata">
  <object type="auto" name="ngc1569_k" prefix="ngc1569">
    <index start="1" stop="9"/>
  </object>
  <calib type="dark" name="darks" prefix="dark"/>
  <calib type="flat" name="flat_on"  prefix="flatkon">
    <property name="flat_type" value="lamp_on"/>
  </calib>
  <calib type="flat" name="flat_off" prefix="flatkoff">
    <property name="flat_type" value="lamp_off"/>
  </calib>
</dataset>
```

`prefix="ngc1569"` matches `/data/2025-03-10/ngc1569*.fits`; `<index start="1" stop="9"/>` keeps frames 1 to 9.
Some instruments put the index first and the date and instrument name after (MIRADAS, EMIR, OSIRIS); use `suffix`
for those. The full story is in the [XML style guide](xml-guide.md).

### Step 3: edit `<parameters>`

At minimum:

```xml
<parameters>
  <param name="outputdir" value="/data/reductions/ngc1569"/>   <!-- must already exist -->
  <param name="gpumode" value="yes"/>                          <!-- "no" without an NVIDIA GPU -->
  <param name="memory_image_limit" value="100"/>               <!-- about 10 x your RAM in GB -->
</parameters>
```

`outputdir` must exist before you run (superFATBOY creates the sub-directories inside it, but not `outputdir` itself).
A larger `memory_image_limit` keeps more intermediate frames in memory instead of spilling them to disk, which is faster.

Most FITS keyword names (exposure time, filter, RA and Dec, and so on) are auto-detected from a list of common names.
If yours differ, override them with `<param name="exptime_keyword" value="EXPOSET"/>` and similar.

### Step 4: choose and tune the processes

The template already lists the right processes in the right order. You can:

- **Remove** a step you do not need (comment it out, or delete it).
- **Reorder** steps. They run top to bottom.
- **Save intermediate products** with `<option name="write_output" value="yes"/>` on any process, and save
  calibration products (master darks, flats, bad pixel masks, slitmasks, ...) with `write_calib_output`.
- **Change an option.** Run `superFatboy3.py -list` or see the [process guide](processes/README.md) for what is available.

Things that most often need changing: the linearity coefficients, `flat_method`, `sky_subtract_method` (imaging) or
`sky_method` (spectroscopy), and `align_method`.

### Step 5: run it

```bash
mkdir -p /data/reductions/ngc1569
cd ~/reductions
superFatboy3.py my_data.xml
```

Progress is printed as it runs and also logged under `flogs/` in the directory you ran from. A full spectroscopic
reduction can take 15 to 25 minutes per dataset; imaging depends on the number of frames.

### Step 6: look at the output

Each process with `write_output = yes` writes into its own sub-directory of `outputdir`, with a short filename prefix:

| Directory | Prefix | Written by |
|---|---|---|
| `linearized/` | `lin_` | `linearity` |
| `darkSubtracted/` | `ds_` | `darkSubtract` |
| `biasSubtracted/` | `bs_` | `biasSubtract` |
| `flatDivided/` | `fd_` | `flatDivide`, `flatDivideSpec` |
| `badPixelMaskApplied/` | `ba_` | `badPixelMask`, `badPixelMaskSpec` |
| `skySubtracted/` | `ss_` | `skySubtract`, `skySubtractSpec` |
| `findSlitlets/` | `slitmask_`, `qa_`, `stats_` | `findSlitlets` (the slitmask and QA files) |
| `rectified/` | `rct_`, `qa_`, `stats_` | `rectify` |
| `alignedStacked/` | `as_`, `exp_`, `objmap_` | `alignStack` (final image, exposure map, object map) |
| `extractedSpectra/` | `es_` | `extractSpectra` (1-d spectra; one row per spectrum) |
| `calibStarDivided/` | `csd_` | `calibStarDivide` |
| `masterDarks/`, `masterFlats/`, `cleanSkies/`, `masterArclamps/` | | master calibrations (`write_calib_output`) |

For imaging, your finished image is `alignedStacked/as_<name>.fits`. For spectroscopy it is in
`extractedSpectra/` (or `calibStarDivided/` if you ran that step).

Re-running is safe: most processes notice that their output file already exists and read it back instead of
recomputing it, unless `overwrite_files` is `yes`. This makes it cheap to fix a problem late in the chain and rerun.
Delete the relevant sub-directory (or set `overwrite_files` to `yes`) to force a step to run again.

## 5. Tips for unattended and batch runs

- Keep `debug_mode` at `no`. With `yes`, several processes pop up an interactive matplotlib window and wait for you.
  Use `write_plots` set to `yes` instead to save the same QA plots as PNG files.
- By default a failure does not stop for keyboard input. If you *want* it to stop and wait after an error
  (for debugging at a terminal), set `<param name="interactive_on_error" value="yes"/>`.
- A failure inside one process disables just the affected frame and the run continues with the rest. A bad input
  file is isolated and logged, but if more than `max_init_failures` (default 3) files fail to read, or all of them do,
  the run aborts, because that almost always means a wrong directory or wrong `datatype`.
- Some processes can prompt for a calibration file when they cannot find a match (`prompt_for_missing_dark`,
  `prompt_for_missing_flat`). Set these to `no` for batch runs, and either comment the step out or supply the
  calibration explicitly.

## 6. Where next

- [Instruments and templates](instruments.md): what has been validated, and what has not.
- [XML style guide](xml-guide.md): every tag and attribute.
- [Process guide](processes/README.md): what each step does.
- [MIRADAS guide](miradas.md): a worked, in-depth example of a spectroscopic reduction.
- [API guide](api.md): running superFATBOY from Python and writing your own processes.
