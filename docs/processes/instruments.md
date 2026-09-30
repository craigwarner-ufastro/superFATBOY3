<p align="center"><img src="../images/superFATBOY.png" alt="superFATBOY" width="110"></p>

# Instrument-specific processes

*[Docs home](../README.md) · [Process guide](README.md) · [Imaging](imaging.md) · [Spectroscopy](spectroscopy.md)*

Some instruments need steps of their own: reading an unusual file layout, removing an instrument-specific artifact, or building
data products (3-d datacubes) that only make sense for that instrument. This page lists them. A process's *mode tag*
(visible with `superFatboy3.py -list <tag>`, for example `-list miradas`) says which family it belongs to.

> **Validation status.** Only the MIRADAS and EMIR-imaging sections below describe processes that have been run end-to-end on the current Python 3 code
> (see [instruments](../instruments.md)). The MEGARA, SINFONI and CIRCE processes were migrated with the rest of the code
> and some were reviewed, but no complete reduction has been run through them yet. Treat them as a starting point, and read the code when
> in doubt.

- [MIRADAS](#miradas)
- [EMIR](#emir)
- [Trimming](#trimming)
- [MEGARA](#megara)
- [SINFONI](#sinfoni)
- [CIRCE](#circe)

---

## MIRADAS

MIRADAS is an image-slicer IFU fed by slits. Each slit (a *slitlet*) has three slices, so a frame of 12 slitlets has 36 spectra. The MIRADAS-specific
processes collapse slices into spaxel images, build 3-d datacubes, measure the PSF, correct differential atmospheric refraction (DAR), and stitch orders together.
The general spectroscopic processes (`findSlitlets`, `rectify`, `wavelengthCalibrate`, ...) do the rest.

**Every process is described, with its options and the recommended settings, in the [MIRADAS guide](../miradas.md).** In brief:

| Process | What it does |
|---|---|
| `miradasCollapseSpaxels` | Collapses each slice of each slitlet into a monochromatic 3-spaxel by N image and centroids it |
| `miradasCreate3dDatacubes` | Builds a 3-d datacube for each slitlet, where each cut is a monochromatic image at one wavelength |
| `miradasCombineSlices` | Combines the three slices of a slitlet into a single 1-d spectrum, weighting for relative slice illumination |
| `miradasStitchOrders` | SOL and SOS only: stitches the orders into one spectrum on a common wavelength scale |
| `miradasCharacterizePSF` | Traces the PSF (centroid, FWHM, peak) at each wavelength cut through the datacube |
| `miradasDARFromConditions` | Estimates DAR from site conditions (pressure, temperature, water vapour, zenith angle) |
| `miradasDARFromData` | Measures DAR from the centroid of a point-like source |
| `miradasRegisterWCS` | Copies probe-arm pointing positions from the header into per-slitlet WCS keywords |

## EMIR

### emirBiasSubtract

Subtracts the detector bias level of an EMIR frame using its reference pixels. It takes the sigma-clipped mean, column by column, of the top rows of the frame, subtracts
it, and then trims the four-pixel border of reference rows and columns from all four edges (so the output is 8 rows and 8 columns smaller). It has no options of its own.
It replaces `darkSubtract` in the [EMIR imaging template](../instruments.md#templates-provided-but-not-yet-in-the-validated-set).
Output: `emirBiasSubtracted/ebs_*.fits`.

## Trimming

### trimOverscan

Trims overscan (and other unwanted) rows and columns from a frame, and flags it with a FITS keyword so that it is not trimmed twice. Mode tags: imaging and spectroscopy.

| Option | Default | Meaning |
|---|---|---|
| `overscan_cols`, `overscan_rows` | `None` | Columns or rows to **remove**. Supports slices: `320:384, 500, 752:768` |
| `python_slicing` | `no` | `no`: `min:max` includes `max` (inclusive, FITS style). `yes`: Python style, excluding `max`. |
| `fits_keyword_flag` | `OTRIMMED` | Keyword added to the header when a frame has been trimmed |
| `flag_true_value` | `1` | If the keyword already has this value, the frame is treated as already trimmed and left alone |

Put this first in the chain, before anything that depends on the image shape (including `findSlitlets`). Trimming the same frames twice with different settings, for example once
as a calibration and again as data, shifts every later coordinate, and is easy to do by accident.
Output: `trimmedOverscan/to_*.fits`.

### trimWindow

Trims an image to a window, given as numbers or as FITS keywords. Used for CIRCE data, where the usable window is recorded in the header. Mode tag: circe.

| Option | Default | Meaning |
|---|---|---|
| `window_xmin`, `window_xmax`, `window_ymin`, `window_ymax` | `None` | Each a number or the name of a FITS keyword that holds it |
| `python_slicing` | `no` | As above |
| `fits_keyword_flag`, `flag_true_value` | `None`, `1` | As above: a keyword that tells whether the data has already been windowed |

## MEGARA

MEGARA is a fibre-fed spectrograph (GTC). Its datatype is `megaraSpectrum`. Fibres are found with `findSlitlets` using the peak-finding options
(`autodetect_peak_local_max`, `trace_peak_local_max`, `fiber_width`); the rest of the chain uses the general spectroscopic processes plus these.

| Process | What it does | Key options |
|---|---|---|
| `trimOverscan` | Removes the overscan regions first (see above). Applying it more than once is a known trap (see the note there). | `overscan_cols`, `overscan_rows` |
| `megaraIdentifyFibers` | Assigns an ID to every fibre found in the slitmask | `missing_fiber_list`: comma-separated fibre numbers (starting at 1) that are dead or absent |
| `collapseFibers` | Uses the slitmask to collapse the data across each fibre's width into a single row per fibre (output: `collapsedFibers/cf_*.fits`) | `collapse_method`: `sum`, `mean` or `median` |
| `megaraSkySubtract` | Builds a master sky from the designated sky fibres and subtracts it | `scaling`: `none`, `peak`, `skylines` (how to scale the sky to each fibre); `scaling_nlines`; `sky_combine_method`: `median` or `mean`; `keep_skies`; `write_plots` |

The MEGARA ThArNe line list is `megara_ThArNe_list.dat`.

## SINFONI

SINFONI is an image-slicer IFU on the VLT. The `sinfoni*` processes handle its 32-slitlet layout and produce 3-d datacubes. All have the mode tag `sinfoni`.

| Process | What it does | Key options |
|---|---|---|
| `sinfoniCalcLinearity` | Calculates the detector's linearity coefficients from lamp on and lamp off exposures | `lamp_method` (`lamp_on-off` by default), `lamp_off_files`, `lamp_off_header_value`, `linearity_fit_order` (2) |
| `sinfoniRemoveBadLines` | Finds and removes bad detector lines from the background | `nsigmaback` (18): sigma for identifying deviant background pixels |
| `sinfoniIdentifySlitlets` | Numbers the slitlets in the slitmask left to right | `slitorder`: a comma-separated list, or a text file, giving the slitlet order |
| `sinfoniCollapseSlitlets` | Uses a clean sky or arclamp line to find the shifts between slitlets and collapses them into a 2-d image | `reference_slit` (2), `collapse_method` (`sum` or `median`), `line_search_box_lo/hi`, `use_arclamps`, `use_integer_shifts`, `do_centroids`, `centroid_method` |
| `sinfoniCreate3dDatacubes` | Builds a 3-d datacube (one monochromatic image per cut). Needs wavelength calibration first. | `reference_slit`, `drihizzle_kernel`, `drihizzle_dropsize`, `use_arclamps`, `use_integer_shifts` |
| `sinfoniCharacterizePSF` | Traces the PSF at each cut through a datacube | `detection_threshold` (2), `centroid_method` |
| `sinfoniRegisterStack` | Aligns and stacks the collapsed images of several frames | The same alignment options as [`alignStack`](imaging.md#alignstack), including `align_method` (default `xregister`) |

## CIRCE

CIRCE is a near-IR imager on the GTC whose FITS files hold several "ramps" per file. Its datatypes are `circeImage` and `circeFastImage`; the
frames in each file are handled as separate *sections* and merged later. The imaging processes (`linearity`, `darkSubtract`, `flatDivide`, `badPixelMask`, `skySubtract`, `cosmicRays`,
`alignStack`) are used for the reduction itself, with these additions. All have the mode tag `circe` unless noted.

| Process | What it does |
|---|---|
| `deboneCirce` | Removes the CIRCE "bone" pattern: the detector's 32 amplifiers (64 columns each) share a repeating column pattern. The process takes the median column profile across amplifiers and subtracts it from each amplifier, then re-masks the bad pixels. Expects a 2048 by 2048 frame. |
| `remergeCirce` | After the per-ramp calibration steps, renames the per-section frames so they are treated as one frame again |
| `mergeObjects` (imaging and circe) | Merges groups of object frames under a common name, from the `merge_name` property you give them in `<queries>`, so they are stacked as one target |
| `trimWindow` | See above |

Useful `alignStack` methods for CIRCE (which were written for its bad-column and bright-star problems) are `xregister_constrained`, `xregister_sep_constrained` and `sep_centroid_constrained`:
all use RA, Dec and pixel scale from the header to make an initial guess and then refine within a window.
