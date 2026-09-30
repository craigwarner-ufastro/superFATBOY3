<p align="center"><img src="images/superFATBOY.png" alt="superFATBOY" width="110"></p>

# Instruments and templates

*[Docs home](README.md)*

superFATBOY is instrument-agnostic: an instrument is described by a **template XML file** (which processes to run and
with what settings) and, for instruments with unusual file structure, a **datatype** (a small Python class that knows
how to read that instrument's FITS files). This page lists what exists, and how much evidence there is that each one works.

## Validated instruments

These configurations have been run end-to-end on the Python 3 version, in **both GPU and CPU mode**, with zero
unhandled errors and sane, finite output. Each has a matching template XML file in
[`superFATBOY/data/templates/`](../superFATBOY/data/templates/).

| Instrument | Mode | Template | `datatype` | What was run |
|---|---|---|---|---|
| **FLAMINGOS-1** | Near-IR imaging | [`FLAMINGOS1_imaging_template.xml`](../superFATBOY/data/templates/FLAMINGOS1_imaging_template.xml) | *(default)* | 9-frame dithered near-IR dataset: dark, flat, bad pixel mask, sky subtraction, cosmic rays, align and stack |
| **FLAMINGOS-1** | Multi-object spectroscopy (MOS) | [`FLAMINGOS1_MOS_template.xml`](../superFATBOY/data/templates/FLAMINGOS1_MOS_template.xml) | `spectrum` | Full chain from linearity through calibration-star division, for 8 science frames plus a standard star |
| **OSIRIS** (GTC) | Longslit spectroscopy | [`OSIRIS_2019_longslit_template.xml`](../superFATBOY/data/templates/OSIRIS_2019_longslit_template.xml) | `osirisSpectrum` | Two separate datasets; bias subtraction through spectral extraction (one of them without flat-fielding, because its flat frames were lost) |
| **KAST** (Lick/Shane 3 m) | Dual-arm (blue + red) longslit | [`KAST_longslit_template.xml`](../superFATBOY/data/templates/KAST_longslit_template.xml) | `spectrum` | Two datasets (blue and red arms), each with its own calibrations |
| **MIRADAS** | SOL (single-object long) | [`MIRADAS_SOL_template.xml`](../superFATBOY/data/templates/MIRADAS_SOL_template.xml) | `miradasSpectrum` | 12 slitlets, through spectral extraction, combined slices and 3-d datacubes |
| **MIRADAS** | SOS (single-object short) | [`MIRADAS_SOS_template.xml`](../superFATBOY/data/templates/MIRADAS_SOS_template.xml) | `miradasSpectrum` | 13 slitlets |
| **MIRADAS** | MOS (multi-object) | [`MIRADAS_MOS_template.xml`](../superFATBOY/data/templates/MIRADAS_MOS_template.xml) | `miradasSpectrum` | 12 slitlets, one MIRADAS order |

How strong is the evidence?

- **FLAMINGOS-1 imaging** is the strongest validation. The same dataset was reduced four ways: by the original Python 2 /
  PyCUDA pipeline (CPU and GPU) and by the Python 3 / CuPy pipeline (CPU and GPU). All four agree on the final image
  alignment shifts to four or more decimal places.
- CPU and GPU runs of every configuration above agree to floating-point rounding.
- The OSIRIS, KAST and MIRADAS datasets exercise the spectroscopy chain (slitlet finding, rectification, wavelength
  calibration, extraction) on very different optical layouts: a classic longslit, a dual-arm longslit with a rotated
  dispersion axis, and an image-slicer IFU fed by multiple slits.

If a dataset or instrument is **not** in this table, treat it as untested. It may well work, since the machinery is
general, but there is no record that it has been run on the current code.

## Templates provided but not yet in the validated set

| Instrument | Mode | Template | Status |
|---|---|---|---|
| **EMIR** (GTC) | Near-IR imaging | [`EMIR_2024_imaging_template.xml`](../superFATBOY/data/templates/EMIR_2024_imaging_template.xml) | Template based on a 2024 EMIR dataset layout (sub-directory style, sky flats). Not yet in the both-modes validated list. |

## Supported in code, not yet exercised on the current version

superFATBOY has datatypes and/or dedicated processes for these instruments. They worked in the Python 2 version, but
there is no end-to-end run of them on the Python 3 code yet, so expect to shake out bugs and to write your own
template (copy the closest validated one):

| Instrument | What exists | Where to read |
|---|---|---|
| **CIRCE** (GTC near-IR imager; multi-ramp FITS) | `circeImage` and `circeFastImage` datatypes; `remergeCirce`, `deboneCirce`, `trimWindow`, `mergeObjects` processes | [instrument-specific processes](processes/instruments.md#circe) |
| **MEGARA** (GTC fiber spectrograph) | `megaraSpectrum` datatype; `trimOverscan`, `megaraIdentifyFibers`, `collapseFibers`, `megaraSkySubtract` processes; fiber options in `findSlitlets` | [instrument-specific processes](processes/instruments.md#megara) |
| **SINFONI** (VLT IFU) | `sinfoni*` processes (linearity calculation, slitlet identification, collapse, datacube, PSF, stacking, bad-line removal) | [instrument-specific processes](processes/instruments.md#sinfoni) |
| **GMOS** | `biasSubtract` (labelled GMOS-specific in the code) | [bias subtract](processes/imaging.md#biassubtract) |
| **FourStar** | in-progress imaging datatype and mosaic process (uncommitted work in the source tree) | not documented |

Parts of the spectroscopic machinery have also been tested on real **LUCI** (LBT) and **EMIR** spectroscopy data while
auditing individual algorithms (slitlet finding, rectification, wavelength calibration), but there is no verified
end-to-end configuration for either, so they are not in the table above.

## Adapting a template to a new instrument

The quickest path is to copy the validated template of the most similar instrument and change the instrument-specific
parts. Usually these are:

1. **`datatype`** on `<dataset>`: `spectrum` for generic spectroscopy (works for FLAMINGOS-1 and KAST), or one of the
   instrument datatypes (`osirisSpectrum`, `miradasSpectrum`, `megaraSpectrum`, `circeImage`). Imaging with a
   single-extension FITS file needs no `datatype` at all.
2. **File selection**: `prefix` or `suffix` (and `subdir`) to match your filenames. See the [XML guide](xml-guide.md#object-and-calib).
3. **Spectroscopy properties**: `specmode` (`longslit`, `ifu`, or anything else meaning MOS) and `dispersion`
   (`horizontal`, the default, or `vertical`). If your dispersion axis runs up the detector, say so.
4. **FITS keyword names** in `<parameters>`: `exptime_keyword`, `filter_keyword`, `gain_keyword`, `grism_keyword`,
   `ra_keyword`, `dec_keyword`, and so on.
5. **Wavelength calibration**: `line_list`, `wavelength_scale_guess`, `min_wavelength` and `max_wavelength` are
   specific to your grating and lamp. Line lists shipped with superFATBOY live in
   [`superFATBOY/data/linelists/`](../superFATBOY/data/linelists/) and can be named without a path.
6. **Tracing geometry** for `findSlitlets` and `rectify`: `slitlet_autodetect_x`, `continuum_trace_xinit`,
   `fit_order`, and the like. These depend on where your data is illuminated and how curved it is. Turn on
   `write_output` and `write_calib_output` for these steps and look at the QA files.

If you get a working configuration for a new instrument, please share it: a template plus a one-line note on what was run
is exactly what extends the table above.

## Line lists and wavelength-calibration files shipped with superFATBOY

Files in [`superFATBOY/data/linelists/`](../superFATBOY/data/linelists/) and
[`superFATBOY/data/config/`](../superFATBOY/data/config/) can be named in an XML file without a path; superFATBOY looks
there if the file is not found as given. `superFatboy3.py -config` lists them.

| Files | Purpose |
|---|---|
| `Redman_UArNe_lines.dat`, `MIRADAS_UArHg_lines.dat` | MIRADAS arclamp line lists |
| `OHlines.dat`, `OHlines_hires_*.dat` | OH sky-line lists at several resolutions (near-IR sky calibration) |
| `henearjhuse_air.dat`, `HeNeAr_vac.dat`, `hklines_mod.dat` | near-IR and optical arc and sky-line lists |
| `NeAr_lines_IR.dat`, `ThAr_lines_IR.dat`, `Xenon_IR.dat` | infrared arclamp lists |
| `KAST_hehgcd.txt`, `KAST_neon.txt` | KAST blue and red arclamps |
| `osiris_HgAr_air_nist.dat` | OSIRIS HgAr lamp |
| `megara_ThArNe_list.dat` | MEGARA ThArNe lamp |
| `wc_miradas_sol.xml`, `wc_miradas_sos.xml`, `wc_miradas_mos_NN.xml` | MIRADAS per-slitlet wavelength-calibration starting guesses (see the [MIRADAS guide](miradas.md)) |
