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
| **LUCI** (LBT) | Near-IR multi-object spectroscopy (MOS) | [`LUCI_MOS_template.xml`](../superFATBOY/data/templates/LUCI_MOS_template.xml) | `spectrum` | 24 slitlets (region file, group trace, flexure correction), A-B nodded H+K frames, through spectral extraction: 19 spectra, 24 wavelength solutions. The standard star (different mask, no calibrations) is not included |

How strong is the evidence?

- **FLAMINGOS-1 imaging** is the strongest validation. The same dataset was reduced four ways: by the original Python 2 /
  PyCUDA pipeline (CPU and GPU) and by the Python 3 / CuPy pipeline (CPU and GPU). All four agree on the final image
  alignment shifts to four or more decimal places.
- CPU and GPU runs of every configuration above agree to floating-point rounding (LUCI: byte-identical output).
- Wavelength calibration is graded per slitlet (RMS in pixels). On these datasets the median is 0.05-0.10 px for the
  arc and sky lists of FLAMINGOS-1, OSIRIS, KAST and LUCI (excellent to good), and 0.22-0.25 px for MIRADAS (see the
  [MIRADAS guide](miradas.md#wavelengthcalibrate)). A second OSIRIS xenon-arc dataset calibrates (0.06 px) with the
  shipped `Xenon_optical_air.dat`; the older Xe list it was configured with lacks the brightest blue Xe I lines.
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

Parts of the spectroscopic machinery have also been tested on real **EMIR** spectroscopy data while auditing individual
algorithms (slitlet finding, rectification, wavelength calibration), but there is no verified end-to-end configuration
for it, so it is not in the table above.

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
| `Xenon_optical_air.dat` | Xe I and Xe II, 2860-8230 A (air), with the strong blue Xe I lines (4501, 4525, 4583, 4624, 4671 A) and intensities measured from a GTC/OSIRIS xenon arc between 3446 and 4606 A (see the file header for how each intensity was derived) |
| `megara_ThArNe_list.dat` | MEGARA ThArNe lamp |
| `wc_miradas_sol.xml`, `wc_miradas_sos.xml`, `wc_miradas_mos_NN.xml` | MIRADAS per-slitlet wavelength-calibration starting guesses (see the [MIRADAS guide](miradas.md)) |

### Making a line list: `makeLineList.py`

`makeLineList.py` (installed with superFATBOY) builds a line list for any set of spectra and wavelength range from the
[NIST Atomic Spectra Database](https://physics.nist.gov/asd):

```bash
makeLineList.py -e "Xe I,Xe II" -r 3400 5000 -o xenon_blue.dat
makeLineList.py -e "Ne I,Ar I" -r 13000 26000 --vacuum --min-intensity 50 -o NeAr_HK.dat
```

It queries NIST in vacuum and converts to air (IAU standard formula) unless `--vacuum` is given, so a list that
crosses 2 microns does not mix the two. Responses are cached in `~/.cache/superFATBOY/nist` (`--cache`), and
`--nist-file "Xe I=file"` reads a saved response instead of querying.

**Intensities are the weak point of any line list**, and superFATBOY's first match depends on them. NIST relative
intensities come from different sources for different spectra (Xe I and Xe II, or Ne and Ar, are not on a common
scale), and some strong lines have none at all. So:

- `--missing estimate` (the default) fills in a missing intensity from the transition probability g*A, scaled to the
  lines of the same spectrum that have both; `--missing skip` drops those lines, or give a number.
- `-s "Ar I=0.5,Ne I=2"` scales each spectrum by hand.
- `-m measured_lines.dat` fits a scale per spectrum to intensities measured in your own data, and `--use-measured`
  uses the measured values wherever a line was measured. `wavelengthCalibrate` writes these for every frame it
  calibrates: `wavelengthCalibrated/measured_lines_<frame>.dat` (see the
  [spectroscopy page](processes/spectroscopy.md#wavelengthcalibrate)). Calibrate once with a NIST list, then rebuild
  the list with the intensities of your lamp and instrument.
- `--blend 1.0` flags lines closer than 1 A (of comparable strength) with -1, so they shape the template but are not
  used in the fit; lines NIST marks as blended are flagged too.

`--min-intensity` and `--max-lines` trim faint lines; a list with many lines the data never shows makes the first
match harder.
