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
| **MEGARA** (GTC) | Fiber IFU (LCB, 623 fibers), HR-I | [`MEGARA_LCB_template.xml`](../superFATBOY/data/templates/MEGARA_LCB_template.xml) | `megaraSpectrum` | 3 science frames from raw: 622 fibers traced (identical to the Python 2 version), 56 sky fibers identified, all fibers wavelength-calibrated (median 0.05 px), sky subtracted. GPU and CPU agree to 5e-6 of the data range |
| **SINFONI** (VLT) | Near-IR image-slicer IFU (32 slitlets), H+K | [`SINFONI_IFU_template.xml`](../superFATBOY/data/templates/SINFONI_IFU_template.xml) | `spectrum` | 30 Doradus and a standard star, full chain to registered, stacked datacubes; slitmask identical to the Python 2 version, stacked image within 0.6% of it |
| **Flamingos-2** (Gemini South) | Near-IR longslit, 2009-era data, JH and HK in one file (two datasets) | [`FLAMINGOS2_longslit_2009_template.xml`](../superFATBOY/data/templates/FLAMINGOS2_longslit_2009_template.xml) | `spectrum` | LMC X-1 and a standard in JH and HK, ABBA nods 100" apart: dark, dome on-off flats, bad pixel masks, dither sky subtraction, rectification, double subtraction, shift-add, HeNeAr wavelength calibration (0.05-0.11 px), extraction, standard-star division |
| **Flamingos-2** (Gemini South) | Near-IR imaging | [`FLAMINGOS2_imaging_template.xml`](../superFATBOY/data/templates/FLAMINGOS2_imaging_template.xml) | *(default)* | Galactic Center J, H, Ks at two position angles: dark, lamp-on dome flats, bad pixel mask (circular field), off-source skies from a list file, cosmic rays, triangles alignment, drizzle + sigma-clipped combine; matches the 2015 reduction |
| **RHO** (Rosemary Hill Observatory 14") | Optical CCD imaging, B V R I | [`RHO_imaging_template.xml`](../superFATBOY/data/templates/RHO_imaging_template.xml) | *(default)* | H Persei, 4 filters x 10 dithered frames: dark (one set per exposure time), lamp-on dome flats, bad pixel mask, cosmic rays, triangles alignment + drizzle stack; GPU and CPU outputs byte-identical |
| **Flamingos-2** (Gemini South) | Near-IR longslit, newer (2017) data, one band | [`FLAMINGOS2_longslit_2017_template.xml`](../superFATBOY/data/templates/FLAMINGOS2_longslit_2017_template.xml) | `spectrum` | XID6592 and a standard (K band): dark, dome flats, on-source dither sky, rectification, HeNeAr wavelength calibration (0.059 px, excellent), extraction, standard-star division; flux agrees with the earlier reduction (correlation 1.0000), GPU and CPU extractions within 1e-4. Older F2 data: use the 2009 template |
| **FISICA** | Near-IR slitlet (MOS-style) spectroscopy, JH grism | [`FISICA_MOS_template.xml`](../superFATBOY/data/templates/FISICA_MOS_template.xml) | `spectrum` | n1569: linearity, darks (the nearest dark with another number of reads is substituted with a warning for the 1-read flats and arcs), region-file slitlets, dome on-off flat, dither sky subtraction, rectification, double subtraction, HeNeAr wavelength calibration (21 of 22 slitlets excellent, 0.02-0.05 px), extraction; matches the 2019 reduction (same spectra, flux correlation 0.996). GPU and CPU agree to about 1% (one marginal spectrum can be found in a different frame) |
| **CIRCE** (GTC) | Near-IR imaging, H band | [`CIRCE_imaging_template.xml`](../superFATBOY/data/templates/CIRCE_imaging_template.xml) | `circeImage` | Crab nebula, two pointings of 27 frames: dark, dome on-off flat, bad pixel mask, off-source nebula sky and sky-surface fit, cosmic rays, section remerge, triangles alignment with `triangles_chain_overlapping_frames` (all frames registered; large dithers over a sparse field), drizzle stack. GPU and CPU stacks identical (correlation 1.00000) |
| **FourStar** (Magellan/Baade) | Near-IR imaging, four chips per exposure | [`FOURSTAR_imaging_template.xml`](../superFATBOY/data/templates/FOURSTAR_imaging_template.xml) | `fourStarImage` | One test exposure: `mosaicFourStar` pastes the four chips into one 4196 x 4196 image (100 px gaps), then dark subtraction; GPU and CPU identical. The later imaging steps were not run |
| **FIRE** (Magellan/Baade) | Near-IR echelle, 21 orders, through rectification only | [`FIRE_echelle_template.xml`](../superFATBOY/data/templates/FIRE_echelle_template.xml) | `spectrum` | Region-file orders, lamp flat, dither sky subtraction, rectification of the 21 curved orders; GPU and CPU agree to rounding. **Wavelength calibration is not validated**: per-order starting guesses (from the FireHose pipeline) are in `data/config/wc_fire_echelle.xml`, and a first test graded 3 marginal and 12 poor of 21 orders |

How strong is the evidence?

- **FLAMINGOS-1 imaging** is the strongest validation. The same dataset was reduced four ways: by the original Python 2 /
  PyCUDA pipeline (CPU and GPU) and by the Python 3 / CuPy pipeline (CPU and GPU). All four agree on the final image
  alignment shifts to four or more decimal places.
- CPU and GPU runs of the configurations above agree to floating-point rounding (LUCI: byte-identical output). The exceptions are the
  ones where tracing or peak finding amplifies rounding: FISICA (rectified frames and wavelength solutions agree to about 1%, one marginal
  spectrum can differ), Flamingos-2 longslit (apertures 1-2 px apart) and FIRE (the CPU rectified frame is one row taller).
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
| **GMOS** | `biasSubtract` (labelled GMOS-specific in the code) | [bias subtract](processes/imaging.md#biassubtract) |

Parts of the spectroscopic machinery have also been tested on real **EMIR** spectroscopy data while auditing individual
algorithms (slitlet finding, rectification, wavelength calibration), but there is no verified end-to-end configuration
for it, so it is not in the table above.

## General templates

For an instrument without its own template, start from the general template for your observing mode. Each has the basic
reduction steps only, with a short comment on every option you are likely to change, and points to the instrument
templates it was built from.

| Template | Steps | Built from |
|---|---|---|
| [`GENERAL_imaging_IR_template.xml`](../superFATBOY/data/templates/GENERAL_imaging_IR_template.xml) | linearity (optional), dark, flat, bad pixel mask, sky subtraction (on-source dithers; off-source shown), cosmic rays, align and stack | FLAMINGOS-1, Flamingos-2 imaging |
| [`GENERAL_imaging_optical_template.xml`](../superFATBOY/data/templates/GENERAL_imaging_optical_template.xml) | bias, (dark), dome flat, bad pixel mask, cosmic rays, align and stack; no sky subtraction | the IR chain with bias, no sky step (not yet run on optical data) |
| [`GENERAL_spectroscopy_longslit_IR_template.xml`](../superFATBOY/data/templates/GENERAL_spectroscopy_longslit_IR_template.xml) | dark, flats, slit, A-B nod sky subtraction, rectify, double subtract, shift-add, wavelength calibration (arcs or OH), extraction, standard | Flamingos-2 longslit |
| [`GENERAL_spectroscopy_longslit_optical_template.xml`](../superFATBOY/data/templates/GENERAL_spectroscopy_longslit_optical_template.xml) | bias, flats, sky from the slit (`median`; `median_boxcar` also works), rectify, shift-add, arc wavelength calibration, extraction, standard | KAST, OSIRIS |
| [`GENERAL_spectroscopy_MOS_IR_template.xml`](../superFATBOY/data/templates/GENERAL_spectroscopy_MOS_IR_template.xml) | as IR longslit, with slitlets from a region file or the flat and per-slitlet calibration | LUCI, FLAMINGOS-1 MOS |
| [`GENERAL_spectroscopy_MOS_optical_template.xml`](../superFATBOY/data/templates/GENERAL_spectroscopy_MOS_optical_template.xml) | as optical longslit, with slitlets from the flat or a region file | optical longslit + LUCI slit handling (not yet run on optical MOS data) |

IFU templates will follow. Overscan trimming is instrument specific and is not in the optical templates.

## Adapting a template to a new instrument

The quickest path is to copy the validated template of the most similar instrument (or the general template for your
mode) and change the instrument-specific parts. Usually these are:

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

| File | Lines | Range (A) | Medium | Purpose |
|---|---|---|---|---|
| `OHlines.dat` | 95 | 10016-18524 | ? | OH sky lines, J and H, low resolution (FLAMINGOS-1, FISICA, Flamingos-2) |
| `OHlines_hires_100.dat`, `_250`, `_500`, `_4000` | 400 / 600 / 838 / 2596 | 10003-24999 | ? | OH sky lines J through K; the number is the number of lines kept (fewer = only the brightest, for lower resolution) |
| `henearjhuse_air.dat` | 65 | 9123-18428 | air | HeNeAr lamp, J and H (FLAMINGOS-1, FISICA) |
| `HeNeAr_vac.dat` | 142 | 9126-45176 | vacuum | HeNeAr lamp, J through L (Flamingos-2) |
| `hklines_mod.dat` | 95 | 1.1-2.4 **microns** | ? | H and K lamp lines; note the unit |
| `NeAr_lines_IR.dat` | 241 | 9489-25794 | ? | NeAr lamp, near-IR |
| `ThAr_lines_IR.dat` | 3938 | 9660-25991 | ? | ThAr lamp, near-IR |
| `Xenon_IR.dat` | 128 | 10027-24832 | ? | Xe lamp, near-IR (SINFONI) |
| `Redman_UArNe_lines.dat` (also `.txt`) | 11741 | 8334-44051 | ? | UArNe lamp (Redman et al.), MIRADAS |
| `Redman_UArNe_lines_MIRADAS.dat` | 11741 | 8334-44051 | ? | `Redman_UArNe_lines.dat` cleaned for MIRADAS with `makeLineList.py --clean`: 50 lines with a consistent offset flagged -1, lowering the fit RMS by up to about a quarter where those lines fall (see the [MIRADAS guide](miradas.md#wavelengthcalibrate)) |
| `MIRADAS_UArHg_lines.dat` | 11696 | 8334-44051 | ? | UArHg lamp, MIRADAS |
| `KAST_hehgcd.txt` | 16 | 3261-5461 | air | KAST blue arm HeHgCd lamps |
| `KAST_neon.txt` | 42 | 5770-8635 | air | KAST red arm Ne (and Ar) lamps |
| `osiris_HgAr_air_nist.dat` | 26 | 3651-9123 | air | OSIRIS HgAr lamp (NIST) |
| `Xenon_optical_air.dat` | 196 | 2865-8232 | air | Xe I and Xe II, with the strong blue Xe I lines (4501, 4525, 4583, 4624, 4671 A) and intensities measured from a GTC/OSIRIS xenon arc between 3446 and 4606 A (see the file header for how each intensity was derived) |
| `megara_ThArNe_list.dat` | 299 | 5111-8998 | ? | MEGARA ThArNe lamp, all VPH filters (sections by filter in the file) |
| `wc_miradas_sol.xml`, `wc_miradas_sos.xml`, `wc_miradas_mos_NN.xml` | | | | MIRADAS per-slitlet wavelength-calibration starting guesses (see the [MIRADAS guide](miradas.md)) |

"?" = not recorded in the file; check before mixing a list with data calibrated in the other medium.

superFATBOY does not assume a wavelength unit: `min_wavelength`, `max_wavelength` and `wavelength_scale_guess` (units per
pixel) just have to be in the units of the line list - Angstrom for every shipped list except `hklines_mod.dat`, which is
in microns. Each list starts with a `#Units:` comment saying which.

### Line lists by instrument

What has been used with each spectrograph, and the starting guesses that went with it (`wavelength_scale_guess` in
Angstrom per pixel - its sign follows the direction of increasing wavelength on the detector - and
`min_wavelength`/`max_wavelength` in Angstrom). The templates carry the alternatives as commented-out options. Lists
marked *not shipped* live with the user's configurations; copy them next to your XML file.

| Instrument | Setup | Lamp list | Sky list | Scale guess | min - max | From |
|---|---|---|---|---|---|---|
| FLAMINGOS-1 | JH grism, MOS | `henearjhuse_air.dat` | `OHlines.dat` | 4.8 (lamp), 4.9 (sky) | 8500 - 19500 | specBench (sky: with `wc_specbench.xml`) |
| FLAMINGOS-1 | HK grism | `hklines_mod.dat` | `OHlines_hires_*.dat` | | | (to be filled in) |
| Flamingos-2 | JH grism, longslit | `HeNeAr_vac.dat` | `OHlines.dat` | -6.6 | 7800 - 19000 | lmcx1 |
| Flamingos-2 | HK grism, longslit | `HeNeAr_vac.dat` | `OHlines_hires_*.dat` | -7.5 | 12500 - 22500 | lmcx1 |
| Flamingos-2 | K, R3000 | `HeNeAr_vac.dat` | | -3.5 | 18000 - 22500/24000 | flamingos2_XID6592 |
| FISICA | JH | `henearjhuse_air.dat` | `OHlines.dat` | 4.9 | | daveFisica (with `wc_specbench.xml`) |
| LUCI (LBT) | H+K, MOS | `NeArXe.dat` (*not shipped*) | `OHlines_hires_100.dat` | -4.5 | 13000 - 26000 | caden_luci_test (sky) |
| SINFONI (VLT) | H+K | `Xenon_IR.dat` | `OHlines_hires_250.dat` | -5.0 | 14000 - 25000 | sinfoni_test_30Dor (lamp); `hklines_mod.dat` also tried |
| MIRADAS | SOL / SOS / MOS | `Redman_UArNe_lines.dat` or `_MIRADAS.dat`; `NeAr_lines_IR.dat`, `ThAr_lines_IR.dat` | `OHlines_hires_4000.dat` | 0.25 | per order: `wc_miradas_*.xml` | verified configs |
| KAST (Lick) | blue arm | `KAST_hehgcd.txt` | | 1.0 | 3200 - 5700 | sarik_quack1 |
| KAST (Lick) | red arm | `KAST_neon.txt` | | 1.2 | 5500 - 9000 | sarik_quack1 |
| OSIRIS (GTC) | R1000B-type, longslit | `osiris_HgAr_air_nist.dat` | | 3.8 | 2400 - 8880 | sarik_osiris |
| OSIRIS (GTC) | R2500U | `Xenon_optical_air.dat` | | -0.57 | 3440 - 4650 | avrajit-osiris |
| MEGARA (GTC) | LCB, HR-I | `megara_ThArNe_list.dat` | | -0.1274 | 8380 - 8890 | mt1 (its local `megara-ThArNe-list.txt` is the HR-I subset of the shipped list) |
| MEGARA (GTC) | LCB, LR-V | `megara_ThArNe_list.dat` | | -0.251 | 5000 - 6180 | J1107 (paul-megara) |
| GMOS (Gemini) | B600 / R400-type, 3 CCDs | `CuAr_GMOS_S_SJ.txt`, `CuAr_homemade.txt` (*not shipped*) | | 0.5 / 1.03 | 3300 - 7000 | OBJ_1 (`n_segments` 3, `wc_gmos.xml`), R0329 |
| FIRE (Magellan) | echelle | | | | | (to be filled in) |

Imaging-only instruments (CIRCE, RHO, FourStar, EMIR imaging) need no line list.

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

**Cleaning an existing list.** If the wavelength fits grade worse than the data deserve, check whether particular lines
are to blame: `measured_lines_*.dat` gives each line's mean offset from the solutions. A line that sits at the same
offset in every fit has a list wavelength that is wrong for your instrument, or is a blend your resolution does not
separate. `--clean` takes an existing list and one or more measured files (several frames, or several datasets with the
same lamp) and flags those lines -1, so they still shape the template but stay out of the fit:

```bash
makeLineList.py --clean Redman_UArNe_lines.dat -m sol/wavelengthCalibrated/measured_lines_lamp.dat \
    -m sos/wavelengthCalibrated/measured_lines_lamp.dat -o Redman_UArNe_cleaned.dat
```

A line is flagged when it was used in at least `--min-fits` fits (3), its mean offset is at least `--flag-offset` px
(0.3) and at least `--min-significance` standard errors (3). With `--correct` such lines are instead moved by their
mean offset (up to `--max-correct` px, 1.5; beyond that they are flagged) - right for a wrong list wavelength, but for a
blend it bakes in the blend's centroid at your resolution. `--use-measured` also replaces the intensities with the
measured ones. Judge a cleaned list on data it was not cleaned with.
