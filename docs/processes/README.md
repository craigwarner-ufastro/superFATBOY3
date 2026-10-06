<p align="center"><img src="../images/superFATBOY.png" alt="superFATBOY" width="110"></p>

# Process guide

*[Docs home](../README.md)*

A **process** is one reduction step. You list processes in the `<processes>` section of your XML file, and they run in that
order on every frame that applies. This guide describes what each process does and which options matter most;
for every option and its default, see the [options reference](../options-reference.md) or run `superFatboy3.py -list`.

| Guide | Covers |
|---|---|
| [Imaging processes](imaging.md) | `linearity`, `darkSubtract`, `biasSubtract`, `flatDivide`, `badPixelMask`, `skySubtract`, `cosmicRays`, `alignStack` |
| [Spectroscopy processes](spectroscopy.md) | `noisemap`, `createCleanSkies`, `createMasterArclamps`, `findSlitlets`, `cosmicRaysSpec`, `flatDivideSpec`, `badPixelMaskSpec`, `skySubtractSpec`, `rectify`, `doubleSubtract`, `shiftAdd`, `slitletAlign`, `wavelengthCalibrate`, `resample`, `extractSpectra`, `calibStarDivide` |
| [Instrument-specific processes](instruments.md) | MIRADAS, MEGARA, SINFONI, CIRCE, EMIR and other instrument processes |
| [MIRADAS guide](../miradas.md) | A complete walk-through of a MIRADAS reduction |

## How processes work together

Three ideas explain most of the behaviour you will see.

**1. Calibrations are found (or built) on demand.** `darkSubtract` does not need to be told which darks to use. For each
science frame it asks the database for dark frames that match the frame's exposure time and number of reads, combines them
into a master dark the first time it is needed, and reuses that master thereafter. Flats, arclamps, skies, bad pixel
masks and slitmasks all work the same way. You only describe the raw frames in `<queries>`.

**2. Earlier steps are replayed on calibrations automatically.** Suppose your chain is `linearity`, `darkSubtract`,
`flatDivide`. To build a master flat, the flat frames must themselves be linearized and dark-subtracted. superFATBOY keeps
a history of which processes were applied to every frame, and `flatDivide` quietly runs the earlier processes on the raw
flats before combining them. You never list a process twice to get this.

**3. A failure degrades gracefully.** If a process cannot do its job for one frame (no matching dark, a fit that would
give a wildly distorted result, an unreadable file), that frame is disabled and the run continues with the rest. Look
at the log for `ERROR` and `WARNING` lines; they say what was dropped and why. Some algorithms also fall back to simpler
behaviour for just the part that failed. For example, a slitlet whose edge trace fails is treated as straight rather than discarding the
whole image.

Processes that need a slitmask, a clean sky or a master arclamp can build them on demand too. For example, `findSlitlets`
needs a master flat and will ask `flatDivideSpec` to build one, while `flatDivideSpec` (when normalizing slitlet by
slitlet) needs a slitmask and will ask `findSlitlets`. Either order works.

## Typical pipelines

Order matters. These are the orders used by the [templates](../instruments.md#validated-instruments).

**Near-IR imaging** (FLAMINGOS-1, EMIR):

```
linearity → darkSubtract → flatDivide → badPixelMask → skySubtract → cosmicRays → alignStack
```

(EMIR replaces `darkSubtract` with `emirBiasSubtract`.)

**Longslit spectroscopy** (OSIRIS, KAST):

```
noisemap → biasSubtract → createMasterArclamps → cosmicRaysSpec → flatDivideSpec → skySubtractSpec
   → rectify → shiftAdd → wavelengthCalibrate → extractSpectra → calibStarDivide
```

**Multi-object spectroscopy** (FLAMINGOS-1 MOS):

```
linearity → noisemap → darkSubtract → createCleanSkies → createMasterArclamps → findSlitlets → cosmicRaysSpec
   → flatDivideSpec → badPixelMaskSpec → skySubtractSpec → rectify → doubleSubtract → shiftAdd
   → wavelengthCalibrate → extractSpectra → calibStarDivide
```

**MIRADAS** adds collapsed-spaxel, datacube and slice-combination steps. See the [MIRADAS guide](../miradas.md).

## All processes

| Process | One-line summary | Guide |
|---|---|---|
| `linearity` | Polynomial linearity correction of every pixel | [imaging](imaging.md#linearity) |
| `darkSubtract` | Subtract a master dark matched by exposure time | [imaging](imaging.md#darksubtract) |
| `biasSubtract` | Subtract a master bias | [imaging](imaging.md#biassubtract) |
| `flatDivide` | Divide by a master flat (dome, sky or twilight) | [imaging](imaging.md#flatdivide) |
| `badPixelMask` | Build and apply a bad pixel mask (imaging) | [imaging](imaging.md#badpixelmask) |
| `skySubtract` | Subtract the sky (on-source or off-source) | [imaging](imaging.md#skysubtract) |
| `cosmicRays` | Remove cosmic rays (imaging) | [imaging](imaging.md#cosmicrays) |
| `alignStack` | Align frames and drizzle them into one image | [imaging](imaging.md#alignstack) |
| `noisemap` | Create a per-pixel noise image that is propagated through later steps | [spectroscopy](spectroscopy.md#noisemap) |
| `createCleanSkies` | Median combine frames into a "clean sky" used to find skylines | [spectroscopy](spectroscopy.md#createcleanskies) |
| `createMasterArclamps` | Combine arclamp frames into a master arclamp | [spectroscopy](spectroscopy.md#createmasterarclamps) |
| `findSlitlets` | Find and trace slitlets (or fibers) and build the slitmask | [spectroscopy](spectroscopy.md#findslitlets) |
| `cosmicRaysSpec` | Remove cosmic rays (spectroscopy: `dcr`, `lacos`, `deepcr`) | [spectroscopy](spectroscopy.md#cosmicraysspec) |
| `flatDivideSpec` | Normalize (per slitlet) and divide by the master flat | [spectroscopy](spectroscopy.md#flatdividespec) |
| `badPixelMaskSpec` | Build and apply a bad pixel mask (spectroscopy) | [spectroscopy](spectroscopy.md#badpixelmaskspec) |
| `skySubtractSpec` | Sky subtraction by dither, step, off-source or median | [spectroscopy](spectroscopy.md#skysubtractspec) |
| `rectify` | Straighten continua and emission lines | [spectroscopy](spectroscopy.md#rectify) |
| `doubleSubtract` | Combine the positive and negative spectra of an A-B pair | [spectroscopy](spectroscopy.md#doublesubtract) |
| `shiftAdd` | Shift and add frames of the same target | [spectroscopy](spectroscopy.md#shiftadd) |
| `slitletAlign` | Align skylines between MOS slitlets so they line up like jail bars | [spectroscopy](spectroscopy.md#slitletalign) |
| `wavelengthCalibrate` | Fit a wavelength solution from sky or arclamp lines | [spectroscopy](spectroscopy.md#wavelengthcalibrate) |
| `resample` | Resample to a linear wavelength scale | [spectroscopy](spectroscopy.md#resample) |
| `extractSpectra` | Find spectra and extract them to 1-d | [spectroscopy](spectroscopy.md#extractspectra) |
| `calibStarDivide` | Divide by a standard-star spectrum | [spectroscopy](spectroscopy.md#calibstardivide) |
| `miradasCollapseSpaxels`, `miradasCreate3dDatacubes`, `miradasCombineSlices`, `miradasStitchOrders`, `miradasCharacterizePSF`, `miradasDARFromConditions`, `miradasDARFromData`, `miradasRegisterWCS` | MIRADAS IFU steps | [instruments](instruments.md#miradas) |
| `trimOverscan`, `trimWindow` | Trim overscan or a sub-window | [instruments](instruments.md#trimming) |
| `emirBiasSubtract` | EMIR bias subtraction | [instruments](instruments.md#emir) |
| `megaraIdentifyFibers`, `collapseFibers`, `megaraSkySubtract` | MEGARA fiber steps | [instruments](instruments.md#megara) |
| `sinfoniCalcLinearity`, `sinfoniRemoveBadLines`, `sinfoniIdentifySlitlets`, `sinfoniCollapseSlitlets`, `sinfoniCreate3dDatacubes`, `sinfoniCharacterizePSF`, `sinfoniRegisterStack` | SINFONI IFU steps | [instruments](instruments.md#sinfoni) |
| `remergeCirce`, `deboneCirce`, `mergeObjects` | CIRCE steps | [instruments](instruments.md#circe) |
| `mosaicFourStar` | FourStar chip mosaic | [instruments](instruments.md#fourstar) |

## A note on algorithm robustness

The spectroscopic tracing algorithms (`findSlitlets`, `rectify`, `wavelengthCalibrate`, `extractSpectra`) were the
subject of a dedicated robustness review. Where that review found that one method works in some regimes but fails
in others, the fix was usually a **new option that tries the old method first and falls back only when it finds nothing**
(for example `edge_detection_method = auto`), and the shipped default was left as it was. Two consequences for you:

- Each of these processes can write **diagnostic files** (`stats_*` files and `qa_*` images) into its output
  sub-directory (`findSlitlets/`, `rectified/`, `extractedSpectra/`) when you turn on `write_calib_output` and `write_output`.
  The `stats_*` files list every traced data point that was rejected and why. When a trace or fit looks wrong, look there first.
- Some options are documented as "validated on a particular dataset" in the option's own help text. Read
  the help text in `-list` before changing a default that affects several instruments.
