<p align="center"><img src="../images/superFATBOY.png" alt="superFATBOY" width="110"></p>

# Imaging processes

*[Docs home](../README.md) · [Process guide](README.md) · [Spectroscopy processes](spectroscopy.md)*

The imaging chain takes raw near-IR (or optical) frames from a dithered sequence to a single aligned, stacked, sky-subtracted,
flat-fielded image. A typical order is:

```
linearity → darkSubtract → flatDivide → badPixelMask → skySubtract → cosmicRays → alignStack
```

Validated on FLAMINGOS-1 (see [instruments](../instruments.md)). Options listed here are the ones you are most likely to
change; everything else is in the [options reference](../options-reference.md).

- [linearity](#linearity)
- [darkSubtract](#darksubtract)
- [biasSubtract](#biassubtract)
- [flatDivide](#flatdivide)
- [badPixelMask](#badpixelmask)
- [skySubtract](#skysubtract)
- [cosmicRays](#cosmicrays)
- [alignStack](#alignstack)

---

## linearity

Applies a polynomial linearity correction to every pixel of every frame (objects, darks, flats, and so on).

The correction is `y = c1·x + c2·x² + c3·x³ + ...`: a polynomial with **no constant term**, where `c1` is the first number you list.
The default, `1`, therefore applies the correction `y = x`, which leaves the data unchanged. If you do not know your detector's coefficients, omit
the process or leave the default.

| Option | Default | Meaning |
|---|---|---|
| `linearity_coeffs` | `1` | Space-separated polynomial coefficients, lowest order first. `0.8 0.1 0.03` means `y = 0.8x + 0.1x² + 0.03x³`. |
| `do_linearity` | `yes` | `no` skips the polynomial (for example if you only want `divide_by_coadds`) |
| `divide_by_coadds` | `no` | Divide each frame by its number of coadds first |

Runs on the GPU when `gpumode` is on. Output: `linearized/lin_*.fits`.

```xml
<process name="linearity">
  <option name="linearity_coeffs" value="1.00425 -1.01413e-6 4.18096e-11"/>
</process>
```

## darkSubtract

Median-combines dark frames into a **master dark** for each exposure time (and number of reads), then subtracts the matching
master dark from every object, flat, sky and arclamp frame.

superFATBOY matches darks to frames by exposure time and number of reads (and detector section, for multi-extension
instruments). If there is no exact match, it can either prompt you for a file or use the master dark with the nearest
exposure time; see below.

| Option | Default | Meaning |
|---|---|---|
| `default_master_dark` | `None` | A master dark file, a comma-separated list, or an ASCII file listing them. Used when the raw darks do not include a match for some frame; the one matching exposure time and number of reads is chosen. |
| `prompt_for_missing_dark` | `no` | If no dark matches a frame's exposure time: `yes` asks you which file to use; `no` uses the dark with the nearest exposure time (matching number of reads first; if none, any number of reads, with a loud WARNING). **Use `no` for unattended runs.** |

You can also supply master darks as `<calib name="masterDark" value="file.fits"/>` inside the process block, which ties them to a
particular object with a `tag`. Output: `darkSubtracted/ds_*.fits`; master darks (with `write_calib_output`) in `masterDarks/`.

## biasSubtract

Median-combines bias frames into a master bias and subtracts it. It is used in place of `darkSubtract` for
CCD-type detectors (KAST, OSIRIS, GMOS-style data) where a bias frame is the relevant calibration.

| Option | Default | Meaning |
|---|---|---|
| `default_master_bias` | `None` | Use this master bias file instead of combining raw bias frames |

Output: `biasSubtracted/bs_*.fits`, master biases in `masterBiases/`.

## flatDivide

Builds a **master flat** (normalized to a median of 1) and divides every object frame by it.

| `flat_method` | Flats you need | How the master flat is made |
|---|---|---|
| `dome_on` (default) | dome flats, lamp on | median of the flats |
| `dome_on-off` | dome flats, lamp on and lamp off | median(lamp on) minus median(lamp off), which removes the thermal background. Mark the two sets with `<property name="flat_type" value="lamp_on"/>` or `lamp_off`. |
| `sky` | sky-flat exposures of the field | each flat is scaled by the reciprocal of its median, then the set is median combined (with optional rejection, see `flat_sky_*`) |
| `twilight` | at least two twilight flats at different intensity | takes the absolute difference of each consecutive pair of twilight flats (`|flat1-flat2|`, `|flat2-flat3|`, ...), scales each difference by the reciprocal of its median and median combines them. Differencing removes the additive background and leaves the pixel-to-pixel response. `twilight_pair_ramps` pairs ramps for multi-ramp data. |

| Option | Default | Meaning |
|---|---|---|
| `flat_method` | `dome_on` | `dome_on`, `dome_on-off`, `sky`, `twilight` |
| `flat_lamp_off_files` | `off` | How to tell lamp-off flats apart when you did not use the `flat_type` property: a filename fragment or an ASCII file listing them |
| `flat_sky_reject_type`, `flat_sky_nlow`, `flat_sky_nhigh`, `flat_sky_lsigma`, `flat_sky_hsigma` | `none`, 1, 1, 5, 5 | Outlier rejection when median combining sky flats |
| `sky_flat_include_list`, `sky_flat_exclude_list` | empty | Files to explicitly include in or exclude from sky flat combination |
| `median_section` | `None` | A sub-region (e.g. `400:1200,500:1000`) whose median is used for normalization |
| `default_master_flat` | `None` | Use this master flat instead of building one |

Output: `flatDivided/fd_*.fits`; master flats in `masterFlats/`.

## badPixelMask

Creates a bad pixel mask and applies it, by default from the normalized master flat. Pixels outside `[clipping_low, clipping_high]` are marked bad. You
can instead supply a mask (a FITS image with bad pixels = 1) as a `<calib type="bad_pixel_mask">`, or name one with `default_bad_pixel_mask`.

| Option | Default | Meaning |
|---|---|---|
| `clipping_method` | `values` | `values` uses the thresholds below. `sigma` uses `clipping_sigma`, which suits a dark frame better than a flat. |
| `clipping_low`, `clipping_high` | `0.5`, `2.0` | Bad if the normalized flat is outside this range |
| `clipping_sigma` | `5` | Threshold for `sigma` |
| `bad_pixel_mask_source` | `None` | Build the mask from this file (or list of files) instead of the master flat |
| `edge_reject` | `5` | Mark this many pixels around the edge as bad |
| `radius_reject` | `0` | Mark everything beyond this radius from the centre as bad (0 = off) |
| `column_reject`, `row_reject` | `None` | Mark columns or rows bad. Supports slices: `320:384, 500, 752:768` |
| `default_bad_pixel_mask` | `None` | A ready-made mask (file, list of files, or an ASCII list), matched to images by filter |

In imaging, bad pixels are **masked** (there is no interpolation option), so they drop out of the stack when frames are combined; every
sky position is observed at several dither positions. Output: `badPixelMaskApplied/ba_*.fits`; masks in `badPixelMasks/`.

## skySubtract

Creates a master sky for each frame and subtracts it after scaling to the frame's own background level. The default,
`remove_objects`, is what you want for most on-source dithered data.

| `sky_subtract_method` | Sky comes from | Use when |
|---|---|---|
| `rough` | The other frames of the same object (or a range around the current one), each scaled by the reciprocal of its median, median combined | Quick look, or sparse fields |
| `remove_objects` *(default)* | The same, but done in two passes: pass one subtracts a rough sky and detects objects (with `sep` or SExtractor); pass two masks those objects out of the *other* frames before making the final sky | Most on-source dithered data. The object masking prevents bright stars imprinting negative holes. |
| `offsource` | Separate off-source sky frames (`<calib type="sky">`), combined in two passes with object masking | Sky frames taken away from the target |
| `offsource_rough` | Off-source skies, one pass only | |
| `offsource_extended` | Off-source skies; the on-source background level is estimated iteratively so an extended object does not bias it | An extended object fills a significant part of the frame |
| `offsource_neb` | Off-source skies; a different algorithm for estimating the on-source background | Nebulosity over most of the frame |

Every method scales the master sky to the frame's own background before subtracting, so sky-brightness changes between
frames are handled.

| Option | Default | Meaning |
|---|---|---|
| `sky_subtract_method` | `remove_objects` | As above |
| `source_extract_method` | `sep` | `sep` or `sextractor`. `sep` works on arrays in memory and is many times faster. |
| `use_sky_files` | `all` | `all`, `range` (use `sky_files_range` frames before and after) or `selected` (use `selected_skies`) |
| `sky_files_range` | `3` | For `range`: this many skies before and after |
| `selected_skies` | `None` | A six-column ASCII file assigning skies to objects |
| `sky_reject_type`, `sky_nlow`, `sky_nhigh`, `sky_lsigma`, `sky_hsigma` | `none`, 1, 1, 5, 5 | Rejection when combining sky frames |
| `two_pass_object_masking` and `two_pass_*` | `yes` | Tuning of the object masks (`two_pass_detect_thresh`, `two_pass_detect_minarea`, `two_pass_boxcar_size`, `two_pass_reject_level`, `two_pass_sep_ellipse_growth`) |
| `sep_detect_thresh` | `1.5` | `sep` detection threshold |
| `onsource_sorting_key` | `full` | How to order frames in time: `full`, `index`, or a FITS keyword (for example `MJD`; recommended for CIRCE) |
| `conserve_memory` | `no` | `yes` on machines with little RAM |
| `keep_skies` | `no` | Keep the master skies on disk |
| `sky_offsource_method`, `sky_offsource_range` | `auto`, `240` | Identify off-source frames by offset (arcsec) from the first frame, or from a six-column file |
| `interp_zeros_sky` | `yes` | Fill the holes left in the master sky by masked objects |
| `interp_zeros_box_size`, `interp_zeros_min_neighbors` | `3`, `2` | A hole pixel becomes the median of the non-zero pixels in a 3x3 (or 5x5) box around it, if there are at least this many; repeated so large holes fill from their edges. GPU and CPU give identical results |

Off-source skies should be added to `<queries>` as `<calib type="sky">`, optionally tied to objects with an `<object>` child.
Output: `skySubtracted/ss_*.fits`.

## cosmicRays

Removes cosmic rays from imaging frames with a simple, fast neighbourhood test. For every pixel it computes the mean and standard deviation of its
eight neighbours; a pixel more than 5 standard deviations away from that neighbour mean is flagged as a cosmic ray and replaced by the median of its neighbours. The
test is repeated for several passes. It runs late in the chain, after sky subtraction, on single frames. The spectroscopic equivalent,
[`cosmicRaysSpec`](spectroscopy.md#cosmicraysspec), offers more sophisticated algorithms.

| Option | Default | Meaning |
|---|---|---|
| `cosmic_ray_passes` | `3` | Number of passes |

## alignStack

Measures the offset between dithered frames, then **drizzle**-combines them into one image (plus an exposure map and an object map),
applying any geometric-distortion correction at the same time.

All frames with the same object name are aligned and stacked together. Give each target and each dither run its own `name`.

| `align_method` | How the shift is measured | Use when |
|---|---|---|
| `xregister` | 2-d cross-correlation of the whole frame (or an `align_box_*` region) | Clean fields with no dominating bright source |
| `xregister_constrained` | The same, limited to a window around a first guess from the header (RA, Dec and pixel scale) | Bright stars or bad columns corrupt the correlation |
| `xregister_sep`, `xregister_sep_constrained` | Builds a clean "dummy image" (a Gaussian at each detected star) and correlates that | Noisy frames |
| `sep_centroid`, `sep_centroid_constrained` | Matches star lists between frames and takes the mean shift with iterative sigma clipping | |
| `triangles` | Matches triangles (asterisms) of detected stars between frames | Many stars in the field; robust when a bright star dithers off the chip |
| `xregister_guesses`, `manual` | Uses shifts you supply in `align_shifts_file` | |

The `triangles` method is the default and is used by the imaging templates. It has its own tuning options:
`triangles_min_angle`, `triangles_max_angle`, `triangles_atol`, `triangles_rtol`, `triangles_max_stars`,
`triangles_use_sigma_clipping` and `triangles_sigma`. By default it uses the 150 brightest stars
(`triangles_max_stars`; `none` = all) and sigma-clips the shifts of the matched triangles
(`triangles_use_sigma_clipping = yes`). With every star, a crowded field is slow and full of chance matches: on the
Flamingos-2 Galactic Center frames (thousands of stars) the unlimited, unclipped shifts were 1-2.5 px off, at about 35 s
per frame; with the defaults they agree with `xregister` to 0.12 px (median), in a few seconds per filter.

**Verification (v2.4.32).** A triangle shift is only accepted if it is also confirmed by the stars themselves: after the triangles vote
(`triangles_verify = yes`, the default), the two star lists are matched at the candidate shift with a coarse-to-fine mutual nearest-neighbour
match (`triangles_match_radius`, 2.5 px), and at least `triangles_min_stars` (3) stars must coincide with a chance probability
below 10^-`triangles_min_significance` (6) once a look-elsewhere correction for the number of possible shifts is applied. Candidate shifts are
ranked by their triangle votes; if the runner-up has more than half the votes the frame is declared ambiguous and fails. Stars alone are never
enough: on 1194 frame pairs that cannot match (mirrored, transposed or from other fields) every false acceptance came from star coincidences
without triangle support, and there were none with it. The original estimate is kept when it agrees with the verified shift (within 1.5 px), so
frames that were right do not change; where it disagreed it was wrong (Crab nebula frames were off by 63-335 px). Before matching,
`triangles_remove_stationary = yes` drops detections that sit at the same pixel in many frames of a dithered sequence (detector or sky-model
artifacts that otherwise produce a competing zero shift); an undithered sequence is left alone. `tri_register` also no longer modifies the
frames it is given (it used to subtract sep's background map from them in place, so triangles-aligned stacks had a smooth background removed).

**Sparse fields and large dithers.** Triangles can only match stars the two frames share. When the field has few
stars (tens at 3 sigma) and the dither moves a frame by a large fraction of the chip, a frame far from the reference
may share fewer than three stars with it, and no triangle can match. A frame with no confirmed match is **discarded with an
ERROR** naming it (it is never given a shift of 0, 0). Set `triangles_chain_overlapping_frames = yes` to rescue such frames: a frame that cannot be
registered against the reference is matched, with the same verification, against other frames that already have a shift, nearest in the sequence
first (up to 10 candidates), and the shifts are composed. Frames that match the reference directly are unchanged, so the option only adds
registered frames. On CIRCE SextansA (22 stars in the reference, dithers up to 720 px) 13 of 32 frames had no confirmed match; with the option all 32
are registered, within 4 px of cross-correlation (26 within 2 px). Lowering `sep_detect_thresh` can also help. At dithers of ~1000 px the shift
is only accurate to a few pixels because plate scale and rotation matter (a warning reports the scatter of the matched stars); a full
distortion model is not attempted.

| Option | Default | Meaning |
|---|---|---|
| `align_method` | `triangles` | See above |
| `align_refframe` | `0` | The frame to align to |
| `align_box_size_x/y`, `align_box_center_x/y` | `-1` | The part of the frame used for correlation (`-1` = full size or centre) |
| `align_constrain_boxsize` | `256` | Window size for the `*_constrained` methods |
| `stack_method` | `drihizzle` | `drihizzle` or `drihizzle_imcombine` (with outlier rejection: `stack_reject_type` and `stack_nlow/nhigh/lsigma/hsigma`) |
| `drihizzle_kernel`, `drihizzle_dropsize` | `point`, `0.01` | Drizzle kernel and drop size |
| `geom_trans_coeffs` | `None` | A coefficient file for distortion correction (polynomial transforms) |
| `keep_indiv_images` | `no` | Also write each aligned frame |
| `keep_exposure_map` | `yes` | Write the exposure map |
| `use_only_selected_indices` | `None` | Only stack these frame indices |

Output: `alignedStacked/as_<name>.fits` (the final image), `exp_<name>.fits` (exposure map), and `objmap_<name>.fits` (object map).
