# Options reference

*[Docs home](README.md)*

**This file is generated.** It is a snapshot of `superFatboy3.py -list` for superFATBOY v2.4.14.
Run `superFatboy3.py -list` yourself for the live list, or regenerate this file with
`python3 docs/gen_options_reference.py`. For prose descriptions of what each process does, see the
[process guide](processes/README.md).

Every process also accepts `write_output`, `write_calib_output` and `create_calib_only`; processes that
carry a noisemap also accept `write_noisemaps`. They are listed here only where the process defines them.

## Contents

- [Global parameters](#global-parameters)
- [alignStack](#alignstack)
- [badPixelMask](#badpixelmask)
- [badPixelMaskSpec](#badpixelmaskspec)
- [biasSubtract](#biassubtract)
- [calibStarDivide](#calibstardivide)
- [collapseFibers](#collapsefibers)
- [cosmicRays](#cosmicrays)
- [cosmicRaysSpec](#cosmicraysspec)
- [createCleanSkies](#createcleanskies)
- [createMasterArclamps](#createmasterarclamps)
- [darkSubtract](#darksubtract)
- [deboneCirce](#debonecirce)
- [doubleSubtract](#doublesubtract)
- [emirBiasSubtract](#emirbiassubtract)
- [extractSpectra](#extractspectra)
- [findSlitlets](#findslitlets)
- [flatDivide](#flatdivide)
- [flatDivideSpec](#flatdividespec)
- [linearity](#linearity)
- [megaraIdentifyFibers](#megaraidentifyfibers)
- [megaraSkySubtract](#megaraskysubtract)
- [mergeObjects](#mergeobjects)
- [miradasCharacterizePSF](#miradascharacterizepsf)
- [miradasCollapseSpaxels](#miradascollapsespaxels)
- [miradasCombineSlices](#miradascombineslices)
- [miradasCreate3dDatacubes](#miradascreate3ddatacubes)
- [miradasDARFromConditions](#miradasdarfromconditions)
- [miradasDARFromData](#miradasdarfromdata)
- [miradasRegisterWCS](#miradasregisterwcs)
- [miradasStitchOrders](#miradasstitchorders)
- [noisemap](#noisemap)
- [rectify](#rectify)
- [remergeCirce](#remergecirce)
- [resample](#resample)
- [shiftAdd](#shiftadd)
- [sinfoniCalcLinearity](#sinfonicalclinearity)
- [sinfoniCharacterizePSF](#sinfonicharacterizepsf)
- [sinfoniCollapseSlitlets](#sinfonicollapseslitlets)
- [sinfoniCreate3dDatacubes](#sinfonicreate3ddatacubes)
- [sinfoniIdentifySlitlets](#sinfoniidentifyslitlets)
- [sinfoniRegisterStack](#sinfoniregisterstack)
- [sinfoniRemoveBadLines](#sinfoniremovebadlines)
- [skySubtract](#skysubtract)
- [skySubtractSpec](#skysubtractspec)
- [slitletAlign](#slitletalign)
- [trimOverscan](#trimoverscan)
- [trimWindow](#trimwindow)
- [wavelengthCalibrate](#wavelengthcalibrate)

## Global parameters

Set in the `<parameters>` section of the XML file with `<param name="..." value="..."/>`.

| Parameter | Default | Notes |
|---|---|---|
| `calibs_only` | `no` |  |
| `convert_mef` | `no` |  |
| `dark_file_list` | `None` |  |
| `date_keyword` | `['DATE', 'DATE-OBS']` |  |
| `dec_keyword` | `['DECOFFSE', 'DEC', 'TELDEC']` |  |
| `exptime_keyword` | `['EXPTIME', 'EXP_TIME', 'EXPCOADD']` |  |
| `filter_keyword` | `['FILTER', 'FILTNAME']` |  |
| `flat_file_list` | `None` |  |
| `gain_keyword` | `['GAIN', 'GAIN_1', 'EGAIN']` |  |
| `gpumode` | `yes` |  |
| `ignore_after_bad_read` | `no` |  |
| `ignore_first_frames` | `no` |  |
| `interactive_on_error` | `no` |  |
| `logdir` | `flogs` |  |
| `max_frame_value` | `None` |  |
| `max_init_failures` | `3` |  |
| `median_kernel` | `None` |  |
| `mef_extension` | `None` |  |
| `memory_image_limit` | `None` |  |
| `min_frame_value` | `None` |  |
| `nreads_keyword` | `['NREADS', 'LNRS', 'FSAMPLE', 'NUMFRAME']` |  |
| `obstype_keyword` | `['OBSTYPE', 'OBS_TYPE', 'IMAGETYP']` |  |
| `outputdir` | `.` |  |
| `overwrite_files` | `no` |  |
| `pixscale_keyword` | `['PIXSCALE']` |  |
| `quick_start_file` | `None` |  |
| `ra_keyword` | `['RAOFFSET', 'RA', 'TELRA']` |  |
| `relative_offset_arcsec` | `no` |  |
| `rotpa_keyword` | `['ROT_PA', 'ROTPA', 'INSTPA']` |  |
| `tempdir` | `temp-fatboy` |  |
| `ut_keyword` | `['UT', 'UTC', 'NOCUTC']` |  |
| `verbosity` | `normal` |  |

## alignStack

| Option | Default | Notes |
|---|---|---|
| `align_box_center_x` | `-1` | center of alignment box; -1 = use x-center |
| `align_box_center_y` | `-1` | center of alignment box; -1 = use y-center |
| `align_box_size_x` | `-1` | size of alignment box; -1 = use full x-size |
| `align_box_size_y` | `-1` | size of alignment box; -1 = use full y-size |
| `align_constrain_boxsize` | `256` |  |
| `align_method` | `triangles` | xregister \| xregister_constrained \| xregister_sep \| xregister_sep_constrained \| xregister_guesses \| sep_centroid \| sep_centroid_constrained \| triangles \| manual |
| `align_refframe` | `0` | number or identifier.index |
| `align_shifts_file` | `None` | shifts for manual or xregister_guesses |
| `create_calib_only` | `no` |  |
| `drihizzle_dropsize` | `0.01` |  |
| `drihizzle_in_units` | `counts` |  |
| `drihizzle_kernel` | `point` | point, turbo, etc. |
| `geom_trans_coeffs` | `None` |  |
| `keep_exposure_map` | `yes` |  |
| `keep_indiv_images` | `no` |  |
| `stack_hsigma` | `3` |  |
| `stack_lsigma` | `3` |  |
| `stack_method` | `drihizzle` | drihizzle \| drihizzle_imcombine |
| `stack_nhigh` | `3` |  |
| `stack_nlow` | `3` |  |
| `stack_reject_type` | `sigclip` |  |
| `triangles` | `delaunay` | delaunay \| all |
| `triangles_atol` | `2.0` | maximum absolute tolerance in pixels for matching triangles |
| `triangles_debug_plots` | `yes` |  |
| `triangles_max_angle` | `110` | max angle for any triangle to have |
| `triangles_max_stars` | `None` | if not None, max stars to compute triangles from, sorted by flux |
| `triangles_min_angle` | `30` | min angle for any triangle to have |
| `triangles_rtol` | `0.025` | maximum relative tolerance in pixels for matching triangles |
| `triangles_sigma` | `3` | Sigma to use for sigma clipping |
| `triangles_use_sigma_clipping` | `no` | Use sigma clipping on shifts from fit triangles |
| `use_only_selected_indices` | `None` | If not None, this can be a list of indices or ASCII file listing indices of frames to align/stack. Others will be ignored. |
| `write_calib_output` | `no` |  |
| `write_output` | `no` |  |
| `xregister_fit_2d_gaussian` | `no` |  |
| `xregister_mask_negatives` | `no` |  |
| `xregister_median_filter2d` | `yes` |  |
| `xregister_pad_align_box_cpu` | `no` |  |
| `xregister_sep_detect_thresh` | `3` |  |
| `xregister_sep_fwhm` | `a` | a=semi-major axis, otherwise a number in pixels |
| `xregister_smooth_correlation` | `no` |  |

## badPixelMask

| Option | Default | Notes |
|---|---|---|
| `bad_pixel_mask_source` | `None` |  |
| `clipping_high` | `2.0` |  |
| `clipping_low` | `0.5` |  |
| `clipping_method` | `values` | values \| sigma |
| `clipping_sigma` | `5` |  |
| `column_reject` | `None` | supports slicing, e.g. 320:384, 500, 752:768 |
| `create_calib_only` | `no` |  |
| `default_bad_pixel_mask` | `None` |  |
| `default_bpm_ignore_header` | `no` |  |
| `edge_reject` | `5` |  |
| `radius_reject` | `0` |  |
| `row_reject` | `None` | supports slicing, e.g. 320:384, 500, 752:768 |
| `write_calib_output` | `no` |  |
| `write_output` | `no` |  |

## badPixelMaskSpec

| Option | Default | Notes |
|---|---|---|
| `bad_pixel_mask_source` | `None` |  |
| `behavior` | `mask` | mask \| interpolate |
| `clipping_high` | `2.0` |  |
| `clipping_low` | `0.5` |  |
| `clipping_method` | `values` | values \| sigma |
| `clipping_sigma` | `5` |  |
| `column_reject` | `None` | supports slicing, e.g. 320:384, 500, 752:768 |
| `create_calib_only` | `no` |  |
| `default_bad_pixel_mask` | `None` |  |
| `default_bpm_ignore_header` | `no` |  |
| `edge_reject` | `5` |  |
| `interpolation_algorithm` | `median_neighbor` | Algorithm used to interpolate replacement value for bad pixels linear_1d_x \| linear_1d_y \| linear_2d \| median_neighbor \| linear_spline \| cubic_spline \| quintic_spline \| weighted \| biharmonic_2d |
| `interpolation_arg` | `None` | interpolation algorithm specific argument(s) |
| `interpolation_iterations` | `1` | Max number of iterations for interpolation algorithms |
| `normalize_source` | `yes` | ensure that bad pixel mask source is normalized before applying clipping_high/clipping_low |
| `radius_reject` | `0` |  |
| `row_reject` | `None` | supports slicing, e.g. 320:384, 500, 752:768 |
| `use_individual_slitlets` | `yes` |  |
| `write_calib_output` | `no` |  |
| `write_output` | `no` |  |

## biasSubtract

| Option | Default | Notes |
|---|---|---|
| `create_calib_only` | `no` |  |
| `default_master_bias` | `None` |  |
| `write_calib_output` | `no` |  |
| `write_noisemaps` | `no` |  |
| `write_output` | `no` |  |

## calibStarDivide

| Option | Default | Notes |
|---|---|---|
| `calib_star_spectrum` | `0` | Which extracted spectrum of the standard is the calibration star (1-based). 0 (default) = the brightest, e.g. the star in a MOS standard that also extracts faint sources from other slitlets. |
| `create_calib_only` | `no` |  |
| `debug_mode` | `no` | Show plots of each slitlet and print out debugging information. |
| `write_calib_output` | `no` |  |
| `write_fits_table` | `no` |  |
| `write_output` | `no` |  |

## collapseFibers

| Option | Default | Notes |
|---|---|---|
| `collapse_method` | `sum` | sum \| mean \| median |
| `create_calib_only` | `no` |  |
| `write_calib_output` | `no` |  |
| `write_output` | `no` |  |

## cosmicRays

| Option | Default | Notes |
|---|---|---|
| `cosmic_ray_passes` | `3` |  |
| `create_calib_only` | `no` |  |
| `write_calib_output` | `no` |  |
| `write_output` | `no` |  |

## cosmicRaysSpec

| Option | Default | Notes |
|---|---|---|
| `cosmic_ray_algorithm` | `dcr` | dcr \| lacos \| deepcr; default = dcr, uses histograms of postage stamp subimages lacos uses laplacian edge detection deepcr uses deep neural net |
| `cosmic_ray_method` | `mask` | mask \| replace |
| `create_calib_only` | `no` |  |
| `dcr_disp_axis` | `1` | Dispersion axis: 0 - no dispersion, 1 - X, 2 - Y |
| `dcr_grow_radius` | `1` | Growing radius |
| `dcr_lower_radius` | `1` | Lower radius of region for replacement statistics |
| `dcr_npass` | `5` | Maximum number of cleaning passes |
| `dcr_threshold` | `4.0` | Threshold (in STDDEV) |
| `dcr_upper_radius` | `3` | Upper radius of region for replacement statistics |
| `dcr_verbosity` | `1` | dcr output: 0 = pixels cleaned per frame, 1 = also per slitlet, 2 = also per-pass counts and frame statistics before/after cleaning |
| `dcr_xradius` | `9` | x-radius of the box (size = 2 * radius) |
| `dcr_yradius` | `9` | y-radius of the box (size = 2 * radius) |
| `deepcr_inpaint_model` | `ACS-WFC-F606W-2-32` | Model to use for inpainting (replacing) in deepCR. Default was trained on HST imaging data. |
| `deepcr_mask_model` | `ACS-WFC-F606W-2-32` | Model to use for masking in deepCR. Default was trained on HST imaging data. |
| `deepcr_threshold` | `0.5` | Threshold to use with deepCR.  Default 0.5 Set higher to avoid false positives. |
| `lacos_cosmic_ray_passes` | `1` | Number of passes for lacos algorithm |
| `lacos_cosmic_ray_sigma` | `10` | Sigma threshold for lacos algorithm |
| `lacos_xorder` | `-1` | Fit order in x-direction (-1 = smooth instead of fit and subtract) |
| `lacos_yorder` | `-1` | Fit order in y-direction (-1 = smooth instead of fit and subtract) |
| `mos_use_whole_chip` | `no` | Set to yes to run CR algorithms on the whole chip rather than each individual slitlet. |
| `remove_stray_light` | `no` | Additionally attempt to remove circular stray light artifcats, useful in KAST red data. Requires sep. |
| `stray_light_max_area` | `225` | Maximum number of pixels for a stray light artifact. |
| `stray_light_method` | `mask` | mask \| replace |
| `stray_light_min_area` | `25` | Minimum number of pixels for a stray light artifact. |
| `stray_light_sigma_threshold` | `100` | Minimum threshold times background RMS to be detected. |
| `stray_light_symmetry_threshold` | `0.9` | Semi-minor axis b / semi-major axis a of feature must be this or higher (1 = circle). |
| `write_calib_output` | `no` |  |
| `write_output` | `no` |  |

## createCleanSkies

| Option | Default | Notes |
|---|---|---|
| `combine_method` | `median` | median \| quartile \| min |
| `create_calib_only` | `no` |  |
| `default_master_clean_sky` | `None` |  |
| `max_frames_to_combine` | `10` |  |
| `write_calib_output` | `no` |  |
| `write_output` | `no` |  |

## createMasterArclamps

| Option | Default | Notes |
|---|---|---|
| `create_calib_only` | `no` |  |
| `default_master_arclamp` | `None` |  |
| `lamp_method` | `lamp_on` | lamp_on \| lamp_on-off |
| `lamp_off_files` | `off` | An ASCII text file listing on and off lamps or a filename fragment or a FITS header keyword for identifying off lamps |
| `lamp_off_header_value` | `OFF` | If lamp_off_files is a FITS keyword, value for off lamps |
| `write_calib_output` | `no` |  |
| `write_noisemaps` | `no` |  |
| `write_output` | `no` |  |

## darkSubtract

| Option | Default | Notes |
|---|---|---|
| `create_calib_only` | `no` |  |
| `default_master_dark` | `None` |  |
| `prompt_for_missing_dark` | `no` |  |
| `write_calib_output` | `no` |  |
| `write_noisemaps` | `no` |  |
| `write_output` | `no` |  |

## deboneCirce

| Option | Default | Notes |
|---|---|---|
| `create_calib_only` | `no` |  |
| `write_calib_output` | `no` |  |
| `write_output` | `no` |  |

## doubleSubtract

| Option | Default | Notes |
|---|---|---|
| `create_calib_only` | `no` |  |
| `find_shift_box_xhi` | `-1` | Used to specify a range of the chip in dispersion direction to sum a 1-d cut across and attemt to find shift. |
| `find_shift_box_xlo` | `0` | Used to specify a range of the chip in dispersion direction to sum a 1-d cut across and attemt to find shift. |
| `find_shift_box_yhi` | `-1` | Used to specify a range of the chip in cross-dispersion direction to sum a 1-d cut across and attemt to find shift. |
| `find_shift_box_ylo` | `0` | Used to specify a range of the chip in cross-dispersion direction to sum a 1-d cut across and attemt to find shift. |
| `find_shift_constrain_boxsize` | `None` | Constrain the fit to a box of this size, centered at the initial guess based on RA and Dec offsets. |
| `min_negative_flux_fraction` | `0.1` | If the sky-subtracted frame's total negative flux is less than this fraction of its positive flux, there is no negative trace to double subtract (e.g. the sky frame had the target nodded off the slit, as for some telluric standards).  Skip double subtraction and use only the positive.  0 = never skip. |
| `use_header` | `no` | Use the information in the header - RA, DEC, PIXSCALE - instead of attempting to find shift. |
| `write_calib_output` | `no` |  |
| `write_noisemaps` | `no` |  |
| `write_output` | `no` |  |

## emirBiasSubtract

| Option | Default | Notes |
|---|---|---|
| `create_calib_only` | `no` |  |
| `write_calib_output` | `no` |  |
| `write_output` | `no` |  |

## extractSpectra

| Option | Default | Notes |
|---|---|---|
| `create_calib_only` | `no` |  |
| `debug_mode` | `no` | Show plots of each slitlet and print out debugging information. |
| `extract_gauss_width` | `None` | Width in sigma of the extraction box based on Gaussian fit to 1-d cut (default 3) |
| `extract_method` | `auto` | auto \| full \| manual \| semi \| filename.xml |
| `extract_min_flux_pct` | `0.001` | If the flux dips below this percent of the peak flux then it will be considered a break between continua when auto-detecting. |
| `extract_min_width` | `5` | Minimum width to be defined as a spectrum |
| `extract_nspec` | `1` | Maximum number of spectra per slitlet to extract |
| `extract_sigma` | `2` | Minimum sigma threshold above background in 1-d cut |
| `extract_weighting` | `linear` | linear \| gaussian \| median |
| `extract_xhi` | `None` | Coordinate for extraction box for 1-d cut to auto-detect |
| `extract_xlo` | `None` | Coordinate for extraction box for 1-d cut to auto-detect |
| `extract_yhi` | `None` | Coordinate for extraction box for 1-d cut to auto-detect |
| `extract_ylo` | `None` | Coordinate for extraction box for 1-d cut to auto-detect |
| `gaussian_box_size` | `25` |  |
| `write_calib_output` | `no` |  |
| `write_fits_table` | `no` |  |
| `write_noisemaps` | `no` |  |
| `write_output` | `no` |  |
| `write_plots` | `no` |  |

## findSlitlets

| Option | Default | Notes |
|---|---|---|
| `autodetect_peak_local_max` | `no` | For fiber data such as MEGARA, use peak local max to find fiber locations |
| `background_boxcar_width` | `25` | Width in pixels of the boxcar used to subtract off background level in 1-d cut.  Should be just under 2 x slit width. |
| `boundary` | `10` | Width in pixels of a boundary to not attempt to fit at the edges of each segment.  Should be 100 for MIRADAS. |
| `create_calib_only` | `no` |  |
| `cut1d_max_threshold` | `2` | Reject a trace datapoint if 1d cut max < this factor * quartile of cut. |
| `debug_mode` | `no` |  |
| `edge_detection_method` | `auto` | Method used at each step to find the slitlet edge position: cross_correlation = cross-correlate 1-d cut with a reference cut and fit a Gaussian to the correlation peak.  Best for slitlets separated by a genuine step edge (flux drops to ~0 between them).  Regresses badly on weak local-minimum boundaries (see local_minimum below) -- typically finds 0 datapoints for that edge. local_minimum = directly find the local minimum flux value in the 1-d cut (with subpixel parabolic refinement) instead of cross-correlating.  Much better for closely-packed slitlets where the boundary is only a weak dip in flux rather than a full step down to 0, which cross_correlation fails on -- but regresses badly on genuine step edges (a step's minimum sits at the edge of the search window, not at an interior parabolic minimum), so it is NOT a safe drop-in replacement for cross_correlation across a whole dataset. auto (default) = try cross_correlation first for every edge (matches cross_correlation exactly for any edge it can trace); only for an edge where that finds literally 0 datapoints (the weak-dip failure mode above) does it retry that same edge with local_minimum instead of giving up.  Recommended over local_minimum whenever a dataset mixes both edge types, which is the common case (see findSlitletProcess algorithm audit notes). |
| `edge_extend_to_chip` | `no` | If set to yes, and one edge of a slitlet is traced out, the other edge if it runs into the chip boundary will not be clipped. |
| `edge_threshold` | `15` | Do not attempt to trace out slitlets within this many pixels of edges |
| `fiber_width` | `5` | Width of fibers, used with peak local max |
| `fit_function` | `polynomial` | Function used to fit the traced (x,y) edge/shift datapoints to a smooth curve Y=f(X): polynomial (default) = single global leastsq polynomial fit of fit_order, as before. A higher fit_order fits real curvature better locally but its extrapolation past the fitted x-range grows increasingly unstable (Runge's phenomenon) -- see rectifyProcess's similar spline-vs-polynomial finding. spline = smoothing B-spline (scipy UnivariateSpline, degree=min(fit_order,5)) through the same datapoints.  Follows local curvature at least as well and extrapolates far more stably at the fitted range's edges/gaps, at the cost of no longer having simple polynomial coefficients to log.  Falls back to polynomial automatically if there are too few datapoints for the requested spline degree. |
| `fit_order` | `2` | Order of polynomial to use to fit slitlet shape. Recommended value = 2 for trace_slitlets_individually, 3 for group mode |
| `flexure_correction` | `none` | none \| shift \| gradient (linear = shift).  Correct for flexure between the flat and each object: measure the shift between the master flat and the object's frames from the slitlet edges (sky-lit), then give that object its own slitmask moved to its frames and a master flat whose slit illumination is moved (pixel response stays in place).  shift = one shift per object; gradient = shift varying linearly along the cross-dispersion direction.  A slitmask that is already aligned with the object (e.g. from a region file drawn on the data) is not moved, only the flat. Writes findSlitlets/flexure_<object>.txt with every edge measurement. |
| `flexure_max_shift` | `5` | Largest flexure shift in pixels searched for by flexure_correction |
| `invert_before_correlating` | `no` | Invert flat field to turn gap trough into a peak for cross correlations |
| `local_min_depth_threshold` | `0.05` | For edge_detection_method=local_minimum only: minimum dip depth required to accept a datapoint, as a fraction of the 1-d cut's local median flux.  Rejects steps where no real dip is present (e.g. pure noise or a genuine data gap). |
| `local_min_search_radius` | `3` | For local_minimum edge tracing: once the trace has accepted a datapoint, only search for the minimum within this many pixels of the predicted position, and reject the point if the minimum is at the edge of that window (no real dip, e.g. a step between two lit slitlets). Stops the trace drifting onto a random point of a fainter neighboring slitlet. |
| `max_residual_error` | `2.0` | Maximum sigma of residuals to fit to be rejected as an invalid fit, default 1.0 |
| `min_coverage_fraction` | `30` | Minimum percentage of a slitlet to trace out to be valid for a fit, default 30% |
| `n_segments` | `1` | Number of piecewise functions to fit.  Should be 2 for MIRADAS, 1 for most other cases. |
| `narrow_gaps_between_slitlets` | `no` | Set to yes for closely packed slitlets whose boundaries are only a dip in flux rather than a drop to background.  cross_correlation edge tracing rejects any datapoint failing the cut1d_max_threshold (peak vs lower quartile) check, which assumes one side of every edge is dark background, so every datapoint along a packed boundary is rejected. With yes, such a datapoint is measured with local_minimum instead.  Unlike auto, which switches a whole edge only when it finds 0 datapoints, this switches point by point, so it also handles an edge that is a step along part of the slit and packed along the rest. |
| `order_step_size` | `5` | Step size in pixels for tracing out orders, default = 5. |
| `padding` | `0` | Number of pixels to pad slitlets by on each side, into the empty rows between slitlets.  A gap narrower than 2*padding is split between its two neighbors so slitlets never overlap.  Applies to all tracing methods.  Default=0 |
| `region_file` | `None` | .reg, .xml, or .txt file describing slitlets |
| `slitlet_attempt_autocorrect` | `no` | If slitlets found does not match slitlet_autodetect_nslits attempt to auto-correct before failing. |
| `slitlet_autocorrect_gap_size` | `None` | Correct auto-detected slitlets to have uniform gaps between slitlets of this size. |
| `slitlet_autodetect_arc_min_corr` | `0.9` | For slitlet_autodetect_source = arclamp or both: minimum correlation between adjacent rows of the high-pass filtered arclamp for them to be part of the same slitlet. |
| `slitlet_autodetect_boxsize` | `5` | Boxsize for auto-detecting slitlets |
| `slitlet_autodetect_min_flux_pct` | `0.001` | When flux drops below this percent of max, force break between slitlets |
| `slitlet_autodetect_min_trough_depth` | `0.3` | Minimum depth of the trough between two adjacent slitlets, as a fraction of the fainter slitlet's height above background, to split them when the flux between them does not drop all the way to background.  Lower (e.g. 0.1) for closely packed slitlets with very shallow boundaries; too low risks splitting slitlets at dust or bad-row dips. |
| `slitlet_autodetect_min_width` | `10` | Minimum width of a slitlet for auto-detection |
| `slitlet_autodetect_nslits` | `0` | Set this to the number of slitlets if auto-detecting them as a check that it found the correct number of slitlets (0 = no check) |
| `slitlet_autodetect_sigma` | `5` | Minimum sigma vs local noise to be a step for slitlet detection |
| `slitlet_autodetect_source` | `flat` | Calibration frame used to auto-detect slitlets if no region file: flat (default) = steps in a 1-d cut of the master flat. arclamp = correlation between adjacent rows of the master arclamp: rows within one slitlet share the same line pattern, so a boundary between closely packed slitlets shows up even when the flat barely dips there, as long as adjacent slitlets are offset in wavelength. Uses the full dispersion range, so is best suited to slitlets that are not strongly tilted. both = slitlets and packed boundaries from the arclamp, outer edges refined to the flat's half-max, and flat regions with no coherent arc spectrum (e.g. a mask ID) reported and dropped. The master arclamp is found or created via createMasterArclamps, which must be in the XML. |
| `slitlet_autodetect_use_median` | `no` | Set to yes to use median rather than sum for auto detection |
| `slitlet_autodetect_use_orig_algorithm` | `no` | Set to yes to auto-detect slitlets with the original extractSpectra algorithm (extractSpectra_orig, versions <= 2.3.29): global sigma-clipped background, no trough splitting. |
| `slitlet_autodetect_x` | `1024` | Central pixel in continuum direction for auto-detecting slitlets if no region file. |
| `slitlet_trace_boxsize` | `21` | Boxsize in cross-dispersion direction of 1-d cut for tracing in individual mode |
| `slitlet_trace_yhi` | `-1` | Upper bound in cross-dispersion direction of 1-d cut for tracing in group mode (-1 = 3/4 ysize) |
| `slitlet_trace_ylo` | `-1` | Lower bound in cross-dispersion direction of 1-d cut for tracing in group mode (-1 = 1/4 ysize) |
| `slitlet_validity_max_flat_roughness` | `0.045` | Flag a slitlet as invalid (e.g. a mask ID or alignment hole) if the robust row-to-row scatter of the flat across it, as a fraction of its flux, exceeds this.  Auto-detected invalid slitlets are dropped; region file slitlets get a warning only.  0 = disable. |
| `slitlet_validity_min_arc_corr` | `0.9` | When a master arclamp is used (slitlet_autodetect_source = arclamp or both), also flag a slitlet as invalid if the mean correlation between its adjacent arclamp rows is below this. 0 = disable. |
| `spline_smoothing` | `-1` | For fit_function=spline only: smoothing factor (scipy UnivariateSpline's s). -1 (default) = let scipy pick its own default smoothing.  Larger values smooth more (fewer, gentler wiggles); 0 = interpolate every point exactly (no smoothing at all). |
| `subtract_background_level` | `no` | Subtract a running boxcar min from the 1-d cut	before attempting to find slitlets |
| `trace_peak_local_max` | `no` | Set to yes for MEGARA or other fiber data np.where the curvature changes between fibers |
| `trace_slitlets_individually` | `yes` | Set to yes for echelle spectra np.where the curvature changes between slitlets. |
| `write_calib_output` | `no` |  |
| `write_output` | `no` |  |
| `write_plots` | `no` |  |

## flatDivide

| Option | Default | Notes |
|---|---|---|
| `create_calib_only` | `no` |  |
| `default_master_flat` | `None` |  |
| `flat_lamp_off_files` | `off` |  |
| `flat_method` | `dome_on` | dome_on \| dome_on-off \| sky \| twilight |
| `flat_sky_hsigma` | `5` |  |
| `flat_sky_lsigma` | `5` |  |
| `flat_sky_nhigh` | `1` |  |
| `flat_sky_nlow` | `1` |  |
| `flat_sky_reject_type` | `none` |  |
| `median_section` | `None` | An optional subsection of the image using slice notation which will be used to calculate the median value for renormalization.  E.g. 400:1200,500:1000 |
| `sky_flat_exclude_list` | `` |  |
| `sky_flat_include_list` | `` |  |
| `twilight_pair_ramps` | `no` |  |
| `write_calib_output` | `no` |  |
| `write_output` | `no` |  |

## flatDivideSpec

| Option | Default | Notes |
|---|---|---|
| `create_calib_only` | `no` |  |
| `default_master_flat` | `None` |  |
| `flat_hi_replace` | `1` | Value for normalized flat pixels above flat_hi_thresh |
| `flat_hi_thresh` | `0` | Pixels of the normalized flat above this value are replaced by flat_hi_replace.  0 = off |
| `flat_lamp_off_files` | `off` | An ASCII text file listing on and off flats or a filename fragment or a FITS header keyword for identifying off flats |
| `flat_lamp_off_header_value` | `OFF` | If flat_lamp_off_files is a FITS keyword, value for off flats |
| `flat_low_replace` | `1` | Value for normalized flat pixels below flat_low_thresh |
| `flat_low_thresh` | `0` | Pixels of the normalized flat (per slitlet for MOS) below this value are replaced by flat_low_replace, e.g. 0.3 so dim slit-edge rows are not amplified by flat division.  0 = off |
| `flat_method` | `dome_on` | dome_on \| dome_on-off |
| `flat_selection` | `all` | all \| object_keyword |
| `normalize_flat` | `yes` |  |
| `prompt_for_missing_flat` | `yes` |  |
| `write_calib_output` | `no` |  |
| `write_noisemaps` | `no` |  |
| `write_output` | `no` |  |

## linearity

| Option | Default | Notes |
|---|---|---|
| `create_calib_only` | `no` |  |
| `divide_by_coadds` | `no` |  |
| `do_linearity` | `yes` |  |
| `linearity_coeffs` | `1` |  |
| `write_calib_output` | `no` |  |
| `write_output` | `no` |  |

## megaraIdentifyFibers

| Option | Default | Notes |
|---|---|---|
| `create_calib_only` | `no` |  |
| `missing_fiber_list` | `None` | Comma separated list of missing fiber numbers (ids start with 1) |
| `write_calib_output` | `no` |  |
| `write_output` | `no` |  |

## megaraSkySubtract

| Option | Default | Notes |
|---|---|---|
| `create_calib_only` | `no` |  |
| `debug_mode` | `no` |  |
| `default_master_sky` | `None` |  |
| `keep_skies` | `no` |  |
| `scaling` | `none` | none \| peak \| skylines |
| `scaling_nlines` | `3` | Number of skylines to use for scaling |
| `sky_combine_method` | `median` | median \| mean |
| `write_calib_output` | `no` |  |
| `write_noisemaps` | `no` |  |
| `write_output` | `no` |  |
| `write_plots` | `no` |  |

## mergeObjects

| Option | Default | Notes |
|---|---|---|
| `create_calib_only` | `no` |  |
| `write_calib_output` | `no` |  |
| `write_output` | `no` |  |

## miradasCharacterizePSF

| Option | Default | Notes |
|---|---|---|
| `centroid_method` | `fit_2d_gaussian` | fit_2d_gaussian \| use_derivatives |
| `create_calib_only` | `no` |  |
| `detection_threshold` | `2` | Minimum sigma above median to detect a source |
| `slitlet_number` | `all` | Set to all (default) to collapse spaxels for all slitlets. Set to a number 1-13 to only select one slitlet. |
| `write_calib_output` | `no` |  |
| `write_output` | `no` |  |

## miradasCollapseSpaxels

| Option | Default | Notes |
|---|---|---|
| `box_hi` | `None` | Bounds of extraction box in dispersion axis where data exists in arclamp. |
| `box_lo` | `None` | Bounds of extraction box in dispersion axis where data exists in arclamp. |
| `centroid_method` | `fit_2d_gaussian` | fit_2d_gaussian \| use_derivatives |
| `create_calib_only` | `no` |  |
| `debug_mode` | `no` |  |
| `do_centroids` | `yes` | Whether to centroid images |
| `max_shift_between_slices` | `4` | max shift between any slice and the mean center. |
| `nslices` | `3` | Number of slices per slitlet |
| `pad_images_to_same_size` | `yes` |  |
| `slice_width` | `auto` | Slice width in pixels or auto to auto-detect. |
| `slitlet_number` | `all` | Set to all (default) to collapse spaxels for all slitlets. Set to a number 1-13 to only select one slitlet. |
| `use_integer_shifts` | `no` | If set to yes, integer shifts will be used and data will not be resampled. |
| `write_calib_output` | `no` |  |
| `write_output` | `no` |  |

## miradasCombineSlices

| Option | Default | Notes |
|---|---|---|
| `create_calib_only` | `no` |  |
| `slice_weighting` | `none` | none \| normalize_flux \| flux_weighted \| noisemap_only |
| `slices_per_slitlet` | `0` | Set this to a nonzero integer to specify number of slices per slitlet. If 0, it will use info in header or a slitlet_list passed as calib. |
| `write_calib_output` | `no` |  |
| `write_noisemaps` | `no` |  |
| `write_output` | `no` |  |

## miradasCreate3dDatacubes

| Option | Default | Notes |
|---|---|---|
| `apply_dar_correction` | `yes` | Whether to apply a DAR correction if it exists |
| `centroid_method` | `fit_2d_gaussian` | fit_2d_gaussian \| use_derivatives |
| `create_calib_only` | `no` |  |
| `do_centroids` | `yes` | Whether to centroid images |
| `drihizzle_dropsize` | `1` |  |
| `drihizzle_kernel` | `turbo` | turbo \| point \| point_replace \| tophat \| gaussian \| fastgauss \| lanczos |
| `max_shift_between_slices` | `4` | max shift between any slice and the mean center. |
| `nslices` | `3` | Number of slices per slitlet |
| `slice_width` | `auto` | Slice width in pixels or auto to auto-detect. |
| `slitlet_number` | `all` | Set to all (default) to collapse spaxels for all slitlets. Set to a number 1-13 to only select one slitlet. |
| `use_integer_shifts` | `no` | If set to yes, integer shifts will be used and data will not be resampled. |
| `write_calib_output` | `no` |  |
| `write_output` | `no` |  |

## miradasDARFromConditions

| Option | Default | Notes |
|---|---|---|
| `create_calib_only` | `no` |  |
| `slitlet_number` | `all` | Set to all (default) to collapse spaxels for all slitlets. Set to a number 1-13 to only select one slitlet. |
| `write_calib_output` | `no` |  |
| `write_output` | `no` |  |

## miradasDARFromData

| Option | Default | Notes |
|---|---|---|
| `create_calib_only` | `no` |  |
| `slitlet_number` | `all` | Set to all (default) to collapse spaxels for all slitlets. Set to a number 1-13 to only select one slitlet. |
| `write_calib_output` | `no` |  |
| `write_output` | `no` |  |

## miradasRegisterWCS

| Option | Default | Notes |
|---|---|---|
| `create_calib_only` | `no` |  |
| `mos_slitlet_1_probe_arm` | `1` | The probe arm for slitlet 1 in MOS mode |
| `single_object_probe_arm` | `5` | The probe arm used for SOL and SOS modes |
| `write_calib_output` | `no` |  |
| `write_output` | `no` |  |

## miradasStitchOrders

| Option | Default | Notes |
|---|---|---|
| `create_calib_only` | `no` |  |
| `order_weighting` | `none` | none \| normalize_flux \| flux_weighted \| noisemap_only |
| `write_calib_output` | `no` |  |
| `write_noisemaps` | `no` |  |
| `write_output` | `no` |  |

## noisemap

| Option | Default | Notes |
|---|---|---|
| `create_calib_only` | `no` |  |
| `write_calib_output` | `no` |  |
| `write_output` | `no` |  |

## rectify

| Option | Default | Notes |
|---|---|---|
| `continuum_find_xhi` | `None` | Defaults to xsize.  Used to specify a range of the chip to sum a 1-d cut across and attemt to find any continua. In the case of highly curved orders, a narrower range is needed. |
| `continuum_find_xlo` | `None` | Defaults to 0.  Used to specify a range of the chip to sum a 1-d cut across and attemt to find any continua. In the case of highly curved orders, a narrower range is needed. |
| `continuum_trace_xinit` | `None` |  |
| `continuum_trace_yhi` | `None` | Used in longslit only.  Highest point on chip to trace out continua in case of bad area on chip. |
| `continuum_trace_ylo` | `None` | Used in longslit only.  Lowest point on chip to trace out continua in case of bad area on chip. |
| `create_calib_only` | `no` |  |
| `debug_mode` | `no` |  |
| `drihizzle_dropsize` | `1` |  |
| `drihizzle_kernel` | `turbo` | turbo \| point \| point_replace \| tophat \| gaussian \| fastgauss \| lanczos |
| `fit_function` | `polynomial` | Function used for the phase-3 outlier-rejection curve fit in calcLongslitContinuaRectification and traceMOSContinuaRectification (this fit's own result is discarded -- only its residuals decide which traced datapoints are kept -- so this mainly affects which points survive to the real rectification fit downstream): polynomial (default) = single leastsq polynomial fit, as before. spline = smoothing B-spline (scipy UnivariateSpline, degree=min(order,5)).  At this function's typical fit_order/mos_fit_order (2-4) a smoothing spline is numerically identical to the polynomial (0 interior knots) -- only useful if you raise the order for a more sharply curved trace.  Falls back to polynomial for too few datapoints. |
| `fit_order` | `2` | Longslit only.  Order of polynomial to use to fit continua. |
| `independent_slitlets_fallback` | `identity` | independent_slitlets mode only.  What to do for a slit with no continuum (or an unusable fit): identity (leave untransformed, previous behavior), pooled_good_slits (fit one whole_chip-style surface from EVERY other slit in this exposure that DOES have usable continuum, and use that instead of identity), or nearest_neighbor_slits (same idea, but pooled from just the independent_slitlets_neighbor_count physically nearest good slits instead of all of them -- can track local curvature better than pooling the whole chip when curvature varies a lot across the field, e.g. MIRADAS).  See the rectify audit notes for a leave-one-out comparison of these on real data before changing the default away from identity for a production run. |
| `independent_slitlets_neighbor_count` | `2` | independent_slitlets_fallback=nearest_neighbor_slits only.  Number of physically nearest good slits (by cross-dispersion center) to pool together for the substitute fit. |
| `longslit_continua_frames` | `None` | FITS file or ASCII list of FITS files to use to trace out continua |
| `max_continua_per_slit` | `1` | Max number of continua to be traced out in each slitlet/order |
| `min_continuum_fwhm` | `1.5` | Minimum FWHM in pixels for tracing out continua. Change to lower than 1.5 if continua are narrow. |
| `min_coverage_fraction` | `30` | Minimum percentage of continuum or skyline that must be traced out in order to be included in fit. |
| `min_sky_threshold` | `2.5` | Minimum sigma threshold compared to noise for sky/lamp line to be detected |
| `min_threshold` | `5` | Minimum sigma threshold compared to noise for continuum to be detected |
| `mos_continua_frames` | `None` | FITS file or ASCII list of FITS files to use to trace out continua |
| `mos_continuum_boundary_size` | `50` | MOS only! Step size in pixels for boundary at edges to not attempt to trace continua. Default: 50 |
| `mos_continuum_step_size` | `5` | MOS only! Step size in pixels for tracing MOS continua within each slitlet. Default: 5 |
| `mos_double_subtract_continua` | `yes` | Set to yes if input data has been sky subtracted. It will double subtract data before tracing out continua	to increase S/N ratio. |
| `mos_faint_floor_pct` | `0.02` | MOS only! While tracing a continuum, a datapoint whose fitted peak flux is below this fraction of the brightest point seen so far on this continuum is rejected as too faint UNLESS it is still locally significant -- see mos_faint_local_sigma.  Default: 0.02 (2%) |
| `mos_faint_local_sigma` | `1.0` | MOS only! Rescues a datapoint that falls below mos_faint_floor_pct if its peak is still this many sigma above its own local background (same test as the standard per-point significance check just above it).  A continuum can legitimately vary in brightness by much more than mos_faint_floor_pct allows across an order (blaze falloff, strong telluric/OH absorption); without this, the trace is lost for good the first time that happens since currY then never updates.  Set very high (e.g. 100) to recover the old fixed-floor-only behavior.  Default: 1.0 (same bar as the local significance test) |
| `mos_find_lines_alternate_boxsize` | `11` | The boxsize in pixels in the center of the slitlet to use to find sky/lamp lines using alternate method. |
| `mos_find_lines_alternate_method` | `no` | Use alternate method to find sky/lamp lines in slitlets. Should be yes if slits are extremely curved like fire data. |
| `mos_fit_order` | `2` | MOS only.  Order of polynomial to use to fit continua. |
| `mos_max_slit_width` | `10` | Anything with a greater width is assumed to be a guide star box and will be blanked out at this stage. |
| `mos_min_continua_global_fit` | `3` | For mos_mode = whole_chip or use_slitpos: minimum number of traced continua, spanning at least 25% of the slitlets, needed to fit the continuum transformation.  With fewer (e.g. a telluric standard with one bright star), reuse the continuum transformation already calculated for another object with the same mask.  0 = always fit. |
| `mos_mode` | `use_slitpos` | independent_slitlets \| use_slitpos \| whole_chip |
| `mos_sky_fallback` | `identity` | What to do for a slit/segment with no traced skylines (or an unusable fit): identity (leave untransformed, previous behavior -- and still the default), pooled_good_slits (fit one whole_chip-style surface from every other slit in this exposure that DOES have traced skylines) or nearest_neighbor_slits (same idea, but pooled from just the mos_sky_neighbor_count physically nearest good slits instead of all of them). Same vocabulary as independent_slitlets_fallback for continua, under its own name since skyline rectification always fits per-slit -- there is no use_slitpos/whole_chip mode here to gate it behind. See the rectify audit notes. |
| `mos_sky_fit_order` | `2` | MOS only! Fit order for MOS skyline rectification within each slitlet |
| `mos_sky_neighbor_count` | `2` | mos_sky_fallback=nearest_neighbor_slits only.  Number of physically nearest good slits (by cross-dispersion center) to pool together for the substitute fit. |
| `mos_sky_step_size` | `5` | MOS only! Step size in pixels for tracing MOS skylines within each slitlet |
| `n_segments` | `1` | Number of piecewise functions to fit.  Should be 2 for MIRADAS, 1 for most other cases. |
| `rect_coeffs_file` | `None` | file describing rectification coefficients |
| `rectify_continua` | `yes` | Turn off to skip continua and only rectify sky. |
| `rectify_max_transform_factor` | `2.0` | If a continuum trace fit extrapolates to transform values more than this many times the image size, retry with a linear fit rather than risk a hugely oversized rectified output image. |
| `rectify_sky` | `yes` | Turn off to skip sky and only rectify continua. |
| `region_file` | `None` | .reg, .xml, or .txt file describing slitlets |
| `sky_boxsize` | `6` | Boxsize in pixels for tracing out sky/lamp lines |
| `sky_fit_order` | `4` | Longslit only!  Fit order for longslit skyline rectification. |
| `sky_max_slope` | `0.04` | Maximum slope of skylines. Change this if skylines are very tilted. |
| `sky_two_pass_detection` | `yes` | Use 2-pass detection to try to ensure skylines are found in both halves of image |
| `skyline_find_yhi` | `None` | Defaults to ysize.  Used to specify a range of the chip to sum a 1-d cut across and attemt to find any sky or lamp lines. In the case of highly curved orders, a narrower range is needed. |
| `skyline_find_ylo` | `None` | Defaults to 0.  Used to specify a range of the chip to sum a 1-d cut across and attemt to find any sky or lamp lines. In the case of highly curved orders, a narrower range is needed. |
| `skyline_mask_radius` | `15` | Number of pixels to mask on either side of a found skyline before attempting to find next one.  Set to 0 to fit and subtract Gaussian instead. |
| `skyline_trace_yinit` | `None` |  |
| `spline_smoothing` | `-1` | For fit_function=spline only: smoothing factor (scipy UnivariateSpline's s). -1 (default) = let scipy pick its own default smoothing.  Larger values smooth more; 0 = interpolate every point exactly (no smoothing at all). |
| `use_arclamps` | `no` | no = use master "clean sky", yes = use master arclamp |
| `use_zero_as_center_fitting` | `no` | Use (0,0) as the center of the chip/slitlet when fitting (e.g. subtract x0 and y0 before least squares fit) |
| `write_calib_output` | `no` |  |
| `write_noisemaps` | `no` |  |
| `write_output` | `no` |  |
| `write_plots` | `no` |  |
| `xtrans_rect_file` | `None` |  |
| `ytrans_rect_file` | `None` |  |

## remergeCirce

| Option | Default | Notes |
|---|---|---|
| `create_calib_only` | `no` |  |
| `write_calib_output` | `no` |  |
| `write_output` | `no` |  |

## resample

| Option | Default | Notes |
|---|---|---|
| `create_calib_only` | `no` |  |
| `output_units` | `cps` | counts \| cps |
| `resample_calibs` | `yes` | Resample slitmask, master lamp, and clean sky if they exist |
| `resample_to_common_scale` | `yes` | Resample all slitlets to common scale for MOS data. If set to no, each slitlet will be resampled to an independent linear scale. |
| `write_calib_output` | `no` |  |
| `write_noisemaps` | `no` |  |
| `write_output` | `no` |  |

## shiftAdd

| Option | Default | Notes |
|---|---|---|
| `create_calib_only` | `no` |  |
| `find_shift_box_xhi` | `-1` | Used to specify a range of the chip in dispersion direction to sum a 1-d cut across and attemt to find shift. |
| `find_shift_box_xlo` | `0` | Used to specify a range of the chip in dispersion direction to sum a 1-d cut across and attemt to find shift. |
| `find_shift_box_yhi` | `-1` | Used to specify a range of the chip in cross-dispersion direction to sum a 1-d cut across and attemt to find shift. |
| `find_shift_box_ylo` | `0` | Used to specify a range of the chip in cross-dispersion direction to sum a 1-d cut across and attemt to find shift. |
| `find_shift_constrain_boxsize` | `None` | Constrain the fit to a box of this size, centered at the initial guess based on RA and Dec offsets. |
| `manual_shifts` | `None` | Set to a number, comma separated list, or ASCII file to specify shifts |
| `mos_use_whole_chip` | `no` | Set to yes to shift/add the whole chip rather than each individual slitlet. |
| `output_rows_between_slitlets` | `3` | Number of blank rows to space slitlets by in output. |
| `use_header` | `no` | Use the information in the header - RA, DEC, PIXSCALE - instead of attempting to find offsets. |
| `write_calib_output` | `no` |  |
| `write_noisemaps` | `no` |  |
| `write_output` | `no` |  |

## sinfoniCalcLinearity

| Option | Default | Notes |
|---|---|---|
| `create_calib_only` | `no` |  |
| `lamp_method` | `lamp_on-off` | lamp_on \| lamp_on-off |
| `lamp_off_files` | `ESO INS1 LAMP5 ST` | An ASCII text file listing on and off lamps or a filename fragment or a FITS header keyword for identifying off lamps |
| `lamp_off_header_value` | `False` | If lamp_off_files is a FITS keyword, value for off lamps |
| `linearity_fit_order` | `2` | Order of the polynomial fit, default=2 |
| `write_calib_output` | `no` |  |
| `write_output` | `no` |  |

## sinfoniCharacterizePSF

| Option | Default | Notes |
|---|---|---|
| `centroid_method` | `fit_2d_gaussian` | fit_2d_gaussian \| use_derivatives |
| `create_calib_only` | `no` |  |
| `detection_threshold` | `2` | Minimum sigma above median to detect a source |
| `write_calib_output` | `no` |  |
| `write_output` | `no` |  |

## sinfoniCollapseSlitlets

| Option | Default | Notes |
|---|---|---|
| `centroid_method` | `fit_2d_gaussian` | fit_2d_gaussian \| use_derivatives |
| `collapse_method` | `sum` | sum \| median |
| `create_calib_only` | `no` |  |
| `debug_mode` | `no` |  |
| `do_centroids` | `yes` | Whether to centroid images |
| `line_search_box_hi` | `-100` | Search box for brightest line in arclamp/sky to use for finding slitlet shifts. |
| `line_search_box_lo` | `100` | Search box for brightest line in arclamp/sky to use for finding slitlet shifts. |
| `reference_slit` | `2` | Reference slit when finding offsets between slits |
| `use_arclamps` | `no` | no = use master "clean sky", yes = use master arclamp |
| `use_integer_shifts` | `no` | If set to yes, integer shifts will be used and data will not be resampled. |
| `write_calib_output` | `no` |  |
| `write_output` | `no` |  |
| `write_plots` | `no` |  |

## sinfoniCreate3dDatacubes

| Option | Default | Notes |
|---|---|---|
| `create_calib_only` | `no` |  |
| `debug_mode` | `no` |  |
| `drihizzle_dropsize` | `1` |  |
| `drihizzle_kernel` | `turbo` | turbo \| point \| point_replace \| tophat \| gaussian \| fastgauss \| lanczos |
| `reference_slit` | `2` | Reference slit when finding offsets between slits |
| `use_arclamps` | `no` | no = use master "clean sky", yes = use master arclamp |
| `use_integer_shifts` | `no` | If set to yes, integer shifts will be used and data will not be resampled. |
| `write_calib_output` | `no` |  |
| `write_output` | `no` |  |
| `write_plots` | `no` |  |

## sinfoniIdentifySlitlets

| Option | Default | Notes |
|---|---|---|
| `create_calib_only` | `no` |  |
| `slitorder` | `None` | Comma separated list or text file listing slitlet numbers from left to right |
| `write_calib_output` | `no` |  |
| `write_output` | `no` |  |

## sinfoniRegisterStack

| Option | Default | Notes |
|---|---|---|
| `align_box_center_x` | `-1` | center of alignment box; -1 = use x-center |
| `align_box_center_y` | `-1` | center of alignment box; -1 = use y-center |
| `align_box_size_x` | `-1` | size of alignment box; -1 = use full x-size |
| `align_box_size_y` | `-1` | size of alignment box; -1 = use full y-size |
| `align_constrain_boxsize` | `256` |  |
| `align_method` | `xregister` | xregister \| xregister_constrained \| xregister_sep \| xregister_sep_constrained \| xregister_guesses \| sep_centroid \| sep_centroid_constrained \| manual |
| `align_refframe` | `0` | number or identifier.index |
| `align_shifts_file` | `None` | shifts for manual or xregister_guesses |
| `create_calib_only` | `no` |  |
| `drihizzle_dropsize` | `0.01` |  |
| `drihizzle_in_units` | `counts` |  |
| `drihizzle_kernel` | `point` | point, turbo, etc. |
| `geom_trans_coeffs` | `None` |  |
| `keep_exposure_map` | `yes` |  |
| `keep_indiv_images` | `no` |  |
| `use_only_selected_indices` | `None` | If not None, this can be a list of indices or ASCII file listing indices of frames to align/stack. Others will be ignored. |
| `write_calib_output` | `no` |  |
| `write_output` | `no` |  |
| `xregister_fit_2d_gaussian` | `no` |  |
| `xregister_mask_negatives` | `no` |  |
| `xregister_median_filter2d` | `yes` |  |
| `xregister_pad_align_box_cpu` | `no` |  |
| `xregister_sep_detect_thresh` | `3` |  |
| `xregister_sep_fwhm` | `a` | a=semi-major axis, otherwise a number in pixels |
| `xregister_smooth_correlation` | `no` |  |

## sinfoniRemoveBadLines

| Option | Default | Notes |
|---|---|---|
| `create_calib_only` | `no` |  |
| `nsigmaback` | `18` | Sigma to identify most of the deviant background pixels |
| `write_calib_output` | `no` |  |
| `write_output` | `no` |  |

## skySubtract

| Option | Default | Notes |
|---|---|---|
| `conserve_memory` | `no` | Set to yes if running on a machine with low RAM |
| `create_calib_only` | `no` |  |
| `default_master_sky` | `None` |  |
| `fit_sky_subtracted_surf` | `no` |  |
| `interp_zeros_sky` | `yes` |  |
| `keep_skies` | `no` |  |
| `onsource_sorting_key` | `full` | full \| index \| a FITS keyword to sort onsource skies by time. For CIRCE data, MJD is recommended. |
| `selected_skies` | `None` | 6 column ASCII file specifying index, start, stop for objects and skies for selected onsource skies. |
| `sep_detect_thresh` | `1.5` |  |
| `sextractor_path` | `/usr/bin/sex` |  |
| `sky_dithering_range` | `2` |  |
| `sky_files_range` | `3` | n skies before AND n skies after will be used |
| `sky_hsigma` | `5` |  |
| `sky_lsigma` | `5` |  |
| `sky_neb_range` | `5` |  |
| `sky_nhigh` | `1` |  |
| `sky_nlow` | `1` |  |
| `sky_offsource_method` | `auto` |  |
| `sky_offsource_range` | `240` |  |
| `sky_reject_type` | `none` |  |
| `sky_subtract_method` | `remove_objects` | rough \| remove_objects \| offsource \| offsource_rough \| offsource_extended \| offsource_neb |
| `source_extract_method` | `sep` | sep \| sextractor |
| `two_pass_boxcar_size` | `51` |  |
| `two_pass_detect_minarea` | `50` |  |
| `two_pass_detect_thresh` | `10` |  |
| `two_pass_object_masking` | `yes` |  |
| `two_pass_reject_level` | `0.95` |  |
| `two_pass_sep_ellipse_growth` | `2.5` | factor for growing ellipses found by sep |
| `use_sky_files` | `all` | all \| range \| selected |
| `write_calib_output` | `no` |  |
| `write_output` | `no` |  |

## skySubtractSpec

| Option | Default | Notes |
|---|---|---|
| `boxcar_nhigh` | `0` | Used in median_boxcar method - to use quartile rather than median set nhigh = width/2 |
| `boxcar_width` | `51` | Used in median_boxcar method.  Maximum width = 51 |
| `create_calib_only` | `no` |  |
| `default_master_sky` | `None` |  |
| `double_subtract_odd_frames` | `no` |  |
| `ignore_odd_frames` | `yes` |  |
| `offsource_multi_dither_ncombine` | `0` | In the case of offsource_multi_dither, set to n short frames at each dither position.  E.g., for AAAAABBBBB pattern set to 5. Default = 0 => auto-detect |
| `onsource_sorting_key` | `full` | full \| index \| a FITS keyword to sort onsource skies, e.g. MJD |
| `remove_residuals` | `no` | Attempt to remove residuals from sky subtraction by chosen method. |
| `residual_removal_method` | `median_boxcar` | median_boxcar \| response_curve |
| `sky_dithering_range` | `2` |  |
| `sky_method` | `dither` | dither \| ifu_onsource_dither \| median \| median_boxcar \| offsource_dither \| offsource_multi_dither \| step |
| `sky_offsource_method` | `auto` |  |
| `sky_offsource_range` | `240` |  |
| `write_calib_output` | `no` |  |
| `write_noisemaps` | `no` |  |
| `write_output` | `no` |  |

## slitletAlign

| Option | Default | Notes |
|---|---|---|
| `create_calib_only` | `no` |  |
| `debug_mode` | `no` | Show plots of each slitlet and print out debugging information. |
| `fit_order` | `3` | Order of polynomial to use to fit wavelength solution. Recommended value = 3. |
| `max_lines` | `None` | Maximum number of lines to fit for each segment. Can be a single value or comma separated list for each segment. Default value of None will impose no maximum. |
| `min_threshold` | `3,2` | Minimum local sigma threshold compared to noise for line to be detected. Can be a single value or comma separated list for each segment. |
| `n_segments` | `2` | Number of segments, default = 2. In JH data, the sky lines in H are much brighter so its best to separately match J lines with lower threshold. |
| `nebular_emission_check` | `no` | In rare cases, your data may have nebular emission lines that appear in some slits but not in others and may interfere with aligning the slits properly. Turn this option to yes for additional nebular emission check. |
| `reference_slit` | `None` | Reference slitlet to be used. Ideally this slitlet should be the centermost slitlet. Default value of None will select slitlet with central x_slit. Can also be a number or "prompt". |
| `reverse_order_of_segments` | `no` | Process segments right to left instead of left to right. Useful when nebular emission contaminates bright lines in leftmost segment. |
| `use_arclamps` | `no` | no = use master "clean sky", yes = use master arclamp |
| `write_calib_output` | `no` |  |
| `write_noisemaps` | `no` |  |
| `write_output` | `no` |  |

## trimOverscan

| Option | Default | Notes |
|---|---|---|
| `create_calib_only` | `no` |  |
| `fits_keyword_flag` | `OTRIMMED` | Set this to a FITS keyword which will be added when images are trimmed |
| `flag_true_value` | `1` | Value indicating image is trimmed.  If not equal to this, image WILL be trimmed. |
| `overscan_cols` | `None` | Columns to be trimmed. Supports slicing, e.g. 320:384, 500, 752:768 |
| `overscan_rows` | `None` | Rows to be trimmed. Supports slicing, e.g. 320:384, 500, 752:768 |
| `python_slicing` | `no` | Set to yes if min:max should be used. Default is min:max+1. |
| `write_calib_output` | `no` |  |
| `write_output` | `no` |  |

## trimWindow

| Option | Default | Notes |
|---|---|---|
| `create_calib_only` | `no` |  |
| `fits_keyword_flag` | `None` | Set this to a FITS keyword flagging whether images are trimmed |
| `flag_true_value` | `1` | Value indicating image is trimmed.  If not equal to this, image WILL be trimmed. |
| `python_slicing` | `no` | Set to yes if min:max should be used. Default is min:max+1. |
| `window_xmax` | `None` | Number or FITS keyword for max x value |
| `window_xmin` | `None` | Number or FITS keyword for min x value |
| `window_ymax` | `None` | Number or FITS keyword for max y value |
| `window_ymin` | `None` | Number or FITS keyword for min y value |
| `write_calib_output` | `no` |  |
| `write_output` | `no` |  |

## wavelengthCalibrate

| Option | Default | Notes |
|---|---|---|
| `bright_line_searchbox_max` | `-100` | Max pixel value of search box, negative notation allowed. |
| `bright_line_searchbox_min` | `100` | Min pixel value of search box. |
| `calibrate_slitlets_individually` | `no` | Set to yes to handle each slitlet individually if image has not been "jailbarred" |
| `create_calib_only` | `no` |  |
| `debug_mode` | `no` | Show plots of each slitlet and print out debugging information. |
| `fit_order` | `3` | Order of polynomial to use to fit wavelength solution. Recommended value = 3. |
| `line_list` | `None` | ASCII file containing line wavelengths and relative intensities. Optional 3rd column contains flag of -1 for blended lines that should not be used in final fit. |
| `max_bright_line_separation` | `None` | Maximum separation in pixels for any two bright lines to be used in matching up 3 brightest. Used when data is nonlinear. |
| `max_shift_tolerance` | `None` | Maximum tolerance in pixels for the shift between the initial guess as to a lines position and the peak of the cross-correlation between data and dummy spectrum. Cross-correlation window is 50 pixels wide for reference. Suggested value 15 to 20 if extraneous lines in line list |
| `max_wavelength` | `18500` | Maximum wavelength of data coverage, for use in constructing "dummy" spectrum. |
| `min_bright_line_separation` | `15` | Minimum separation in pixels for two bright lines to be used in matching up 3 brightest. |
| `min_intensity_percent` | `0.5` | Minimum intensity as a percent of the brightest line for a line to be detected |
| `min_lines_to_refine_nonlinear_guess` | `12` | Minimum number of lines that must be matched before refining a nonlinear initial guess. |
| `min_threshold` | `3` | Minimum local sigma threshold compared to noise for line to be detected |
| `min_wavelength` | `10000` | Minimum wavelength of data coverage, for use in constructing "dummy" spectrum. |
| `n_brightest_data` | `5` | If the 3 brightest lines in the image cannot be matched, try permutations of up to the n brightest. Should rarely need to be changed. Useful if line is missing from line list or scaling is vastly off for a line. |
| `n_brightest_lines` | `14` | Use the n brightest lines in the line list to compare to the 3 brightest in the image. Should rarely need to be changed. |
| `n_segments` | `1` | Number of piecewise functions to fit.  Should be 2 for MIRADAS, 1 for most other cases. |
| `resample_to_common_scale` | `yes` | Resample all slitlets to common scale for MOS data. If set to no, each slitlet will be resampled to an independent linear scale. |
| `slitlets_to_debug` | `None` | Set to a list - 3,5,7 - to debug only certain slitlets |
| `slitlets_to_write_plots` | `None` | Set to a list - 3,5,7 - to plot only certain slitlets |
| `use_arclamps` | `no` | no = use master "clean sky", yes = use master arclamp |
| `use_initial_guess_on_fail` | `no` | Use the initial guess to the wavelength solution if a solution cannot be found. |
| `wavecal_blind_scale_range` | `0.5,2` | Range of scales searched by the pattern and blind fallbacks, as factors of wavelength_scale_guess |
| `wavecal_fallback` | `learned,neighbor,trend,pattern,blind` | If the 3 brightest lines cannot be matched with wavelength_scale_guess and min/max_wavelength (or the solution is graded wavecal_retry_grade or worse), try these guesses, in order (comma-separated, or none).  Each uses line intensities measured in the calibrated slitlets when there are any.  learned = the configured guess with those measured intensities; neighbor = the solution of a calibrated slitlet covering the same range, shifted by cross-correlating the 1-d cuts; trend = predicted from the calibrated slitlets on either side (orders, e.g. MIRADAS); pattern = matching the spacing ratios of neighboring bright lines (no intensities); blind = cross-correlation over scales and zero points.  A solution from a fallback is kept only if graded satisfactory or better with enough lines. |
| `wavecal_quality_thresholds` | `0.1,0.2,0.3,0.4` | RMS of each wavelength fit in PIXELS separating excellent, good, satisfactory, marginal and poor (printed per slitlet, in the qa_*.dat file and the WCQUAL header keyword) |
| `wavecal_retry_grade` | `poor` | Once all slitlets have been tried, try the wavecal_fallback guesses again for slitlets that failed or were graded this or worse (excellent, good, satisfactory, marginal, poor, or none = only failures); a new solution replaces the old one only if it is clearly better. |
| `wavelength_calibration_file` | `None` | An XML file with wavelength calibration info for the image as a whole or for each order individually. Any options can be passed as attributes of <dataset> tag And <order> subtag, which also has optional attribute "slitlet", which refers to index in slitmask |
| `wavelength_fit_function` | `polynomial` | Function fit to the lines: polynomial, legendre or chebyshev (of order fit_order).  The same functions of pixel, so PORDER/PCOEFF still hold the equivalent polynomial; legendre/chebyshev also write WCFUNC and their own coefficients NCOEFF_i (MOS: WCFUNxx, NCFi_Sxx), pixels 0..WCXMAX mapped to [-1,1]. |
| `wavelength_line_1` | `None` | The wavelength (in output units) of a particular line, for use in constructing "dummy" spectrum. |
| `wavelength_line_2` | `None` | The wavelength (in output units) of a particular line, for use in constructing "dummy" spectrum. |
| `wavelength_line_separation` | `None` | The separation in pixels between line_1 and line_2, for use in constructing "dummy" spectrum. |
| `wavelength_scale_guess` | `None` | Initial guess of linear wavelength scale, for use in constructing "dummy" spectrum. Can also be space delmited list of polynomail coefficients, starting with linear term. |
| `write_calib_output` | `no` |  |
| `write_noisemaps` | `no` |  |
| `write_output` | `no` |  |
| `write_plots` | `no` |  |
