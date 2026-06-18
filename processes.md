# SuperFATBOY3 Processes

This document details the processes in the SuperFATBOY3 pipeline.

## General Processes

### linearity
Performs a linearity correction using a polynomial transformation.

### noisemap
Creates a noise map of the linearized data: `nm_i = sqrt(counts_i / gain)`. Propagates through future processes.

### darkSubtract
Median combines individual dark frames and subtracts them from objects, flats, arclamps, and skies.

### createMasterArclamps
Median combines individual arclamp frames into a master arclamp frame for rectification and wavelength calibration.

### findSlitlets
Traces out edges of individual slitlets to create a master slitmask image for MOS/IFU data.

### cosmicRaysSpec
Removes cosmic rays using either L.A. Cosmic (Laplacian edge detection) or DCR (histogram-based) algorithms.

### flatDivideSpec
Normalizes flat fields (optionally within each slitlet) and divides object frames by the master flat.

### badPixelMaskSpec
Creates and/or applies a bad pixel mask based on master flat field thresholds or source files.

### skySubtractSpec
Performs sky subtraction. Supports AB dither patterns (onsource or offsource).

### rectify
Straightens continua and emission lines by fitting 2D polynomials to trace the geometric distortion.

### extractSpectra
Extracts 1D spectra from wavelength-calibrated 3D data, producing 2D Row Stacked Spectra (RSS) files.

## MIRADAS-Specific Processes

### miradasCollapseSpaxels
Collapses each slice of a slitlet into a monochromatic image (3xN).

### miradasCreate3dDatacubes
Creates a 3D datacube where each cut is a 3xNxN image at a given wavelength.

### miradasCombineSlices
Combines the 3 slices in each slitlet into a single 1D spectrum, accounting for relative slice illumination.

### miradasStitchOrders
(SOL/SOS only) Stitches individual orders together into one spectrum, resampling to a common wavelength scale.

### miradasDARFromConditionsProcess
Estimates differential atmospheric refraction based on site conditions.

### miradasDARFromDataProcess
Measures DAR from centroid position of point-like sources.

### miradasCharacterizePSF
Characterizes PSF at cuts through the 3D datacube.

### miradasRegisterWCSProcess
Registers WCS based on probe arm pointing positions.
