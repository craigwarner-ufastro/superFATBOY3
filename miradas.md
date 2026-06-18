# MIRADAS Data Reduction Pipeline

The MIRADAS DRP is a specialized pipeline built on the SuperFATBOY3 framework.

## Key Features
- Written in Python with GPU-optimized algorithms via CUDA/CuPy.
- Highly customizable via XML configuration files.
- Supports SOL, SOS, and MOS observing modes.

## Quick-Start Workflow
1.  **Templates:** Download the MIRADAS template XML files.
2.  **Configuration:** Copy the template matching your mode (SOL, SOS, or MOS) to `my_data.xml`.
3.  **Setup:**
    - Edit the `<queries>` section to point to your data directory and set correct suffixes/indices.
    - Set `outputdir` in the `<parameters>` section.
    - Set `gpumode` to `yes` if an nVidia GPU is available.
4.  **Run:**
    ```bash
    python superFatboy.py my_data.xml
    ```

## Important Notes
- Always set the `datatype="miradasSpectrum"` attribute in the `<dataset>` tag.
- MIRADAS data requires the `dispersion` property to be set to `vertical`.
- Data includes multiple ramps, which are handled automatically by the pipeline.

Refer to the "MIRADAS Quick-start Tutorial" in the original documentation for detailed XML parameter settings.
