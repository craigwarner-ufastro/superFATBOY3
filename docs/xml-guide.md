<p align="center"><img src="images/superFATBOY.png" alt="superFATBOY" width="110"></p>

# XML style guide

*[Docs home](README.md)*

Everything superFATBOY does is driven by one XML configuration file. This page describes every tag and attribute,
how options are resolved, and conventions that keep configuration files readable.

- [File structure](#file-structure)
- [Queries: describing your data](#queries-describing-your-data)
  - [`<dataset>`](#dataset)
  - [`<object>` and `<calib>`](#object-and-calib)
  - [Children of `<object>` and `<calib>`](#children-of-object-and-calib)
  - [Calibration types](#calibration-types)
  - [Master calibrations](#master-calibrations)
  - [Legacy `<filelist>`](#legacy-filelist)
- [Processes: what to do](#processes-what-to-do)
- [Parameters: global settings](#parameters-global-settings)
- [Tags: different settings for different data](#tags-different-settings-for-different-data)
- [A complete annotated example](#a-complete-annotated-example)
- [Style conventions](#style-conventions)
- [Common mistakes](#common-mistakes)

## File structure

A configuration file has one root element, `<fatboy>`, with three children:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<fatboy>
  <queries>
    <!-- describe your data -->
  </queries>

  <processes>
    <!-- reduction steps, in the order they run -->
  </processes>

  <parameters>
    <!-- global settings -->
  </parameters>
</fatboy>
```

Standard XML comments (`<!-- ... -->`) can go anywhere and are used heavily in the shipped
[templates](../superFATBOY/data/templates/). A malformed file stops the run immediately with an XML parse error, so
a stray `&` or an unclosed tag is reported before any work is done.

## Queries: describing your data

### `<dataset>`

The preferred way to describe data. One `<dataset>` describes one directory tree; you can have several in one file
(for example, the blue and red arms of a spectrograph, or several targets that share nothing).

| Attribute | Meaning |
|---|---|
| `dir` | Directory where the data lives. If the data is spread over sub-directories, use the lowest common directory. |
| `tag` | A label for this dataset, used to give different options to different datasets. See [Tags](#tags-different-settings-for-different-data). |
| `datatype` | The datatype class used to read these files. Omit for ordinary single-extension imaging. See the table below. |
| `delim` | Optional prefix delimiter, used to split a filename into a prefix and an index. Usually omit it. |

Datatypes available out of the box:

| `datatype` | Use for |
|---|---|
| *(omitted)* | ordinary images (`fatboyImage`), for example FLAMINGOS-1 or EMIR imaging |
| `spectrum` | generic spectroscopy, for example FLAMINGOS-1 MOS/longslit and KAST |
| `osirisSpectrum` | OSIRIS (GTC) spectroscopy |
| `miradasSpectrum` | MIRADAS spectroscopy (handles multiple ramps automatically) |
| `megaraSpectrum` | MEGARA fiber spectroscopy |
| `circeImage`, `circeFastImage` | CIRCE imaging (multiple ramps per file) |
| `specCalib` | a stored spectroscopic calibration (used internally) |

You can add your own; see the [API guide](api.md#adding-a-datatype).

### `<object>` and `<calib>`

Inside a `<dataset>`, `<object>` tags describe science observations and `<calib>` tags describe calibration frames.
Both take the same attributes:

| Attribute | Meaning |
|---|---|
| `name` | The name of this set of frames. It becomes part of output filenames and is how the set is referred to everywhere else. Make it meaningful: `crab_h`, `dark20s`. |
| `type` | `auto` (default) reads the type from the FITS header (`obstype_keyword`). For `<calib>` you usually set it explicitly; see [Calibration types](#calibration-types). |
| `prefix` | Match files named `<dir>/<subdir>/<prefix>*.fits` |
| `suffix` | Match files named `<dir>/<subdir>/*<suffix>*.fits` |
| `pattern` | Match files by a glob pattern you supply (may contain wildcards) |
| `subdir` | A sub-directory of `dir` to look in, for data organized into `object/`, `dark/`, `flat/`... folders |
| `filename` | **`<calib>` only.** A specific file, typically a pre-made master calibration. |
| `tag` | A sub-label for this set, to give it its own options. See [Tags](#tags-different-settings-for-different-data). |

Notes on matching:

- Give exactly one of `prefix`, `suffix` or `pattern` (or `filename` for a calib). If you give several, prefix wins, then suffix.
- Compressed files (`.fits.fz`, `.fits.gz`) are matched as well as plain `.fits`.
- **Indices.** With `<index>` or `<value>` (below), superFATBOY looks for frames whose *index* matches. The index is the number
  in the filename. Files look like `ori.0001.fits` or, for instruments that put the index first,
  `000101-20190101-MIRADAS-flat.fits`. Zero-padding is optional in the XML: `68`, `068` and `000068` all match.
- With no `<index>` or `<value>` children at all, *every* file matching the prefix, suffix or pattern is used.
- Use `prefix` when the filename *starts* with something distinctive (`flatkon.0001.fits`). Use `suffix` when the
  distinctive part is after the index (`2270993-20240509-EMIR-EmirDark.fits`), usually together with `subdir`.

### Children of `<object>` and `<calib>`

**`<index>`** selects a range of frames. It is the most common child.

```xml
<object type="auto" name="5k1" prefix="oribcf5k1">
  <index start="1" stop="9">
    <except>5</except>       <!-- skip frame 5 -->
  </index>
</object>
```

`<index>` takes `start` and `stop`. Inside it, `<value>` adds an index outside the range and `<except>` removes one.

**`<value>`** adds a single index (an alternative to `<index>` for a single frame):

```xml
<calib name="dark20s" type="dark" prefix="13Dec2014/CIRCE2014-12-14">
  <value>68</value>
</calib>
```

**`<timestamp>`** selects frames by the time in their header instead of by index (`start` and `stop`, plus an optional
`format`, default `%Y-%m-%dT%H:%M:%S.%f`). This is useful when the index resets, for example at midnight UTC.

**`<property>`** attaches a named value to the frames. Processes read properties to learn things the FITS header
does not say. Any name is allowed; a process ignores properties it does not recognise. The properties in regular use:

| Property | Values | Used for |
|---|---|---|
| `flat_type` | `lamp_on`, `lamp_off` | telling dome-flat-on frames from dome-flat-off frames (`flat_method = dome_on-off`) |
| `lamp_type` | `lamp_on`, `lamp_off` | the same for arclamps (`lamp_method = lamp_on-off`) |
| `specmode` | `longslit`, `ifu`, or anything else (meaning MOS) | what kind of spectroscopy this is. The default is MOS. |
| `dispersion` | `horizontal` (default), `vertical` | which way the spectra run on the detector |
| `sky_method` | as for `skySubtractSpec`'s `sky_method` | per-object override of the sky method (used in the FLAMINGOS-1 MOS template for a standard star) |

```xml
<calib name="flat_on" type="flat" prefix="flatkon">
  <index start="108" stop="110"/>
  <property name="flat_type" value="lamp_on"/>
</calib>
<calib name="flat_off" type="flat" prefix="flatkoff">
  <value>105</value>
  <property name="flat_type" value="lamp_off"/>
</calib>
```

**`<object>`** (inside a `<calib>` only) ties a set of calibration frames to particular science objects. Normally a
process looks for calibrations tagged to *this* object first, then falls back to untagged calibrations with a matching
exposure time, filter and so on. Use this to give one target its own darks or skies:

```xml
<!-- offsource skies only for crab_h1 and crab_h2 -->
<calib type="sky" name="crab_h_sky" prefix="14Dec2014/CIRCE2014-12-15">
  <object>
    <value>crab_h1</value>
    <value>crab_h2</value>
  </object>
  <index start="72" stop="80"/>
</calib>
```

```xml
<!-- a general set of 3 s darks, and a special set used only for gc_j and gc_j_rot -->
<calib name="dark3s" type="dark" prefix="2009sep12/S2009">
  <index start="14" stop="22"/>
</calib>
<calib name="dark3s_for_j_only" prefix="2009sep12/S2009">
  <object><value>gc_j</value><value>gc_j_rot</value></object>
  <index start="15" stop="19"/>
</calib>
```

### Calibration types

`type` on a `<calib>` overrides whatever the header says. The type names are matched by their content, so spelling variants
are tolerated (anything containing `dark` is a dark, anything containing `flat` a flat, and so on). The ones processes look for:

| `type` | Meaning | Used by |
|---|---|---|
| `dark` | dark frames (matched by exposure time, number of reads, section) | `darkSubtract` |
| `bias` | bias frames | `biasSubtract`, `emirBiasSubtract` |
| `flat` | flat fields (dome, sky or twilight) | `flatDivide`, `flatDivideSpec` |
| `sky` | off-source sky frames | `skySubtract`, `skySubtractSpec` |
| `arclamp` | arclamp frames (anything containing `lamp` or `arc`) | `createMasterArclamps` |
| `standard` | a spectroscopic standard star | `calibStarDivide` |
| `bad_pixel_mask` | an existing bad-pixel mask | `badPixelMask`, `badPixelMaskSpec` |
| `master_dark`, `master_flat`, ... | a ready-made master calibration (anything containing `master`) | see below |
| `continuum_source` | a bright continuum frame (for example a standard star) used to locate spectra when your science frames have no bright continuum | `extractSpectra` |

A `<calib>` can also use any type name a process you wrote looks for.

### Master calibrations

To supply a finished master calibration (a master dark, flat, slitmask...) instead of making one, use a `<calib>` with
a `filename`:

```xml
<calib type="master_dark" filename="masterDarks/mdark-10s-1rd-dark10s.fits"/>
```

Processes also accept a named calibration inside their own `<process>` block, which ties it to one object (or all of them):

```xml
<process name="darkSubtract">
  <calib name="masterDark" value="master_dark_5s.fits" tag="object1"/>
  <calib name="masterDark" value="master_dark_10s.fits"/>
</process>
```

and many take a `default_*` *option* (`default_master_dark`, `default_master_flat`, `default_bad_pixel_mask`, ...) that
may be a file, a comma-separated list of files, or an ASCII file with one filename per line. The difference: a `<calib>`
is tied to the objects you name, whereas a `default_*` option is searched for a frame whose exposure time, filter or grating matches.

### Legacy `<filelist>`

Kept for old FATBOY users. It reads an ASCII file instead of a `<dataset>`:

```xml
<queries>
  <filelist type="list" dir="/data/oriImages/" delim=".">files.dat</filelist>
</queries>
```

- `type="list"`: a one-column file of filenames.
- `type="grouping"`: the classic FATBOY file, four columns, `input_prefix  start_index  stop_index  output_prefix`:

```
2009sep12/S2009  014  022  dark3s
2009sep12/S2009  080  099  gc_j
```

Attributes: `type`, `tag`, `datatype`, `delim`, `dir`. New files should use `<dataset>`.

## Processes: what to do

`<processes>` holds one `<process>` tag per reduction step. **They run in the order you list them.**

```xml
<process name="flatDivide">
  <option name="flat_method" value="twilight"/>
  <option name="twilight_pair_ramps" value="yes" tag="circe"/>
  <option name="write_calib_output" value="yes"/>
  <option name="write_output" value="yes"/>
</process>
```

`<process>` attributes:

| Attribute | Meaning |
|---|---|
| `name` | The process to run (see the [process guide](processes/README.md) or `superFatboy3.py -list`) |
| `tag` | Optional. Run this process only on datasets or objects carrying this tag. To run a process on two tags, repeat the block. |

`<process>` children:

| Child | Meaning |
|---|---|
| `<option name="..." value="..." [tag="..."]/>` | Set an option |
| `<calib name="..." value="..." [tag="..."]/>` | Pass a calibration file to this process |

Every option not set takes its default. `superFatboy3.py -list` prints them all; the
[options reference](options-reference.md) has a snapshot.

### Options every process has

| Option | Default | Meaning |
|---|---|---|
| `write_output` | `no` | Write this process's output frames to disk (into the process's own sub-directory of `outputdir`). |
| `write_calib_output` | `no` | Write the calibration built or used by this process (a master flat, a bad pixel mask, a slitmask...). |
| `create_calib_only` | `no` | Build the calibration and stop, without applying it to the science frames. |
| `write_noisemaps` | `no` | For processes that carry a noisemap (those after `noisemap`): write the propagated noisemap too. |

If you only want the final product, leave `write_output` off everywhere except the last process. Intermediate frames that
are not written are kept in memory, up to `memory_image_limit`, which is faster.

## Parameters: global settings

`<parameters>` holds `<param name="..." value="..." [tag="..."]/>` tags.

```xml
<parameters>
  <param name="outputdir" value="/data/reductions/dataset_name"/>
  <param name="overwrite_files" value="no"/>
  <param name="memory_image_limit" value="100"/>
  <param name="gpumode" value="yes"/>
</parameters>
```

The important ones (the full list with defaults is in the [options reference](options-reference.md#global-parameters)):

| Parameter | Default | Meaning |
|---|---|---|
| `outputdir` | `.` | Where output goes. Sub-directories are created inside it; `outputdir` itself must already exist. |
| `gpumode` | `yes` | `yes` uses the GPU (CuPy); `no` runs on the CPU. |
| `overwrite_files` | `no` | `no` reuses output files already on disk; `yes` recomputes and overwrites. |
| `memory_image_limit` | none | Roughly 10 times your free RAM in GB. Higher means fewer intermediate frames written to disk. |
| `quick_start_file` | none | A file where superFATBOY caches per-file shapes and medians so later runs start faster. |
| `verbosity` | `normal` | `brief`, `normal` or `verbose` |
| `logdir` | `flogs` | Directory for log files |
| `tempdir` | `temp-fatboy` | Scratch directory, removed at the end of a run |
| `interactive_on_error` | `no` | `yes` pauses and waits for ENTER after an error (useful at a terminal, fatal when unattended) |
| `max_init_failures` | `3` | Abort if more than this many input files fail to read |
| `min_frame_value`, `max_frame_value` | none | Reject frames whose median is outside this range (bad or saturated frames) |
| `ignore_first_frames`, `ignore_after_bad_read` | `no` | Skip frames known to be unreliable |
| `processdir`, `datatypedir` | none | Directories of your own processes and datatypes (see the [API guide](api.md)). May be repeated. |

### FITS keyword parameters

Parameters ending in `_keyword` tell superFATBOY which header keyword holds a quantity. Defaults are *lists*: each
name is tried in turn and the first one present in the header is used.

| Parameter | Default list |
|---|---|
| `exptime_keyword` | `EXPTIME`, `EXP_TIME`, `EXPCOADD` |
| `filter_keyword` | `FILTER`, `FILTNAME` |
| `obstype_keyword` | `OBSTYPE`, `OBS_TYPE`, `IMAGETYP` |
| `gain_keyword` | `GAIN`, `GAIN_1`, `EGAIN` |
| `nreads_keyword` | `NREADS`, `LNRS`, `FSAMPLE`, `NUMFRAME` |
| `ra_keyword`, `dec_keyword` | `RAOFFSET`, `RA`, `TELRA` and `DECOFFSE`, `DEC`, `TELDEC` |
| `ut_keyword`, `date_keyword` | `UT`, `UTC`, `NOCUTC` and `DATE`, `DATE-OBS` |
| `pixscale_keyword`, `rotpa_keyword` | `PIXSCALE` and `ROT_PA`, `ROTPA`, `INSTPA` |

If your keyword is already in the list you need do nothing. If your exposure time is in `EXPOSET`, add
`<param name="exptime_keyword" value="EXPOSET"/>`. Spectroscopy templates add `grism_keyword`, `object_keyword`,
`readnoise_keyword` and so on; see the template for your instrument. `relative_offset_arcsec` tells superFATBOY the
RA/Dec keywords are offsets in arcseconds rather than absolute coordinates.

## Tags: different settings for different data

When one XML file holds several datasets (or one dataset with differently-treated objects), tags let a single `<option>`
or `<param>` apply to only some of them.

Put `tag="label"` on the `<dataset>`, `<object>` or `<calib>`, then use the same `tag` on an `<option>` or `<param>`:

```xml
<dataset dir="/data/ori/" tag="ori">...</dataset>
<dataset dir="/data/gc/"  tag="gc">...</dataset>

<param name="exptime_keyword" value="EXP_TIME" tag="ori"/>
<param name="exptime_keyword" value="EXPTIME"  tag="gc"/>

<process name="skySubtract">
  <option name="sky_subtract_method" value="remove_objects" tag="ori"/>
  <option name="sky_subtract_method" value="offsource"      tag="gc"/>
</process>
```

The lookup for any option or parameter, for the frame being processed, is:

1. a value defined in the XML with a `tag` matching the frame's tag;
2. else a value defined in the XML with **no** tag;
3. else the built-in default.

So an untagged value is a default for everything, and a tagged value overrides it for one dataset. The KAST template
uses this heavily: the red and blue arms share most options, and differ with `tag="red"` and `tag="blue"` overrides.

A `tag` on a `<process>` itself restricts the whole process to frames with that tag.

## A complete annotated example

A minimal near-IR imaging reduction with an on-source sky:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<fatboy>
<queries>
  <!-- Everything is in one directory, told apart by filename prefix. -->
  <dataset dir="/data/2025-03-10/" tag="ori">

    <!-- Science frames ori1.0001.fits ... ori1.0009.fits -->
    <object type="auto" name="ori1" prefix="ori1">
      <index start="1" stop="9"/>
    </object>

    <!-- All files starting "dark" are darks; superFATBOY matches them to
         each science frame by exposure time, so one <calib> is enough. -->
    <calib type="dark" prefix="dark"/>

    <!-- Lamp-on and lamp-off dome flats, told apart by a property. -->
    <calib type="flat" name="flat_on" prefix="flatkon">
      <property name="flat_type" value="lamp_on"/>
    </calib>
    <calib type="flat" name="flat_off" prefix="flatkoff">
      <property name="flat_type" value="lamp_off"/>
    </calib>
  </dataset>
</queries>

<processes>
  <process name="darkSubtract">
    <option name="prompt_for_missing_dark" value="no"/>
  </process>

  <process name="flatDivide">
    <option name="flat_method" value="dome_on-off"/>
  </process>

  <process name="badPixelMask"/>

  <process name="skySubtract">
    <option name="sky_subtract_method" value="remove_objects"/>
  </process>

  <process name="cosmicRays"/>

  <process name="alignStack">
    <option name="align_method" value="triangles"/>
    <option name="write_output" value="yes"/>   <!-- only the final image is kept -->
  </process>
</processes>

<parameters>
  <param name="outputdir" value="/data/reductions/ori1"/>
  <param name="gpumode" value="yes"/>
  <param name="memory_image_limit" value="100"/>
  <param name="exptime_keyword" value="EXP_TIME"/>
  <param name="obstype_keyword" value="OBS_TYPE"/>
</parameters>
</fatboy>
```

## Style conventions

These keep configuration files easy to read, diff and hand to someone else.

- **Start from a template** and keep its comments. Add your own next to anything you change.
- **One process per block, in execution order**, with options indented below it. Group `write_*` options together at the bottom of each block.
- **Comment out, don't delete**, a step you may want later. A commented-out block documents what you chose not to do.
- **Name things meaningfully**: `name="crab_h"`, not `name="obj1"`. Names end up in output filenames.
- **Keep the tag vocabulary small** and meaningful (`blue`, `red`, `ori`, `gc`). Drop the tag entirely if there is only one dataset.
- **Put data paths only in `<dataset dir=...>` and `outputdir`**, so moving the reduction to another machine is a two-line edit.
- **Use `<param>` for keyword names, `<option>` for algorithm behaviour.** If the answer is "what is the header keyword called", it is a param.
- **Leave `debug_mode` at `no`** in anything you commit or run unattended. Use `write_plots` for QA images.
- **Record the dataset and what you verified** in a leading comment, in the way the templates do.

## Common mistakes

| Symptom | Likely cause |
|---|---|
| "node ... has invalid attribute" warning | A typo in an attribute name (for example `prefex`). The attribute is ignored, so the wrong files (or none) are matched. |
| A setting has no effect | A misspelled `<param>` or `<option>` name is silently accepted and ignored (for example `overwite_files`); only unknown XML *attributes* and *tags* produce warnings. Compare names against `superFatboy3.py -list`. |
| "does not specify a prefix, suffix, or pattern attribute" | An `<object>` or `<calib>` with no way to find its files. |
| Nothing matched by an `<index>` | The index in the XML is compared with the number in the filename. Check zero-padding and the `prefix`/`suffix` choice against `ls`. |
| Output directory errors at start | `outputdir` does not exist. Create it first. |
| Process silently does nothing for one dataset | A `tag` on the `<process>` that does not match the dataset's tag. |
| A process cannot find a master calibration | Missing `<calib>` for that exposure time, filter or grating, or a `flat_type` or `lamp_type` property missing on a dome-on-off set. |
| Run hangs waiting for input | A `prompt_for_missing_*` option, or `interactive_on_error`, or `debug_mode`, is set to `yes` in an unattended run. |
| Spectra processed as though horizontal | Missing `<property name="dispersion" value="vertical"/>` on the object and all its calibs. |
| Misordered results | Processes run in the order listed. For example, `rectify` needs the slitmask from `findSlitlets` already built (or buildable on demand) and `wavelengthCalibrate` needs a rectified arclamp. |
