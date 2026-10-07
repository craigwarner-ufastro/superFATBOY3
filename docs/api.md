<p align="center"><img src="images/superFATBOY.png" alt="superFATBOY" width="110"></p>

# API guide

*[Docs home](README.md)*

This guide is for people who want to **script superFATBOY from Python**, **write their own process or datatype**, or **understand how the framework is put
together** before changing it. Every code example on this page that is marked *(tested)* was run against the current code.

- [Architecture in one page](#architecture-in-one-page)
- [Running a pipeline from Python](#running-a-pipeline-from-python)
- [`fatboyDatabase`](#fatboydatabase)
- [`fatboyDataUnit` and its subclasses](#fatboydataunit-and-its-subclasses)
- [`fatboyProcess`](#fatboyprocess)
- [Writing a process](#writing-a-process)
- [Adding a datatype](#adding-a-datatype)
- [Supporting CPU and GPU](#supporting-cpu-and-gpu)
- [Library modules](#library-modules)
- [Logging and error handling](#logging-and-error-handling)
- [Conventions and pitfalls](#conventions-and-pitfalls)

## Architecture in one page

```
superFatboy3.py ──▶ fatboyDatabase(xml)
                        │  parseXML()          reads <parameters>, <queries>, <processes>
                        │  executeQueries()    fatboyQuery turns <dataset> tags into fatboyDataUnit objects
                        │  preprocessAll()     read headers, set types/keywords, apply rejection rules
                        │  executeProcesses()  for each science frame, for each process in order:
                        │                          process.execute(fdu, prevProc)
                        └─ cleanUp()

   fatboyDataUnit (FDU)  one frame: header, data, properties, history, tags
        ├── fatboyImage                       ordinary imaging frame
        │     └── fatboyCalib                 a master calibration made (or loaded) by a process
        ├── fatboySpectrum                    adds specmode, dispersion, slitmask support
        │     └── fatboySpecCalib             spectroscopic master calibration (slitmask, clean sky, ...)
        ├── circeImage, circeFastImage        CIRCE multi-ramp imaging
        ├── miradasSpectrum, megaraSpectrum, osirisSpectrum   instrument spectrum types

   fatboyProcess        one reduction step; ~50 subclasses in superFATBOY/fatboyProcesses/
```

The three pieces you interact with:

- **`fatboyDatabase`** owns everything: parameters, every frame (FDU), every master calibration, the list of processes, the log.
  Inside a process or datatype it is available as `self._fdb`.
- **`fatboyDataUnit` (FDU)** is one frame. It lazily reads its data from disk, knows its header, carries named *properties*
  (values from `<property>` tags or set by processes), a *history* of processes applied, and optionally several *tagged* copies of its data (for example `"cleanFrame"`).
- **`fatboyProcess`** is one step. The framework calls `execute(fdu, prevProc)` once per science frame; the process asks the database for the calibrations it needs
  (`getCalibs`), builds them if they do not exist yet, and updates the FDU in place.

How calibrations work is what makes the design unusual. A process never loops over calibration frames itself. It asks the database for darks that match *this* science frame, and if only raw darks exist it builds a master dark, stores it
in the database (`appendCalib`) so the next frame finds it, and uses it. Before combining raw calibration frames it calls `recursivelyExecute` to run all earlier processes on them, which
is why you only list each process once.

## Running a pipeline from Python

This example *(tested)* runs a whole configuration and then inspects the frames.

```python
from superFATBOY.fatboyDatabase import fatboyDatabase

fd = fatboyDatabase("my_data.xml")   # parses the XML and sets everything up
fd.execute()                         # queries, preprocessing, every process, cleanup

for fdu in fd.getObjects():          # science frames still in use afterwards
    print(fdu.getFullId(), fdu.getShape(), fdu.getProperty("instrument"))
```

`execute()` is just three calls you can make yourself if you want to stop part-way:

```python
fd = fatboyDatabase("my_data.xml")
fd.executeQueries()      # find files and create FDUs
fd.preprocessAll()       # read headers, set types, apply min/max frame rejection
fd.executeProcesses()    # run the process chain
fd.cleanUp()             # delete the temp directory
```

`fatboyDatabase(modeTag=None)` with no config file is the `-list` mode: it prints every parameter, process and option and does nothing else. Pass a mode tag
(`"imaging"`, `"spectroscopy"`, `"miradas"`, `"megara"`, `"sinfoni"`, `"emir"`, `"circe"`) to print just that family.

Use the library functions directly, without a database, if you only need an algorithm: see [Library modules](#library-modules).

## Wavelength-calibrating one cut: `wavecal`

`superFATBOY.wavecal` runs `wavelengthCalibrate` on a single slitlet or order outside the pipeline, for example to try
a line list or a starting guess on one slitlet quickly. It uses the pipeline's own matching, fitting, fallbacks and
quality grades, so a cut gets the same solution it would get in a pipeline run (checked on all 24 LUCI arc slitlets).
Output goes to `./wavelengthCalibrated/`. *(tested)*

```python
from superFATBOY import wavecal

options = {"line_list": "NeArXe.dat", "min_wavelength": "13000", "max_wavelength": "26000",
           "wavelength_scale_guess": "-4.5"}
solved = []      #calibrated cuts so far: lets the neighbor and trend fallbacks work
measured = {}    #line intensities measured so far: the learned fallback
for slitlet in range(1, 25):
    fdu = wavecal.extract2DFromImageWithSlitmask("rct_lamp.fits", "rct_slitmask.fits", slitlet=slitlet)
    wavecal.executeWavelengthCalibration(fdu, dict(options), {"solvedCuts": solved, "lineMeasures": measured})
    if (fdu.hasProperty("solvedCuts")):
        solved = fdu.getProperty("solvedCuts")
        measured = fdu.getProperty("lineMeasures")
    print(slitlet, fdu.getProperty("wcHeader"))
```

`extract1DFromImage(image, ylo, yhi)` takes a cut between two rows instead, and `read1DFromImage(file)` reads a 1-d
spectrum. Any `wavelengthCalibrate` option can go in the options dict; for example `"wavecal_initial_method": "vote"`
when the scale guess is rough (see the [line-identification vote](processes/spectroscopy.md#line-identification-vote)). With a single cut there is no second pass;
instead a solution graded `wavecal_retry_grade` or worse is retried with the fallbacks at once and replaced only if
clearly better. After a successful fit the FDU has the properties `wcHeader`, `wcQuality` (RMS, RMS px, grade,
number of lines), `solvedCuts` and `lineMeasures`.

## `fatboyDatabase`

`superFATBOY/fatboyDatabase.py`. The main framework class. Constructor: `fatboyDatabase(config=None, modeTag=None)`.

**Running**

| Method | Purpose |
|---|---|
| `execute()` | Run the whole pipeline |
| `executeQueries()` | Turn the XML `<queries>` into FDUs in the database |
| `preprocessAll()` | Read headers, set obstype and keywords, apply `min_frame_value` and related rejection |
| `executeProcesses()` | Run the processes in order on every science frame (or every frame with `calibs_only = yes`) |
| `initializeAll()` | Initialize every FDU; isolates a bad file, and aborts if more than `max_init_failures` fail |
| `cleanUp()` | Remove the temporary directory |

**Getting frames and calibrations**: the heart of the API for process authors. All take keyword criteria, and match only frames still *in use*.

| Method | Returns |
|---|---|
| `getObjects()` | List of science FDUs |
| `getFDUs(ident=None, obstype=None, filter=None, section=None, exptime=None, tag=None, shape=None, properties=None, headerVals=None, inUse=True)` | List of FDUs matching all the criteria you give |
| `getSortedFDUs(..., sortby='full')` | As `getFDUs`, sorted by identifier, index or a header keyword (used to pair frames for sky subtraction) |
| `getIndividualFDU(identFull)` | One FDU by its full identifier |
| `getCalibs(ident, obstype, filter, section, exptime, nreads, tag, shape, properties, headerVals, inUse)` | Raw calibration frames (darks, flats, skies, arclamps, ...) |
| `getTaggedCalibs(ident, ...)` | Raw calibrations tagged to *this object* with an `<object>` child |
| `getMasterCalib(pname, ident, obstype, ...)` | The one master calibration (a `fatboyCalib`) matching the criteria, or `None` |
| `getMasterCalibs(...)` | A list of matching master calibrations |
| `getTaggedMasterCalib(pname, ident, obstype, ...)` | A master calibration tagged to this object |
| `hasMasterCalib(pname, ident, obstype, filename)` | `True` if such a master exists |
| `appendCalib(calib)` | Store a master calibration so later frames and processes find it |
| `addNewSlitmask(oldSlitmask, newData, pname)` | Register a slitmask with a new shape (after rectification or resampling) |

`obstype` is an FDU type constant (`fatboyDataUnit.FDU_TYPE_DARK`, `FDU_TYPE_FLAT`, `FDU_TYPE_SKY`, `FDU_TYPE_ARCLAMP`, `FDU_TYPE_BIAS`, ...) or a string such as `"master_dark"`.
`properties` and `headerVals` are dicts that must match the frame's `<property>` values or header values.

**Parameters, log and mode**

| Method | Purpose |
|---|---|
| `getParam(pname, ptag=None)` | A global parameter, with the tag lookup of the [XML guide](xml-guide.md#tags-different-settings-for-different-data) |
| `getGPUMode()` | `True` if GPU mode is on |
| `getLog()`, `getShortLog()` | The `fatboyLog` objects (full and short) |
| `getProcessByName(pname)` | The process class registered under this name, or `None` |
| `getDatatypeByName(dname)` | An instance of a registered datatype |
| `printParams()`, `printProcesses(fdu, modeTag)` | What `-list` calls |

## `fatboyDataUnit` and its subclasses

`superFATBOY/fatboyDataUnit.py` is the base class of everything that represents a frame. Constructor: `fatboyDataUnit(filename, log=None)`; `fatboyImage(filename, log=None, tag=None)` is the usual concrete class, and
`fatboyCalib(pname, obstype, source, filename=None, data=None, tagname=None, headerExt=None, log=None)` makes a master calibration from data, or loads one from a file, using `source` for its header.

**Identity and state**

| Member | Meaning |
|---|---|
| `getFullId()` | The full identifier, for example `ori1.0003.fits`. Used for output filenames. |
| `_id` | The object name without the index; `_identFull` is the full form |
| `getFilename()`, `getShortName()` | Current file path; file name without the path |
| `getObsType(value=False)` | Type name (`"object"`, `"dark"`, ...), or with `True` the integer constant |
| `getTag(mode='all')`, `setTag(tag, subtag=False)` | The dataset/object tags used for option lookup |
| `inUse` | `False` once the FDU is disabled |
| `disable()`, `enable()` | Take the frame out of (or put it back into) the reduction |
| `exptime`, `nreads`, `section`, `filter` | Attributes read from the header at `initialize()` |

**Data**

| Method | Purpose |
|---|---|
| `getData(tag=None, force_cpu=False)` | The data array, read from disk if necessary. In GPU mode it is a CuPy array; `force_cpu=True` always returns NumPy. `tag` selects a tagged copy. |
| `updateData(data)` | Replace the data (this is how a process returns its result) |
| `tagDataAs(tagname, data=None)` | Save a named copy of the data, for example `"cleanFrame"` (the pre-flat-field frame) |
| `getMedian(tag=None)`, `getMaskedMedian(tag)`, `getMaskedData(tag)` | Median, and versions that honor the bad pixel mask |
| `getShape()`, `setShape()` | Array shape |
| `getBadPixelMask()`, `applyBadPixelMask(bpm)`, `setMask(mask)` | Bad pixel mask access |
| `renormalize(bpm=None)` | Re-normalize (used after applying a mask to flats) |
| `forgetData()` | Release the data from memory but keep the header |
| `min()`, `max()` | Convenience |

**Header, properties, history**

| Method | Purpose |
|---|---|
| `_header` | The `astropy.io.fits.Header`. Add history with `fdu._header.add_history("...")`. |
| `getHeaderValue(key)`, `hasHeaderValue(key)` | `key` is a *parameter name* such as `'exptime_keyword'`; the real keyword comes from the params and is tried from the list |
| `setKeyword(index, keyword, value)`, `updateHeader(dict)`, `removeHeaderKeyword(key)` | Edit the header |
| `setProperty(key, value)`, `getProperty(key)`, `hasProperty(key)`, `removeProperty(key)` | Named values, set from `<property>` tags or by processes. A missing property reads as `None`. |
| `addProcessToHistory(pname)`, `hasProcessInHistory(pname)` | Which processes have run on this frame (this is how "already done" is decided) |
| `setHistory(key, value)`, `getHistory(key)`, `hasHistory(key)` | Key/value history entries (for example `cosmic_rays_removed`) |

**I/O**

| Method | Purpose |
|---|---|
| `writeTo(outfile, tag=None, headerExt=None)` | Write the data (or a tagged copy) to a FITS file |
| `updateFrom(updateFile, tag=None, headerTag=None, pname=None)` | Load data and history from a previously written file (how processes reuse their output) |
| `toHDUList(tag, headerExt, prefix)` | As an `astropy` `HDUList` |
| `writeToAndForget(outfile)` | Spill to disk to free memory |

### `fatboySpectrum` (spectroscopy base type)

`superFATBOY/datatypeExtensions/fatboySpectrum.py`. Adds to the above:

| Member | Purpose |
|---|---|
| `_specmode`, `dispersion` | `FDU_TYPE_LONGSLIT` / `FDU_TYPE_IFU` / `FDU_TYPE_MOS`; `DISPERSION_HORIZONTAL` / `DISPERSION_VERTICAL` (from the `specmode` and `dispersion` properties) |
| `getSlitmask(pname=None, shape=None, properties=None, headerVals=None, tagname=None, ignoreShape=False)` | The slitmask `fatboySpecCalib` for this frame |
| `setSlitmask(smdata, pname=None, properties=None, tagname=None)` | Attach a slitmask built from an array |
| `renormalize(slitmask=None, bpm=None)` | Slitlet-aware renormalization |

`fatboySpecCalib` is a spectroscopic master calibration (slitmask, clean sky, master arclamp, ...).

## `fatboyProcess`

`superFATBOY/fatboyProcess.py`. The base class of every process.

| Method | Override? | Purpose |
|---|---|---|
| `execute(self, fdu, prevProc=None)` | **Yes** | Do the work on one frame. Return `True` on success, `False` on failure. |
| `setDefaultOptions(self)` | **Yes** | Register the options and their defaults (and their `-list` help text) |
| `getCalibs(self, fdu, prevProc=None)` | Usually | Find or build the calibrations; return a `dict` of name to calibration |
| `writeOutput(self, fdu)` | Usually | Write output when `write_output = yes` |
| `checkValidDatatype(self, fdu)` | Sometimes | Return `False` to keep this process from being applied to some frames or calibrations (default `True`). For instance flats should never be sky subtracted. |
| `getOption(oname, otag=None)` | | The option's value for a frame's tag (see the [lookup rules](xml-guide.md#tags-different-settings-for-different-data)). Values are strings unless the default was a number or `None`. |
| `hasOption(oname, otag=None)`, `setOption(name, value)` | | Test for, or set, an option |
| `getCalib(cname, ctag=None)` | | A `<calib>` passed to this process in the XML (a filename) |
| `checkOutputExists(fdu, outfile, tag=None, headerTag=None)` | | If `outputdir/outfile` exists and `overwrite_files` is `no`, load it into the FDU and return `True`. Call it at the top of `execute` to make a step re-runnable. |
| `recursivelyExecute(calibs, prevProc)` | | Run all earlier processes on a list of raw calibration frames before you combine them |
| `checkTag(fdu)`, `getTag()` | | Does the `<process tag=...>` apply to this frame? |

Class attributes and members:

| Name | Meaning |
|---|---|
| `_modeTags` | List of mode tags (`["imaging"]`, `["spectroscopy", "miradas"]`, ...) used by `-list <tag>` |
| `_options`, `_optioninfo` | Option values and their help text. Populate both in `setDefaultOptions`. |
| `_pname` | The process name as it appears in the XML |
| `_fdb` | The `fatboyDatabase` |
| `_log`, `_shortlog` | Loggers |
| `_outputdir` | Output directory |

Every process automatically gets `write_output`, `write_calib_output` and `create_calib_only`. `pass_number` (an option you set yourself) lets the same process appear twice in a chain: the history records `name::pass_number`.

### What the framework does around your `execute`

For every science frame, for every process in order, `executeProcesses`:

1. skips the process if the frame has been disabled, is not a science frame, or the process's `tag` does not match;
2. skips it if the frame's history already contains it (`Already executed process ...`);
3. calls `setDefaultOptions()`, then `execute(fdu, prevProc)` inside a `try`;
4. if `execute` returns `False`, or raises, **disables that frame** and continues with the next frame (an exception is logged with its traceback, and waits for ENTER only if `interactive_on_error = yes`);
5. after a `True` return checks that the frame still has readable data, and disables it if not;
6. records the process in the frame's history, and if `write_output = yes` calls `writeOutput`.

So **return `False` (and `fdu.disable()`) when you cannot do your job**, rather than letting the pipeline continue with data you did not produce.

## Writing a process

A complete, working process *(tested)*. It divides every image by a number given in the XML.

`myFatboyProcesses/divideByNProcess.py`

```python
import os
import numpy as np
from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.fatboyLog import fatboyLog

# Divide every image by a constant N, given by the option "divisor".
class divideByNProcess(fatboyProcess):
    _modeTags = ["imaging"]

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("divideByN process")
        print(fdu.getFullId())
        # Skip the work if the output is already on disk (and overwrite_files is no)
        if (self.checkOutputExists(fdu, "divideByN/div_"+fdu.getFullId())):
            return True
        # Get the option, honoring any tag="..." override for this frame
        divisor = float(self.getOption('divisor', fdu.getTag()))
        if (divisor == 0):
            print("divideByNProcess::execute> ERROR: divisor is zero for "+fdu.getFullId()+".  Discarding image!")
            self._log.writeLog(__name__, "divisor is zero for "+fdu.getFullId()+".  Discarding image!", type=fatboyLog.ERROR)
            fdu.disable()
            return False
        fdu.updateData(fdu.getData()/divisor)
        fdu._header.add_history('Divided by '+str(divisor))
        return True

    ## OVERRIDE setDefaultOptions
    def setDefaultOptions(self):
        self._options.setdefault('divisor', 1)
        self._optioninfo.setdefault('divisor', 'The data will be divided by this number')

    ## OVERRIDE writeOutput
    def writeOutput(self, fdu):
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/divideByN", os.F_OK)):
            os.mkdir(outdir+"/divideByN", 0o755)
        outfile = outdir+"/divideByN/div_"+fdu.getFullId()
        if (os.access(outfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(outfile)
        if (not os.access(outfile, os.F_OK)):
            fdu.writeTo(outfile)
```

`myFatboyProcesses/__init__.py` can be empty, but it must exist. `myFatboyProcesses/processDict.py` maps names used in XML to classes:

```python
from myFatboyProcesses import divideByNProcess

def getProcessDict():
    processDict = dict()
    processDict['divideByN'] = divideByNProcess.divideByNProcess
    return processDict
```

Tell superFATBOY where the directory is, and use the process by its name:

```xml
<process name="divideByN">
  <option name="divisor" value="4"/>
  <option name="write_output" value="yes"/>
</process>
...
<parameters>
  <param name="processdir" value="/home/user/myFatboyProcesses"/>
</parameters>
```

`processdir` may appear more than once. The directory's *parent* is added to `sys.path` and `<dirname>.processDict` is imported, so the directory name must be a valid Python package name.

Things a process author needs to know:

- **Return value.** `True` = success; `False` = failure. Call `fdu.disable()` yourself if the frame should be dropped.
- **Options are strings.** `getOption` returns what was in the XML as a string, so convert: `float(...)`, `int(...)`, `.lower() == "yes"`. A default of `None` stays `None`, which is how you detect "not set".
  Always pass the frame's tag: `self.getOption('divisor', fdu.getTag())`.
- **Use `fdu.updateData(...)`** to store a result, not assignment. To add a *named* extra copy (for example a pre-flat-field frame) use `fdu.tagDataAs(name, data)`.
- **Propagate the noisemap.** If `fdu.hasProperty("noisemap")`, update it alongside the data (the spectroscopic processes do this through a `noisemap` tagged copy of the data); otherwise the final error bars will be wrong.
- **Keep the history honest.** `fdu._header.add_history(...)` records in the FITS header what you did.
- **Name your output directory and prefix** as the built-in processes do (`divideByN/div_`), and call `checkOutputExists` first so reruns are cheap.

### A process that needs calibrations

When your process needs a calibration that the database does not already hold, follow the shape of `darkSubtractProcess.getCalibs`, here simplified:

```python
from superFATBOY.fatboyCalib import fatboyCalib
from superFATBOY.fatboyDataUnit import fatboyDataUnit

def getCalibs(self, fdu, prevProc=None):
    calibs = dict()
    # 1. A calibration given in the XML with <calib name="masterDark" value="file.fits"/>
    filename = self.getCalib("masterDark", fdu.getTag())
    if (filename is not None and os.access(filename, os.F_OK)):
        calibs['masterDark'] = fatboyCalib(self._pname, "master_dark", fdu, filename=filename, log=self._log)
        return calibs
    # 2. An already-built master calibration matching this frame
    masterDark = self._fdb.getMasterCalib(self._pname, obstype="master_dark", exptime=fdu.exptime,
                                          nreads=fdu.nreads, section=fdu.section, tag=fdu.getTag())
    if (masterDark is not None):
        calibs['masterDark'] = masterDark
        return calibs
    # 3. Raw frames to build one from
    darks = self._fdb.getCalibs(obstype=fatboyDataUnit.FDU_TYPE_DARK, exptime=fdu.exptime,
                                nreads=fdu.nreads, section=fdu.section, tag=fdu.getTag())
    if (len(darks) > 0):
        self.recursivelyExecute(darks, prevProc)        # run earlier processes (linearity, ...) on the darks first
        masterDark = self.createMasterDark(fdu, darks)  # a helper defined in your class: combine and return a fatboyCalib
        self._fdb.appendCalib(masterDark)               # so the next frame finds it
        calibs['masterDark'] = masterDark
    return calibs    # an empty dict means "not found"; execute() decides what to do
```

`execute` then calls `calibs = self.getCalibs(fdu, prevProc)` and checks for the key it needs. The real `darkSubtractProcess` adds steps for
calibrations tagged to a single object, `default_master_dark`, and a nearest-exposure-time fallback; it is the best worked example in the code base. The `fatboyLibs` helpers
(`imcombine`, `arraymedian`, ...) do the combining.

### Processes that build a calibration without applying it

`create_calib_only = yes` makes the framework call your `getCalibs` and stop; a non-empty dict counts as success. Nothing extra is needed in your process.

## Adding a datatype

A datatype tells superFATBOY how to read an instrument's files. Most instruments need only `fatboyImage` (imaging) or `spectrum` (spectroscopy) and a few parameters or properties. Write a datatype only when the file
layout itself is unusual: multiple extensions or ramps per file (CIRCE, MIRADAS), instrument-specific headers, or extra state.

`myFatboyDatatypes/myImage.py`

```python
from superFATBOY.fatboyImage import fatboyImage

# An image type that tags every frame as coming from "myscope"
class myImage(fatboyImage):
    ## OVERRIDE initialize: let the base class read the header, then add to it
    def initialize(self):
        super().initialize()
        self.setProperty("instrument", "myscope")
```

`myFatboyDatatypes/datatypeDict.py`

```python
from myFatboyDatatypes import myImage

def getDatatypeDict():
    datatypeDict = dict()
    datatypeDict['myImage'] = myImage.myImage
    return datatypeDict
```

Plus an empty `__init__.py`. Then *(tested)*:

```xml
<dataset dir="/path/to/data" datatype="myImage"> ... </dataset>
...
<param name="datatypedir" value="/home/user/myFatboyDatatypes"/>
```

Extend `fatboyImage` for imaging, `fatboySpectrum` for spectroscopy (`superFATBOY/datatypeExtensions/osirisSpectrum.py` is a compact example), or `fatboyDataUnit` for something unusual. Methods you
typically override: `initialize()` (read instrument keywords, determine the shape), `getData()` (custom reading; accept `tag=None, force_cpu=False`), `hasMultipleExtensions()` and
`getMultipleExtensions()` (for multiple detectors or ramps, each becomes its own FDU with a different `section`), and `setPropertiesFromHeader()`.

> If you override `getData`, keep the `force_cpu` parameter in its signature. Several algorithms call `fdu.getData(force_cpu=True)` on every datatype.

## Supporting CPU and GPU

The code base runs in two modes from one source. The conventions:

- Import NumPy as `np` always. Import CuPy as `cp` inside a `try`, and only use it when `fdb.getGPUMode()` (or `self._fdb.getGPUMode()` in a process) is `True`.
- In GPU mode `fdu.getData()` returns a CuPy array. Code that is inherently CPU-bound (small medians, `np.correlate`, SciPy fits) should ask for `fdu.getData(force_cpu=True)`.
- Many algorithms exist as a CPU/GPU pair (`xregister` and `gpu_xregister`, `imcombine` and `gpu_imcombine`, `arraymedian` and `gpu_arraymedian`, `drihizzle` and `gpu_drihizzle`); the caller picks with the mode flag.
- CUDA kernels are C source strings compiled with `cp.RawModule` (see `fatboyLibs.get_fatboy_mod`). C-level `sqrt` and `exp` inside those strings are C, not NumPy.
- `arraymedian` and `gpu_arraymedian` are a hand-written quickselect, kept on purpose so CPU and GPU agree exactly; do not replace them with `np.median`.

## Library modules

These are importable on their own (`from superFATBOY.xregister import xregister`) and are the algorithms the processes are built from.

| Module | Key functions | What for |
|---|---|---|
| `fatboyLibs` | `createNoisemap`, `createSlitmask`, `findRegions`, `fitGaussian`, `fitGaussian2d`, `fwhm1d`, `fwhm2d`, `getCentroid`, `findAndFitLines`, `extractSpectra`, `medfilt2d`, `mediansmooth1d`, `blkavg`, `blkrep`, `convolve2d`, `fit1d`, `lacos_spec`, `dcr`, `linterp_cpu`, `linterp_gpu`, `getWavelengthSolution`, `hasWavelengthSolution`, `formatList`, `createFitsTable` | General-purpose array, fitting, cosmic-ray and spectroscopy helpers. Most processes `from superFATBOY.fatboyLibs import *`. |
| `arraymedian`, `gpu_arraymedian` | `arraymedian(input, axis='both', lthreshold, hthreshold, nlow, nhigh, nonzero)`, `gpu_arraymedian(...)` (also weighted and sigma-clipped variants) | Fast medians and means of arrays with rejection, along an axis or over everything |
| `imcombine`, `gpu_imcombine` | `imcombine(frames, outfile=None, method='median', reject='none', lsigma=3, hsigma=3, nlow=0, nhigh=0, scale, zero, weight, ...)` | Combine frames with median or mean and several rejection schemes (the IRAF `imcombine` semantics) |
| `xregister`, `gpu_xregister` | `xregister(frames, refframe=0, xcenter, ycenter, xboxsize, yboxsize, constrain_boxsize, method, ...)` | 2-d cross-correlation image alignment, with constrained and SEP-assisted variants |
| `tri_register` | `tri_register(frames, method, min_angle, max_angle, atol, rtol, ...)` | Triangle (asterism) matching alignment |
| `drihizzle`, `gpu_drihizzle` | `drihizzle(frames, outfile, kernel='point', dropsize=1, geomDist, xsh, ysh, xtrans, ytrans, inunits, outunits, ...)`, `drihizzle3d` | Drizzle-style resampling and stacking, with several kernels; also used to apply rectification maps |
| `pysurfit`, `gpu_pysurfit` | `pysurfit(input, order=1, niter=3, lower, upper, ...)` | Iterative 2-d polynomial surface fit (used for sky-surface removal) |
| `wavecal` | `executeWavelengthCalibration(fdu, options, calibs, gpumode)`, `extract1DFromImage`, `extract2DFromImageWithSlitmask` | The wavelength-calibration engine behind `wavelengthCalibrate` |
| `fatboyLog` | `fatboyLog.writeLog(name, message, type=...)` | Logging |
| `fatboyQuery` | | The XML `<dataset>` and `<filelist>` parser that finds files |

`imcombine`, `xregister` and `drihizzle` take a list of FDUs as `frames`. Read the function signature in the source for the full parameter list:
most of these functions have a dozen or more keyword arguments, with defaults that match the process defaults.

## Logging and error handling

Write to the log, not just to the terminal:

```python
from superFATBOY.fatboyLog import fatboyLog
self._log.writeLog(__name__, "Something happened to "+fdu.getFullId())                       # INFO
self._log.writeLog(__name__, "Careful...", type=fatboyLog.WARNING)
self._log.writeLog(__name__, "That failed.  Discarding image!", type=fatboyLog.ERROR)
```

Warnings and errors are shown at every verbosity (`brief`, `normal`, `verbose`); INFO lines at `normal` and above. Pass `verbosity=fatboyLog.VERBOSE` to show a line only at `verbose`. The full log is written
under `logdir` (default `flogs/`), along with a short log of which process ran on which frame. By convention, messages start with `<processName>::<method>>` and failures say what was discarded.

Guidelines for code that behaves well inside the framework:

- **Fail one frame, not the run.** On an unrecoverable problem for one frame: log an `ERROR`, `fdu.disable()`, `return False`.
- **Fall back for one piece, not all.** When one slitlet or one fit fails inside a larger process, degrade that piece (a straight edge, an identity transform) and keep the rest, as `findSlitlets` and `rectify` do. Log an `ERROR` or `WARNING` so it cannot scroll by unnoticed.
- **Do not prompt.** Do not call `input()` by default. Put an opt-in option such as `prompt_for_missing_dark` in front of any interaction, defaulting to `no`.
- **Do not open windows by default.** Keep `debug_mode = no` by default and offer `write_plots` for saving QA figures.
- **Be honest about success.** Return `True` only if you really left valid data behind; the framework checks, but the error message is more useful from you.

## Conventions and pitfalls

Collected from debugging the Python 3 migration; each of these produced a real bug that was silent or confusing.

- **`np.min(a, b)` is not `min(a, b)`.** The second positional argument of `np.min` and `np.max` is `axis`. To compare two scalars, use the builtin `min(a, b)` or `max(a, b)`.
- **`.astype()` is for arrays.** Calling it on a Python int or float (`ncoeffs.astype(np.int32)`) raises; use `np.int32(x)`.
- **A dtype string is just a string.** Write `.astype("int32")`, not `.astype("np.int32")`. A comparison like `array.dtype != "np.float32"` never raises and is always true.
- **CuPy kernels read scalars by the Python object's own type.** `RawKernel` packs each scalar argument according to the Python/NumPy type you pass, not the kernel's C signature. A `float64` passed where the kernel expects `float` silently shifts
  every later argument. Cast the *whole* expression, `np.float32(a - b)`, not one operand, `a - np.float32(b)` (which promotes back to `float64`).
- **CuPy has no `drv.Out()`.** PyCUDA copied a device result back into a host array for you. With CuPy, allocate a device buffer (`cp.empty_like(x)`), pass that to the kernel, and `.get()` the result back. `cp.empty(existing_array)` does *not* copy the array; it treats its values as a shape.
- **`astropy` will not take a CuPy array.** Convert with `cp.asnumpy(data)` before assigning to an HDU's `.data` or calling `fits.writeto`.
- **Array-ness matters.** Some helpers return one-element arrays on purpose (`whereEqual` returns a tuple of arrays, like `np.where`). Do not collapse to a scalar without checking the callers.
- **Negative slice indices wrap.** `a[i-10:i+11]` with `i < 10` silently takes the wrong slice. Clamp with `max(i-10, 0)`.
- **Mind the first and second pass.** Trace and fit functions that run on the CPU should get their data with `force_cpu=True` everywhere inside the function, not just at the top.
