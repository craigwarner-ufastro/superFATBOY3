# Project: superFATBOY3 

## General Instructions
- This project is a general purpose astronomical data pipeline framework with "processes" or "reciepes" for data reduction of various IR and optical imaging and spectroscopic instruments.

- We are making some refactors and improvements listed below:
	1. The original code was written in the early python days - some dates back to when numpy was numarray!  I have some "early python errors" most proiminetly `from numpy import *`.  Can we redo this in the proper python syntax - I know this requires a little delicacy determining exactly which functions are numpy funcitons vs math functions - which is why I haven't spent the required time to go through it yet.
	2. When the original code was written there were many numpy bugs - such that .sum()/N was faster than .mean() or calling .sort() and then selecting element N/2 was faster than calling median.  These were due to early bugs in numpy and the speed difference was substantial so I have a lot of places in the code where it now doesn't make any sense to use these tricks.  The one caveat I will make is I do want to keep my gpu_arraymedian and arraymedian libraries - and my CPU and CUDA implementations of quickselect.  But please get rid of any m = x.sum()/N and replace with m = x.mean().
	3. The original was written using PyCUDA which is what was available at the time.  Cupy is now the standard so can you refactor the code to use cupy instead of PyCUDA?
	4. Error handling - if there's anywhere that errors can be more elegantly handled we should do this.  More on that below.

- Ensure all new functions and classes have appropriate comments

- We should work on a new branch "refactor" which will need to be created.

- We should do a git commit on the refactor branch after every major code change with a meaningful commit message summarizing the changes

- Coding style should avoid multiple commands on a line for better readability.  We should also avoid lines like x, y, z = True, False, True when defining variables for readability.

## Specific Algorithms
- Improvements can be made to the following specific algorithms both for error handling, checking for bad data, and algorithmic improvements:
	- findSlitletProcess.py - this is where a flat field is used to trace out the curvature of each slitlet for MOS/IFU data and create a "slitmask" to identify pixels belonging to each slitlet.  Improved trace algorithms, improved creation of slitmasks, and functions other than regular polynomials are all possible improvements.
	- removeCosmicRaysSpecProcess.py - the imaging cosmic ray removal is fairly robust but if there are additional algorithms to get better cosmic ray removal than we currently have, I am very open to them.
	- rectfyProcess.py - this is the process of tracing out the curvature of continua and skylines / lamplines and doing a fit to remove the curvature so that spectra can be extracted and wavelength calibraiton can be performed.  Similar to findSlitletProcess - improved tracing, and functions producing better fits are welcomed!
	- wavelengthCalibrateProcess.py - this process takes a list of wavelengths and relative intensities and tries to create a dummy spectrum and match up the lines with the arclamp or sky frame and identify lines and solve for a wavelength solution.  Better algorithms are welcomed!

## Documentation

- The original HTML which is no longer on a server anywhere can be found at /home/cwarner/FATBOY/superFATBOY/.  Also look at /home/cwarner/FATBOY/miradas/ for MIRADAS-specific documentation.  And you can look at the whole /home/cwarner/FATBOY to see older documentation.

- Running superFatboy3.py -list will also list all processes in alphabetical order and all options for each process.

- Let's collate all this into new git style .md files - a main one with install and how to run info including XML style guide.  A processes.md file listing specifics for each process and a miradas.md file for MIRADAS specific documentation.  We could also consider broadly an imaging.md and spectroscopy.md instead of a single processes.md if you think it is better.

## Coding style:
- Standard python coding style with 4 spaces for indents
