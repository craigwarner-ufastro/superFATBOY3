from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY.fatboyLibs import *
from numpy import *
import os, time

block_size = 512

class noisemapProcess(fatboyProcess):
    _modeTags = ["spectroscopy", "miradas"]

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("Noisemap")
        print(fdu._identFull)

        #Check if output exists first
        nmfile = "noisemaps/NM_"+fdu.getFullId()
        if (self.checkOutputExists(fdu, nmfile)):
            return True

        #Select cpu/gpu option
        if (self._fdb.getGPUMode()):
            #from fatboyLibs
            noisemap = createNoisemap
        else:
            noisemap = self.noisemap_cpu

        if (fdu.gain is None):
            print("noisemapProcess::execute> WARNING: GAIN is not specified.  Using 1 but results MAY BE WRONG!")
            self._log.writeLog(__name__, "GAIN is not specified.  Using 1 but results MAY BE WRONG!", type=fatboyLog.WARNING)
            fdu.gain = 1

        #Use tagDataAs to store noisemap in _properties dict
        fdu.tagDataAs("noisemap", noisemap(fdu.getData(), fdu.gain))
        return True
    #end execute

    def noisemap_cpu(self, data, gain):
        t = time.time()
        nm = sqrt(abs(data/gain))
        if (self._fdb._verbosity == fatboyLog.VERBOSE):
            print("CPU noisemap: ",time.time()-t)
        return nm
    #end noisemap_cpu

    noisemap = createNoisemap

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/noisemaps", os.F_OK)):
            os.mkdir(outdir+"/noisemaps",0o755)
        #Create output filename
        nmfile = outdir+"/noisemaps/NM_"+fdu.getFullId()
        #Check to see if it exists
        if (os.access(nmfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(nmfile)
        if (not os.access(nmfile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(nmfile, tag="noisemap")
    #end writeOutput
