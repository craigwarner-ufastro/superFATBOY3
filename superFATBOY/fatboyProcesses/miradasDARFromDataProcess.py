from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY import drihizzle
import os, time

class miradasDARFromDataProcess(fatboyProcess):
    """ Measure DAR from centroid position of a point-like source
          Need: airmass, in header
          """
    _modeTags = ["miradas"]

    def calculateDAR(self, fdu):
        #Read options
        slitlet_number = self.getOption("slitlet_number", fdu.getTag())
        doAllSlitlets = False
        if (slitlet_number == 'all'):
            doAllSlitlets = True
        elif (isInt(slitlet_number)):
            slitlet_number = int(slitlet_number)

        if (doAllSlitlets):
            slitlets = arange(1, nslits+1)
        else:
            slitlets = [slitlet_number]

        #Get size in dispersion direction
        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            xsize = fdu.getShape()[1]
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            xsize = fdu.getShape()[0]

        #Get header info
        airmass = fdu.getHeaderValue("airmass_keyword")
        #Loop over slitlets (could be one pass or nslits passes)
        for islit in slitlets:
            if (fdu.hasProperty("psf_"+str(islit))):
                psf = fdu.getProperty("psf_"+str(islit))
                refidx = where(psf[:,1] != -1)[0][0]
                refwave = psf[refidx, 0]
                dar_i = psf[:,1]-psf[refidx, 1]
                fdu.tagDataAs("dar_slit_"+str(islit), dar_i)
                fdu.setProperty("dar_ref_wave_slit_"+str(islit), refwave)
    #end calculateDAR

    #Override checkValidDatatype
    def checkValidDatatype(self, fdu):
        #Should only be done for objects or continuum sources
        if (fdu.getObsType(True) == fatboyDataUnit.FDU_TYPE_OBJECT or fdu.getObsType(True) == fdu.FDU_TYPE_STANDARD):
            return True
        if (fdu.getObsType(True) == fdu.FDU_TYPE_CONTINUUM_SOURCE):
            return True
        return False
    #end checkValidDatatype

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("MIRADAS: compute DAR from data")
        print(fdu._identFull)

        #Check if output exists first and update from disk
        darfile = "DAR/dar_"+fdu.getFullId()
        if (self.checkOutputExists(fdu, darfile)):
            return True

        #There are no calibs to get.  Simply calculate dar.
        self.calculateDAR(fdu)

        #There are no calibs to get.
        return True
    #end execute

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('slitlet_number', 'all')
        self._optioninfo.setdefault('slitlet_number', 'Set to all (default) to collapse spaxels for all slitlets.\nSet to a number 1-13 to only select one slitlet.')
    #end setDefaultOptions

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/DAR", os.F_OK)):
            os.mkdir(outdir+"/DAR",0o755)
        nslits = fdu.getProperty("nslits")
        for j in range(1, nslits+1):
            if (fdu.hasProperty("dar_slit_"+str(j))):
                #Create output filename
                darfile = outdir+"/DAR/dar_slit_"+str(j)+"_"+fdu.getFullId()
                #Check to see if it exists
                if (os.access(darfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                    os.unlink(darfile)
                if (not os.access(darfile, os.F_OK)):
                    #Use fatboyDataUnit writeTo method to write
                    fdu.writeTo(darfile, tag="dar_slit_"+str(j))
    #end writeOutput
