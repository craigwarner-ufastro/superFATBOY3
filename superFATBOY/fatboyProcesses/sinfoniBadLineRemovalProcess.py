from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY.fatboyLibs import *
from numpy import *
import os, time

class sinfoniBadLineRemovalProcess(fatboyProcess):
    _modeTags = ["sinfoni"]

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("Sinfoni Bad Line Removal")
        print(fdu._identFull)

        #Check if output exists first
        blrfile = "badLinesRemoved/blr_"+fdu.getFullId()
        if (self.checkOutputExists(fdu, blrfile)):
            return True

        success = self.removeBadLines(fdu)
        return success
    #end execute

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('nsigmaback', '18')
        self._optioninfo.setdefault('nsigmaback', 'Sigma to identify most of the deviant background pixels')
    #end setDefaultOptions


    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/badLinesRemoved", os.F_OK)):
            os.mkdir(outdir+"/badLinesRemoved",0o755)
        #Create output filename
        blrfile = outdir+"/badLinesRemoved/blr_"+fdu.getFullId()
        #Check to see if it exists
        if (os.access(blrfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(blrfile)
        if (not os.access(blrfile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(blrfile)
    #end writeOutput

    def removeBadLines(self, fdu):
        nsigmaback = int(self.getOption("nsigmaback", fdu.getTag()))
        width = 4
        xsize = fdu.getShape()[1]
        ysize = fdu.getShape()[0]
        #build the template mask for non illuminated edge pixels (background pixels)
        mask_back = zeros(fdu.getShape(), int32)
        #mask_back[0:width, width:xsize-width] = 1
        #mask_back[ysize-width:ysize, width:xsize-width] = 1
        mask_back[width:ysize-width, 0:width] = 1
        mask_back[width:ysize-width, xsize-width:xsize] = 1
        #define subscripts of background pixels
        backpos = where(mask_back == 1)
        #define y pos of back pixels
        ybackpix = backpos[0]

        data = fdu.getData()
        newdata = data.copy()
        #reset the background mask
        backpix = data[backpos]
        #search for back pixels too deviant
        diffbackpix = backpix-gpu_arraymedian(backpix)
        sigmaback = medianfilterCPU(diffbackpix, 3).std()
        bad = abs(diffbackpix) > nsigmaback*sigmaback
        nbad = bad.sum()
        #if no bad pixel, do nothing
        if (nbad == 0):
            print("sinfoniRemoveBadLinesProcess::removeBadLines> No bad lines found for "+fdu.getFullId())
            self._log.writeLog(__name__, "No bad lines found for "+fdu.getFullId())
            return True
        #define background median value of good back pixels
        ybad = ybackpix[bad]
        good = abs(diffbackpix) <= nsigmaback*sigmaback
        medvalue = gpu_arraymedian(backpix[good])
        yprev = -1
        #cycle through rows containing bad back pixels and correct them
        for k in range(nbad):
            yval = ybad[k]
            if (yval != yprev):
                yprev = yval
                #kline = [data[0:width,yval], data[ysize-width:ysize, yval]]
                kline = [data[yval, 0:width], data[yval, xsize-width:xsize]]
                kline_mean = array(kline).sum()/(2.*width)
                #newdata[width:ysize-width, yval] = data[width:ysize-width, yval]+kline_mean-medvalue
                newdata[yval, width:xsize-width] = data[yval, width:xsize-width]+kline_mean-medvalue
        fdu.updateData(newdata)
        return True
