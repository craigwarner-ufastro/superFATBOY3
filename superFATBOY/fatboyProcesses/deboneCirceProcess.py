from superFATBOY.fatboyLog import fatboyLog
from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.gpu_arraymedian import *
from numpy import *
import os, time

class deboneCirceProcess(fatboyProcess):
    _modeTags = ["circe"]

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("Debone Circe")
        print(fdu._identFull)

        #Check if output exists first
        dbfile = "debonedCirce/db_"+fdu.getFullId()
        if (self.checkOutputExists(fdu, dbfile)):
            return True

        fdu.updateData(self.deboneCirce(fdu))
        fdu._header.add_history('deboned')
        return True
    #end execute

    def deboneCirce(self, fdu):
        data = fdu.getData()
        if (data.shape != (2048,2048)):
            print("deboneCirceProcess::deboneCirce> Warning: Data shape for "+fdu.getFullId()+" is "+str(data.shape)+" not (2048, 2048)!  Cannot debone data!")
            self._log.writeLog(__name__, "Data shape for "+fdu.getFullId()+" is "+str(data.shape)+" not (2048, 2048)!  Cannot debone data!", type=fatboyLog.WARNING)
            return data
        newData = empty(data.shape, data.dtype)
        im_amp = empty((2048,64), data.dtype)
        im_stride = arange(32)*64 # stride
        #Loop over amps
        for i in range(64):
            im_amp[:,i] = gpu_arraymedian(data[:,im_stride+i].copy(), axis="X", even=True, kernel=fatboyclib.median, kernel2d=fatboyclib.median2d)
        #Update data
        for i in range(32):
            i1=i*64
            i2=i1+64
            newData[:,i1:i2]=data[:,i1:i2]-im_amp
        #remask bad pixels
        newData[fdu.getBadPixelMask().getData()] = 0
        return newData
    #end deboneCirce

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/debonedCirce", os.F_OK)):
            os.mkdir(outdir+"/debonedCirce",0o755)
        #Create output filename
        dbfile = outdir+"/debonedCirce/db_"+fdu.getFullId()
        #Check to see if it exists
        if (os.access(dbfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(dbfile)
        if (not os.access(dbfile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(dbfile)
    #end writeOutput
