from superFATBOY.fatboyLog import fatboyLog
from superFATBOY.fatboyProcess import fatboyProcess
import numpy as np
from astropy.stats import sigma_clipped_stats
import os, time

class emirBiasSubtractProcess(fatboyProcess):
    _modeTags = ["emir"]

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("Emir Bias Subtract")
        print(fdu._identFull)

        #Check if output exists first
        ebsfile = "emirBiasSubtracted/ebs_"+fdu.getFullId()
        if (self.checkOutputExists(fdu, ebsfile)):
            return True

        fdu.updateData(self.emirBiasSubtract(fdu))
        fdu._header.add_history('emir bias subtracted')
        return True
    #end execute

    def emirBiasSubtract(self, fdu):
        data = fdu.getData().astype(np.float32)
        #Computes a 1D array of sigma clipped mean and median values from the top 4 rows of the image
        #Can also do the same with the bottom 4 rows, but the values are not consistent and advice from GTC was to use the top
        #Can test to see which works best, but start with top

        mean,median,std=sigma_clipped_stats(data[0:3,:],mask=None,mask_value=None,sigma=3.0,maxiters=3,cenfunc='median',stdfunc='std',axis=0,grow=False)
        #Subtracting off the mean for now. Minimal difference to median from what I can see
        data = data-mean 
        #Note that there are also 4 columns on each side which could be used to correct for variations in that direction. Omitting until it proves necessary.
        #Finally, we should strip off all of these dark rows and columns
        data = np.ascontiguousarray(data[4:-4,4:-4])

        return data 
    #end emirBiasSubtract

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/emirBiasSubtracted", os.F_OK)):
            os.mkdir(outdir+"/emirBiasSubtracted",0o755)
        #Create output filename
        ebsfile = outdir+"/emirBiasSubtracted/ebs_"+fdu.getFullId()
        #Check to see if it exists
        if (os.access(ebsfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(ebsfile)
        if (not os.access(ebsfile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(ebsfile)
    #end writeOutput
