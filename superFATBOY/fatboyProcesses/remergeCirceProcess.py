from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import fatboyLog
import os, time

class remergeCirceProcess(fatboyProcess):
    _modeTags = ["circe"]

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("Remerge")
        print(fdu._identFull)
        ##This fdu and presumably all others in its group have been remerged already
        if (fdu.section == -1):
            return True
        #Call get calibs to return dict() of calibration frames.
        #For remergeProcess, this dict should have one entry 'frameList' which is an fdu list (including the current fdu)
        calibs = self.getCalibs(fdu, prevProc)
        if (not 'frameList' in calibs):
            #Failed to obtain framelist
            #Issue error message and disable this FDU
            print("remergeProcess::execute> ERROR: Remerging not done for "+fdu.getFullId()+" (exp time="+str(fdu.exptime)+"; nreads="+str(fdu.nreads)+").  Discarding Image!")
            self._log.writeLog(__name__, "Remerging not done for "+fdu.getFullId()+" (exp time="+str(fdu.exptime)+"; nreads="+str(fdu.nreads)+").  Discarding Image!", type=fatboyLog.ERROR)
            #disable this FDU
            fdu.disable()
            return False

        #get framelist
        frameList = calibs['frameList']
        zeros = '0000'
        for image in frameList:
            if (image.inUse and image.section is not None and image.section >= 0):
                spos = -1-len(str(image.section)) #-2 for 1 digit sections but allow multiple digits
                if (image._id[spos] == 'S'):
                    image._id = image._id[:spos]
                    sramp = str(image.ramp)
                    if (image._expmode == image.EXPMODE_URG):
                        #trailing index should be section number not ramp number for URG data
                        sramp = str(image.section)
                    sramp = zeros[len(sramp):]+sramp
                    image._identFull = image._id+'.'+image._index+sramp+'.fits'
                    image.section = -1
                    updateHeaderEntry(image._header, 'SECTION', -1)
        return True
    #end execute

    ## OVERRIDE getCalibs
    def getCalibs(self, fdu, prevProc = None):
        calibs = dict()
        #get FDUs matching this identifier and filter, sorted by index
        fdus = self._fdb.getFDUs(ident = fdu._id, filter=fdu.filter)
        if (len(fdus) > 0):
            #Found other objects associated with this fdu.
            print("remergeProcess::getCalibs> Remerging object "+fdu._id+", exposure time "+str(fdu.exptime)+", and "+str(fdu.nreads)+" reads...")
            #First recursively process before changing section number
            self.recursivelyExecute(fdus, prevProc)
            calibs['frameList'] = fdus
            return calibs
        return calibs
    #end getCalibs

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/remerged", os.F_OK)):
            os.mkdir(outdir+"/remerged",0o755)
        #Create output filename
        rmfile = outdir+"/remerged/rm_"+fdu.getFullId()
        #Check to see if it exists
        if (os.access(rmfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(rmfile)
        if (not os.access(rmfile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(rmfile)
    #end writeOutput
