from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.fatboyLog import fatboyLog
import os, time

class mergeObjectsProcess(fatboyProcess):
    _modeTags = ["imaging", "circe"]

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("Merge objects")
        print(fdu._identFull)

        if (not fdu.hasProperty("merge_name")):
            #This fdu has no merge_name property
            print("mergeObjectsProcess::execute> Warning: No merge_name property defined for "+fdu.getFullId()+".")
            self._log.writeLog(__name__, "No merge_name property defined for "+fdu.getFullId()+".", type=fatboyLog.WARNING)
            return False

        ##This fdu and presumably all others in its group have been merged already
        if (fdu.getProperty("merge_name") == fdu._id):
            return True

        #Call get calibs to return dict() of calibration frames.
        #For mergeObjectsProcess, this dict should have one entry 'frameList' which is an fdu list (including the current fdu)
        calibs = self.getCalibs(fdu, prevProc)
        if (not 'frameList' in calibs):
            #Failed to obtain framelist
            #Issue warning message but don't disable this FDU
            print("mergeObjectsProcess::execute> Warning: Merging not done for "+fdu.getFullId()+" (exp time="+str(fdu.exptime)+"; nreads="+str(fdu.nreads)+").")
            self._log.writeLog(__name__, "Merging not done for "+fdu.getFullId()+" (exp time="+str(fdu.exptime)+"; nreads="+str(fdu.nreads)+").", type=fatboyLog.WARNING)
            return False

        #get framelist
        frameList = calibs['frameList']
        mergeid = fdu.getProperty("merge_name")
        for image in frameList:
            if (image.inUse):
                #update _identFull first, the _id
                image._identFull = image._identFull.replace(image._id, mergeid)
                image._id = mergeid
        return True
    #end execute

    ## OVERRIDE getCalibs
    def getCalibs(self, fdu, prevProc = None):
        calibs = dict()
        properties = dict()
        properties['merge_name'] = fdu.getProperty("merge_name")
        #get FDUs matching this merge_name and filter
        fdus = self._fdb.getFDUs(properties=properties, filter=fdu.filter)
        if (len(fdus) > 0):
            #Found FDUs
            print("mergeObjectsProcess::getCalibs> Merging "+str(len(fdus))+" objects into "+properties['merge_name']+" reads...")
            #First recursively process before "merging"
            self.recursivelyExecute(fdus, prevProc)
            calibs['frameList'] = fdus
            return calibs
        return calibs
    #end getCalibs

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/mergedObjects", os.F_OK)):
            os.mkdir(outdir+"/mergedObjects",0o755)
        #Create output filename
        mofile = outdir+"/mergedObjects/mo_"+fdu.getFullId()
        #Check to see if it exists
        if (os.access(mofile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(mofile)
        if (not os.access(mofile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(mofile)
    #end writeOutput
