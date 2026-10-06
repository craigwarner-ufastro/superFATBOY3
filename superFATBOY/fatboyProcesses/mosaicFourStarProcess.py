import numpy as np
from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import fatboyLog
import os, time

#Mosaic the four FourStar chips (sections 1-4, one file each) of an exposure into one 4196 x 4196 image.
class mosaicFourStarProcess(fatboyProcess):

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("Mosaic FourStar")
        print(fdu._identFull)
        ##This fdu and presumably all others in its group have been mosaicFourStar already
        if (fdu.section == -1):
            return True
        #Call get calibs to return dict() of calibration frames.
        #For mosaicFourStarProcess, this dict should have one entry 'frameList' which is an fdu list (including the current fdu)
        calibs = self.getCalibs(fdu, prevProc)
        if (not 'frameList' in calibs):
            #Failed to obtain framelist
            #Issue error message and disable this FDU
            print("mosaicFourStarProcess::execute> ERROR: Remerging not done for "+fdu.getFullId()+", index "+str(fdu._index)) 
            self._log.writeLog(__name__, "Remerging not done for "+fdu.getFullId()+", index "+str(fdu._index)+".  Discarding Image!", type=fatboyLog.ERROR)
            #disable this FDU
            fdu.disable()
            return False

        #get framelist
        frameList = calibs['frameList']
        newData = np.zeros((4196,4196), dtype=np.float32)
        for image in frameList:
            if (image.inUse and image.section is not None and image.section >= 0):
                x0 = ((image.section-1)//2)*2148 #1 and 2 => 0, 3 and 4 => 2148
                y0 = ((image.section-1)%2)*2148 #1 and 3 => 0, 2 and 4 => 2148
                #force_cpu: the mosaic is assembled in host memory even in a GPU-mode run
                newData[y0:y0+2048,x0:x0+2048] = image.getData(force_cpu=True)
                if (image.section > 1):
                    image.disable() #disable this FDU
        #Update data in section 1 and remove section info
        for image in frameList:
            if (image.inUse and image.section == 1):
                image.updateData(newData)
                image.section = -1
                updateHeaderEntry(image._header, 'SECTION', -1)
        return True
    #end execute

    ## OVERRIDE getCalibs
    def getCalibs(self, fdu, prevProc = None):
        calibs = dict()
        #get FDUs matching this identifier and filter, sorted by index
        fdus = self._fdb.getFDUs(ident = fdu._id, filter=fdu.filter)
        #Narrow to only FDUs matching this index
        matched_index_fdus = []
        for currFDU in fdus:
            if (currFDU._index == fdu._index):
                matched_index_fdus.append(currFDU)
        if (len(matched_index_fdus) > 0):
            #Found other objects associated with this fdu.
            print("mosaicFourStarProcess::getCalibs> Mosaicing sections for object "+fdu._id+", index "+str(fdu._index)) 
            #First recursively process before changing section number
            self.recursivelyExecute(matched_index_fdus, prevProc)
            calibs['frameList'] = matched_index_fdus
            return calibs
        return calibs
    #end getCalibs

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/mosaicFourStar", os.F_OK)):
            os.mkdir(outdir+"/mosaicFourStar",0o755)
        #Create output filename
        mfsfile = outdir+"/mosaicFourStar/mfs_"+fdu.getFullId()
        #Check to see if it exists
        if (os.access(mfsfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(mfsfile)
        if (not os.access(mfsfile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(mfsfile)
    #end writeOutput
