from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY.fatboyLibs import *
from numpy import *
import os, time

class sinfoniIdentifySlitletsProcess(fatboyProcess):
    _modeTags = ["sinfoni"]

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("Sinfoni Identify Slitlets")
        print(fdu._identFull)

        #Call get calibs to return dict() of calibration frames.
        #For sinfoniIdentifySlitlets, this should get a slitmask if MOS data
        calibs = self.getCalibs(fdu, prevProc)
        if (not 'slitmask' in calibs):
            #Failed to obtain slitmask
            #Issue error message and disable this FDU
            print("sinfoniIdentifySlitletsProcess::execute> ERROR: Slitmask not found for "+fdu.getFullId()+".  Discarding Image!")
            self._log.writeLog(__name__, "Slitmask not found for "+fdu.getFullId()+".  Discarding Image!", type=fatboyLog.ERROR)
            #disable this FDU
            fdu.disable()
            return False

        #Use fdu.getSlitmask() here to look for slitmask specfic to the FDU
        slitmask = fdu.getSlitmask()
        if (slitmask.hasProperty("SlitletsIdentified")):
            return True

        #check output exists for this FDU's slitmask
        smfile = "sinfoniIdentifySlitlets/slitmask_"+fdu.getFullId()
        if (self.checkOutputExists(fdu, smfile, tag="slitmask")):
            slitmask = fdu.getSlitmask()
            #Check for slitmask for lamp/sky
            if (not calibs['slitmask'].hasProperty("SlitletsIdentified")):
                sisfile = "sinfoniIdentifySlitlets/sis_"+slitmask.getFullId()
                self.checkOutputExists(calibs['slitmask'], sisfile)
            #Resampled slitmask
            resampFile = "sinfoniIdentifySlitlets/resamp_slitmask_"+fdu.getFullId()
            self.checkOutputExists(fdu, resampFile, tag="resampled_slitmask", headerTag="resampledHeader")

            #Update regions!
            if (slitmask.hasProperty("regions") and not slitmask.hasProperty("SlitletsIdentified")):
                (ylos, yhis, slitx, slitw) = slitmask.getProperty("regions")
                slitnums = array([9, 8, 10, 7, 11, 6, 12, 5, 13, 4, 14, 3, 15, 2, 16, 1, 32, 17, 31, 18, 30, 19, 29, 20, 28, 21, 27, 22, 26, 23, 25, 24])
                slitorder = self.getOption("slitorder", fdu.getTag())
                if (slitorder is not None):
                    if (os.access(slitorder, os.F_OK)):
                        slitnums = readFileIntoList(slitorder)
                        for j in range(len(slitnums)):
                            slitnums = int(slitnums)
                        slitnums = array(slitnums)
                    elif (slitorder.find(",") != -1):
                        slitorder = slitorder.split(",")
                        try:
                            for j in range(len(slitorder)):
                                slitorder[j] = int(slitorder[j].strip())
                            slitnums = array(slitorder)
                        except Exception as ex:
                            print("sinfoniIdentifySlitletsProcess::execute> Warning: Misformatted option slitordre: "+self.getOption("slitorder", fdu.getTag()))
                            self._log.writeLog(__name__, "Misformatted option slitorder: "+self.getOption("slitorder", fdu.getTag()), type=fatboyLog.WARNING)
                            return True
                b = slitnums.argsort()
                ylos = array(ylos)[b]
                yhis = array(yhis)[b]
                slitx = array(slitx)[b]
                slitw = array(slitw)[b]
                slitmask.setProperty("regions", (ylos, yhis, slitx, slitw))
                calibs['slitmask'].setProperty("regions", (ylos, yhis, slitx, slitw))
            slitmask.setProperty("SlitletsIdentified", True)
            calibs['slitmask'].setProperty("SlitletsIdentified", True)
            if (fdu.hasProperty("resampled_slitmask")):
                fdu.getSlitmask(tagname="resampled_slitmask", ignoreShape=True).setProperty("SlitletsIdentified", True)
            return True

        if (slitmask.hasHeaderValue("SLITS_ID")):
            slitmask.setProperty("SlitletsIdentified", True)

#    if (slitmask.hasProperty("SlitletsIdentified")):
#      #Update regions and SlitletsIdentified in other slitmasks
#      if (fdu.hasProperty("slitmask")):
#       fdu.getProperty("slitmask").setProperty("regions", slitmask.getProperty("regions"))
#        fdu.getProperty("slitmask").setProperty("SlitletsIdentified", True)
#      if (fdu.hasProperty("resampled_slitmask")):
#        fdu.getProperty("resampled_slitmask").setProperty("regions", slitmask.getProperty("regions"))
#        fdu.getProperty("resampled_slitmask").setProperty("SlitletsIdentified", True)
            return True

        success = self.identifySlitlets(fdu, calibs)
        return success
    #end execute

    ## OVERRIDE getCalibs
    def getCalibs(self, fdu, prevProc = None):
        calibs = dict()

        #Look for each master calib passed from XML
        smfilename = self.getCalib("slitmask", fdu.getTag())
        if (smfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(smfilename, os.F_OK)):
                print("sinfoniIdentifySlitletsProcess::getCalibs> Using slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Using slitmask "+smfilename+"...")
                calibs['slitmask'] = fatboySpecCalib(self._pname, "slitmask", fdu, filename=smfilename, log=self._log)
            else:
                print("sinfoniIdentifySlitletsProcess::getCalibs> Warning: Could not find slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Could not find slitmask "+smfilename+"...", type=fatboyLog.WARNING)

        #Find master clean sky and master arclamp associated with this object
        headerVals = dict()
        headerVals['grism_keyword'] = fdu.grism
        properties = dict()
        properties['specmode'] = fdu.getProperty("specmode")
        properties['dispersion'] = fdu.getProperty("dispersion")

        skyShape = None
        if (not 'cleanSky' in calibs):
            #Check for an already created clean sky frame frame matching specmode/filter/grism/ident
            #cleanSky = self._fdb.getMasterCalib(ident = "cleanSky_"+fdu._id, filter=fdu.filter, obstype="master_clean_sky", properties=properties, headerVals=headerVals, tag=fdu.getTag())
            cleanSky = self._fdb.getMasterCalib(ident = "cleanSky_ds_"+fdu._id, filter=fdu.filter, section=fdu.section, obstype="master_clean_sky", properties=properties, headerVals=headerVals, tag=fdu.getTag())
            if (cleanSky is not None):
                #add to calibs for rectification below
                calibs['cleanSky'] = cleanSky
                skyShape = cleanSky.getShape()

        if (not 'masterLamp' in calibs):
            #1) Check for an already created master arclamp frame matching specmode/filter/grism and TAGGED for this object
            masterLamp = self._fdb.getTaggedMasterCalib(ident=fdu._id, obstype="master_arclamp", filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals)
            if (masterLamp is None):
                #2) Check for an already created master arclamp frame frame matching specmode/filter/grism
                masterLamp = self._fdb.getMasterCalib(obstype="master_arclamp", filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals, tag=fdu.getTag())
            if (masterLamp is not None):
                #add to calibs for rectification below
                calibs['masterLamp'] = masterLamp
                skyShape = masterLamp.getShape()

        if (not 'slitmask' in calibs):
            #Find slitmask associated with this fdu
            #Use new fdu.getSlitmask method
            fdu.printAllSlitmasks()
            slitmask = fdu.getSlitmask(pname=None, shape=skyShape, properties=properties, headerVals=headerVals)
            if (slitmask is None):
                print("sinfoniIdentifySlitletsProcess::getCalibs> ERROR: Could not find slitmask associated with "+fdu.getFullId()+"!  Unable to identify slitlets!")
                self._log.writeLog(__name__, "Could not find slitmask associated with "+fdu.getFullId()+"!  Unable to identify slitlets!", type=fatboyLog.ERROR)
                return calibs
            calibs['slitmask'] = slitmask

        #Check for individual FDUs matching specmode/filter/grism/ident
        #fdus can not be [] as it will always at least return the current FDU itself
        fdus = self._fdb.getSortedFDUs(ident = fdu._id, obstype=fdu.FDU_TYPE_OBJECT, filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals, tag=fdu.getTag())
        if (len(fdus) > 0):
            #Found other objects associated with this fdu. Recursively process
            print("sinfoniIdentifySlitletsProcess::getCalibs> Recursivley processing other images for object "+fdu._id+"...")
            self._log.writeLog(__name__, "Recursivley processing other images for object "+fdu._id+"...")
            #First recursively process
            self.recursivelyExecute(fdus, prevProc)

        return calibs
    #end getCalibs

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('slitorder', None)
        self._optioninfo.setdefault('slitorder', 'Comma separated list or text file listing slitlet numbers\nfrom left to right')
    #end setDefaultOptions


    def identifySlitlets(self, fdu, calibs):
        #default slit order
        slitnums = array([9, 8, 10, 7, 11, 6, 12, 5, 13, 4, 14, 3, 15, 2, 16, 1, 32, 17, 31, 18, 30, 19, 29, 20, 28, 21, 27, 22, 26, 23, 25, 24])
        slitorder = self.getOption("slitorder", fdu.getTag())
        if (slitorder is not None):
            if (os.access(slitorder, os.F_OK)):
                slitnums = readFileIntoList(slitorder)
                for j in range(len(slitnums)):
                    slitnums = int(slitnums)
                slitnums = array(slitnums)
            elif (slitorder.find(",") != -1):
                slitorder = slitorder.split(",")
                try:
                    for j in range(len(slitorder)):
                        slitorder[j] = int(slitorder[j].strip())
                    slitnums = array(slitorder)
                except Exception as ex:
                    print("sinfoniIdentifySlitletsProcess::identifySlitlets> Warning: Misformatted option slitordre: "+self.getOption("slitorder", fdu.getTag()))
                    self._log.writeLog(__name__, "Misformatted option slitorder: "+self.getOption("slitorder", fdu.getTag()), type=fatboyLog.WARNING)
                    return False

        #Use fdu.getSlitmask() here to look for slitmask specfic to the FDU
        slitmask = fdu.getSlitmask()
        data = slitmask.getData()
        #create a new slitmask of zeros
        newmask = zeros(data.shape, data.dtype)
        doCalib = False
        doResampFDU = False
        #Main slitmask is FDU.  Need to update calib slitmask for lamp/sky
        #in first pass as well.
        #And get resampled FDU shape slitmask
        #Do not check properties - just use getSlitmask and call with correct shapes
        if (slitmask.getShape() != calibs['slitmask'].getShape()):
            doCalib = True
            calibData = calibs['slitmask'].getData()
            newCalibData = zeros(calibData.shape, calibData.dtype)

        if (not fdu.hasProperty("is_resampled") and fdu.hasProperty("resampled")):
            doResampFDU = True
            resamp_slitmask = fdu.getSlitmask(tagname="resampled_slitmask", shape=fdu.getProperty("resampled").shape)
            fduResampData = resamp_slitmask.getData()
            newResampFDUData = zeros(fduResampData.shape, fduResampData.dtype)

        #populate with new slitnums array
        for j in range(len(slitnums)):
            newmask[data == (j+1)] = slitnums[j]
            if (doCalib):
                newCalibData[calibData == (j+1)] = slitnums[j]
            if (doResampFDU):
                newResampFDUData[fduResampData == (j+1)] = slitnums[j]

        #Use setSlitmask to update slitmask
        smprops = dict()
        smprops["SlitletsIdentified"] = True

        #slitmask.updateData(newmask)
        #slitmask.setProperty("SlitletsIdentified", True)
        #slitmask._header["SLITS_ID"] = "T"
        slitmask = fdu.setSlitmask(newmask, pname=self._pname, properties=smprops)
        slitmask._header["SLITS_ID"] = "T"
        slitmask._header.add_history('Slitlets Identified')
        if (doCalib):
            #calibs['slitmask'].setProperty("SlitletsIdentified", True)
            calibs['slitmask'] = fdu.setSlitmask(newCalibData, pname=self._pname, properties=smprops)
            calibs['slitmask']._header["SLITS_ID"] = "T"
            calibs['slitmask']._header.add_history('Slitlets Identified')
        if (doResampFDU):
            #resamp_slitmask.setProperty("SlitletsIdentified", True)
            resamp_slitmask = fdu.setSlitmask(newResampFDUData, pname=self._pname, properties=smprops, tagname="resampled_slitmask")
            resamp_slitmask._header["SLITS_ID"] = "T"
            resamp_slitmask._header.add_history('Slitlets Identified')

        #Update regions!
        if (slitmask.hasProperty("regions")):
            (ylos, yhis, slitx, slitw) = slitmask.getProperty("regions")
            b = slitnums.argsort()
            ylos = array(ylos)[b]
            yhis = array(yhis)[b]
            slitx = array(slitx)[b]
            slitw = array(slitw)[b]
        else:
            #Use helper method to all ylo, yhi for each slit in each frame
            nslits = slitmask.getData().max()
            slitmask.setProperty("nslits", nslits)
            (ylos, yhis, slitx, slitw) = findRegions(slitmask.getData(), nslits, fdu, gpu=self._fdb.getGPUMode(), log=self._log)
        slitmask.setProperty("regions", (ylos, yhis, slitx, slitw))
        if (doCalib):
            calibs['slitmask'].setProperty("nslits", nslits)
            calibs['slitmask'].setProperty("regions", (ylos, yhis, slitx, slitw))
        if (doResampFDU):
            resamp_slitmask.setProperty("nslits", nslits)
            resamp_slitmask.setProperty("regions", (ylos, yhis, slitx, slitw))

        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
            if (not os.access(outdir+"/sinfoniIdentifySlitlets", os.F_OK)):
                os.mkdir(outdir+"/sinfoniIdentifySlitlets",0o755)
            sisfile = outdir+"/sinfoniIdentifySlitlets/slitmask_"+fdu.getFullId()
            if (os.access(sisfile, os.F_OK) and  self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(sisfile)
            if (not os.access(sisfile, os.F_OK)):
                slitmask.writeTo(sisfile)
            if (doCalib):
                sisfile = outdir+"/sinfoniIdentifySlitlets/sis_"+slitmask.getFullId()
                if (os.access(sisfile, os.F_OK) and  self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                    os.unlink(sisfile)
                if (not os.access(sisfile, os.F_OK)):
                    calibs['slitmask'].writeTo(sisfile)
            if (doResampFDU):
                sisfile = outdir+"/sinfoniIdentifySlitlets/resamp_slitmask_"+fdu.getFullId()
                if (os.access(sisfile, os.F_OK) and  self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                    os.unlink(sisfile)
                if (not os.access(sisfile, os.F_OK)):
                    resamp_slitmask.writeTo(sisfile)
        return True
