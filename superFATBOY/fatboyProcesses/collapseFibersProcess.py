from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY.datatypeExtensions.fatboySpecCalib import fatboySpecCalib
from numpy import *
import os, time

class collapseFibersProcess(fatboyProcess):
    _modeTags = ["spectroscopy"]

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("Collapse Fibers")
        print(fdu._identFull)

        #Call get calibs to return dict() of calibration frames.
        #For collapse fibers, this should get a slitmask if MOS data
        #And optionally an arclamp and clean sky frame
        calibs = self.getCalibs(fdu, prevProc)

        #Check if output exists first
        cffile = "collapsedFibers/cf_"+fdu.getFullId()
        if (self.checkOutputExists(fdu, cffile)):
            #Also check if "cleanFrame" exists
            cleanfile = "collapsedFibers/clean_cf_"+fdu.getFullId()
            self.checkOutputExists(fdu, cleanfile, tag="cleanFrame")
            #Also check if noisemap exists
            nmfile = "collapsedFibers/NM_cf_"+fdu.getFullId()
            self.checkOutputExists(fdu, nmfile, tag="noisemap")

            #Check for cleanSky and masterLamp frames to update from disk too
            if ('cleanSky' in calibs and not calibs['cleanSky'].hasProperty("collapsed")):
                #Check if output exists
                cffile = "collapsedFibers/cf_"+calibs['cleanSky'].getFullId()
                if (self.checkOutputExists(calibs['cleanSky'], cffile)):
                    #output file already exists and overwrite = no.  Update data from disk and set "collapsed" = True
                    calibs['cleanSky'].setProperty("collapsed", True)

            if ('masterLamp' in calibs and not calibs['masterLamp'].hasProperty("collapsed")):
                #Check if output exists first
                cffile = "collapsedFibers/cf_"+calibs['masterLamp'].getFullId()
                if (self.checkOutputExists(calibs['masterLamp'], cffile)):
                    #output file already exists and overwrite = no.  Update data from disk and set "collapsed" = True
                    calibs['masterLamp'].setProperty("collapsed", True)

            #Check for slitmask frames to update from disk too
            if ('slitmask' in calibs and not calibs['slitmask'].hasProperty("collapsed")):
                #Check if output exists
                cffile = "collapsedFibers/cf_"+calibs['slitmask'].getFullId()
                #This will append new slitmask
                if (self.checkOutputExists(calibs['slitmask'], cffile)):
                    calibs['slitmask'].setProperty("collapsed", True)
                    #Now get new slitmask with correct shape - FDU data has been updated with checkOutputExists above
                    calibs['slitmask'] = fdu.getSlitmask(pname=None, properties=properties, headerVals=headerVals)
                    #output file already exists and overwrite = no.  Update data from disk and set "collapsed" = True
                    calibs['slitmask'].setProperty("collapsed", True)
                    #Update nslits property
                    nslits = calibs['slitmask'].getData().max()
                    calibs['slitmask'].setProperty("nslits", nslits)
                    #Update regions
                    if (calibs['slitmask'].hasProperty("regions")):
                        (sylo, syhi, slitx, slitw) = calibs['slitmask'].getProperty("regions")
                        #Use helper method to all ylo, yhi for each slit in each frame
                        #Keep original slitx, slitw - use temp vars to receive return values
                        (sylo, syhi, tempx, tempw) = findRegions(calibs['slitmask'].getData(), nslits, calibs['slitmask'], gpu=self._fdb.getGPUMode(), log=self._log)
                    else:
                        #Use helper method to all ylo, yhi for each slit in each frame
                        (sylo, syhi, slitx, slitw) = findRegions(calibs['slitmask'].getData(), nslits, calibs['slitmask'], gpu=self._fdb.getGPUMode(), log=self._log)
                    calibs['slitmask'].setProperty("regions", (sylo, syhi, slitx, slitw))
            return True

        self.collapseFibers(fdu, calibs)
        fdu._header.add_history('Collapsed Fibers')
        return True
    #end execute

    ## OVERRIDE getCalibs
    def getCalibs(self, fdu, prevProc = None):
        calibs = dict()

        #Look for each master calib passed from XML
        smfilename = self.getCalib("slitmask", fdu.getTag())
        if (smfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(smfilename, os.F_OK)):
                print("collapseFibersProcess::getCalibs> Using slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Using slitmask "+smfilename+"...")
                calibs['slitmask'] = fatboySpecCalib(self._pname, "slitmask", fdu, filename=smfilename, log=self._log)
            else:
                print("collapseFibersProcess::getCalibs> Warning: Could not find slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Could not find slitmask "+smfilename+"...", type=fatboyLog.WARNING)
        #Look for each master calib passed from XML
        csfilename = self.getCalib("master_clean_sky", fdu.getTag())
        if (csfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(csfilename, os.F_OK)):
                print("collapseFibersProcess::getCalibs> Using master clean sky frame "+csfilename+"...")
                self._log.writeLog(__name__, "Using master clean sky frame "+csfilename+"...")
                calibs['cleanSky'] = fatboySpecCalib(self._pname, "master_clean_sky", fdu, filename=csfilename, log=self._log)
            else:
                print("collapseFibersProcess::getCalibs> Warning: Could not find master clean sky frame "+csfilename+"...")
                self._log.writeLog(__name__, "Could not find master clean sky frame "+csfilename+"...", type=fatboyLog.WARNING)
        #Look for each master calib passed from XML
        mlfilename = self.getCalib("master_arclamp", fdu.getTag())
        if (mlfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(mlfilename, os.F_OK)):
                print("collapseFibersProcess::getCalibs> Using master arclamp frame "+mlfilename+"...")
                self._log.writeLog(__name__, "Using master arclamp frame "+mlfilename+"...")
                calibs['masterLamp'] = fatboySpecCalib(self._pname, "master_arclamp", fdu, filename=mlfilename, log=self._log)
            else:
                print("collapseFibersProcess::getCalibs> Warning: Could not find master arclamp frame "+mlfilename+"...")
                self._log.writeLog(__name__, "Could not find master arclamp frame "+mlfilename+"...", type=fatboyLog.WARNING)

        #Find master clean sky and master arclamp associated with this object
        headerVals = dict()
        headerVals['grism_keyword'] = fdu.grism
        properties = dict()
        properties['specmode'] = fdu.getProperty("specmode")
        properties['dispersion'] = fdu.getProperty("dispersion")

        #Arclamp / clean sky frame
        if (not 'cleanSky' in calibs):
            #Check for an already created clean sky frame frame matching specmode/filter/grism/ident
            cleanSky = self._fdb.getMasterCalib(ident = "cleanSky_ds_"+fdu._id, filter=fdu.filter, obstype="master_clean_sky", shape=fdu.getShape(), properties=properties, headerVals=headerVals, tag=fdu.getTag())
            if (cleanSky is not None):
                #add to calibs for collapse below
                calibs['cleanSky'] = cleanSky

        if (not 'masterLamp' in calibs):
            #1) Check for an already created master arclamp frame matching specmode/filter/grism and TAGGED for this object
            masterLamp = self._fdb.getTaggedMasterCalib(ident=fdu._id, obstype="master_arclamp", filter=fdu.filter, shape=fdu.getShape(), properties=properties, headerVals=headerVals)
            if (masterLamp is None):
                #2) Check for an already created master arclamp frame frame matching specmode/filter/grism
                masterLamp = self._fdb.getMasterCalib(obstype="master_arclamp", filter=fdu.filter, shape=fdu.getShape(), properties=properties, headerVals=headerVals, tag=fdu.getTag())
            if (masterLamp is not None):
                #add to calibs for collapse below
                calibs['masterLamp'] = masterLamp

        if (fdu._specmode != fdu.FDU_TYPE_LONGSLIT):
            if (not 'slitmask' in calibs):
                #Find slitmask associated with this fdu
                #Use new fdu.getSlitmask method
                slitmask = fdu.getSlitmask(pname=None, properties=properties, headerVals=headerVals)
                if (slitmask is None):
                    print("collapseFibersProcess::getCalibs> ERROR: Could not find slitmask associated with "+fdu.getFullId()+"!  Unable to collapse fibers!")
                    self._log.writeLog(__name__, "Could not find slitmask associated with "+fdu.getFullId()+"!  Unable to collapse fibers!", type=fatboyLog.ERROR)
                    return calibs
                calibs['slitmask'] = slitmask

        return calibs
    #end getCalibs

    def collapseData(self, fdu, data, method):
        if (method.lower() == "sum"):
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                return data.sum(0)
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                return data.sum(1)
        elif (method.lower() == "mean"):
            #Exclude 0s
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                return data.sum(0)/(data != 0).sum(0)
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                return data.sum(1)/(data != 0).sum(1)
        elif (method.lower() == "median"):
            #Select kernel for 2d median
            kernel2d = fatboyclib.median2d
            if (self._fdb.getGPUMode()):
                #Use GPU for medians
                kernel2d=gpumedian2d
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                if (self._fdb.getGPUMode()):
                    #Use GPU
                    return gpu_arraymedian(data, axis="Y", nonzero=True, kernel2d=kernel2d)
                else:
                    #Use CPU
                    return kernel2d(data.transpose().copy(), nonzero=True)
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                return gpu_arraymedian(data, axis="X", nonzero=True, kernel2d=kernel2d)
        print("collapseFibersProcess::collapseData> ERROR: Invalid collapse method "+method+"!")
        self._log.writeLog(__name__, "Invalid collapse method "+method+"!", type=fatboyLog.ERROR)
        return data
    #end collapseData

    def collapseFibers(self, fdu, calibs):
        collapse_method = self.getOption('collapse_method', fdu.getTag())

        writeCalibs = False
        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            writeCalibs = True
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/collapsedFibers", os.F_OK)):
            os.mkdir(outdir+"/collapsedFibers",0o755)

        #Check for cleanSky and masterLamp frames to update from disk too
        if ('cleanSky' in calibs and not calibs['cleanSky'].hasProperty("collapsed")):
            #Check if output exists
            cffile = "collapsedFibers/cf_"+calibs['cleanSky'].getFullId()
            if (self.checkOutputExists(calibs['cleanSky'], cffile)):
                #output file already exists and overwrite = no.  Update data from disk and set "collapsed" = True
                calibs['cleanSky'].setProperty("collapsed", True)

        if ('masterLamp' in calibs and not calibs['masterLamp'].hasProperty("collapsed")):
            #Check if output exists first
            cffile = "collapsedFibers/cf_"+calibs['masterLamp'].getFullId()
            if (self.checkOutputExists(calibs['masterLamp'], cffile)):
                #output file already exists and overwrite = no.  Update data from disk and set "collapsed" = True
                calibs['masterLamp'].setProperty("collapsed", True)

        if (fdu._specmode == fdu.FDU_TYPE_LONGSLIT):
            fdu.updateData(self.collapseData(fdu, fdu.getData(), collapse_method))
            if (fdu.hasProperty("cleanFrame")):
                fdu.tagDataAs("cleanFrame", data=self.collapseData(fdu, fdu.getData(tag="cleanFrame"), collapse_method))
            #collapse noisemap for spectrocsopy data
            if (fdu.hasProperty("noisemap")):
                #Square data, collapse, take sqare root
                nmData = fdu.getData(tag="noisemap")**2
                fdu.tagDataAs("noisemap", data=sqrt(self.collapseData(fdu, nmData, collapse_method)))
            #Look for "cleanSky" frame to collapse
            if ('cleanSky' in calibs and not calibs['cleanSky'].hasProperty("collapsed")):
                cleanSky = calibs['cleanSky']
                #update data, set "collapsed" property
                cleanSky.updateData(self.collapseData(fdu, cleanSky.getData(), collapse_method))
                cleanSky.setProperty("collapsed", True)
                #Write to disk if requested
                if (writeCalibs):
                    cffile = outdir+"/collapsedFibers/cf_"+cleanSky.getFullId()
                    #Remove existing files if overwrite = yes
                    if (os.access(cffile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                        os.unlink(cffile)
                    #Write to disk
                    if (not os.access(cffile, os.F_OK)):
                        cleanSky.writeTo(cffile)
            #Look for "masterLamp" frame to collapse
            if ('masterLamp' in calibs and not calibs['masterLamp'].hasProperty("collapsed")):
                masterLamp = calibs['masterLamp']
                #update data, set "collapsed" property
                masterLamp.updateData(self.collapseData(fdu, masterLamp.getData(), collapse_method))
                masterLamp.setProperty("collapsed", True)
                #Write to disk if requested
                if (writeCalibs):
                    cffile = outdir+"/collapsedFibers/cf_"+masterLamp.getFullId()
                    #Remove existing files if overwrite = yes
                    if (os.access(cffile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                        os.unlink(cffile)
                    #Write to disk
                    if (not os.access(cffile, os.F_OK)):
                        masterLamp.writeTo(cffile)
        else:
            #MOS/IFU data, loop over slitlets
            if (not 'slitmask' in calibs):
                print("collapseFibersProcess::collapseFibers> ERROR: Could not find slitmask associated with "+fdu.getFullId()+"!  Unable to collapse fibers!")
                self._log.writeLog(__name__, "Could not find slitmask associated with "+fdu.getFullId()+"!  Unable to collapse fibers!", type=fatboyLog.ERROR)
                return
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                xsize = fdu.getShape()[1]
                ysize = fdu.getShape()[0]
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                ##xsize should be size across dispersion direction
                xsize = fdu.getShape()[0]
                ysize = fdu.getShape()[1]
            if (not calibs['slitmask'].hasProperty("nslits")):
                calibs['slitmask'].setProperty("nslits", calibs['slitmask'].getData().max())
            nslits = calibs['slitmask'].getProperty("nslits")
            #Use helper method to all ylo, yhi for each slit in each frame
            (ylos, yhis, slitx, slitw) = findRegions(calibs['slitmask'].getData(), nslits, calibs['slitmask'], gpu=self._fdb.getGPUMode(), log=self._log)

            #Create new output arrays
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                data = zeros((nslits, xsize), float32)
                if (fdu.hasProperty("cleanFrame")):
                    cleanData = zeros((nslits, xsize), float32)
                if (fdu.hasProperty("noisemap")):
                    nmData = zeros((nslits, xsize), float32)
                if ('cleanSky' in calibs and not calibs['cleanSky'].hasProperty("collapsed")):
                    skyData = zeros((nslits, xsize), float32)
                if ('masterLamp' in calibs and not calibs['masterLamp'].hasProperty("collapsed")):
                    lampData = zeros((nslits, xsize), float32)
                if ('slitmask' in calibs and not calibs['slitmask'].hasProperty("collapsed")):
                    slitData = zeros((nslits, xsize), int32)
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                data = zeros((xsize, nslits), float32)
                if (fdu.hasProperty("cleanFrame")):
                    cleanData = zeros((xsize, nslits), float32)
                if (fdu.hasProperty("noisemap")):
                    nmData = zeros((xsize, nslits), float32)
                if ('cleanSky' in calibs and not calibs['cleanSky'].hasProperty("collapsed")):
                    skyData = zeros((xsize, nslits), float32)
                if ('masterLamp' in calibs and not calibs['masterLamp'].hasProperty("collapsed")):
                    lampData = zeros((xsize, nslits), float32)
                if ('slitmask' in calibs and not calibs['slitmask'].hasProperty("collapsed")):
                    slitData = zeros((xsize, nslits), int32)

            #Loop over slitlets
            for slitidx in range(nslits):
                ylo = ylos[slitidx]
                yhi = yhis[slitidx]
                #Find the data corresponding to this slit and take 1-d cut
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    currMask = calibs['slitmask'].getData()[ylo:yhi+1,:] == (slitidx+1)
                    slit = fdu.getData()[ylo:yhi+1,:]*currMask
                    data[slitidx,:] = self.collapseData(fdu, slit, collapse_method)
                    if (fdu.hasProperty("cleanFrame")):
                        slit = fdu.getData(tag="cleanFrame")[ylo:yhi+1,:]*currMask
                        cleanData[slitidx,:] = self.collapseData(fdu, slit, collapse_method)
                    #collapse noisemap for spectrocsopy data
                    if (fdu.hasProperty("noisemap")):
                        #Square data, collapse, take sqare root
                        slit = (fdu.getData(tag="noisemap")[ylo:yhi+1,:]*currMask)**2
                        nmData[slitidx,:] = sqrt(self.collapseData(fdu, slit, collapse_method))
                    if ('cleanSky' in calibs and not calibs['cleanSky'].hasProperty("collapsed")):
                        slit = calibs['cleanSky'].getData()[ylo:yhi+1,:]*currMask
                        skyData[slitidx,:] = self.collapseData(fdu, slit, collapse_method)
                    if ('masterLamp' in calibs and not calibs['masterLamp'].hasProperty("collapsed")):
                        slit = calibs['masterLamp'].getData()[ylo:yhi+1,:]*currMask
                        lampData[slitidx,:] = self.collapseData(fdu, slit, collapse_method)
                    if ('slitmask' in calibs and not calibs['slitmask'].hasProperty("collapsed")):
                        slitData[slitidx,:] = slitidx+1
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    currMask = calibs['slitmask'].getData()[:,ylo:yhi+1] == (slitidx+1)
                    slit = fdu.getData()[:,ylo:yhi+1]*currMask
                    data[:,slitidx] = self.collapseData(fdu, slit, collapse_method)
                    if (fdu.hasProperty("cleanFrame")):
                        slit = fdu.getData(tag="cleanFrame")[:,ylo:yhi+1]*currMask
                        cleanData[:,slitidx] = self.collapseData(fdu, slit, collapse_method)
                    #collapse noisemap for spectrocsopy data
                    if (fdu.hasProperty("noisemap")):
                        #Square data, collapse, take sqare root
                        slit = (fdu.getData(tag="noisemap")[:,ylo:yhi+1]*currMask)**2
                        nmData[:,slitidx] = sqrt(self.collapseData(fdu, slit, collapse_method))
                    if ('cleanSky' in calibs and not calibs['cleanSky'].hasProperty("collapsed")):
                        slit = calibs['cleanSky'].getData()[:,ylo:yhi+1]*currMask
                        skyData[:,slitidx] = self.collapseData(fdu, slit, collapse_method)
                    if ('masterLamp' in calibs and not calibs['masterLamp'].hasProperty("collapsed")):
                        slit = calibs['masterLamp'].getData()[:,ylo:yhi+1]*currMask
                        lampData[:,slitidx] = self.collapseData(fdu, slit, collapse_method)
                    if ('slitmask' in calibs and not calibs['slitmask'].hasProperty("collapsed")):
                        slitData[:,slitidx] = slitidx+1

            #Update arrays
            fdu.updateData(data)
            if (fdu.hasProperty("cleanFrame")):
                fdu.tagDataAs("cleanFrame", data=cleanData)
            #collapse noisemap for spectrocsopy data
            if (fdu.hasProperty("noisemap")):
                fdu.tagDataAs("noisemap", data=nmData)
            if ('cleanSky' in calibs and not calibs['cleanSky'].hasProperty("collapsed")):
                cleanSky = calibs['cleanSky']
                #update data, set "collapsed" property
                cleanSky.updateData(skyData)
                cleanSky.setProperty("collapsed", True)
                #Write to disk if requested
                if (writeCalibs):
                    cffile = outdir+"/collapsedFibers/cf_"+cleanSky.getFullId()
                    #Remove existing files if overwrite = yes
                    if (os.access(cffile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                        os.unlink(cffile)
                    #Write to disk
                    if (not os.access(cffile, os.F_OK)):
                        cleanSky.writeTo(cffile)
            if ('masterLamp' in calibs and not calibs['masterLamp'].hasProperty("collapsed")):
                masterLamp = calibs['masterLamp']
                #update data, set "collapsed" property
                masterLamp.updateData(lampData)
                masterLamp.setProperty("collapsed", True)
                #Write to disk if requested
                if (writeCalibs):
                    cffile = outdir+"/collapsedFibers/cf_"+masterLamp.getFullId()
                    #Remove existing files if overwrite = yes
                    if (os.access(cffile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                        os.unlink(cffile)
                    #Write to disk
                    if (not os.access(cffile, os.F_OK)):
                        masterLamp.writeTo(cffile)
            if ('slitmask' in calibs and not calibs['slitmask'].hasProperty("collapsed")):
                #Use new slitmask methods to create new slitmask
                slitmask = calibs['slitmask']
                cf_slitmask = self._fdb.addNewSlitmask(slitmask, slitData, self._pname)
                #update properties
                cf_slitmask.setProperty("nslits", nslits)
                #Use helper method to all ylo, yhi for each slit in each frame
                (sylo, syhi, slitx, slitw) = findRegions(cf_slitmask.getData(), nslits, cf_slitmask, gpu=self._fdb.getGPUMode(), log=self._log)
                cf_slitmask.setProperty("regions", (sylo, syhi, slitx, slitw))
                cf_slitmask.setProperty("collapsed", True)
                #Set "collapsed" property
                slitmask.setProperty("collapsed", True)
                #Write to disk if requested
                if (writeCalibs):
                    cffile = outdir+"/collapsedFibers/cf_"+cf_slitmask.getFullId()
                    #Remove existing files if overwrite = yes
                    if (os.access(cffile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                        os.unlink(cffile)
                    #Write to disk
                    if (not os.access(cffile, os.F_OK)):
                        cf_slitmask.writeTo(cffile)
    #end collapseFibers

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('collapse_method', 'sum')
        self._optioninfo.setdefault('collapse_method', 'sum | mean | median')
    #end setDefaultOptions

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/collapsedFibers", os.F_OK)):
            os.mkdir(outdir+"/collapsedFibers",0o755)
        #Create output filename
        cffile = outdir+"/collapsedFibers/cf_"+fdu.getFullId()
        #Check to see if it exists
        if (os.access(cffile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(cffile)
        if (not os.access(cffile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(cffile)
        #Write out clean frame if it exists
        if (fdu.hasProperty("cleanFrame")):
            cleanfile = outdir+"/collapsedFibers/clean_cf_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(cleanfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(cleanfile)
            if (not os.access(cleanfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.writeTo(cleanfile, tag="cleanFrame")
        #Write noisemap for spectrocsopy data if requested
        if (self.getOption("write_noisemaps", fdu.getTag()).lower() == "yes" and fdu.hasProperty("noisemap")):
            nmfile = outdir+"/collapsedFibers/NM_cf_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(nmfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(nmfile)
            if (not os.access(nmfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.writeTo(nmfile, tag="noisemap")
    #end writeOutput
