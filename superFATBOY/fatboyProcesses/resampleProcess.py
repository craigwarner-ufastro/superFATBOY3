from superFATBOY.fatboyDataUnit import fatboyDataUnit
from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.datatypeExtensions.fatboySpecCalib import fatboySpecCalib
from superFATBOY import gpu_drihizzle, drihizzle
from numpy import *

usePlot = True
try:
    import matplotlib.pyplot as plt
except Exception as ex:
    print("Warning: Could not import matplotlib!")
    usePlot = False

block_size = 512

class resampleProcess(fatboyProcess):
    _modeTags = ["spectroscopy"]

    #Override checkValidDatatype
    def checkValidDatatype(self, fdu):
        if (fdu.getObsType(True) == fatboyDataUnit.FDU_TYPE_OBJECT or fdu.getObsType(True) == fdu.FDU_TYPE_STANDARD):
            #Make sure rectify isn't called on offsource skies when re-reading data from disk
            return True
        return False
    #end checkValidDatatype

    ## resample data and calibs to linear scale
    def resampleData(self, fdu, calibs):
        ###*** For purposes of resample algorithm, X = dispersion direction and Y = cross-dispersion direction ***###
        print("resampleProcess::resampleData> Resampling "+fdu.getFullId()+" to linear scale...")
        self._log.writeLog(__name__, "Resampling "+fdu.getFullId()+" to linear scale...")
        #Read options
        resampleCalibs = False
        if (self.getOption("resample_calibs", fdu.getTag()).lower() == "yes"):
            resampleCommonScale = True
        resampleCommonScale = False
        if (self.getOption("resample_to_common_scale", fdu.getTag()).lower() == "yes"):
            resampleCommonScale = True
        writeCalibs = False
        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            writeCalibs = True
        outunits = 'cps'
        if (self.getOption("output_units", fdu.getTag()).lower() == "counts"):
            outunits = 'counts'

        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            xsize = fdu.getShape()[1]
            ysize = fdu.getShape()[0]
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            ##xsize should be size across dispersion direction
            xsize = fdu.getShape()[0]
            ysize = fdu.getShape()[1]

        headerVals = dict()
        headerVals['grism_keyword'] = fdu.grism
        properties = dict()
        properties['specmode'] = fdu.getProperty("specmode")
        properties['dispersion'] = fdu.getProperty("dispersion")

        #Create output dir if it doesn't exist
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/resampled", os.F_OK)):
            os.mkdir(outdir+"/resampled",0o755)

        #Defaults for longslit - treat whole image as 1 slit
        nslits = 1
        ylos = [0]
        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            yhis = [fdu.getShape()[0]]
        else:
            yhis = [fdu.getShape()[1]]
        slitmask = None

        doSlitmask = True
        if (fdu._specmode != fdu.FDU_TYPE_LONGSLIT):
            ###MOS/IFU data -- get slitmask
            #Use new fdu.getSlitmask method
            fdu.printAllSlitmasks()
            slitmask = fdu.getSlitmask(pname=None, properties=properties, headerVals=headerVals)
            if (slitmask is None):
                print("resampleProcess::resampleData> ERROR: Could not find slitmask for object "+fdu.getFullId()+".  Cannot resample!")
                self._log.writeLog(__name__, "Could not find slitmask for object "+fdu.getFullId()+".  Cannot resample!", type=fatboyLog.ERROR)
                return False
            if (slitmask.hasProperty("is_resampled")):
                doSlitmask = False
            #Find nslits and regions
            if (not fdu.hasProperty("nslits")):
                fdu.setProperty("nslits", slitmask.getData().max())
            nslits = fdu.getProperty("nslits")
            if (fdu.hasProperty("regions")):
                (ylos_data, yhis_data, slitx_data, slitw_data) = fdu.getProperty("regions")
            else:
                #Use helper method to all ylo, yhi for each slit in each frame
                (ylos_data, yhis_data, slitx_data, slitw_data) = findRegions(slitmask.getData(), nslits, fdu, gpu=self._fdb.getGPUMode(), log=self._log)
                fdu.setProperty("regions", (ylos_data, yhis_data, slitx_data, slitw_data))

        if ('slitmask' in calibs and calibs['slitmask'].hasProperty("xout_data") and calibs['slitmask'].hasProperty("fdus_to_resample")):
            xout_data = calibs['slitmask'].getProperty("xout_data")
            resampHeader = calibs['slitmask'].getProperty("resampHeader")
            fdu.updateHeader(resampHeader) #Update header
        else:
            if (not hasWavelengthSolution(fdu)):
                print("resampleProcess::resampleData> ERROR: Could not find wavelength solution for object "+fdu.getFullId()+".  Cannot resample!")
                self._log.writeLog(__name__, "Could not find wavelength solution for object "+fdu.getFullId()+".  Cannot resample!", type=fatboyLog.ERROR)
                return False

            minLambda = None
            for j in range(nslits):
                wave = getWavelengthSolution(fdu, j, xsize)
                if (minLambda is None):
                    #First pass
                    minLambda = wave.min()
                    maxLambda = wave.max()
                else:
                    minLambda = min(minLambda, wave.min())
                    maxLambda = max(maxLambda, wave.max())

            xout_data = zeros(fdu.getShape(), dtype=float32)
            #Linear wavelength scale
            scale_data = (maxLambda-minLambda)/xsize
            xs_data = arange(xsize, dtype=float32)
            #New header keywords for resampled data
            resampHeader = dict()
            resampHeader['RESAMPLD'] = 'YES'
            resampHeader['CRVAL1'] = minLambda
            if (scale_data < 0):
                resampHeader['CRVAL1'] = maxLambda
            resampHeader['CDELT1'] = scale_data
            resampHeader['CRPIX1'] = 1
            if (fdu.hasHeaderValue('CRVAL1') and fdu.hasHeaderValue('CDELT1')):
                resampHeader['CRVAL1'] = fdu.getHeaderValue('CRVAL1')
                resampHeader['CDELT1'] = fdu.getHeaderValue('CDELT1')
                scale = fdu.getHeaderValue('CDELT1')
                scale_data = fdu.getHeaderValue('CDELT1')
            elif (fdu.hasProperty("resampledHeader")):
                resampHeader['CRVAL1'] = fdu.getProperty("resampledHeader")['CRVAL1']
                resampHeader['CDELT1'] = fdu.getProperty("resampledHeader")['CDELT1']
                scale = fdu.getProperty("resampledHeader")['CDELT1']
                scale_data = fdu.getProperty("resampledHeader")['CDELT1']

            #Find wavelengths corresponding to each pixel and set up output linear wavelength grid
            for j in range(nslits):
                lambdaIn_data = getWavelengthSolution(fdu, j, xsize)
                if (not resampleCommonScale):
                    #Use min/max wavelength from this input slitlet and segment
                    maxLambda = lambdaIn_data.max()
                    minLambda = lambdaIn_data.min()
                    scale_data = (maxLambda-minLambda)/(float)(xsize)

                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    if (slitmask is not None):
                        #MOS data - add to xout array, mutiply input by currMask
                        #Apply mask to slit - based on if individual slitlets are being calibrated or not
                        currMask = slitmask.getData()[int(ylos_data[j]):int(yhis_data[j]+1),:] == (j+1)
                        xout_data[int(ylos_data[j]):int(yhis_data[j]+1),:] += ((lambdaIn_data-minLambda)/scale_data)*currMask
                    else:
                        #Longslit data, calculate entire xout frame at once
                        xout_data[:,:] = (lambdaIn_data-minLambda)/scale_data
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    if (slitmask is not None):
                        #Use this FDU's slitmask to calculate currMask for xout_data
                        #Apply mask to slit - based on if individual slitlets are being calibrated or not
                        currMask = slitmask.getData()[:,int(ylos_data[j]):int(yhis_data[j]+1)] == (j+1)
                        xout_data[:,int(ylos_data[j]):int(yhis_data[j]+1)] += ((lambdaIn_data.reshape((len(lambdaIn_data), 1))-minLambda)/scale_data)*currMask
                    else:
                        #Longslit data, calculate entire xout frame at once
                        xout_data[:,:] = (lambdaIn_data.reshape((len(lambdaIn_data), 1))-minLambda)/scale

            #Update header as the data will be resampled
            fdu.updateHeader(resampHeader)

            #Tag slitmask if exists
            if ('slitmask' in calibs):
                calibs['slitmask'].setProperty("xout_data", xout_data)
                calibs['slitmask'].setProperty("resampHeader", resampHeader)

        #Select cpu/gpu option
        drihizzle_method = gpu_drihizzle.drihizzle
        if (not self._fdb.getGPUMode()):
            drihizzle_method = drihizzle.drihizzle

        properties["is_resampled"] = True

        #Resampled frames should be in cps not counts!
        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            #Use drihizzle to resample with xtrans=xout_data
            inMask = (xout_data != 0).astype(int32)

            #First update slitmask before anything else.  Use "uniform" kernel.
            if (slitmask is not None):
                (smData, header, expmap, pixmap) = drihizzle_method(slitmask, None, None, inmask=inMask, kernel="uniform", xtrans=xout_data, log=self._log, mode=gpu_drihizzle.MODE_FDU)
                smData = ascontiguousarray(smData[:ysize,:]) #Remove extraneous top row
                if (doSlitmask):
                    if ('slitmask' in calibs and not calibs['slitmask'].hasProperty("is_resampled")):
                        #calibs['slitmask'].setProperty("presampled_shape", calibs['slitmask'].getShape())
                        resampSlitmask = self._fdb.addNewSlitmask(calibs['slitmask'], smData, self._pname)
                        calibs['slitmask'].setProperty("is_resampled", True)
                        resampSlitmask.updateData(smData)
                        resampSlitmask.updateHeader(header)
                        resampSlitmask.setProperty("is_resampled", True)
                        #Write to disk if requested
                        if (writeCalibs):
                            rsfile = outdir+"/resampled/resamp_"+resampSlitmask.getFullId()
                            #Remove existing files if overwrite = yes
                            if (os.access(rsfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                                os.unlink(rsfile)
                            #Write to disk
                            if (not os.access(rsfile, os.F_OK)):
                                resampSlitmask.writeTo(rsfile)
                #Update "slitmask" data tag - this should be outside of doSlitmask and done for all FDUs
                slitmask.setProperty("is_resampled", True)
                fdu.setSlitmask(smData, pname=self._pname, properties=properties)

            #Look for "cleanSky" frame to resample
            if ('cleanSky' in calibs and not calibs['cleanSky'].hasProperty("is_resampled")):
                cleanSky = calibs['cleanSky']
                cleanSky.setProperty("presampled_shape", cleanSky.getShape())
                #Use turbo kernel for cleanSky
                (data, header, expmap, pixmap) = drihizzle_method(cleanSky, None, None, inmask=inMask, kernel="turbo", dropsize=1, xtrans=xout_data, inunits='counts', outunits=outunits, log=self._log, mode=gpu_drihizzle.MODE_FDU)
                data = ascontiguousarray(data[:ysize,:]) #Remove extraneous top row
                #update data, header, set "is_resampled" property
                cleanSky.updateData(data)
                cleanSky.updateHeader(header)
                cleanSky.setProperty("is_resampled", True)
                #Write to disk if requested
                if (writeCalibs):
                    rsfile = outdir+"/resampled/resamp_"+cleanSky.getFullId()
                    #Remove existing files if overwrite = yes
                    if (os.access(rsfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                        os.unlink(rsfile)
                    #Write to disk
                    if (not os.access(rsfile, os.F_OK)):
                        cleanSky.writeTo(rsfile)

            #Look for "masterLamp" frame to resample
            if ('masterLamp' in calibs and not calibs['masterLamp'].hasProperty("is_resampled")):
                masterLamp = calibs['masterLamp']
                masterLamp.setProperty("presampled_shape", masterLamp.getShape())
                #Use turbo kernel for masterLamp
                (data, header, expmap, pixmap) = drihizzle_method(masterLamp, None, None, inmask=inMask, kernel="turbo", dropsize=1, xtrans=xout_data, inunits='counts', outunits=outunits, log=self._log, mode=gpu_drihizzle.MODE_FDU)
                data = ascontiguousarray(data[:ysize,:]) #Remove extraneous top row
                #update data, header, set "is_resampled" property
                masterLamp.updateData(data)
                masterLamp.updateHeader(header)
                masterLamp.setProperty("is_resampled", True)
                #Write to disk if requested
                if (writeCalibs):
                    rsfile = outdir+"/resampled/resamp_"+masterLamp.getFullId()
                    #Remove existing files if overwrite = yes
                    if (os.access(rsfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                        os.unlink(rsfile)
                    #Write to disk
                    if (not os.access(rsfile, os.F_OK)):
                        masterLamp.writeTo(rsfile)

            #Look for "cleanFrame" to resample
            if (fdu.hasProperty("cleanFrame")):
                (cleanData, header, expmap, pixmap) = drihizzle_method(fdu, None, None, inmask=inMask, kernel="turbo", dropsize=1, xtrans=xout_data, inunits='counts', outunits=outunits, log=self._log, mode=gpu_drihizzle.MODE_FDU_TAG, dataTag="cleanFrame")
                cleanData = ascontiguousarray(cleanData[:ysize,:]) #Remove extraneous top row
                #Update "cleanFrame" data tag
                fdu.tagDataAs("cleanFrame", data=cleanData)

            #Resample noisemap for spectrocsopy data
            if (fdu.hasProperty("noisemap")):
                #Square data, resample, take sqare root
                nmData = fdu.getData(tag="noisemap")**2
                #Update data tag before passing to drihizzle
                fdu.tagDataAs("noisemap", nmData)
                (nmData, header, expmap, pixmap) = drihizzle_method(fdu, None, None, inmask=inMask, kernel="turbo", dropsize=1, xtrans=xout_data, inunits='counts', outunits='counts', log=self._log, mode=gpu_drihizzle.MODE_FDU_TAG, dataTag="noisemap")
                #Take sqrt then divide by expmap
                nmData = sqrt(nmData)
                expmap[expmap == 0] = 1
                nmData /= expmap
                nmData = ascontiguousarray(nmData[:ysize,:]) #Remove extraneous top row
                #Update "noisemap" data tag
                fdu.tagDataAs("noisemap", data=nmData)

            #Resample exposure map for spectrocsopy data
            if (fdu.hasProperty("exposure_map")):
                (expData, header, expmap, pixmap) = drihizzle_method(fdu, None, None, inmask=inMask, kernel="turbo", dropsize=1, xtrans=xout_data, inunits='counts', outunits=outunits, log=self._log, mode=gpu_drihizzle.MODE_FDU_TAG, dataTag="exposure_map")
                expData = ascontiguousarray(expData[:ysize,:]) #Remove extraneous top row
                #Update "exposure_map" data tag
                fdu.tagDataAs("exposure_map", data=sqrt(expData))

            #Resample real data last
            (resampData, header, expmap, pixmap) = drihizzle_method(fdu, None, None, inmask=inMask, kernel="turbo", dropsize=1, xtrans=xout_data, inunits="counts", outunits=outunits, log=self._log, mode=gpu_drihizzle.MODE_FDU)
            resampData = ascontiguousarray(resampData[:ysize,:]) #Remove extraneous top row
            #Update data
            fdu.updateData(resampData)
            fdu.setProperty("is_resampled", True)

        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            #ytrans = xout_data
            inMask = (xout_data != 0).astype(int32)

            #First update slitmask before anything else.  Use "uniform" kernel.
            if (slitmask is not None):
                (smData, header, expmap, pixmap) = drihizzle_method(slitmask, None, None, inmask=inMask, kernel="uniform", ytrans=xout_data, log=self._log, mode=gpu_drihizzle.MODE_FDU)
                smData = ascontiguousarray(smData[:,:ysize]) #Remove extraneous top row
                if (doSlitmask):
                    if ('slitmask' in calibs and not calibs['slitmask'].hasProperty("is_resampled")):
                        #calibs['slitmask'].setProperty("presampled_shape", calibs['slitmask'].getShape())
                        resampSlitmask = self._fdb.addNewSlitmask(calibs['slitmask'], smData, self._pname)
                        calibs['slitmask'].setProperty("is_resampled", True)
                        resampSlitmask.updateData(smData)
                        resampSlitmask.updateHeader(header)
                        resampSlitmask.setProperty("is_resampled", True)
                        #Write to disk if requested
                        if (writeCalibs):
                            rsfile = outdir+"/resampled/resamp_"+resampSlitmask.getFullId()
                            #Remove existing files if overwrite = yes
                            if (os.access(rsfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                                os.unlink(rsfile)
                            #Write to disk
                            if (not os.access(rsfile, os.F_OK)):
                                resampSlitmask.writeTo(rsfile)
                #Update "slitmask" data tag - this should be outside of doSlitmask and done for all FDUs
                slitmask.setProperty("is_resampled", True)
                fdu.setSlitmask(smData, pname=self._pname, properties=properties)

            #Look for "cleanSky" frame to resample
            if ('cleanSky' in calibs and not calibs['cleanSky'].hasProperty("is_resampled")):
                cleanSky = calibs['cleanSky']
                cleanSky.setProperty("presampled_shape", cleanSky.getShape())
                #Use turbo kernel for cleanSky
                (data, header, expmap, pixmap) = drihizzle_method(cleanSky, None, None, inmask=inMask, kernel="turbo", dropsize=1, ytrans=xout_data, inunits='counts', outunits=outunits, log=self._log, mode=gpu_drihizzle.MODE_FDU)
                data = ascontiguousarray(data[:,:ysize]) #Remove extraneous top row
                #update data, header, set "is_resampled" property
                cleanSky.updateData(data)
                cleanSky.updateHeader(header)
                cleanSky.setProperty("is_resampled", True)
                #Write to disk if requested
                if (writeCalibs):
                    rsfile = outdir+"/resampled/resamp_"+cleanSky.getFullId()
                    #Remove existing files if overwrite = yes
                    if (os.access(rsfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                        os.unlink(rsfile)
                    #Write to disk
                    if (not os.access(rsfile, os.F_OK)):
                        cleanSky.writeTo(rsfile)

            #Look for "masterLamp" frame to resample
            if ('masterLamp' in calibs and not calibs['masterLamp'].hasProperty("is_resampled")):
                masterLamp = calibs['masterLamp']
                masterLamp.setProperty("presampled_shape", masterLamp.getShape())
                #Use turbo kernel for masterLamp
                (data, header, expmap, pixmap) = drihizzle_method(masterLamp, None, None, inmask=inMask, kernel="turbo", dropsize=1, ytrans=xout_data, inunits='counts', outunits=outunits, log=self._log, mode=gpu_drihizzle.MODE_FDU)
                data = ascontiguousarray(data[:,:ysize]) #Remove extraneous top row
                #update data, header, set "is_resampled" property
                masterLamp.updateData(data)
                masterLamp.updateHeader(header)
                masterLamp.setProperty("is_resampled", True)
                #Write to disk if requested
                if (writeCalibs):
                    rsfile = outdir+"/resampled/resamp_"+masterLamp.getFullId()
                    #Remove existing files if overwrite = yes
                    if (os.access(rsfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                        os.unlink(rsfile)
                    #Write to disk
                    if (not os.access(rsfile, os.F_OK)):
                        masterLamp.writeTo(rsfile)

            #Look for "cleanFrame" to resample
            if (fdu.hasProperty("cleanFrame")):
                (cleanData, header, expmap, pixmap) = drihizzle_method(fdu, None, None, inmask=inMask, kernel="turbo", dropsize=1, ytrans=xout_data, inunits='counts', outunits=outunits, log=self._log, mode=gpu_drihizzle.MODE_FDU_TAG, dataTag="cleanFrame")
                cleanData = ascontiguousarray(cleanData[:,:ysize]) #Remove extraneous top row
                #Update "cleanFrame" data tag
                fdu.tagDataAs("cleanFrame", data=cleanData)

            #Resample noisemap for spectrocsopy data
            if (fdu.hasProperty("noisemap")):
                #Square data, resample, take sqare root
                nmData = fdu.getData(tag="noisemap")**2
                #Update data tag before passing to drihizzle
                fdu.tagDataAs("noisemap", nmData)
                (nmData, header, expmap, pixmap) = drihizzle_method(fdu, None, None, inmask=inMask, kernel="turbo", dropsize=1, ytrans=xout_data, inunits='counts', outunits=outunits, log=self._log, mode=gpu_drihizzle.MODE_FDU_TAG, dataTag="noisemap")
                nmData = ascontiguousarray(nmData[:,:ysize]) #Remove extraneous top row
                #Update "noisemap" data tag
                fdu.tagDataAs("noisemap", data=sqrt(nmData))
            #Resample exposure map for spectrocsopy data
            if (fdu.hasProperty("exposure_map")):
                (expData, header, expmap, pixmap) = drihizzle_method(fdu, None, None, inmask=inMask, kernel="turbo", dropsize=1, ytrans=xout_data, inunits='counts', outunits=outunits, log=self._log, mode=gpu_drihizzle.MODE_FDU_TAG, dataTag="exposure_map")
                expData = ascontiguousarray(expData[:,:ysize]) #Remove extraneous top row
                #Update "exposure_map" data tag
                fdu.tagDataAs("exposure_map", data=sqrt(expData))

            #Resample real data last
            (resampData, header, expmap, pixmap) = drihizzle_method(fdu, None, None, inmask=inMask, kernel="turbo", dropsize=1, ytrans=xout_data, inunits="counts", outunits=outunits, log=self._log, mode=gpu_drihizzle.MODE_FDU)
            resampData = ascontiguousarray(resampData[:,:ysize]) #Remove extraneous top row
            #Update data
            fdu.updateData(resampData)
            fdu.setProperty("is_resampled", True)

        #Free memory if this is last FDU
        if ('slitmask' in calibs and calibs['slitmask'].hasProperty("xout_data") and calibs['slitmask'].hasProperty("fdus_to_resample")):
            if (fdu.getFullId() in calibs['slitmask'].getProperty("fdus_to_resample")):
                calibs['slitmask'].getProperty("fdus_to_resample").remove(fdu.getFullId())
                if (len(calibs['slitmask'].getProperty("fdus_to_resample")) == 0):
                    calibs['slitmask'].removeProperty("fdus_to_resample")
                    calibs['slitmask'].removeProperty("xout_data")
                    calibs['slitmask'].removeProperty("resampHeader")
        return True
    #end resampleData

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("Resample")
        print(fdu._identFull)

        #Get original shape before checkOutputExists
        origShape = fdu.getShape()

        #Check if output exists first and update from disk
        rsfile = "resampled/resamp_"+fdu.getFullId()
        if (self.checkOutputExists(fdu, rsfile, headerTag="resampHeader")):
            #Also check if "cleanFrame" exists
            cleanfile = "resampled/clean_resamp_"+fdu.getFullId()
            self.checkOutputExists(fdu, cleanfile, tag="cleanFrame")
            #Also check if "noisemap" exists
            nmfile = "resampled/NM_resamp_"+fdu.getFullId()
            self.checkOutputExists(fdu, nmfile, tag="noisemap")
            #Also check if exposure map exists
            expfile = "resampled/exp_resamp_"+fdu.getFullId()
            self.checkOutputExists(fdu, expfile, tag="exposure_map")
            #Also check if slitmask
            smfile = "resampled/slitmask_resamp_"+fdu.getFullId()
            self.checkOutputExists(fdu, smfile, tag="slitmask")

            #Need to get calibration frames - cleanSky, masterLamp, and slitmask to update from disk too
            calibs = dict()
            headerVals = dict()
            headerVals['grism_keyword'] = fdu.grism
            properties = dict()
            properties['specmode'] = fdu.getProperty("specmode")
            properties['dispersion'] = fdu.getProperty("dispersion")
            if (not 'cleanSky' in calibs):
                #Check for an already created clean sky frame frame matching specmode/filter/grism/ident
                #cleanSky = self._fdb.getMasterCalib(ident = "cleanSky_"+fdu._id, filter=fdu.filter, obstype="master_clean_sky", properties=properties, headerVals=headerVals, tag=fdu.getTag())
                cleanSky = self._fdb.getMasterCalib(ident = "cleanSky_ds_"+fdu._id, filter=fdu.filter, section=fdu.section, obstype="master_clean_sky", properties=properties, headerVals=headerVals, tag=fdu.getTag())
                if (cleanSky is not None):
                    #add to calibs for rectification below
                    calibs['cleanSky'] = cleanSky

            if (not 'masterLamp' in calibs):
                #1) Check for an already created master arclamp frame matching specmode/filter/grism and TAGGED for this object
                masterLamp = self._fdb.getTaggedMasterCalib(ident=fdu._id, obstype="master_arclamp", filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals)
                if (masterLamp is None):
                    #2) Check for an already created master arclamp frame frame matching specmode/filter/grism
                    masterLamp = self._fdb.getMasterCalib(obstype="master_arclamp", filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals, tag=fdu.getTag())
                if (masterLamp is not None):
                    #add to calibs for rectification below
                    calibs['masterLamp'] = masterLamp

            if (not 'slitmask' in calibs):
                #Use new fdu.getSlitmask method
                slitmask = fdu.getSlitmask(pname=None, properties=properties, headerVals=headerVals)
                if (slitmask is None):
                    #Now check for one with origShape -- will be updated below
                    slitmask = fdu.getSlitmask(pname=None, shape=origShape, properties=properties, headerVals=headerVals)
                if (slitmask is not None):
                    #add to calibs for rectification below
                    calibs['slitmask'] = slitmask

            #Check for cleanSky and masterLamp frames to update from disk too
            if ('cleanSky' in calibs and not calibs['cleanSky'].hasProperty("is_resampled")):
                #Check if output exists
                cleanfile = "resampled/resamp_"+calibs['cleanSky']._id+".fits"
                if (self.checkOutputExists(calibs['cleanSky'], cleanfile)):
                    #output file already exists and overwrite = no.  Update data from disk and set "is_resampled" = True
                    calibs['cleanSky'].setProperty("is_resampled", True)

            if ('masterLamp' in calibs and not calibs['masterLamp'].hasProperty("is_resampled")):
                #Check if output exists first
                lampfile = "resampled/resamp_"+calibs['masterLamp'].getFullId()
                if (self.checkOutputExists(calibs['masterLamp'], lampfile)):
                    #output file already exists and overwrite = no.  Update data from disk and set "is_resampled" = True
                    calibs['masterLamp'].setProperty("is_resampled", True)

            if ('slitmask' in calibs and not calibs['slitmask'].hasProperty("is_resampled")):
                #Check if output exists first
                smfile = "resampled/resamp_"+calibs['slitmask'].getFullId()
                if (self.checkOutputExists(calibs['slitmask'], smfile)):
                    #Now get new slitmask with correct shape
                    calibs['slitmask'] = fdu.getSlitmask(pname=None, properties=properties, headerVals=headerVals)
                    #output file already exists and overwrite = no.  Update data from disk and set "is_resampled" = True
                    calibs['slitmask'].setProperty("is_resampled", True)

            return True

        #Call get calibs to return dict() of calibration frames.
        #For resampled, this dict should have cleanSky and/or masterLamp
        #and optionally slitmask if this is not a property of the FDU at this point.
        #These are obtained by tracing slitlets using the master flat
        calibs = self.getCalibs(fdu, prevProc)

        success = self.resampleData(fdu, calibs)
        return success
    #end execute

    ## OVERRIDE getCalibs
    def getCalibs(self, fdu, prevProc = None):
        calibs = dict()

        #Look for each calib passed from XML
        smfilename = self.getCalib("slitmask", fdu.getTag())
        if (smfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(smfilename, os.F_OK)):
                print("resampleProcess::getCalibs> Using slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Using slitmask "+smfilename+"...")
                calibs['slitmask'] = fatboySpecCalib(self._pname, "slitmask", fdu, filename=smfilename, log=self._log)
            else:
                print("resampleProcess::getCalibs> Warning: Could not find slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Could not find slitmask "+smfilename+"...", type=fatboyLog.WARNING)
        #Look for each master calib passed from XML
        csfilename = self.getCalib("master_clean_sky", fdu.getTag())
        if (csfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(csfilename, os.F_OK)):
                print("resampleProcess::getCalibs> Using master clean sky frame "+csfilename+"...")
                self._log.writeLog(__name__, "Using master clean sky frame "+csfilename+"...")
                calibs['cleanSky'] = fatboySpecCalib(self._pname, "master_clean_sky", fdu, filename=csfilename, log=self._log)
            else:
                print("resampleProcess::getCalibs> Warning: Could not find master clean sky frame "+csfilename+"...")
                self._log.writeLog(__name__, "Could not find master clean sky frame "+csfilename+"...", type=fatboyLog.WARNING)
        #Look for each master calib passed from XML
        mlfilename = self.getCalib("master_arclamp", fdu.getTag())
        if (mlfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(mlfilename, os.F_OK)):
                print("resampleProcess::getCalibs> Using master arclamp frame "+mlfilename+"...")
                self._log.writeLog(__name__, "Using master arclamp frame "+mlfilename+"...")
                calibs['masterLamp'] = fatboySpecCalib(self._pname, "master_arclamp", fdu, filename=mlfilename, log=self._log)
            else:
                print("resampleProcess::getCalibs> Warning: Could not find master arclamp frame "+mlfilename+"...")
                self._log.writeLog(__name__, "Could not find master arclamp frame "+mlfilename+"...", type=fatboyLog.WARNING)

        #Look for matching grism_keyword, specmode, and dispersion
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
                #if (cleanSky.hasProperty("presampled_shape")):
                #  skyShape = cleanSky.getProperty("presampled_shape")

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
                #if (masterLamp.hasProperty("presampled_shape")):
                #  skyShape = masterLamp.getProperty("presampled_shape")

        if (fdu._specmode != fdu.FDU_TYPE_LONGSLIT and not 'slitmask' in calibs):
            #Use new fdu.getSlitmask method
            fdu.printAllSlitmasks()
            slitmask = fdu.getSlitmask(pname=None, shape=skyShape, properties=properties, headerVals=headerVals)
            if (slitmask is not None):
                #Found slitmask
                calibs['slitmask'] = slitmask

        #Check for individual FDUs matching specmode/filter/grism/ident to shift and add
        #fdus can not be [] as it will always at least return the current FDU itself
        fdus = self._fdb.getSortedFDUs(ident = fdu._id, obstype=fatboyDataUnit.FDU_TYPE_OBJECT, filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals, tag=fdu.getTag())
        if (len(fdus) > 0):
            #Found other objects associated with this fdu. Recursively process
            print("resampleProcess::getCalibs> Recursivley processing other images for object "+fdu._id+"...")
            self._log.writeLog(__name__, "Recursivley processing other images for object "+fdu._id+"...")
            #First recursively process
            self.recursivelyExecute(fdus, prevProc)
            if ('slitmask' in calibs):
                fduIdList = []
                for j in range(len(fdus)):
                    if (fdus[j].inUse):
                        fduIdList.append(fdus[j].getFullId())
                calibs['slitmask'].setProperty("fdus_to_resample", fduIdList)

        return calibs
    #end getCalibs

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('output_units', 'cps')
        self._optioninfo.setdefault('output_units', 'counts | cps')
        self._options.setdefault('resample_calibs', 'yes')
        self._optioninfo.setdefault('resample_calibs', 'Resample slitmask, master lamp, and clean sky if they exist')
        self._options.setdefault('resample_to_common_scale', 'yes')
        self._optioninfo.setdefault('resample_to_common_scale', 'Resample all slitlets to common scale for MOS data.\nIf set to no, each slitlet will be resampled to an\nindependent linear scale.')
        self._options.setdefault('write_noisemaps', 'no')
    #end setDefaultOptions

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/resampled", os.F_OK)):
            os.mkdir(outdir+"/resampled",0o755)
        #Create output filename
        rsfile = outdir+"/resampled/resamp_"+fdu.getFullId()
        #Check to see if it exists
        if (os.access(rsfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(rsfile)
        if (not os.access(rsfile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(rsfile, headerExt=fdu.getProperty("resampHeader"))
        #Write out clean frame if it exists
        if (fdu.hasProperty("cleanFrame")):
            cleanfile = outdir+"/resampled/clean_resamp_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(cleanfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(cleanfile)
            if (not os.access(cleanfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.writeTo(cleanfile, tag="cleanFrame")
        #Write out exposure map if it exists
        if (fdu.hasProperty("exposure_map")):
            expfile = outdir+"/resampled/exp_resamp_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(expfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(expfile)
            if (not os.access(expfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.writeTo(expfile, tag="exposure_map", headerExt=fdu.getProperty("resampHeader"))
        #Write noisemap for spectrocsopy data if requested
        if (self.getOption("write_noisemaps", fdu.getTag()).lower() == "yes" and fdu.hasProperty("noisemap")):
            nmfile = outdir+"/resampled/NM_resamp_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(nmfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(nmfile)
            if (not os.access(nmfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.writeTo(nmfile, tag="noisemap", headerExt=fdu.getProperty("resampHeader"))
        #Write out slitmask if it exists
        if (fdu.hasProperty("slitmask")):
            smfile = outdir+"/resampled/slitmask_resamp_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(smfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(smfile)
            if (not os.access(smfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                #Use new fdu.getSlitmask to get FDU and then write
                fdu.getSlitmask().writeTo(smfile)
    #end writeOutput
