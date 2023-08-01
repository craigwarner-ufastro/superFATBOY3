from superFATBOY.fatboyCalib import fatboyCalib
from superFATBOY.fatboyDataUnit import fatboyDataUnit
from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY.fatboyProcess import fatboyProcess
hasCuda = True
hasSep = True
try:
    import superFATBOY
    if (not superFATBOY.gpuEnabled()):
        hasCuda = False
    else:
        import pycuda.driver as drv
        if (not superFATBOY.threaded()):
            #If not threaded mode, import autoinit.  Otherwise assume context exists.
            #Code will crash if in threaded mode and context does not exist.
            import pycuda.autoinit
        from pycuda.compiler import SourceModule
except Exception:
    print("skySubtractProcess> Warning: PyCUDA not installed")
    hasCuda = False
import math
from superFATBOY import gpu_imcombine, imcombine
from superFATBOY import gpu_pysurfit, pysurfit
from superFATBOY.gpu_arraymedian import *
try:
    import sep
except Exception:
    print("skySubtractProcess> Warning: sep not installed")
    hasSep = False

block_size = 512

class skySubtractProcess(fatboyProcess):
    _modeTags = ["imaging", "circe"]

    ssmethods = ["remove_objects", "rough", "offsource", "offsource_extended", "offsource_neb", "offsource_rough"]
    lastIdent = None #Track last identifier for onsource skies
    fduCount = 0 #Track count of fdus within this identifier
    identTotal = 0 #Track number of frames for this identifier

    #Override checkValidDatatype
    def checkValidDatatype(self, fdu):
        if (fdu.getObsType(True) == fatboyDataUnit.FDU_TYPE_OBJECT):
            #If sky subtract is done before flat divide, it will attempt to
            #recursively process flats.  Make sure it only tries to sky subtract objects
            return True
        return False
    #end checkValidDatatype

    def createRoughSkyAndSextractedImage(self, fdu, prevProc=None):
        sxtMethod = self.getOption("source_extract_method", fdu.getTag()).lower()
        if (sxtMethod == "sep" and not hasSep):
            print("skySubtractProcess::createRoughSkyAndSextractedImage> WARNING: sep is not installed.  Using sextractor for source extraction.")
            self._log.writeLog(__name__, "sep is not installed.  Using sextractor for source extraction.", type=fatboyLog.WARNING)
            sxtMethod = "sextractor"

        #Get skies for this FDU (even if it is the FDU being processed)
        skies = self.findOnsourceSkies(fdu)
        if (prevProc is not None):
            #First recursively process
            self.recursivelyExecute(skies, prevProc)

        #Select cpu/gpu option
        imcombine_method = gpu_imcombine.imcombine
        if (not self._fdb.getGPUMode()):
            imcombine_method = imcombine.imcombine

        #Simply imcombine sky files and scale by median
        (data, header) = imcombine_method(skies, outfile=None, method="median", scale="median", mef=skies[0]._mef, log=self._log, returnHeader=True, mode=gpu_imcombine.MODE_FDU_TAG, dataTag="preSkySubtraction")
        msname = "onsourceSkies/sky-rough-"+str(fdu.getFullId()[:-5]) #strip off .fits
        masterSky = fatboyCalib(self._pname, "master_sky", skies[0], data=data, tagname=msname, headerExt=header, log=self._log)
        fdu.tagDataAs("preSkySubtraction")

        #call skySubtractImage helper function to do gpu/cpu scale and subtract
        data = self.skySubtractImage(fdu.getData().copy(), masterSky.getData(), fdu.getMedian()/masterSky.getMedian())

        if (sxtMethod == "sep"):
            self.createSepImage(fdu, data) #call helper method to run sep
        else:
            #write file to disk and run sextractor
            tempSSfile = self._fdb._tempdir+"/TEMPss_"+fdu.getFullId() #sky subtracted version
            write_fits_file(tempSSfile, data, data.dtype, overwrite=True, log=self._log)
            self.createSextractedImage(fdu, tempSSfile) #call helper method to run sextractor
        del data
    #end createRoughSkyAndSextractedImage

    ##Helper method to run sextractor
    def createSextractedImage(self, fdu, tempSSfile=None):
        #Read options
        sxtPath = self.getOption("sextractor_path", fdu.getTag())
        twoPass = self.getOption("two_pass_object_masking", fdu.getTag()).lower()
        two_pass_detect_thresh = str(self.getOption('two_pass_detect_thresh', fdu.getTag()))
        two_pass_detect_minarea = str(self.getOption('two_pass_detect_minarea', fdu.getTag()))
        two_pass_boxcar_size = int(self.getOption('two_pass_boxcar_size', fdu.getTag()))
        two_pass_reject_level = float(self.getOption('two_pass_reject_level', fdu.getTag()))

        print("skySubtractProcess::createSextractedImage> Finding objects and creating object masks for "+fdu.getFullId())
        self._log.writeLog(__name__, "Finding objects and creating object masks for "+fdu.getFullId())

        wimage = self._fdb._tempdir+'/TEMPsxt_weight_image_'+fdu._id+'.fits'
        if (not os.access(wimage, os.F_OK)):
            #Only need to write weighting image once as BPM is shared by all frames
            goodPixelMask = (1-fdu.getBadPixelMask().getData()).astype("int32")
            write_fits_file(wimage, goodPixelMask, uint8, overwrite=True, log=self._log)

        #Run SExtractor on rough sky subtracted images to extract sources
        #Create object mask.  Treat as BPM and apply to flat divided images.
        #set sxt filename
        t = time.time()
        sxtfile = self._fdb._tempdir+"/TEMPsxt_"+fdu.getFullId()
        if (tempSSfile is None):
            #default file for running sextractor on.  Should have been created already
            tempSSfile = self._fdb._tempdir+"/TEMPss_"+fdu.getFullId() #sky subtracted version

        if (not os.access(tempSSfile, os.F_OK)):
            print("skySubtractProcess::createSextractedImage> ERROR: Could not find file "+tempSSfile+"! Failed to run sextractor!")
            self._log.writeLog(__name__, "Could not find file "+tempSSfile+"! Failed to run sextractor!", type=fatboyLog.ERROR)
            return

        if (os.access(sxtfile, os.F_OK)):
            os.unlink(sxtfile)
        #sextractor command
        if (not os.access(self._fdb._tempdir+'/default.sex', os.F_OK)):
            os.system(sxtPath+" -d > "+self._fdb._tempdir+"/default.sex")
        if (not os.access(self._fdb._tempdir+'/default.param', os.F_OK)):
            writeSexParam(self._fdb._tempdir+'/default.param')
        if (not os.access(self._fdb._tempdir+'/default.conv', os.F_OK)):
            writeSexConv(self._fdb._tempdir+'/default.conv')
        sxtCom = sxtPath+" " + tempSSfile + " -c "+self._fdb._tempdir+"/default.sex -CATALOG_TYPE NONE -PARAMETERS_NAME "+self._fdb._tempdir+"/default.param -FILTER_NAME "+self._fdb._tempdir+"/default.conv -DETECT_MINAREA 9 -CHECKIMAGE_TYPE SEGMENTATION -CHECKIMAGE_NAME " + sxtfile + " -WEIGHT_IMAGE " + wimage + " -WEIGHT_TYPE MAP_WEIGHT -VERBOSE_TYPE QUIET"
        print("\tSextracting "+tempSSfile+" with command "+sxtCom)
        self._log.writeLog(__name__, "Sextracting "+tempSSfile+" with command "+sxtCom, printCaller=False, tabLevel=1)
        #run sextractor
        os.system(sxtCom)

        #try opening output segmentation image file from sextractor as object mask
        objMask = None
        try:
            maskFile = pyfits.open(sxtfile)
            objmef = findMef(maskFile)
            objMask = maskFile[objmef].data
            maskFile.close()
        except Exception as ex:
            print(ex)
            print("skySubtractProcess::createSextractedImage> WARNING: sextractor was not successful.  Check that it is properly installed in "+sxtPath)
            print("\t\tAs a result, OBJECTS HAVE NOT BEEN REMOVED in second pass of sky subtraction for "+fdu.getFullId())
            self._log.writeLog(__name__, "sextractor was not successful.  Check that it is properly installed in "+sxtPath, type=fatboyLog.WARNING)
            self._log.writeLog(__name__, "As a result, OBJECTS HAVE NOT BEEN REMOVED in second pass of sky subtraction for "+fdu.getFullId(), printCaller=False, tabLevel=2)
            self._shortlog.writeLog(__name__, "Objects have not been removed in second pass of sky subtraction for "+fdu.getFullId(), type=fatboyLog.WARNING)
            objMask = zeros(fdu._shape, int32)
        #Apply object mask to image
        data = fdu.getData().copy()
        if (self._fdb.getGPUMode()):
            applyObjMask(data, objMask)
        else:
            data[objMask > 0] = 0
        #No need to write out data here, just keep in memory

        #Two pass object masking to grow mask around bright stars
        if (twoPass == "yes"):
            tpImage = self._fdb._tempdir+"/TEMP_2pass_sxt_"+fdu.getFullId()
            if (os.access(tpImage, os.F_OK)):
                os.unlink(tpImage)
            #sextractor command for 2nd pass
            sxtCom = sxtPath+" " + tempSSfile + " -c "+self._fdb._tempdir+"/default.sex -DETECT_MINAREA " + two_pass_detect_minarea + " -DETECT_THRESH " + two_pass_detect_thresh + " -CATALOG_TYPE NONE -PARAMETERS_NAME "+self._fdb._tempdir+"/default.param -FILTER_NAME "+self._fdb._tempdir+"/default.conv -CHECKIMAGE_TYPE SEGMENTATION -CHECKIMAGE_NAME " + tpImage + " -WEIGHT_IMAGE " + wimage + " -WEIGHT_TYPE MAP_WEIGHT -VERBOSE_TYPE QUIET"
            print("\tSextracting "+tempSSfile+" for two pass object masking with command "+sxtCom)
            self._log.writeLog(__name__, "Sextracting "+tempSSfile+" for two pass object masking with command "+sxtCom, printCaller=False, tabLevel=1)
            #run sextractor 2nd pass
            os.system(sxtCom)
            try:
                maskFile = pyfits.open(tpImage)
                objmef = findMef(maskFile)
                objMask = maskFile[objmef].data
                maskFile.close()
            except Exception:
                print("skySubtractProcess::createSextractedImage> WARNING: sextractor was not successful.  Check that it is properly installed in "+sxtPath)
                print("\t\tAs a result, OBJECTS HAVE NOT BEEN REMOVED in second pass of sky subtraction for "+fdu.getFullId())
                self._log.writeLog(__name__, "sextractor was not successful.  Check that it is properly installed in "+sxtPath, type=fatboyLog.WARNING)
                self._log.writeLog(__name__, "As a result, OBJECTS HAVE NOT BEEN REMOVED in second pass of sky subtraction for "+fdu.getFullId(), printCaller=False, tabLevel=2)
                self._shortlog.writeLog(__name__, "Objects have not been removed in second pass of sky subtraction for "+fdu.getFullId(), type=fatboyLog.WARNING)
                objMask = zeros(fdu._shape, int32)
            #Generate 2nd object mask, grow it with smoothing function, and apply mask to image
            #No need to read in from disk, data is still in memory, apply this mask too
            if (self._fdb.getGPUMode()):
                apply2PassObjMask(data, objMask, two_pass_boxcar_size, two_pass_reject_level)
            else:
                #Use CPU to grow and apply mask
                objMask[objMask > 0] = 1
                objMask = smooth_cpu(1-objMask, two_pass_boxcar_size, 1)
                objMask = where(objMask > two_pass_reject_level, 1, 0)
                data *= objMask

        fdu.tagDataAs("sextractedImage", data)
        fdu.writeTo(self._fdb._tempdir+'/sxtImage_'+fdu.getFullId(), tag="sextractedImage")
        self._fdb.totalSextractorTime += (time.time()-t)
        print("\tSextractor time: ", time.time()-t)
        self._log.writeLog(__name__, "Sextractor time: "+str(time.time()-t), printCaller=False, tabLevel=1)
    #end createSextractedImage

    ##Helper method to run sep
    def createSepImage(self, fdu, ssdata):
        #Read options
        sep_thresh = float(self.getOption("sep_detect_thresh", fdu.getTag()).lower())
        twoPass = self.getOption("two_pass_object_masking", fdu.getTag()).lower()
        two_pass_detect_thresh = int(self.getOption('two_pass_detect_thresh', fdu.getTag()))
        two_pass_detect_minarea = int(self.getOption('two_pass_detect_minarea', fdu.getTag()))
        two_pass_sep_ellipse_growth = float(self.getOption('two_pass_sep_ellipse_growth', fdu.getTag()))

        print("skySubtractProcess::createSepImage> Finding objects and creating object masks for "+fdu.getFullId())
        self._log.writeLog(__name__, "Finding objects and creating object masks for "+fdu.getFullId())

        #Run sep rough sky subtracted images to extract sources.  Sep uses BPM not GPM as input mask.
        #Create object mask.  Treat as BPM and apply to flat divided images.
        t = time.time()
        bkg = sep.Background(ssdata, mask=fdu.getBadPixelMask().getData().astype(bool))
        print("\tsep background = "+str(bkg.globalback)+" rms = "+str(bkg.globalrms))
        self._log.writeLog(__name__, "sep background = "+str(bkg.globalback)+" rms = "+str(bkg.globalrms), printCaller=False, tabLevel=1)
        thresh = sep_thresh*bkg.globalrms #default 1.5
        #subtract background from data
        bkg.subfrom(ssdata)
        #extract objects
        #Use try-except to catch any sep crashes
        try:
            objects = sep.extract(ssdata, thresh, minarea=9)
        except Exception as ex:
            #Sep crahsed.  Print notification and try again with higher thresh and minarea
            print("skySubtractProcess::createSepImage> Error running sep on "+fdu.getFullId()+": "+str(ex)+" Attempting to run with thresh="+str(2*thresh)+" and minarea 16.")
            self._log.writeLog(__name__, "Error running sep on "+fdu.getFullId()+": "+str(ex)+" Attempting to run with thresh="+str(2*thresh)+" and minarea 16.", type=fatboyLog.ERROR)
            try:
                objects = sep.extract(ssdata, 2*thresh, minarea=15)
            except Exception as ex:
                #Sep crashed again.  Return without masking objects out
                print("skySubtractProcess::createSepImage> ERROR: sep failed twice. Unable to mask objects for "+fdu.getFullId())
                self._log.writeLog(__name__, "sep failed twice. Unable to mask objects for "+fdu.getFullId(), type=fatboyLog.ERROR)
                #sepMask = all zero
                fdu.tagDataAs("sepMask", zeros(ssdata.shape, bool))
                #Tag current data as "sextractedImage"
                fdu.tagDataAs("sextractedImage")
                return
        print("\tsep extracted "+str(len(objects))+" objects")
        self._log.writeLog(__name__, "sep extracted "+str(len(objects))+" objects", printCaller=False, tabLevel=1)
        #Create object mask
        objMask = zeros(ssdata.shape, bool)
        #use r = 2.5 to grow ellipse because sep makes ellipses too small
        sep.mask_ellipse(objMask, objects['x'], objects['y'], objects['a'], objects['b'], objects['theta'], r=2.5)
        #Apply object mask to image
        data = fdu.getData().copy()
        if (self._fdb.getGPUMode()):
            applyObjMask(data, objMask)
        else:
            data[objMask > 0] = 0
        #Tag mask here to be reused in fitSkySubtractedSurf
        fdu.tagDataAs("sepMask", objMask)
        #No need to write out data here, just keep in memory

        #Two pass object masking to grow mask around bright stars
        if (twoPass == "yes"):
            thresh = two_pass_detect_thresh*bkg.globalrms
            ob2 = sep.extract(ssdata, thresh, minarea = two_pass_detect_minarea)
            print("\tsep pass 2 extracted "+str(len(ob2))+" objects")
            self._log.writeLog(__name__, "sep pass 2 extracted "+str(len(ob2))+" objects", printCaller=False, tabLevel=1)
            #Create object mask
            objMask = zeros(ssdata.shape, bool)
            #Generate 2nd object mask, grow it with sep instead of smoothing function, and apply mask to image
            #No need to read in from disk, data is still in memory, apply this mask too
            sep.mask_ellipse(objMask, ob2['x'], ob2['y'], ob2['a'], ob2['b'], ob2['theta'], r=2.5*two_pass_sep_ellipse_growth)
            if (self._fdb.getGPUMode()):
                applyObjMask(data, objMask)
            else:
                data[objMask > 0] = 0

        fdu.tagDataAs("sextractedImage", data)
        self._fdb.totalSextractorTime += (time.time()-t)
        print("\tSep time: ", time.time()-t)
        self._log.writeLog(__name__, "Sep time: "+str(time.time()-t), printCaller=False, tabLevel=1)
    #end createSepImage

    #Convenience method so that code doesn't have to be rewritten several times in getCalib
    #Propagates to different methods depending on sky_method
    def createMasterSky(self, fdu, skies, properties, prevProc=None):
        masterSky = None
        msfilename = None
        skymethod = properties['sky_method']
        if (fdu.hasProperty("sky_offsource_name")):
            properties['sky_offsource_name'] = fdu.getProperty("sky_offsource_name")

        #Check number of skies
        if (len(skies) == 1):
            print("skySubtractProcess::createMasterSky> Only 1 sky found!  Master sky not created for "+fdu.getFullId())
            self._log.writeLog(__name__, "Only 1 sky found!  Master sky not created for "+fdu.getFullId(), type=fatboyLog.ERROR)
            return None

        #Check to see if file already exists
        skyDir = "onsourceSkies"
        msname = skyDir+"/sky-"+str(skymethod)+"-"+str(fdu.getFullId()[:-5]) #strip off .fits
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (skymethod.startswith("offsource")):
            #use skies[0] for ident in case of offsource
            skyDir = "offsourceSkies"
            msname = skyDir+"/sky-"+str(skymethod)+"-"+str(skies[0].getFullId()[:-5]) #strip off .fits
            if ('sky_offsource_name' in properties):
                msname = properties['sky_offsource_name']
        elif ('sky_selected_name' in properties):
            msname = properties['sky_selected_name']
        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            #Optionally save if write_calib_output = yes
            if (not os.access(outdir+"/"+skyDir, os.F_OK)):
                os.mkdir(outdir+"/"+skyDir,0o755)
            msfilename = outdir+"/"+msname+".fits"
        #Check to see if master sky exists already from a previous run
        prevmsfilename = outdir+"/"+msname+".fits"
        if (os.access(prevmsfilename, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(prevmsfilename)
        elif (os.access(prevmsfilename, os.F_OK)):
            #file already exists
            print("skySubtractProcess::createMasterSky> Master sky "+prevmsfilename+" already exists!  Re-using...")
            self._log.writeLog(__name__, "Master sky "+prevmsfilename+" already exists!  Re-using...")
            masterSky = fatboyCalib(self._pname, "master_sky", skies[0], filename=prevmsfilename, log=self._log)
            if (skymethod.startswith("offsource")):
                #disable these skies as master sky has been created
                for skyfdu in skies:
                    skyfdu.disable()
            return masterSky

        if (skymethod == "rough"):
            masterSky = self.createMasterSkyRough(fdu, skies, msname, msfilename)
        elif (skymethod == "remove_objects"):
            masterSky = self.createMasterSkyRemoveObjects(fdu, skies, msname, msfilename, prevProc = prevProc)
        elif (skymethod.startswith("offsource")):
            masterSky = self.createMasterSkyOffsource(fdu, skies, skymethod, msname, msfilename)
        else:
            print("skySubtractProcess::createMasterSky> Error: invalid sky_method: "+skymethod)
            self._log.writeLog(__name__, " invalid sky_method: "+skymethod, type=fatboyLog.ERROR)
            return None

        #set sky_method property
        masterSky.setProperty("sky_method", skymethod)
        if ('sky_offsource_name' in properties):
            masterSky.setProperty('sky_offsource_name', properties['sky_offsource_name'])
        if ('sky_name' in properties):
            masterSky.setProperty('sky_name', properties['sky_name'])
        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes" and not os.access(msfilename, os.F_OK)):
            #Optionally save if write_calib_output = yes
            masterSky.writeTo(msfilename)
        return masterSky
    #end createMasterSky

    def createMasterSkyRough(self, fdu, skies, msname, msfilename):
        #Select cpu/gpu option
        imcombine_method = gpu_imcombine.imcombine
        if (not self._fdb.getGPUMode()):
            imcombine_method = imcombine.imcombine

        #Simply imcombine sky files and scale by median
        (data, header) = imcombine_method(skies, outfile=msfilename, method="median", scale="median", mef=skies[0]._mef, log=self._log, returnHeader=True, mode=gpu_imcombine.MODE_FDU_TAG, dataTag="preSkySubtraction")
        #(data, header) = imcombine_method(skies, outfile=msfilename, method="median", mef=skies[0]._mef, log=self._log, returnHeader=True, mode=gpu_imcombine.MODE_FDU_TAG, dataTag="preSkySubtraction")
        masterSky = fatboyCalib(self._pname, "master_sky", skies[0], data=data, tagname=msname, headerExt=header, log=self._log)
        fdu.tagDataAs("preSkySubtraction")
        del data
        del header
        return masterSky
    #end createMasterSkyRough

    def createMasterSkyRemoveObjects(self, fdu, skies, msname, msfilename, prevProc=None):
        sky_reject_type = str(self.getOption("sky_reject_type", fdu.getTag()))
        sky_nlow = int(self.getOption("sky_nlow", fdu.getTag()))
        sky_nhigh = int(self.getOption("sky_nhigh", fdu.getTag()))
        sky_lsigma = int(self.getOption("sky_lsigma", fdu.getTag()))
        sky_hsigma = int(self.getOption("sky_hsigma", fdu.getTag()))

        #Select cpu/gpu option
        imcombine_method = gpu_imcombine.imcombine
        if (not self._fdb.getGPUMode()):
            imcombine_method = imcombine.imcombine

        #if this image hasn't been rough sky subtracted and sextracted, create rough sky now
        if (not fdu.hasProperty("sextractedImage")):
            self.createRoughSkyAndSextractedImage(fdu, prevProc=prevProc)

        for skyfdu in skies:
            if (not skyfdu.hasProperty("sextractedImage")):
                self.createRoughSkyAndSextractedImage(skyfdu, prevProc=prevProc)

        #Pass 2: create master skies with objects removed
        #Combine skies with imcombine.  Scale each sky by the reciprocal of its median and then median combine them.
        (data, header) = imcombine_method(skies, msfilename, method="median", scale="median", reject=sky_reject_type, nlow=sky_nlow, nhigh=sky_nhigh, lsigma=sky_lsigma, hsigma=sky_hsigma, nonzero=True, log=self._log, returnHeader=True, mode=gpu_imcombine.MODE_FDU_TAG, dataTag="sextractedImage")
        masterSky = fatboyCalib(self._pname, "master_sky", skies[0], data=data, tagname=msname, headerExt=header, log=self._log)

        #Interpolate over holes in master sky
        if (self.getOption('interp_zeros_sky', fdu.getTag()).lower() == 'yes'):
            goodPixelMask = (1-fdu.getBadPixelMask().getData()).astype("int32")
            if (self._fdb.getGPUMode()):
                masterSky.updateData(linterp_gpu(masterSky.getData(), 0, goodPixelMask, log=self._log))
            else:
                masterSky.updateData(linterp_cpu(masterSky.getData(), 0, goodPixelMask, log=self._log))
            del goodPixelMask
        return masterSky
    #end createMasterSkyRemoveObjects

    def createMasterSkyOffsource(self, fdu, skies, skymethod, msname, msfilename):
        #read options
        sky_reject_type = str(self.getOption("sky_reject_type", fdu.getTag()))
        sky_nlow = int(self.getOption("sky_nlow", fdu.getTag()))
        sky_nhigh = int(self.getOption("sky_nhigh", fdu.getTag()))
        sky_lsigma = int(self.getOption("sky_lsigma", fdu.getTag()))
        sky_hsigma = int(self.getOption("sky_hsigma", fdu.getTag()))
        sxtMethod = self.getOption("source_extract_method", fdu.getTag()).lower()
        if (sxtMethod == "sep" and not hasSep):
            print("skySubtractProcess::createMasterSkyOffsource> WARNING: sep is not installed.  Using sextractor for source extraction.")
            self._log.writeLog(__name__, "sep is not installed.  Using sextractor for source extraction.", type=fatboyLog.WARNING)
            sxtMethod = "sextractor"

        #Select cpu/gpu option
        imcombine_method = gpu_imcombine.imcombine
        if (not self._fdb.getGPUMode()):
            imcombine_method = imcombine.imcombine

        if (skymethod == "offsource_rough"):
            #Write sky to disk if requested for offsource_rough and return here
            (data, header) = imcombine_method(skies, outfile=msfilename, method="median", scale="median", mef=skies[0]._mef, log=self._log, returnHeader=True)
            masterSky = fatboyCalib(self._pname, "master_sky", skies[0], data=data, tagname=msname, headerExt=header, log=self._log)
            #disable these skies as master sky has been created
            for skyfdu in skies:
                skyfdu.disable()
            return masterSky

        #All other cases - first pass master sky created by simply imcombining sky files and scaling by median
        #Do not write this first pass sky to disk
        (data, header) = imcombine_method(skies, outfile=None, method="median", scale="median", mef=skies[0]._mef, log=self._log, returnHeader=True)
        masterSky = fatboyCalib(self._pname, "master_sky", skies[0], data=data, tagname=msname, headerExt=header, log=self._log)

        #Run SExtractor on rough sky subtracted images to extract sources
        #Create object mask.  Treat as BPM and apply to flat divided images.
        for skyfdu in skies:
            #set sxt filename
            t = time.time()
            #call skySubtractImage helper function to do gpu/cpu scale and subtract
            data = self.skySubtractImage(skyfdu.getData().copy(), masterSky.getData(), skyfdu.getMedian()/masterSky.getMedian())

            if (sxtMethod == "sep"):
                self.createSepImage(skyfdu, data) #call helper method to run sep and create sextractedImage for each
            else:
                #write file to disk and run sextractor
                tempSSfile = self._fdb._tempdir+"/TEMPss_"+skyfdu.getFullId() #sky subtracted version
                write_fits_file(tempSSfile, data, data.dtype, overwrite=True, log=self._log)
                self.createSextractedImage(skyfdu, tempSSfile) #call helper method to run sextractor and create sextractedImage for each
        #endfor

        #Pass 2: create master off skies with objects removed
        #Combine skies with imcombine.  Scale each sky by the reciprocal of its median and then median combine them.
        (data, header) = imcombine_method(skies, msfilename, method="median", scale="median", reject=sky_reject_type, nlow=sky_nlow, nhigh=sky_nhigh, lsigma=sky_lsigma, hsigma=sky_hsigma, nonzero=True, log=self._log, returnHeader=True, mode=gpu_imcombine.MODE_FDU_TAG, dataTag="sextractedImage")
        masterSky = fatboyCalib(self._pname, "master_sky", skies[0], data=data, tagname=msname, headerExt=header, log=self._log)

        #Interpolate over holes in master sky
        if (self.getOption('interp_zeros_sky', fdu.getTag()).lower() == 'yes'):
            goodPixelMask = (1-fdu.getBadPixelMask().getData()).astype("int32")
            if (self._fdb.getGPUMode()):
                masterSky.updateData(linterp_gpu(masterSky.getData(), 0, goodPixelMask, log=self._log))
            else:
                masterSky.updateData(linterp_cpu(masterSky.getData(), 0, goodPixelMask, log=self._log))
            del goodPixelMask

        #disable these skies as master sky has been created
        for skyfdu in skies:
            skyfdu.disable()
        if (skymethod == "offsource_neb"):
            for skyfdu in skies:
                skyfdu.setProperty("sky_method", "offsource_neb")
        return masterSky
    #end createMasterSkyOffsource

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("Sky Subtract")
        print(fdu._identFull)

        #Call get calibs to return dict() of calibration frames.
        #For skySubtract, this dict should have one entry 'masterSky' which is an fdu
        calibs = self.getCalibs(fdu, prevProc)
        if ('skipFrame' in calibs):
            #This image is an offsource sky.  Return false and skip to next object.  Do not disable yet as it will be used in creating offsource sky.
            return False
        if (not 'masterSky' in calibs):
            #Failed to obtain master sky frame
            #Issue error message and disable this FDU
            print("skySubtractProcess::execute> ERROR: Sky not subtracted for "+fdu.getFullId()+" (filter="+str(fdu.filter)+").  Discarding Image!")
            self._log.writeLog(__name__, "Sky not subtracted for "+fdu.getFullId()+" (filter="+str(fdu.filter)+").  Discarding Image!", type=fatboyLog.ERROR)
            #disable this FDU
            fdu.disable()
            return False

        #Check if output exists first
        ssfile = "skySubtracted/ss_"+fdu.getFullId()
        if (self.checkOutputExists(fdu, ssfile)):
            return True

        #get master sky
        masterSky = calibs['masterSky']
        #determine scaling based on sky subtract method
        method = fdu.getProperty("sky_method")
        if (method == 'offsource_extended'):
            immed = self.extObjSky(fdu.getData(), masterSky.getData())
        elif (method == 'offsource_neb'):
            immed = self.nebObjSky(fdu)
            if (immed == 0):
                #Could not find skies associated with this FDU
                print("skySubtractProcess::execute> WARNING: Could not find skies for offsource_neb method for "+fdu.getFullId()+". Using offsource method instead.")
                self._log.writeLog(__name__, " Could not find skies for offsource_neb method for "+fdu.getFullId()+". Using offsource method instead.", type=fatboyLog.WARNING)
                immed = fdu.getMedian()
        else:
            immed = fdu.getMedian()
        scale = immed / masterSky.getMedian()
        #call skySubtractImage helper function to do gpu/cpu scale and subtract
        fdu.updateData(self.skySubtractImage(fdu.getData().copy(), masterSky.getData(), scale))
        fdu._header.add_history('Sky subtracted using '+masterSky.filename)

        if (self.getOption('fit_sky_subtracted_surf', fdu.getTag()).lower() == 'yes'):
            #call helper method to fit sky subtracted surface to this FDU and updateData
            self.fitSkySubtractedSurf(fdu)

        if (fdu._id == self.lastIdent):
            self.fduCount += 1
            if (self.fduCount == self.identTotal):
                #get FDUs matching this identifier and filter, sorted by index
                fduList = self._fdb.getSortedFDUs(ident = fdu._id, filter=fdu.filter, section=fdu.section)
                #Loop over fduList and remove preSkySubtraction and sextractedImage data tags to free memory
                for image in fduList:
                    image.removeProperty("preSkySubtraction")
                    image.removeProperty("sextractedImage")
                    image.removeProperty("sepMask")
        else:
            #first FDU in new identifier
            self.lastIdent = fdu._id
            self.fduCount = 1
            self.identTotal = len(self._fdb.getFDUs(ident = fdu._id, filter=fdu.filter, section=fdu.section))
        return True
    #end execute

    def extObjSky(self, image, msky):
        #Find median of master sky and image, scale and subtract sky.
        kernel = fatboyclib.median
        if (self._fdb.getGPUMode()):
            #Use GPU for medians
            kernel=self._fdb.getParam('median_kernel')

        msmed = gpu_arraymedian(msky, nonzero=True, kernel=kernel)
        immed = gpu_arraymedian(image, nonzero=True, kernel=kernel)
        ss_image = image-(immed/msmed)*msky

        #Create array of 100 stddevs of subimages
        xsize = int(ss_image.shape[0]//10)
        ysize = int(ss_image.shape[1]//10)
        sds = zeros(100)
        for j in range(10):
            for l in range(10):
                temp = ss_image[j*xsize:(j+1)*xsize,l*ysize:(l+1)*ysize]
                if len(temp[where(temp != 0)] != 0):
                    sds[j*10+l] = temp[where(temp != 0)].std()
                else:
                    sds[j*10+l] = 0.
        stddev = gpu_arraymedian(sds, kernel=kernel)

        good = logical_and(ss_image < 5*stddev, ss_image != 0)
        immed = gpu_arraymedian(image[good], kernel=kernel)
        change = 0
        while (change < 0.99 or change > 1.01):
            oldGood = good
            stddev = ss_image[where(ss_image != 0)].std()
            ss_image = image-(immed/msmed)*msky
            good = logical_and(ss_image < 5*stddev, ss_image != 0)
            immed = gpu_arraymedian(image[good], kernel=kernel)
            change = image[oldGood].size/float(image[good].size)
        return immed
    #end extObjSky

    #Find individual sky frames for this object to create master sky
    def findSkies(self, fdu, properties):
        skymethod = properties['sky_method'].lower()
        if (skymethod.startswith("offsource")):
            skies = self.findOffsourceSkies(fdu, properties)
        else:
            skies = self.findOnsourceSkies(fdu, properties)
        return skies
    #end findSkies

    #find offsource skies
    def findOffsourceSkies(self, fdu, properties):
        if (fdu.hasProperty("sky_offsource_name")):
            properties["sky_offsource_name"] = fdu.getProperty("sky_offsource_name")
        #1) Check for individual sky frames matching filter/section/sky_subtract_method and possibly sky_offsource_name to create master sky
        skies = self._fdb.getCalibs(obstype=fatboyDataUnit.FDU_TYPE_SKY, filter=fdu.filter, section=fdu.section, tag=fdu.getTag(), properties=properties)
        if (len(skies) > 0):
            #Found skies associated with this fdu
            return skies
        #2) Check sky_offsource_method
        skyOffMethod = self.getOption("sky_offsource_method", fdu.getTag())
        if (skyOffMethod.lower() == "auto"):
            #get FDUs matching this identifier and filter, sorted by index
            #if property sky_offsource_name not set, identify first object as onsource and objects at least sky_offsource_range away as offsource
            sky_offsource_range = float(self.getOption('sky_offsource_range', fdu.getTag()))/3600.
            for skyfdu in self._fdb.getSortedFDUs(ident = fdu._id, filter=fdu.filter, section=fdu.section):
                if (skyfdu.hasProperty("sky_offsource_name")):
                    #this FDU already has sky_offsource_name set
                    continue
                skyfdu.setProperty("sky_offsource_name", "offsourceSkies/sky_"+fdu._id+"_"+fdu.filter)
                #RA difference from object fdu
                diffRA = abs((skyfdu.ra - fdu.ra)*math.cos(fdu.dec*math.pi/180))
                #Dec difference from object fdu
                diffDec = abs(skyfdu.dec - fdu.dec)
                if (diffRA >= sky_offsource_range or diffDec >= sky_offsource_range):
                    #This is an offsource sky
                    skyfdu.setType(fatboyDataUnit.FDU_TYPE_SKY)
        elif (os.access(skyOffMethod, os.F_OK)):
            #This is an ASCII file listing identifier_object start_index stop_index identifier_sky start_index stop_index
            #Process entire file here
            methodList = readFileIntoList(skyOffMethod)
            #loop over methodList do a split on each line
            for j in range(len(methodList)-1, -1, -1):
                methodList[j] = methodList[j].split()
                #remove misformatted lines
                if (len(methodList[j]) != 6):
                    print("skySubtractProcess::findOffsourceSkies> Warning: line "+str(j)+" misformatted in "+skyOffMethod)
                    self._log.writeLog(__name__, " line "+str(j)+" misformatted in "+skyOffMethod, type=fatboyLog.WARNING)
                    methodList.pop(j)
                    continue
                try:
                    methodList[j][1] = int(methodList[j][1])
                    methodList[j][2] = int(methodList[j][2])
                    methodList[j][4] = int(methodList[j][4])
                    methodList[j][5] = int(methodList[j][5])
                except Exception as ex:
                    print("skySubtractProcess::findOffsourceSkies> Warning: line "+str(j)+" misformatted in "+skyOffMethod)
                    self._log.writeLog(__name__, " line "+str(j)+" misformatted in "+skyOffMethod, type=fatboyLog.WARNING)
                    methodList.pop(j)
                    continue
            #loop over dataset and assign property to all fdus that don't already have 'sky_offsource_name' property.
            #some FDUs may have used xml to define sky_offsource_name already
            #if tag is None, this loops over all FDUs
            for skyfdu in self._fdb.getFDUs(tag=fdu.getTag()):
                if (skyfdu.hasProperty("sky_offsource_name")):
                    #this FDU already has sky_offsource_name set
                    continue
                i = 0 #line number
                #method = [ 'identifier', start_idx, end_idx, 'sky_identifier', start_idx, end_idx ]
                for method in methodList:
                    i += 1
                    skypfix = method[3]+"_"+str(i)
                    if (skyfdu._id == method[0] and int(skyfdu._index) >= method[1] and int(skyfdu._index) <= method[2]):
                        #exact match. this is an object, set sky_offsource_name property
                        skyfdu.setProperty("sky_offsource_name", "offsourceSkies/sky_"+skypfix+"_"+skyfdu.filter)
                    elif (skyfdu._id == method[3] and int(skyfdu._index) >= method[4] and int(skyfdu._index) <= method[5]):
                        #exact match. this is a sky.  set sky_offsource_name property and set type to SKY
                        skyfdu.setProperty("sky_offsource_name", "offsourceSkies/sky_"+skypfix+"_"+skyfdu.filter)
                        skyfdu.setType(fatboyDataUnit.FDU_TYPE_SKY)
                    elif (skyfdu._id.find(method[0]) != -1 and int(skyfdu._index) >= method[1] and int(skyfdu._index) <= method[2]):
                        #partial match. this is an object, set sky_offsource_name property
                        skyfdu.setProperty("sky_offsource_name", "offsourceSkies/sky_"+skypfix+"_"+skyfdu.filter)
                    elif (skyfdu._id.find(method[3]) != -1 and int(skyfdu._index) >= method[4] and int(skyfdu._index) <= method[5]):
                        #partial match. this is a sky.  set sky_offsource_name property and set type to SKY
                        skyfdu.setProperty("sky_offsource_name", "offsourceSkies/sky_"+skypfix+"_"+skyfdu.filter)
                        skyfdu.setType(fatboyDataUnit.FDU_TYPE_SKY)
        else:
            print("skySubtractProcess::findOffsourceSkies> Error: invalid sky_offsource_method: "+skyOffMethod)
            self._log.writeLog(__name__, " invalid sky_offsource_method: "+skyOffMethod, type=fatboyLog.ERROR)
            return skies
        #Now Check for individual sky frames matching filter/section/sky_subtract_method/sky_offsource_name to create master sky
        properties['sky_offsource_name'] = fdu.getProperty('sky_offsource_name')
        skies = self._fdb.getCalibs(obstype=fatboyDataUnit.FDU_TYPE_SKY, filter=fdu.filter, section=fdu.section, tag=fdu.getTag(), properties=properties)
        return skies
    #end findOffsourceSkies

    #find onsource skies
    def findOnsourceSkies(self, fdu, properties=dict()):
        skies = []
        use_sky_files = self.getOption('use_sky_files', fdu.getTag()).lower()
        sky_files_range = int(self.getOption('sky_files_range', fdu.getTag()))
        #convert to arcsec
        sky_dithering_range = float(self.getOption('sky_dithering_range', fdu.getTag()))/3600.
        #full | index | FITS keyword
        sort_key = self.getOption('onsource_sorting_key', fdu.getTag())
        conserveMem = False
        if (self.getOption('conserve_memory', fdu.getTag()).lower() == 'yes'):
            conserveMem = True

        #get FDUs matching this identifier and filter, sorted by index
        skyfdus = self._fdb.getSortedFDUs(ident = fdu._id, filter=fdu.filter, section=fdu.section, sortby=sort_key)
        #Find index of current FDU
        idx = skyfdus.index(fdu)

        #For data where a series of exposures is taken at each point in dither pattern, identify index within series
        currIdx = 0
        if (not skyfdus[0].hasProperty('series_index')):
            skyfdus[0].setProperty('series_index', 0)
        for j in range(1, len(skyfdus)):
            if (not skyfdus[j].hasProperty('series_index')):
                if (skyfdus[j].ra == skyfdus[j-1].ra and skyfdus[j].dec == skyfdus[j-1].dec):
                    currIdx += 1
                else:
                    #Reset index on dither
                    currIdx = 0
                skyfdus[j].setProperty('series_index', currIdx)

        #Find skies at different dither positions before target fdu
        beforefdus = []
        lastRA = 0
        lastDec = 0
        #Loop backwards from idx-1 to 0, inclusively
        for j in range(idx-1, -1, -1):
            #RA difference from current fdu and last RA position
            diffRA = abs((skyfdus[j].ra - fdu.ra)*math.cos(fdu.dec*math.pi/180))
            diffLastRA = abs((skyfdus[j].ra - lastRA)*math.cos(lastDec*math.pi/180))
            #Dec difference from current fdu and last Dec position
            diffDec = abs(skyfdus[j].dec - fdu.dec)
            diffLastDec = abs(skyfdus[j].dec - lastDec)
            #Add object to list if it is at least sky_dithering_range arcsec away in either direction from both target and last fdu
            if ((diffRA >= sky_dithering_range or diffDec >= sky_dithering_range) and (diffLastRA >= sky_dithering_range or diffLastDec >= sky_dithering_range)):
                lastRA = skyfdus[j].ra
                lastDec = skyfdus[j].dec
                beforefdus.append(skyfdus[j])
            elif ((diffRA >= sky_dithering_range or diffDec >= sky_dithering_range) and skyfdus[j].ra == lastRA and skyfdus[j].dec == lastDec):
                #This object is at least sky_dithering_range arcsec away from target and in same position as last fdu.
                #Check its index within this exposure series and if it matches target's series index, replace previous matching fdu with this one.
                #E.g. If 5 observes are done per position, 2, 7, and 12 will match for image.0017.fits instead of 15, 10, and 5.
                if (skyfdus[j].getProperty('series_index') == fdu.getProperty('series_index')):
                    beforefdus.pop(-1)
                    beforefdus.append(skyfdus[j])

        #Find skies at different dither positions after target fdu
        afterfdus = []
        lastRA = 0
        lastDec = 0
        #Loop backwards from idx-1 to 0, inclusively
        for j in range(idx+1, len(skyfdus)):
            #RA difference from current fdu and last RA position
            diffRA = abs((skyfdus[j].ra - fdu.ra)*math.cos(fdu.dec*math.pi/180))
            diffLastRA = abs((skyfdus[j].ra - lastRA)*math.cos(lastDec*math.pi/180))
            #Dec difference from current fdu and last Dec position
            diffDec = abs(skyfdus[j].dec - fdu.dec)
            diffLastDec = abs(skyfdus[j].dec - lastDec)
            #Add object to list if it is at least sky_dithering_range arcsec away in either direction from both target and last fdu
            if ((diffRA >= sky_dithering_range or diffDec >= sky_dithering_range) and (diffLastRA >= sky_dithering_range or diffLastDec >= sky_dithering_range)):
                lastRA = skyfdus[j].ra
                lastDec = skyfdus[j].dec
                afterfdus.append(skyfdus[j])
            elif ((diffRA >= sky_dithering_range or diffDec >= sky_dithering_range) and skyfdus[j].ra == lastRA and skyfdus[j].dec == lastDec):
                #This object is at least sky_dithering_range arcsec away from target and in same position as last fdu.
                #Check its index within this exposure series and if it matches target's series index, replace previous matching fdu with this one.
                #E.g. If 5 observes are done per position, 2, 7, and 12 will match for image.0017.fits instead of 15, 10, and 5.
                if (skyfdus[j].getProperty('series_index') == fdu.getProperty('series_index')):
                    afterfdus.pop(-1)
                    afterfdus.append(skyfdus[j])

        if (use_sky_files == "all"):
            #Use all matching skies
            skies = beforefdus
            skies.extend(afterfdus)
        elif (use_sky_files == "range"):
            #Use a range of sky_files_range before and after but attempt to get 2*sky_files_range total.
            nbefore = min(sky_files_range, len(beforefdus))
            nafter = min(sky_files_range, len(afterfdus))
            if (nbefore < sky_files_range):
                nafter = min(2*sky_files_range - nbefore, len(afterfdus))
            elif (nafter < sky_files_range):
                nbefore = min(2*sky_files_range - nafter, len(beforefdus))
            if (nbefore+nafter < 2*sky_files_range):
                #Issue warning
                print("skySubtractProcess::findOnsourceSkies> Warning: Only able to use "+str(nbefore+nafter)+" files for sky for "+fdu.getFullId())
                self._log.writeLog(__name__, "Only able to use "+str(nbefore+nafter)+" files for sky for "+fdu.getFullId(), type=fatboyLog.WARNING)
            skies = beforefdus[:nbefore]
            skies.extend(afterfdus[:nafter])
            if (conserveMem):
                for j in range(nbefore, len(beforefdus)):
                    beforefdus[j].removeProperty("preSkySubtraction")
                    beforefdus[j].removeProperty("sextractedImage")
                    beforefdus[j].removeProperty("sepMask")
        elif (use_sky_files == "selected"):
            if (fdu.hasProperty("sky_offsource_name")):
                properties["sky_offsource_name"] = fdu.getProperty("sky_offsource_name")
            selectedSkies = str(self.getOption("selected_skies", fdu.getTag()))
            if (selectedSkies is None or not os.access(selectedSkies, os.F_OK)):
                #Issue error and return
                print("skySubtractProcess::findOnsourceSkies> ERROR: Could not find selected skies file "+str(selectedSkies)+" for "+fdu.getFullId())
                self._log.writeLog(__name__, "Could not find selected skies file "+str(selectedSkies)+" for "+fdu.getFullId(), type=fatboyLog.ERROR)
                return skies
            #This is an ASCII file listing identifier_object start_index stop_index identifier_sky start_index stop_index
            #Process entire file here
            skyList = readFileIntoList(selectedSkies)
            #loop over skyList do a split on each line
            for j in range(len(skyList)-1, -1, -1):
                skyList[j] = skyList[j].split()
                #remove misformatted lines
                if (len(skyList[j]) != 6):
                    print("skySubtractProcess::findOffsourceSkies> Warning: line "+str(j)+" misformatted in "+selectedSkies)
                    self._log.writeLog(__name__, " line "+str(j)+" misformatted in "+selectedSkies, type=fatboyLog.WARNING)
                    skyList.pop(j)
                    continue
                try:
                    skyList[j][1] = int(skyList[j][1])
                    skyList[j][2] = int(skyList[j][2])
                    skyList[j][4] = int(skyList[j][4])
                    skyList[j][5] = int(skyList[j][5])
                except Exception as ex:
                    print("skySubtractProcess::findOffsourceSkies> Warning: line "+str(j)+" misformatted in "+selectedSkies)
                    self._log.writeLog(__name__, " line "+str(j)+" misformatted in "+skyOffMethod, type=fatboyLog.WARNING)
                    skyList.pop(j)
                    continue
            #loop over dataset and assign property to all fdus that don't already have 'sky_selected_name' property.
            #some FDUs may have used xml to define sky_selected_name already
            #if tag is None, this loops over all FDUs
            for skyfdu in self._fdb.getFDUs(tag=fdu.getTag()):
                if (skyfdu.hasProperty("sky_selected_name")):
                    #this FDU already has sky_sky_selected_name set
                    continue
                i = 0 #line number
                #skyLine = [ 'identifier', start_idx, end_idx, 'sky_identifier', start_idx, end_idx ]
                for skyLine in skyList:
                    i += 1
                    skypfix = skyLine[3]+"_"+str(i)
                    if (skyfdu._id == skyLine[0] and int(skyfdu._index) >= skyLine[1] and int(skyfdu._index) <= skyLine[2]):
                        #exact match. this is an object, set sky_selected_name property
                        skyfdu.setProperty("sky_selected_name", "onsourceSkies/sky_"+skypfix+"_"+skyfdu.filter)
                    elif (skyfdu._id.find(skyLine[0]) != -1 and int(skyfdu._index) >= skyLine[1] and int(skyfdu._index) <= skyLine[2]):
                        #partial match. this is an object, set sky_selected_name property
                        skyfdu.setProperty("sky_selected_name", "onsourceSkies/sky_"+skypfix+"_"+skyfdu.filter)
                    if (skyfdu._id == skyLine[3] and int(skyfdu._index) >= skyLine[4] and int(skyfdu._index) <= skyLine[5]):
                        #exact match. this is a sky.  set sky_selected_name and selected_for_sky properties
                        skyfdu.setProperty("sky_selected_name", "onsourceSkies/sky_"+skypfix+"_"+skyfdu.filter)
                        skyfdu.setProperty("selected_for_sky", "yes")
                    elif (skyfdu._id.find(skyLine[3]) != -1 and int(skyfdu._index) >= skyLine[4] and int(skyfdu._index) <= skyLine[5]):
                        #partial match. this is a sky.  set sky_selected_name and selected_for_sky properties
                        skyfdu.setProperty("sky_selected_name", "onsourceSkies/sky_"+skypfix+"_"+skyfdu.filter)
                        skyfdu.setProperty("selected_for_sky", "yes")
            #Now Check for individual sky frames matching filter/section/sky_subtract_method/sky_selected_name to create master sky
            properties['sky_selected_name'] = fdu.getProperty('sky_selected_name')
            properties['selected_for_sky'] = "yes"
            skies = self._fdb.getFDUs(filter=fdu.filter, section=fdu.section, tag=fdu.getTag(), properties=properties)
        return skies
    #end findOnsourceSkies

    #Look at XML to determine sky subtraction methods
    def findSkySubtractMethods(self, fdu):
        skymethod = self.getOption("sky_subtract_method", fdu.getTag())
        if (skymethod.lower() in self.ssmethods):
            skymethod = skymethod.lower()
            #loop over dataset and assign property to all fdus that don't already have 'sky_method' property.
            #some FDUs may have used xml to define sky_method already
            #if tag is None, this loops over all FDUs
            for fdu in self._fdb.getFDUs(tag=fdu.getTag()):
                if (not fdu.hasProperty("sky_method")):
                    fdu.setProperty("sky_method", skymethod)
        elif (os.access(skymethod, os.F_OK)):
            #This is an ASCII file listing filter/identifier and flat method
            #Process entire file here
            methodList = readFileIntoList(skymethod)
            #loop over methodList do a split on each line
            for j in range(len(methodList)-1, -1, -1):
                methodList[j] = methodList[j].split()
                #remove misformatted lines
                if (len(methodList[j]) < 2):
                    print("skySubtractProcess::findSkySubtractMethods> Warning: line "+str(j)+" misformatted in "+skymethod)
                    self._log.writeLog(__name__, " line "+str(j)+" misformatted in "+skymethod, type=fatboyLog.WARNING)
                    methodList.pop(j)
                    continue
                methodList[j][1] = methodList[j][1].lower()
                if (not methodlist[j][1] in ssmethods):
                    print("skySubtractProcess::findSkySubtractMethods> Warning: line "+str(j)+" misformatted in "+skymethod)
                    self._log.writeLog(__name__, " line "+str(j)+" misformatted in "+skymethod, type=fatboyLog.WARNING)
                    methodList.pop(j)
                    continue
            #loop over dataset and assign property to all fdus that don't already have 'sky_method' property.
            #some FDUs may have used xml to define sky_method already
            #if tag is None, this loops over all FDUs
            for fdu in self._fdb.getFDUs(tag=fdu.getTag()):
                if (fdu.hasProperty("sky_method")):
                    #this FDU already has sky_method set
                    continue
                #method = [ 'Filter/identifier', 'method' ]
                for method in methodList:
                    if (method[0].lower() == fdu.filter.lower()):
                        fdu.setProperty("sky_method", method[1])
                        #Exact match for filter
                    elif (len(method[0]) > 2 and fdu._id.lower().find(method[0].lower()) != -1):
                        #Partial match for identifier
                        fdu.setProperty("sky_method", method[1])
        else:
            print("skySubtractProcess::findSkySubtractMethods> Error: invalid sky_method: "+skymethod)
            self._log.writeLog(__name__, " invalid sky_method: "+skymethod, type=fatboyLog.ERROR)
    #end findSkySubtractMethods

    def fitSkySubtractedSurf(self, fdu):
        sxtMethod = self.getOption("source_extract_method", fdu.getTag()).lower()
        if (sxtMethod == "sep" and not hasSep):
            print("skySubtractProcess::fitSkySubtractedSurf> WARNING: sep is not installed.  Using sextractor for source extraction.")
            self._log.writeLog(__name__, "sep is not installed.  Using sextractor for source extraction.", type=fatboyLog.WARNING)
            sxtMethod = "sextractor"

        if (sxtMethod == "sep"):
            if (not fdu.hasProperty("sepMask")):
                #Presumably this is an image with offsource skies so sextractor has not been run on it yet.
                self.createSepImage(fdu, fdu.getData().copy()) #call helper method to run sep
            objMask = fdu.getData(tag="sepMask") #Simply get tagged mask
        else:
            #Sextractor is used.  Look for existing sxtfile
            sxtfile = self._fdb._tempdir+"/TEMPsxt_"+fdu.getFullId()
            if (not os.access(sxtfile, os.F_OK)):
                #Presumably this is an image with offsource skies so sextractor has not been run on it yet.
                tempSSfile = self._fdb._tempdir+"/TEMPss_"+fdu.getFullId() #sky subtracted version
                #image is already sky subtracted but must write out image to disk for sextractor to run on
                write_fits_file(tempSSfile, fdu.getData(), overwrite=True, log=self._log)
                self.createSextractedImage(fdu, tempSSfile) #call helper method to run sextractor
            #Open sextractor segmentation image
            maskFile = pyfits.open(sxtfile)
            objmef = findMef(maskFile)
            objMask = maskFile[objmef].data
            maskFile.close()

        #Apply object mask to image
        data = fdu.getData().copy()
        if (self._fdb.getGPUMode()):
            applyObjMask(data, objMask)
        else:
            data[objMask > 0] = 0

        #Select cpu/gpu option
        pysurfit_method = gpu_pysurfit.pysurfit
        if (not self._fdb.getGPUMode()):
            pysurfit_method = pysurfit.pysurfit

        goodPixelMask = (1-fdu.getBadPixelMask().getData()).astype(bool)
        surf = pysurfit_method(data, order=1, niter=2, lower=2.5, upper=2.5, inmask=goodPixelMask, log=self._log, mode=gpu_pysurfit.MODE_RAW)

        #Subtract surf and update data
        if (self._fdb.getGPUMode()):
            fdu.updateData(subtractImages(fdu.getData(), surf, gpm=goodPixelMask.astype("int32")))
        else:
            fdu.updateData(fdu.getData() - surf*goodPixelMask.astype("int32"))
        del goodPixelMask
        fdu._header.add_history('Fit sky subtracted surface')
    #end fitSkySubtractedSurf

    ## OVERRIDE getCalibs
    def getCalibs(self, fdu, prevProc = None):
        calibs = dict()

        msfilename = self.getCalib("masterSky", fdu.getTag())
        if (msfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(msfilename, os.F_OK)):
                print("skySubtractProcess::getCalibs> Using master sky "+msfilename+"...")
                self._log.writeLog(__name__, "Using master sky "+msfilename+"...")
                calibs['masterSky'] = fatboyCalib(self._pname, "master_sky", fdu, filename=msfilename, log=self._log)
                return calibs
            else:
                print("skySubtractProcess::getCalibs> Warning: Could not find master sky "+msfilename+"...")
                self._log.writeLog(__name__, "Could not find master sky "+msfilename+"...", type=fatboyLog.WARNING)

        #First look for property sky_method
        properties = dict()
        if (not fdu.hasProperty("sky_method")):
            #Look at XML options to find sky method and assign it to FDUs
            self.findSkySubtractMethods(fdu)
        properties['sky_method'] = fdu.getProperty("sky_method")
        if (properties['sky_method'] is None):
            print("skySubtractProcess::getCalibs> Error: Could not find sky_method for "+fdu.getFullId())
            self._log.writeLog(__name__, " Could not find sky_method for "+fdu.getFullId(), type=fatboyLog.ERROR)
            return calibs
        if (fdu.hasProperty("sky_offsource_name")):
            #Use properties to find master sky associated with this offsource frame
            properties['sky_offsource_name'] = fdu.getProperty("sky_offsource_name")
        elif (fdu.hasProperty("sky_selected_name")):
            #Use properties to find master sky associated with this onsource frame
            properties['sky_selected_name'] = fdu.getProperty("sky_selected_name")
        elif (not fdu.getProperty("sky_method").startswith("offsource")):
            #set sky_name property for onsource skies
            properties['sky_name'] = "sky-"+str(properties['sky_method'])+"-"+str(fdu._id)+"."+str(fdu._index)

        #1) Check for an already created (OFFSOURCE only) master sky frame matching filter/section and TAGGED for this object
        masterSky = self._fdb.getTaggedMasterCalib(self._pname, fdu._id, obstype="master_sky", filter=fdu.filter, section=fdu.section, properties=properties)
        if (masterSky is not None):
            #Found master sky.  Return here
            calibs['masterSky'] = masterSky
            return calibs
        #2) Find any individual sky frames TAGGED for this object (OFFSOURCE only) to create master sky
        skies = self._fdb.getTaggedCalibs(fdu._id, obstype=fatboyDataUnit.FDU_TYPE_SKY, filter=fdu.filter, section=fdu.section, properties=properties)
        if (len(skies) > 0):
            #Found skies associated with this fdu.  Create master sky.
            print("skySubtractProcess::getCalibs> Creating Master Sky for tagged object "+fdu._id+", filter "+str(fdu.filter)+" using METHOD "+properties['sky_method']+"...")
            self._log.writeLog(__name__, "Creating Master Sky for tagged object "+fdu._id+", filter "+str(fdu.filter)+" using METHOD "+properties['sky_method']+"...")
            #First recursively process
            self.recursivelyExecute(skies, prevProc)
            #This method creates the master sky dependent on sky_method
            masterSky = self.createMasterSky(fdu, skies, properties)
            if (masterSky is None):
                return calibs
            self._fdb.appendCalib(masterSky)
            calibs['masterSky'] = masterSky
            return calibs
        #3) Check for an already created (OFFSOURCE only) master sky frame matching filter/section
        masterSky = self._fdb.getMasterCalib(self._pname, obstype="master_sky", filter=fdu.filter, section=fdu.section, tag=fdu.getTag(), properties=properties)
        if (masterSky is not None):
            #Found master flat.  Return here
            print("skySubtractProcess::getCalibs> Using "+masterSky.getFilename()+" as master sky for "+fdu.getFullId())
            self._log.writeLog(__name__, "Using "+masterSky.getFilename()+" as master sky for "+fdu.getFullId())
            calibs['masterSky'] = masterSky
            return calibs
        #4) Check default_master_sky for matching filter/section
        #### Unlike flats and darks, default sky takes priority since objects themselves are used and can't be omitted
        #### This option will rarely be used
        defaultMasterSkies = []
        if (self.getOption('default_master_sky', fdu.getTag()) is not None):
            dmslist = self.getOption('default_master_sky', fdu.getTag())
            if (dmslist.count(',') > 0):
                #comma separated list
                defaultMasterSkies = dmslist.split(',')
                removeEmpty(defaultMasterSkies)
                for j in range(len(defaultMasterSkies)):
                    defaultMasterSkies[j] = defaultMasterSkies[j].strip()
            elif (dmslist.endswith('.fit') or dmslist.endswith('.fits')):
                #FITS file given
                defaultMasterSkies.append(dmslist)
            elif (dmslist.endswith('.dat') or dmslist.endswith('.list') or dmslist.endswith('.txt')):
                #ASCII file list
                defaultMasterSkies = readFileIntoList(dmslist)
            for mskyfile in defaultMasterSkies:
                #Loop over list of default master skies
                #masterSky = fatboyImage(mskyfile)
                masterSky = fatboyCalib(self._pname, "master_sky", fdu, filename=mskyfile, log=self._log)
                #read header and initialize
                masterSky.readHeader()
                masterSky.initialize()
                if (masterSky.filter != fdu.filter):
                    #does not match filter
                    continue
                if (fdu.section is not None):
                    #check section if applicable
                    section = -1
                    if (masterSky.hasHeaderValue('SECTION')):
                        section = masterSky.getHeaderValue('SECTION')
                    else:
                        idx = masterSky.getFilename().rfind('.fit')
                        if (masterSky.getFilename()[idx-2] == 'S' and isDigit(masterSky.getFilename()[idx-1])):
                            section = int(masterSky.getFilename()[idx-1])
                    if (section != fdu.section):
                        continue
                masterSky.setType("master_sky")
                #Found matching master sky
                print("skySubtractProcess::getCalibs> Using default master sky "+masterSky.getFilename())
                self._log.writeLog(__name__, "Using default master sky "+masterSky.getFilename())
                self._fdb.appendCalib(masterSky)
                calibs['masterSky'] = masterSky
                return calibs
        #5) Find individual sky frames for this object to create master sky
        skies = self.findSkies(fdu, properties)
        if (len(skies) > 0):
            #Check if this fdu is an offsource sky.  Only possible for sky_offsource_method from file
            if (fdu in skies and not fdu.hasProperty("sky_selected_name")):
                print("skySubtractProcess::getCalibs> Object "+fdu.getFullId()+" is actually an OFFSOURCE sky!  Skipping.")
                self._log.writeLog(__name__, "Object "+fdu.getFullId()+" is actually an OFFSOURCE sky!  Skipping.")
                calibs['skipFrame'] = True
                return calibs
            #Found skies associated with this fdu.  Create master sky.
            print("skySubtractProcess::getCalibs> Creating Master Sky for filter: "+str(fdu.filter)+" using METHOD "+properties['sky_method']+"...")
            self._log.writeLog(__name__, "Creating Master Sky for filter: "+str(fdu.filter)+" using METHOD "+properties['sky_method']+"...")
            #First recursively process
            self.recursivelyExecute(skies, prevProc)
            #This method creates the master sky dependent on sky_method
            masterSky = self.createMasterSky(fdu, skies, properties, prevProc=prevProc)
            if (masterSky is None):
                return calibs
            skymethod = properties['sky_method'].lower()
            if (skymethod.startswith("offsource")):
                #only append master calib for offsource skies!
                self._fdb.appendCalib(masterSky)
            elif(fdu.hasProperty("sky_selected_name")):
                #Also append master calib for onsource with use_sky_files = selected
                self._fdb.appendCalib(masterSky)
            calibs['masterSky'] = masterSky
            return calibs
        print("skySubtractProcess::getCalibs> ERROR: No skies found for "+fdu.getFullId())
        self._log.writeLog(__name__, "No skies found for "+fdu.getFullId(), type=fatboyLog.ERROR)
        return calibs
    #end getCalibs

    def nebObjSky(self, fdu):
        sky_neb_range = int(self.getOption("sky_neb_range", fdu.getTag()))
        #First look for property sky_method
        properties = dict()
        #We can assume fdu has sky_method property since this was set in getCalibs.
        properties['sky_method'] = fdu.getProperty("sky_method")
        if (fdu.hasProperty("sky_offsource_name")):
            properties['sky_offsource_name'] = fdu.getProperty("sky_offsource_name")

        #Now we need to find date/time/median values of individual skies associated with this object, even if master sky has already been created!
        #1) Find any individual sky frames TAGGED for this object
        skies = self._fdb.getTaggedCalibs(fdu._id, obstype=fatboyDataUnit.FDU_TYPE_SKY, filter=fdu.filter, section=fdu.section, properties=properties, inUse=False)
        if (len(skies) > 0):
            #Found skies associated with this fdu.
            print("skySubtractProcess::nebObjSky> Calculating median background for tagged object "+fdu.getFullId())
            self._log.writeLog(__name__, "Calculating median background for tagged object "+fdu.getFullId()+"...")
        else:
            #2) Check for individual sky frames matching filter/section/sky_subtract_method and possibly sky_offsource_name to create master sky
            skies = self._fdb.getCalibs(obstype=fatboyDataUnit.FDU_TYPE_SKY, filter=fdu.filter, section=fdu.section, tag=fdu.getTag(), properties=properties, inUse=False)
            if (len(skies) > 0):
                #Found skies associated with this fdu
                print("skySubtractProcess::nebObjSky> Calculating median background for "+fdu.getFullId())
                self._log.writeLog(__name__, "Calculating median background for "+fdu.getFullId()+"...")
            else:
                #return 0 if can't find any skies associated with this fdu.  Image median will be used instead
                return 0
        objDate = fdu.getHeaderValue("date_keyword")
        objUT = getRADec(fdu.getHeaderValue("ut_keyword"))*60.
        deltat = zeros(len(skies))
        #Calculate time difference between object and each sky
        for j in range(len(skies)):
            date = skies[j].getHeaderValue("date_keyword")
            ut = getRADec(skies[j].getHeaderValue("ut_keyword"))*60.
            deltat[j] = abs(objUT-ut)
            if (objDate != date):
                deltat[j] += 1.e+6
        skyIndices = deltat.argsort()[0:sky_neb_range]
        medtotal = 0.0
        weight = 0.0
        #Perform weighted average of medians of skies
        for i in skyIndices:
            medtotal += skies[i].getMedian()/deltat[i]
            weight += 1.0/deltat[i]
        medtotal = medtotal/weight
        return medtotal
    #end nebObjSky

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('default_master_sky', None)
        self._options.setdefault('sky_subtract_method', "remove_objects") #remove_objects, rough, offsource, offsource_extended, offsource_neb, offsource_rough
        self._optioninfo.setdefault('sky_subtract_method', 'rough | remove_objects | offsource | offsource_rough |\noffsource_extended | offsource_neb')
        self._options.setdefault('sky_reject_type','none')
        self._options.setdefault('sky_nlow','1')
        self._options.setdefault('sky_nhigh','1')
        self._options.setdefault('sky_lsigma','5')
        self._options.setdefault('sky_hsigma','5')
        self._options.setdefault('use_sky_files','all')
        self._optioninfo.setdefault('use_sky_files', 'all | range | selected')
        self._options.setdefault('sky_files_range','3')
        self._optioninfo.setdefault('sky_files_range', 'n skies before AND n skies after will be used')
        self._options.setdefault('selected_skies', None)
        self._optioninfo.setdefault('selected_skies', '6 column ASCII file specifying index, start, stop\nfor objects and skies for selected onsource skies.')
        self._options.setdefault('onsource_sorting_key', 'full') #full, index, or a FITS keyword to sort onsource skies by time
        self._optioninfo.setdefault('onsource_sorting_key', 'full | index | a FITS keyword to sort onsource skies\nby time. For CIRCE data, MJD is recommended.')
        self._options.setdefault('sky_dithering_range','2')
        self._options.setdefault('sky_offsource_range','240')
        self._options.setdefault('sky_offsource_method','auto')
        self._options.setdefault('keep_skies','no')
        self._options.setdefault('sep_detect_thresh', '1.5')
        self._options.setdefault('sky_neb_range','5')
        self._options.setdefault('source_extract_method', 'sep') #sep or sextractor
        self._optioninfo.setdefault('source_extract_method', 'sep | sextractor')
        self._options.setdefault('sextractor_path','/usr/bin/sex')
        self._options.setdefault('two_pass_object_masking', 'yes')
        self._options.setdefault('two_pass_detect_thresh', '10')
        self._options.setdefault('two_pass_detect_minarea', '50')
        self._options.setdefault('two_pass_boxcar_size', '51')
        self._options.setdefault('two_pass_reject_level', '0.95')
        self._options.setdefault('two_pass_sep_ellipse_growth', '2.5') #multiplicative factor for semimajor axes of ellipses found by sep
        self._optioninfo.setdefault('two_pass_sep_ellipse_growth', 'factor for growing ellipses found by sep')
        self._options.setdefault('interp_zeros_sky', 'yes')
        self._options.setdefault('fit_sky_subtracted_surf', 'no')
        self._options.setdefault('conserve_memory', 'no')
        self._optioninfo.setdefault('conserve_memory', 'Set to yes if running on a machine with low RAM')
    #end setDefaultOptions

    # Actually perform sky subtraction
    def skySubtractImage(self, image, sky, scale):
        t = time.time()
        blocks = image.size//512
        if (image.size % 512 != 0):
            blocks += 1
        #Select cpu/gpu option
        if (self._fdb.getGPUMode()):
            #Use GPU
            if (not superFATBOY.threaded()):
                global fatboy_mod
            else:
                fatboy_mod = get_fatboy_mod()
            skySubtractFunc = fatboy_mod.get_function("subtractArrays_scaled_float")
            skySubtractFunc(drv.InOut(image), drv.In(sky), float32(scale), grid=(blocks,1), block=(block_size,1,1))
        else:
            #Use CPU
            image -= sky*scale
        print("Scale and subtract time: ",time.time()-t)
        return image
    #end skySubtractImage

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/skySubtracted", os.F_OK)):
            os.mkdir(outdir+"/skySubtracted",0o755)
        #Create output filename
        ssfile = outdir+"/skySubtracted/ss_"+fdu.getFullId()
        #Check to see if it exists
        if (os.access(ssfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(ssfile)
        if (not os.access(ssfile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(ssfile)
    #end writeOutput
