from superFATBOY.fatboyCalib import fatboyCalib
from superFATBOY.fatboyDataUnit import fatboyDataUnit
from superFATBOY.fatboyImage import fatboyImage
from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY import gpu_imcombine, imcombine
from superFATBOY import gpu_xregister, xregister
from superFATBOY import gpu_drihizzle, drihizzle

class sinfoniRegisterStackProcess(fatboyProcess):
    _modeTags = ["sinfoni"]

    def alignFrames(self, fdu, frameList):
        alignMethod = self.getOption("align_method", fdu.getTag()).lower()
        refframe = self.getOption("align_refframe", fdu.getTag())
        if (isInt(refframe)):
            refframe = int(refframe)
        xboxsize = int(self.getOption('align_box_size_x', fdu.getTag()))
        yboxsize = int(self.getOption('align_box_size_y', fdu.getTag()))
        xcenter = int(self.getOption('align_box_center_x', fdu.getTag()))
        ycenter = int(self.getOption('align_box_center_y', fdu.getTag()))
        constrain_box = int(self.getOption('align_constrain_boxsize', fdu.getTag()))
        shiftsFile = self.getOption("align_shifts_file", fdu.getTag())
        sepDetectThresh = float(self.getOption('xregister_sep_detect_thresh', fdu.getTag()))
        sepfwhm = self.getOption('xregister_sep_fwhm', fdu.getTag())

        if (not fdu.hasProperty("collapsedSlitlets")):
            print("sinfoniRegisterStackProcess::alignFrames> Object "+fdu._id+" has no collapsedSlitlets property.  No alignment or stacking possible.")
            self._log.writeLog(__name__, "Object "+fdu._id+" has no collapsedSlitlets property.  No alignment or stacking possible.", type=fatboyLog.WARNING)
            return None
        shp = fdu.getProperty("collapsedSlitlets").shape
        if (xboxsize == -1):
            #xboxsize = fdu.getShape()[1]
            xboxsize = shp[1]
        if (yboxsize == -1):
            #yboxsize = fdu.getShape()[0]
            yboxsize = shp[0]
        if (xcenter == -1):
            #xcenter = fdu.getShape()[1]/2
            xcenter = shp[1]/2
        if (ycenter == -1):
            #ycenter = fdu.getShape()[0]/2
            ycenter = shp[0]/2

        doMedianFilter = True
        if (self.getOption('xregister_median_filter2d', fdu.getTag()).lower() == "no"):
            doMedianFilter = False
        doMaskNegatives = False
        if (self.getOption('xregister_mask_negatives', fdu.getTag()).lower() == "yes"):
            doMaskNegatives = True
        doSmoothCorrelation = False
        if (self.getOption('xregister_smooth_correlation', fdu.getTag()).lower() == "yes"):
            doSmoothCorrelation = True
        doFit2dGaussian = False
        if (self.getOption('xregister_fit_2d_gaussian', fdu.getTag()).lower() == "yes"):
            doFit2dGaussian = True

        #Check number of frames
#    if (len(frameList) == 1):
#      print "sinfoniRegisterStackProcess::alignFrames> Object "+fdu._id+" has only one frame!  No alignment or stacking possible."
#      self._log.writeLog(__name__, "Object "+fdu._id+" has only one frame!  No alignment or stacking possible.", type=fatboyLog.WARNING)
#      return None
        print("sinfoniRegisterStackProcess::alignFrames> Aligning "+str(len(frameList))+" frames for object "+fdu._id)
        self._log.writeLog(__name__, "Aligning "+str(len(frameList))+" frames for object "+fdu._id)

        #Select cpu/gpu option
        #xregister_method = gpu_xregister.xregister
        xregister_method = xregister.xregister
        if (not self._fdb.getGPUMode()):
            xregister_method = xregister.xregister
            if (self.getOption('xregister_pad_align_box_cpu', fdu.getTag()).lower() == "yes"):
                xregister.doPad = True

        #xregister, xregister_constrained, xregister_sep, xregister_sep_constrained, xregister_guesses, manual
        if (alignMethod == "xregister"):
            #Use python xregister to calculate the shifts
            shifts = xregister_method(frameList, xcenter=xcenter, ycenter=ycenter, xboxsize=xboxsize, yboxsize=yboxsize, refframe=refframe, log=self._log, median_filter2d=doMedianFilter, doMaskNegatives=doMaskNegatives, doSmoothCorrelation=doSmoothCorrelation, doFit2dGaussian=doFit2dGaussian, method=xregister.METHOD_REGULAR, dataTag="collapsedSlitlets", mode=xregister.MODE_FDU_TAG)
        elif (alignMethod == "xregister_constrained"):
            #use RA, Dec, pixscale for initial guess
            shifts = xregister_method(frameList, xcenter=xcenter, ycenter=ycenter, xboxsize=xboxsize, yboxsize=yboxsize, constrain_boxsize=constrain_box, refframe=refframe, log=self._log, median_filter2d=doMedianFilter, doMaskNegatives=doMaskNegatives, doSmoothCorrelation=doSmoothCorrelation, doFit2dGaussian=doFit2dGaussian, method=xregister.METHOD_CONSTRAINED, dataTag="collapsedSlitlets", mode=xregister.MODE_FDU_TAG)
        elif (alignMethod == "xregister_sep"):
            #generate dummy image using sep
            shifts = xregister_method(frameList, xcenter=xcenter, ycenter=ycenter, xboxsize=xboxsize, yboxsize=yboxsize, refframe=refframe, log=self._log, median_filter2d=doMedianFilter, doMaskNegatives=False, doSmoothCorrelation=doSmoothCorrelation, doFit2dGaussian=doFit2dGaussian, sepDetectThresh=sepDetectThresh, sepfwhm=sepfwhm, method=xregister.METHOD_SEP, dataTag="collapsedSlitlets", mode=xregister.MODE_FDU_TAG)
        elif (alignMethod == "xregister_sep_constrained"):
            #Use RA, Dec, pixscale for initial guess and generate dummy image using sep
            shifts = xregister_method(frameList, xcenter=xcenter, ycenter=ycenter, xboxsize=xboxsize, yboxsize=yboxsize, constrain_boxsize=constrain_box, refframe=refframe, log=self._log, median_filter2d=doMedianFilter, doMaskNegatives=False, doSmoothCorrelation=doSmoothCorrelation, doFit2dGaussian=doFit2dGaussian, sepDetectThresh=sepDetectThresh, sepfwhm=sepfwhm, method=xregister.METHOD_SEP_CONSTRAINED, dataTag="collapsedSlitlets", mode=xregister.MODE_FDU_TAG)
        elif (alignMethod == "xregister_guesses"):
            #Use python xregister to calculate the shifts
            shifts = xregister_method(frameList, xcenter=xcenter, ycenter=ycenter, xboxsize=xboxsize, yboxsize=yboxsize, constrain_boxsize=constrain_box, constrain_guesses=shiftsFile, refframe=refframe, log=self._log, median_filter2d=doMedianFilter, doMaskNegatives=doMaskNegatives, doSmoothCorrelation=doSmoothCorrelation, doFit2dGaussian=doFit2dGaussian, method=xregister.METHOD_GUESSES, dataTag="collapsedSlitlets", mode=xregister.MODE_FDU_TAG)
        elif (alignMethod == "sep_centroid"):
            #First cross correlate dummy images from sep, then perform sigma clipping of differences between centroids for objects
            shifts = xregister_method(frameList, xcenter=xcenter, ycenter=ycenter, xboxsize=xboxsize, yboxsize=yboxsize, refframe=refframe, log=self._log, median_filter2d=doMedianFilter, doMaskNegatives=False, doSmoothCorrelation=doSmoothCorrelation, doFit2dGaussian=doFit2dGaussian, sepDetectThresh=sepDetectThresh, sepfwhm=sepfwhm, method=xregister.METHOD_SEP_CENTROID, dataTag="collapsedSlitlets", mode=xregister.MODE_FDU_TAG)
        elif (alignMethod == "sep_centroid_constrained"):
            #Use RA, Dec, pixscale for initial guess and generate dummy image using sep
            #First cross correlate dummy images from sep, then perform sigma clipping of differences between centroids for objects
            shifts = xregister_method(frameList, xcenter=xcenter, ycenter=ycenter, xboxsize=xboxsize, yboxsize=yboxsize, constrain_boxsize=constrain_box, refframe=refframe, log=self._log, median_filter2d=doMedianFilter, doMaskNegatives=False, doSmoothCorrelation=doSmoothCorrelation, doFit2dGaussian=doFit2dGaussian, sepDetectThresh=sepDetectThresh, sepfwhm=sepfwhm, method=xregister.METHOD_SEP_CENTROID_CONSTRAINED, dataTag="collapsedSlitlets", mode=xregister.MODE_FDU_TAG)
        elif (alignMethod == "manual"):
            if (not os.access(shiftsFile, os.F_OK)):
                print("sinfoniRegisterStackProcess::alignFrames> ERROR: align_shifts_file "+shiftsFile+" not found! Alignment and stacking not done!")
                self._log.writeLog(__name__, "align_shifts_file "+shiftsFile+" not found! Alignment and stacking not done!", type=fatboyLog.ERROR)
                return None
            shifts = loadtxt(shiftsFile).transpose() #need to transpose to get proper shape
        else:
            print("sinfoniRegisterStackProcess::alignFrames> ERROR: Invalid align method "+alignMethod+"!  Alignment and stacking not done!")
            self._log.writeLog(__name__, "Invalid align method "+alignMethod+"!  Alignment and stacking not done!", type=fatboyLog.ERROR)
            return None
        return shifts
    #end alignFrames

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("Align Stack")
        print(fdu._identFull)

        #Call get calibs to return dict() of calibration frames.
        #For darkSubtract, this dict should have one entry 'masterDark' which is an fdu
        calibs = self.getCalibs(fdu, prevProc)
        if (not 'frameList' in calibs):
            #Failed to obtain framelist
            #Issue error message and disable this FDU
            print("sinfoniRegisterStackProcess::execute> ERROR: Alignment and stacking not done for "+fdu.getFullId()+" (exp time="+str(fdu.exptime)+"; nreads="+str(fdu.nreads)+").  Discarding Image!")
            self._log.writeLog(__name__, "Alignment and stacking not done for "+fdu.getFullId()+" (exp time="+str(fdu.exptime)+"; nreads="+str(fdu.nreads)+").  Discarding Image!", type=fatboyLog.ERROR)
            #disable this FDU
            fdu.disable()
            return False

        #get framelist
        frameList = calibs['frameList']

        if (self.getOption('use_only_selected_indices', fdu.getTag()) is not None):
            #Check for special case where fdu is not in frameList
            if (not fdu in frameList):
                #Use new first frame as current FDU!
                fdu = frameList[0]

        #Check if output exists first
        csfile = "registeredStacked/cs_"+fdu._id+".fits"
        if (self.checkOutputExists(fdu, csfile)):
            frameList = calibs['frameList']
            #Disable all frames but first one
            for image in frameList:
                if (image.getFullId() != fdu.getFullId()):
                    image.disable()
            return True

        #Find shifts
        shifts = self.alignFrames(fdu, frameList)
        #Stack frames
        if (shifts is not None):
            self.stackFrames(fdu, frameList, shifts)
        #Disable all frames but first one
        for image in frameList:
            if (image.getFullId() != fdu.getFullId()):
                image.disable()
        return True
    #end execute

    ## OVERRIDE getCalibs
    def getCalibs(self, fdu, prevProc = None):
        calibs = dict()
        #get FDUs matching this identifier and filter, sorted by index
        fdus = self._fdb.getSortedFDUs(ident = fdu._id, filter=fdu.filter, section=fdu.section)

        #New option 3/31/16 - use_only_selected_indices for align/stack
        if (self.getOption('use_only_selected_indices', fdu.getTag()) is not None):
            #use_only_selected_indices of format "5, 20:34, 47, 50:80" OR ASCII file
            use_indices = self.getOption('use_only_selected_indices', fdu.getTag())
            indList = []
            if (os.access(use_indices, os.F_OK)):
                indList = readFileIntoList(use_indices)
                for j in range(len(indList)):
                    try:
                        indList[j] = int(indList[j])
                    except ValueError as ex:
                        print("sinfoniRegisterStackProcess::getCalibs> Warning: invalid index in file "+use_indices+": "+str(indList[j]))
                        self._log.writeLog(__name__, " invalid index in file "+use_indices+": "+str(indList[j]), type=fatboyLog.WARNING)
            else:
                try:
                    #Parse out into list
                    use_indices = use_indices.split(",")
                    for j in range(len(use_indices)):
                        use_indices[j] = use_indices[j].strip().split(":")
                        #Add to list
                        if (len(use_indices[j]) == 1):
                            indList.append(int(use_indices[j][0]))
                        elif (len(use_indices[j]) == 2):
                            indList.extend(list(range(int(use_indices[j][0]), int(use_indices[j][1]))))
                except ValueError as ex:
                    print("sinfoniRegisterStackProcess::getCalibs> Error: invalid format in use_indices: "+str(ex)+": No Frames will be stacked!")
                    self._log.writeLog(__name__, " invalid format in use_indices: "+str(ex)+": No Frames will be stacked!", type=fatboyLog.ERROR)
                    indList = []
            #Loop backwards over fdu list so as to be able to remove cleanly
            for j in range(len(fdus)-1, -1, -1):
                if (indList.count(int(fdus[j]._index)) == 0):
                    #FDU is not in indList
                    print("sinfoniRegisterStackProcess::getCalibs> Frame "+fdus[j].getFullId()+" is not in index list.  Discarding!")
                    self._log.writeLog(__name__, "Frame "+fdus[j].getFullId()+" is not in index list.  Discarding!")
                    #Disable so its not picked up by sinfoniRegisterStack later and then remove from fdu list
                    fdus[j].disable()
                    fdus.pop(j)

        if (len(fdus) > 0):
            #Found other objects associated with this fdu.  Create registered stacked image
            print("sinfoniRegisterStackProcess::getCalibs> Creating registered stacked image for object "+fdu._id+", exposure time "+str(fdu.exptime)+", and "+str(fdu.nreads)+" reads...")
            self._log.writeLog(__name__, "Creating registered stacked image for object "+fdu._id+", exposure time "+str(fdu.exptime)+", and "+str(fdu.nreads)+" reads...")
            #First recursively process
            self.recursivelyExecute(fdus, prevProc)
            #Loop over fdus and pop out any that have been disabled at sky subtraction stage by pairing up
            #Loop backwards!
            for j in range(len(fdus)-1, -1, -1):
                currFDU = fdus[j]
                if (not currFDU.inUse):
                    fdus.remove(currFDU)
            #convenience method
            calibs['frameList'] = fdus
            return calibs
        return calibs
    #end getCalibs

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('align_box_size_x', '-1')
        self._optioninfo.setdefault('align_box_size_x', 'size of alignment box; -1 = use full x-size')
        self._options.setdefault('align_box_size_y', '-1')
        self._optioninfo.setdefault('align_box_size_y', 'size of alignment box; -1 = use full y-size')
        self._options.setdefault('align_box_center_x', '-1')
        self._optioninfo.setdefault('align_box_center_x', 'center of alignment box; -1 = use x-center')
        self._options.setdefault('align_box_center_y', '-1')
        self._optioninfo.setdefault('align_box_center_y', 'center of alignment box; -1 = use y-center')
        self._options.setdefault('align_constrain_boxsize', '256')
        self._options.setdefault('align_method', 'xregister') #xregister, xregister_constrained, xregister_sep, xregister_sep_constrained, xregister_guesses, sep_centroid, sep_centroid_constrained, manual
        self._optioninfo.setdefault('align_method', 'xregister | xregister_constrained | xregister_sep |\nxregister_sep_constrained | xregister_guesses |\nsep_centroid | sep_centroid_constrained | manual')
        self._options.setdefault('align_refframe', '0') #number, identifier.index
        self._optioninfo.setdefault('align_refframe', 'number or identifier.index')
        self._options.setdefault('align_shifts_file', None) #shifts for manual or xregister_guesses
        self._optioninfo.setdefault('align_shifts_file', 'shifts for manual or xregister_guesses')
        self._options.setdefault('drihizzle_dropsize', '0.01')
        self._options.setdefault('drihizzle_in_units', 'counts')
        self._options.setdefault('drihizzle_kernel', 'point') #point, turbo, etc
        self._optioninfo.setdefault('drihizzle_kernel', 'point, turbo, etc.')
        self._options.setdefault('geom_trans_coeffs', None)
        self._options.setdefault('keep_exposure_map', 'yes') #whether to save exposure_map and pixel_map so they can be used in a later step
        self._options.setdefault('keep_indiv_images', 'no')
        self._options.setdefault('use_only_selected_indices', None)
        self._optioninfo.setdefault('use_only_selected_indices', 'If not None, this can be a list of indices or ASCII\nfile listing indices of frames to align/stack.\nOthers will be ignored.')
        self._options.setdefault('xregister_fit_2d_gaussian', 'no') #fit a 2-d gaussian instead of two 1-d gaussians
        self._options.setdefault('xregister_mask_negatives', 'no') #mask negatives out before cross correlating
        self._options.setdefault('xregister_median_filter2d', 'yes') #median filter cross correlation
        self._options.setdefault('xregister_smooth_correlation', 'no') #smooth cross correlation for purposes of finding max
        self._options.setdefault('xregister_pad_align_box_cpu', 'no') #pad align box
        self._options.setdefault('xregister_sep_detect_thresh', '3') #detect threshold
        self._options.setdefault('xregister_sep_fwhm', 'a') #fwhm (default = 'a', semi-major axis, otherwise in pixels)
        self._optioninfo.setdefault('xregister_sep_fwhm', 'a=semi-major axis, otherwise a number in pixels')
    #end setDefaultOptions

    def stackFrames(self, fdu, frameList, shifts):
        refframe = self.getOption("align_refframe", fdu.getTag())
        dropsize = float(self.getOption('drihizzle_dropsize', fdu.getTag()))
        inunits = self.getOption('drihizzle_in_units', fdu.getTag())
        kernel = self.getOption('drihizzle_kernel', fdu.getTag())
        geomFile = self.getOption('geom_trans_coeffs', fdu.getTag())
        keepImages = self.getOption('keep_indiv_images', fdu.getTag())

        #Select cpu/gpu option
        drihizzle_method = gpu_drihizzle.drihizzle
        if (not self._fdb.getGPUMode()):
            drihizzle_method = drihizzle.drihizzle

        drihizzle_method3d = gpu_drihizzle.drihizzle3d
        if (not self._fdb.getGPUMode()):
            drihizzle_method3d = drihizzle.drihizzle3d

        #Drihizzle each frame into the final image and exposure mask
        imgdir = str(self._fdb.getParam("outputdir", fdu.getTag()))+'/registeredStacked'
        if (not os.access(imgdir, os.F_OK)):
            os.mkdir(imgdir)
        xshifts = shifts[0]
        yshifts = shifts[1]
        xshift0 = -1*int(max(xshifts)+min(xshifts))//2
        yshift0 = -1*int(max(yshifts)+min(yshifts))//2
        #goodPixelMask = (1-fdu.getBadPixelMask().getData()).astype("int32")
        for i in range(len(frameList)):
            xshifts[i]+=xshift0
            yshifts[i]+=yshift0
            if (frameList[i].getShape() != frameList[0].getShape()):
                goodPixelMask = gpu_drihizzle.MODE_FDU_USE_INDIVIDUAL_GPMS
        #Stack 2d images
        (data, header, expmap, pixmap) = drihizzle_method(frameList, None, None, inmask=None, weight='exptime', kernel=kernel, dropsize=dropsize, geomDist=geomFile, xsh=xshifts, ysh=yshifts, inunits=inunits, outunits='cps', keepImages=keepImages, imgdir=imgdir, log=self._log, mode=gpu_drihizzle.MODE_FDU_TAG, dataTag="collapsedSlitlets")
        fdu.tagDataAs("stacked2d", data)
        fdu.updateHeader(header)
        if (self.getOption('write_output', fdu.getTag()).lower() == "yes"):
            #only tag these if writing output -- then free up memory after writing output
            fdu.tagDataAs("exposure_map", data=expmap)
            fdu.tagDataAs("pixel_map", data=pixmap)
        #Stack 3d datacubes
        (data, header, expmap, pixmap) = drihizzle_method3d(frameList, None, None, inmask=None, weight='exptime', kernel=kernel, dropsize=dropsize, geomDist=geomFile, xsh=xshifts, ysh=yshifts, inunits=inunits, outunits='cps', keepImages=keepImages, imgdir=imgdir, log=self._log, mode=gpu_drihizzle.MODE_FDU_TAG, dataTag="datacube")
        fdu.tagDataAs("stacked3d", data)
        fdu.updateHeader(header)
        if (self.getOption('write_output', fdu.getTag()).lower() == "yes"):
            #only tag these if writing output -- then free up memory after writing output
            fdu.tagDataAs("exposure_map3d", data=expmap)
            fdu.tagDataAs("pixel_map3d", data=pixmap)
    #end stackFrames

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/registeredStacked", os.F_OK)):
            os.mkdir(outdir+"/registeredStacked",0o755)
        #Create output filenames
        csfile = outdir+"/registeredStacked/cs_"+fdu._id+".fits"
        dcfile = outdir+"/registeredStacked/3ds_"+fdu._id+".fits"
        expfile = outdir+"/registeredStacked/exp_"+fdu._id+".fits"
        objfile = outdir+"/registeredStacked/objmap_"+fdu._id+".fits"
        expfile3 = outdir+"/registeredStacked/exp3_"+fdu._id+".fits"
        objfile3 = outdir+"/registeredStacked/objmap3_"+fdu._id+".fits"
        #Check to see if it exists
        if (os.access(csfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(csfile)
        if (os.access(expfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(expfile)
        if (os.access(objfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(objfile)
        if (os.access(dcfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(dcfile)
        if (os.access(expfile3, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(expfile3)
        if (os.access(objfile3, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(objfile3)
        if (not os.access(csfile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(csfile, tag="stacked2d")
        if (not os.access(expfile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(expfile, tag="exposure_map")
        if (not os.access(objfile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(objfile, tag="pixel_map")
        if (not os.access(dcfile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(dcfile, tag="stacked3d")
        if (not os.access(expfile3, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(expfile3, tag="exposure_map3d")
        if (not os.access(objfile3, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(objfile3, tag="pixel_map3d")
        #free memory from exposure_map and pixel_map tags if requested
        if (self.getOption('keep_exposure_map', fdu.getTag()).lower() != "yes"):
            fdu.removeProperty("exposure_map")
            fdu.removeProperty("pixel_map")
            fdu.removeProperty("exposure_map3d")
            fdu.removeProperty("pixel_map3d")
    #end writeOutput
