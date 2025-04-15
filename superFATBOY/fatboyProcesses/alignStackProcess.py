from superFATBOY.fatboyCalib import fatboyCalib
from superFATBOY.fatboyDataUnit import fatboyDataUnit
from superFATBOY.fatboyImage import fatboyImage
from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY import gpu_imcombine, imcombine
from superFATBOY import gpu_xregister, xregister
from superFATBOY import gpu_drihizzle, drihizzle
from superFATBOY import tri_register

class alignStackProcess(fatboyProcess):
    _modeTags = ["imaging", "circe"]

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

        if (xboxsize == -1):
            xboxsize = fdu.getShape()[1]
        if (yboxsize == -1):
            yboxsize = fdu.getShape()[0]
        if (xcenter == -1):
            xcenter = fdu.getShape()[1]//2
        if (ycenter == -1):
            ycenter = fdu.getShape()[0]//2

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
        if (len(frameList) == 1):
            print("alignStackProcess::alignFrames> Object "+fdu._id+" has only one frame!  No alignment or stacking possible.")
            self._log.writeLog(__name__, "Object "+fdu._id+" has only one frame!  No alignment or stacking possible.", type=fatboyLog.WARNING)
            return None
        print("alignStackProcess::alignFrames> Aligning "+str(len(frameList))+" frames for object "+fdu._id)
        self._log.writeLog(__name__, "Aligning "+str(len(frameList))+" frames for object "+fdu._id)

        #Select cpu/gpu option
        xregister_method = gpu_xregister.xregister
        if (not self._fdb.getGPUMode()):
            xregister_method = xregister.xregister
            if (self.getOption('xregister_pad_align_box_cpu', fdu.getTag()).lower() == "yes"):
                xregister.doPad = True
        triregister_method = tri_register.tri_register

        #xregister, xregister_constrained, xregister_sep, xregister_sep_constrained, xregister_guesses, manual
        if (alignMethod == "xregister"):
            #Use python xregister to calculate the shifts
            shifts = xregister_method(frameList, xcenter=xcenter, ycenter=ycenter, xboxsize=xboxsize, yboxsize=yboxsize, refframe=refframe, log=self._log, median_filter2d=doMedianFilter, doMaskNegatives=doMaskNegatives, doSmoothCorrelation=doSmoothCorrelation, doFit2dGaussian=doFit2dGaussian, method=xregister.METHOD_REGULAR)
        elif (alignMethod == "xregister_constrained"):
            #use RA, Dec, pixscale for initial guess
            shifts = xregister_method(frameList, xcenter=xcenter, ycenter=ycenter, xboxsize=xboxsize, yboxsize=yboxsize, constrain_boxsize=constrain_box, refframe=refframe, log=self._log, median_filter2d=doMedianFilter, doMaskNegatives=doMaskNegatives, doSmoothCorrelation=doSmoothCorrelation, doFit2dGaussian=doFit2dGaussian, method=xregister.METHOD_CONSTRAINED)
        elif (alignMethod == "xregister_sep"):
            #generate dummy image using sep
            shifts = xregister_method(frameList, xcenter=xcenter, ycenter=ycenter, xboxsize=xboxsize, yboxsize=yboxsize, refframe=refframe, log=self._log, median_filter2d=doMedianFilter, doMaskNegatives=False, doSmoothCorrelation=doSmoothCorrelation, doFit2dGaussian=doFit2dGaussian, sepDetectThresh=sepDetectThresh, sepfwhm=sepfwhm, method=xregister.METHOD_SEP)
        elif (alignMethod == "xregister_sep_constrained"):
            #Use RA, Dec, pixscale for initial guess and generate dummy image using sep
            shifts = xregister_method(frameList, xcenter=xcenter, ycenter=ycenter, xboxsize=xboxsize, yboxsize=yboxsize, constrain_boxsize=constrain_box, refframe=refframe, log=self._log, median_filter2d=doMedianFilter, doMaskNegatives=False, doSmoothCorrelation=doSmoothCorrelation, doFit2dGaussian=doFit2dGaussian, sepDetectThresh=sepDetectThresh, sepfwhm=sepfwhm, method=xregister.METHOD_SEP_CONSTRAINED)
        elif (alignMethod == "xregister_guesses"):
            #Use python xregister to calculate the shifts
            shifts = xregister_method(frameList, xcenter=xcenter, ycenter=ycenter, xboxsize=xboxsize, yboxsize=yboxsize, constrain_boxsize=constrain_box, constrain_guesses=shiftsFile, refframe=refframe, log=self._log, median_filter2d=doMedianFilter, doMaskNegatives=doMaskNegatives, doSmoothCorrelation=doSmoothCorrelation, doFit2dGaussian=doFit2dGaussian, method=xregister.METHOD_GUESSES)
        elif (alignMethod == "sep_centroid"):
            #First cross correlate dummy images from sep, then perform sigma clipping of differences between centroids for objects
            shifts = xregister_method(frameList, xcenter=xcenter, ycenter=ycenter, xboxsize=xboxsize, yboxsize=yboxsize, refframe=refframe, log=self._log, median_filter2d=doMedianFilter, doMaskNegatives=False, doSmoothCorrelation=doSmoothCorrelation, doFit2dGaussian=doFit2dGaussian, sepDetectThresh=sepDetectThresh, sepfwhm=sepfwhm, method=xregister.METHOD_SEP_CENTROID)
        elif (alignMethod == "sep_centroid_constrained"):
            #Use RA, Dec, pixscale for initial guess and generate dummy image using sep
            #First cross correlate dummy images from sep, then perform sigma clipping of differences between centroids for objects
            shifts = xregister_method(frameList, xcenter=xcenter, ycenter=ycenter, xboxsize=xboxsize, yboxsize=yboxsize, constrain_boxsize=constrain_box, refframe=refframe, log=self._log, median_filter2d=doMedianFilter, doMaskNegatives=False, doSmoothCorrelation=doSmoothCorrelation, doFit2dGaussian=doFit2dGaussian, sepDetectThresh=sepDetectThresh, sepfwhm=sepfwhm, method=xregister.METHOD_SEP_CENTROID_CONSTRAINED)
        elif (alignMethod == "triangles"):
            triangles = self.getOption("triangles", fdu.getTag()).lower()
            trimethod = tri_register.METHOD_DELAUNAY
            if (triangles == "all"):
                trimethod = tri_register.METHOD_ALL
            triangles_atol = float(self.getOption('triangles_atol', fdu.getTag()))
            triangles_rtol = float(self.getOption('triangles_rtol', fdu.getTag()))
            max_stars = None
            if (self.getOption('triangles_max_stars', fdu.getTag()) is not None):
                max_stars = int(self.getOption('triangles_max_stars', fdu.getTag()))
            debug_plots = False
            if (self.getOption('triangles_debug_plots', fdu.getTag()).lower() == "yes"):
                debug_plots = True
            triangles_min_angle = float(self.getOption('triangles_min_angle', fdu.getTag()))
            triangles_max_angle = float(self.getOption('triangles_max_angle', fdu.getTag()))
            triangles_sigma = float(self.getOption('triangles_sigma', fdu.getTag()))
            triangles_use_sigma_clipping = False
            if (self.getOption('triangles_use_sigma_clipping', fdu.getTag()).lower() == 'yes'):
                triangles_use_sigma_clipping = True

            outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
            if (not os.access(outdir+"/alignedStacked", os.F_OK)):
                os.mkdir(outdir+"/alignedStacked",0o755)
            shifts = triregister_method(frameList, xcenter=xcenter, ycenter=ycenter, xboxsize=xboxsize, yboxsize=yboxsize, refframe=refframe, log=self._log, sepDetectThresh=sepDetectThresh, method=trimethod, min_angle=triangles_min_angle, max_angle=triangles_max_angle, max_stars=max_stars, doplots=debug_plots, plotdir=outdir+"/alignedStacked/", atol=triangles_atol, rtol=triangles_rtol, sigma_clipping=triangles_use_sigma_clipping, sig_to_clip=triangles_sigma)
        elif (alignMethod == "manual"):
            if (not os.access(shiftsFile, os.F_OK)):
                print("alignStackProcess::alignFrames> ERROR: align_shifts_file "+shiftsFile+" not found! Alignment and stacking not done!")
                self._log.writeLog(__name__, "align_shifts_file "+shiftsFile+" not found! Alignment and stacking not done!", type=fatboyLog.ERROR)
                return None
            shifts = loadtxt(shiftsFile).transpose() #need to transpose to get proper shape
        else:
            print("alignStackProcess::alignFrames> ERROR: Invalid align method "+alignMethod+"!  Alignment and stacking not done!")
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
            print("alignStackProcess::execute> ERROR: Alignment and stacking not done for "+fdu.getFullId()+" (exp time="+str(fdu.exptime)+"; nreads="+str(fdu.nreads)+").  Discarding Image!")
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
        asfile = "alignedStacked/as_"+fdu._id+".fits"
        if (self.checkOutputExists(fdu, asfile)):
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
                        print("alignStackProcess::getCalibs> Warning: invalid index in file "+use_indices+": "+str(indList[j]))
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
                    print("alignStackProcess::getCalibs> Error: invalid format in use_indices: "+str(ex)+": No Frames will be stacked!")
                    self._log.writeLog(__name__, " invalid format in use_indices: "+str(ex)+": No Frames will be stacked!", type=fatboyLog.ERROR)
                    indList = []
            #Loop backwards over fdu list so as to be able to remove cleanly
            for j in range(len(fdus)-1, -1, -1):
                if (indList.count(int(fdus[j]._index)) == 0):
                    #FDU is not in indList
                    print("alignStackProcess::getCalibs> Frame "+fdus[j].getFullId()+" is not in index list.  Discarding!")
                    self._log.writeLog(__name__, "Frame "+fdus[j].getFullId()+" is not in index list.  Discarding!")
                    #Disable so its not picked up by alignStack later and then remove from fdu list
                    fdus[j].disable()
                    fdus.pop(j)

        if (len(fdus) > 0):
            #Found other objects associated with this fdu.  Create aligned stacked image
            print("alignStackProcess::getCalibs> Creating aligned stacked image for object "+fdu._id+", exposure time "+str(fdu.exptime)+", and "+str(fdu.nreads)+" reads...")
            self._log.writeLog(__name__, "Creating aligned stacked image for object "+fdu._id+", exposure time "+str(fdu.exptime)+", and "+str(fdu.nreads)+" reads...")
            #First recursively process
            self.recursivelyExecute(fdus, prevProc)
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
        self._optioninfo.setdefault('align_method', 'xregister | xregister_constrained | xregister_sep |\nxregister_sep_constrained | xregister_guesses |\nsep_centroid | sep_centroid_constrained | triangles | manual')
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
        self._options.setdefault('stack_method', 'drihizzle') #drihizzle, drihizzle_imcombine, etc
        self._optioninfo.setdefault('stack_method', 'drihizzle | drihizzle_imcombine')
        self._options.setdefault('stack_hsigma', '3')
        self._options.setdefault('stack_lsigma', '3')
        self._options.setdefault('stack_nhigh', '3')
        self._options.setdefault('stack_nlow', '3')
        self._options.setdefault('stack_reject_type', 'sigclip')
        self._options.setdefault('triangles', 'delaunay') #delaunay tringles only or all triangles within angle limits
        self._optioninfo.setdefault('triangles', 'delaunay | all')
        self._options.setdefault('triangles_atol', '2.0') #maximum absolute tolerance in pixels for matching triangles
        self._optioninfo.setdefault('triangles_atol', 'maximum absolute tolerance in pixels for matching triangles')
        self._options.setdefault('triangles_debug_plots', 'yes')
        self._options.setdefault('triangles_max_angle', '110')
        self._optioninfo.setdefault('triangles_max_angle', 'max angle for any triangle to have')
        self._options.setdefault('triangles_max_stars', None)
        self._optioninfo.setdefault('triangles_max_stars', 'if not None, max stars to compute triangles from, sorted by flux')
        self._options.setdefault('triangles_min_angle', '30')
        self._optioninfo.setdefault('triangles_min_angle', 'min angle for any triangle to have')
        self._options.setdefault('triangles_rtol', '0.025') #maximum relative tolerance in pixels for matching triangles
        self._optioninfo.setdefault('triangles_rtol', 'maximum relative tolerance in pixels for matching triangles')
        self._options.setdefault('triangles_sigma', '3')
        self._optioninfo.setdefault('triangles_sigma', 'Sigma to use for sigma clipping') 
        self._options.setdefault('triangles_use_sigma_clipping', 'no')
        self._optioninfo.setdefault('triangles_use_sigma_clipping', 'Use sigma clipping on shifts from fit triangles')
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
        stackMethod = self.getOption("stack_method", fdu.getTag()).lower()
        refframe = self.getOption("align_refframe", fdu.getTag())
        dropsize = float(self.getOption('drihizzle_dropsize', fdu.getTag()))
        inunits = self.getOption('drihizzle_in_units', fdu.getTag())
        kernel = self.getOption('drihizzle_kernel', fdu.getTag())
        geomFile = self.getOption('geom_trans_coeffs', fdu.getTag())
        keepImages = self.getOption('keep_indiv_images', fdu.getTag())
        stack_hsigma = int(self.getOption('stack_hsigma', fdu.getTag()))
        stack_lsigma = int(self.getOption('stack_lsigma', fdu.getTag()))
        stack_nhigh = int(self.getOption('stack_nhigh', fdu.getTag()))
        stack_nlow = int(self.getOption('stack_nlow', fdu.getTag()))
        stack_reject_type = self.getOption('stack_reject_type', fdu.getTag())

        #Select cpu/gpu option
        drihizzle_method = gpu_drihizzle.drihizzle
        if (not self._fdb.getGPUMode()):
            drihizzle_method = drihizzle.drihizzle

        if (stackMethod == 'drihizzle' or stackMethod == 'drihizzle_imcombine'):
            #Drihizzle each frame into the final image and exposure mask
            #or drihizzle frames into individual files.
            imgdir = str(self._fdb.getParam("outputdir", fdu.getTag()))+'/alignedStacked'
            if (not os.access(imgdir, os.F_OK)):
                os.mkdir(imgdir)
            xshifts = shifts[0]
            yshifts = shifts[1]
            xshift0 = -1*int(max(xshifts)+min(xshifts))//2
            yshift0 = -1*int(max(yshifts)+min(yshifts))//2
            goodPixelMask = (1-fdu.getBadPixelMask().getData()).astype("int32")
            for i in range(len(frameList)):
                xshifts[i]+=xshift0
                yshifts[i]+=yshift0
                if (frameList[i].getShape() != frameList[0].getShape()):
                    goodPixelMask = gpu_drihizzle.MODE_FDU_USE_INDIVIDUAL_GPMS
        if (stackMethod == 'drihizzle'):
            (data, header, expmap, pixmap) = drihizzle_method(frameList, None, None, inmask=goodPixelMask, weight='exptime', kernel=kernel, dropsize=dropsize, geomDist=geomFile, xsh=xshifts, ysh=yshifts, inunits=inunits, outunits='cps', keepImages=keepImages, imgdir=imgdir, log=self._log, mode=gpu_drihizzle.MODE_FDU)
            fdu.updateData(data)
            fdu.updateHeader(header)
            #Update bad pixel mask to be true where pixmap == 0
            badPixelMask = fatboyCalib(self._pname, fatboyDataUnit.FDU_TYPE_BAD_PIXEL_MASK, fdu, data=(pixmap == 0), tagname="badPixelMasks/BPM-"+str(fdu.filter)+"-"+str(fdu._id), log=self._log)
            fdu.applyBadPixelMask(badPixelMask)
            #fdu.getBadPixelMask().updateData(pixmap == 0)
            if (self.getOption('write_output', fdu.getTag()).lower() == "yes"):
                #only tag these if writing output -- then free up memory after writing output
                fdu.tagDataAs("exposure_map", data=expmap)
                fdu.tagDataAs("pixel_map", data=pixmap)
        elif (stackMethod == 'drihizzle_imcombine'):
            #Use updateFDUs flag to tag individual FDUs with result and expmap
            (data, header, expmap, pixmap) = drihizzle_method(frameList, None, None, inmask=goodPixelMask, weight='exptime', kernel=kernel, dropsize=dropsize, geomDist=geomFile, xsh=xshifts, ysh=yshifts, inunits=inunits, outunits='cps', keepImages=keepImages, imgdir=imgdir, log=self._log, mode=gpu_drihizzle.MODE_FDU, updateFDUs=True)
            for i in range(len(frameList)):
                data = frameList[i].getData(tag="drihizzled")
                #Set bad pixels to -1e+7 so they will be rejected by imcombine
                data[data == 0] = -1e+7
                #Overwrite tag
                frameList[i].tagDataAs("drihizzled", data)

            #Select cpu/gpu option
            imcombine_method = gpu_imcombine.imcombine
            if (not self._fdb.getGPUMode()):
                imcombine_method = imcombine.imcombine

            #imcombine frames and take mean
            (data, imexpmap, imheader) = imcombine_method(frameList, outfile=None, expmask='return_expmask', method="mean", reject=stack_reject_type, nlow=stack_nlow, nhigh=stack_nhigh, lsigma=stack_lsigma, hsigma=stack_hsigma, lthreshold=-1e+6, mef=frameList[0]._mef, log=self._log, returnHeader=True, mode=gpu_imcombine.MODE_FDU_TAG, dataTag="drihizzled")
            expmap = imcombine_method(frameList, outfile=None, method="sum", mef=frameList[0]._mef, log=self._log, mode=gpu_imcombine.MODE_FDU_TAG, dataTag="exposure_map")
            expdiff = zeros(expmap.shape, float32)
            for i in range(len(frameList)):
                fullexp = frameList[i].exptime
                currExp = frameList[i].getData(tag="exposure_map")
                #boolean mapping of where exposure map > total exposure time (more than 1 input pixel contributing flux)
                b = currExp > fullexp
                expdiff += b*(currExp-fullexp)
                #done with dataTags.  Free memory.
                frameList[i].removeProperty("drihizzled")
                frameList[i].removeProperty("exposure_map")
            imexpmap += expdiff
            #Find where sum of expmaps < current total imexpmap
            b = expmap < imexpmap
            imexpmap -= b*(imexpmap-expmap)
            #Free memory
            del expmap
            del expdiff
            #Now update fdu with new data, header
            fdu.updateData(data)
            fdu.updateHeader(header) #drihizzle keywords
            fdu.updateHeader(imheader) #imcombine keywords
            #Update bad pixel mask to be true where pixmap == 0
            badPixelMask = fatboyCalib(self._pname, fatboyDataUnit.FDU_TYPE_BAD_PIXEL_MASK, fdu, data=(pixmap == 0), tagname="badPixelMasks/BPM-"+str(fdu.filter)+"-"+str(fdu._id), log=self._log)
            fdu.applyBadPixelMask(badPixelMask)
            #fdu.getBadPixelMask().updateData(pixmap == 0)
            if (self.getOption('write_output', fdu.getTag()).lower() == "yes" or self.getOption('keep_exposure_map', fdu.getTag()).lower() == "yes"):
                #only tag these if writing output -- then free up memory after writing output
                #also save these if requested via keep_exposure_map
                fdu.tagDataAs("exposure_map", data=imexpmap)
                fdu.tagDataAs("pixel_map", data=pixmap)
    #end alignFrames

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/alignedStacked", os.F_OK)):
            os.mkdir(outdir+"/alignedStacked",0o755)
        #Create output filenames
        asfile = outdir+"/alignedStacked/as_"+fdu._id+".fits"
        expfile = outdir+"/alignedStacked/exp_"+fdu._id+".fits"
        objfile = outdir+"/alignedStacked/objmap_"+fdu._id+".fits"
        #Check to see if it exists
        if (os.access(asfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(asfile)
        if (os.access(expfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(expfile)
        if (os.access(objfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(objfile)
        if (not os.access(asfile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(asfile)
        if (not os.access(expfile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(expfile, tag="exposure_map")
        if (not os.access(objfile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(objfile, tag="pixel_map")
        #free memory from exposure_map and pixel_map tags if requested
        if (self.getOption('keep_exposure_map', fdu.getTag()).lower() != "yes"):
            fdu.removeProperty("exposure_map")
            fdu.removeProperty("pixel_map")
    #end writeOutput
