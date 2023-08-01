from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY import gpu_drihizzle, drihizzle
import os, time

class miradasCreate3dDatacubesProcess(fatboyProcess):
    """ Create a 3-d datacube where each "cut" is a 3xn monochromatic image at a given wavelength """
    _modeTags = ["miradas"]

    def createDatacubes(self, fdu, calibs):
        #read options
        nslices = int(self.getOption("nslices", fdu.getTag()))
        slice_width = self.getOption("slice_width", fdu.getTag())
        max_shift = int(self.getOption("max_shift_between_slices", fdu.getTag()))
        if (isInt(slice_width)):
            slice_width = int(slice_width)
            findSliceWidth = False
        else:
            findSliceWidth = True
        slitlet_number = self.getOption("slitlet_number", fdu.getTag())
        doAllSlitlets = False
        if (slitlet_number == 'all'):
            doAllSlitlets = True
        elif (isInt(slitlet_number)):
            slitlet_number = int(slitlet_number)
        integerShifts = False
        if (self.getOption("use_integer_shifts", fdu.getTag()).lower() == "yes"):
            integerShifts = True
        box_lo = 0

        drihizzleKernel = self.getOption("drihizzle_kernel", fdu.getTag()).lower()
        dropsize = float(self.getOption("drihizzle_dropsize", fdu.getTag()))

        if (fdu.hasProperty("nslits")):
            nslits = fdu.getProperty("nslits")
        else:
            nslits = calibs['slitmask'].getData().max()
            fdu.setProperty("nslits", nslits)

        if (doAllSlitlets):
            slitlets = arange(1, nslits+1)
        else:
            slitlets = [slitlet_number]

        hasCleanFrame = False
        if (fdu.hasProperty("cleanFrame")):
            hasCleanFrame = True

        #Collapse image across wavelengths
        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            data1d = fdu.getData().sum(1)
            clean1d = fdu.getData(tag="cleanFrame").sum(1)
            xsize = fdu.getShape()[1]
            box_hi = data1d.shape[0]+1
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            data1d = fdu.getData().sum(0)
            clean1d = fdu.getData(tag="cleanFrame").sum(0)
            xsize = fdu.getShape()[0]
            box_hi = data1d.shape[0]+1

        if (self.getOption("box_lo", fdu.getTag()) is not None):
            box_lo = int(self.getOption("box_lo", fdu.getTag()))
        if (self.getOption("box_hi", fdu.getTag()) is not None):
            box_hi = int(self.getOption("box_hi", fdu.getTag()))

        lampkey = 'cleanSky'
        if ('masterLamp' in calibs):
            #Use master arclamp if it exists
            lampkey = 'masterLamp'

        #Loop over slitlets (could be one pass or nslits passes)
        for islit in slitlets:
            #Check if image_slice_coords already saved from collapseSpaxels
            if (fdu.hasProperty("image_slice_coords_"+str(islit))):
                (ys, slice_width) = fdu.getProperty("image_slice_coords_"+str(islit))
            else:
                #Rerun same calcs as in miradasCollapseSpaxels
                #Take 1-d cut of arclamp in each slitlet
                #b = where(calibs['slitmask'].getData() == islit)
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    b = where(calibs['slitmask'].getData()[:,box_lo:box_hi] == islit)
                    ylo = min(b[0])
                    yhi = max(b[0])
                    lamp1d = (calibs[lampkey].getData()[ylo:yhi,box_lo:box_hi]*(calibs['slitmask'].getData()[ylo:yhi,box_lo:box_hi] == islit)).sum(1)
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    b = where(calibs['slitmask'].getData()[box_lo:box_hi,:] == islit)
                    ylo = min(b[1])
                    yhi = max(b[1])
                    lamp1d = (calibs[lampkey].getData()[box_lo:box_hi,ylo:yhi]*(calibs['slitmask'].getData()[box_lo:box_hi,ylo:yhi] == islit)).sum(0)
                #Median filter 1-d cut and invert so that "gaps" between slices turn into peaks
                z = medianfilterCPU(lamp1d)
                #Correct for values on the edges of the slitlet which will now be very negative
                b = where(z > 0)
                z[:b[0][0]] = 0
                z[b[0][-1]:] = 0
                if (findSliceWidth):
                    offset = len(z)//nslices-1
                else:
                    offset = slice_width-1
                if (nslices % 2 == 1):
                    zx = len(z)//2
                else:
                    zx = len(z)//2-offset//2
                z = -1*z[zx-offset:zx+offset+1]
                slo = where(z == max(z[:offset]))[0][0]
                shi = where(z == max(z[offset:]))[0][0]
                #Try fitting Gaussians to "peaks"
                lsq_lo = fitGaussian(z[:offset])
                lsq_hi = fitGaussian(z[offset:])
                fit_slo = lsq_lo[0][1]
                fit_shi = lsq_hi[0][1]+offset
                if (lsq_lo[1] == False or lsq_hi[1] == False):
                    print("miradasCreate3dDatacubesProcess::createDatacubes> Warning: Could not identify spaxels in slit "+str(islit)+". Guessing at slice width.")
                    self._log.writeLog(__name__, "Could not identify spaxels in slit "+str(islit)+". Guessing at slice width.", type=fatboyLog.WARNING)
                    fit_slo = len(z)//nslices
                    fit_shi = fit_slo*2

                if (abs(fit_slo-slo) <= 1 and abs(fit_shi-shi) <= 1):
                    #Use fits
                    if (findSliceWidth):
                        slice_width = int(ceil(fit_shi - fit_slo))
                    yinit = ylo+zx-offset+int(round(fit_slo))
                else:
                    if (findSliceWidth):
                        slice_width = shi - slo
                    yinit = ylo+zx-offset+slo
                yinit -= slice_width * ((nslices-1)//2)

                #Loop over slices and fit each with Gaussian to find shifts to subpixel accuracy
                ys = []
                cen = 0

                #Always compute integer shifts first in case non integer fails
                #Concat string below
                image2d = zeros((nslices, slice_width), float32)
                for j in range(nslices):
                    image2d[j,:] = data1d[yinit+j*slice_width:yinit+(j+1)*slice_width]
                    ys.append(yinit+j*slice_width)

                if (not integerShifts):
                    (success, nis_image2d, nis_ys, nis_cen) = self.computeNonIntegerShifts(fdu, nslices, slice_width, clean1d, data1d, hasCleanFrame, yinit, max_shift)
                    if (success):
                        image2d = nis_image2d
                        ys = nis_ys
                        cen = nis_cen
                    else:
                        print("miradasCreate3dDatacubesProcess::createDatacubes> Warning: Failed to compute non-integer shifts for slit "+str(islit)+". Using integer shifts instead.")
                        self._log.writeLog(__name__, "Failed to compute non-integer shifts for slit "+str(islit)+". Using integer shifts instead.", type=fatboyLog.WARNING)

            s = 'Slitlet '+str(islit)+': Slices of width '+str(slice_width)+' at ['
            for j in range(len(ys)):
                if (j != 0):
                    s += ', '
                s += str(ys[j])
            #Print info to screen and log
            s += ']'
            print("miradasCreate3dDatacubesProcess::createDatacubes> "+s)
            self._log.writeLog(__name__, s)

            image3d = []
            data = fdu.getData()
            ysh = arange(nslices, dtype=float32)
            yinit = int(ys[0])
            xsh = yinit+ysh*slice_width-ys
            images = []
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                for j in range(nslices):
                    images.append(data[yinit+j*slice_width:yinit+(j+1)*slice_width,:].transpose())
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                for j in range(nslices):
                    images.append(data[:, yinit+j*slice_width:yinit+(j+1)*slice_width])
            #Create yind array of indices to layer slices in a xsize*nslices x slice_width 2-d array
            #This will later be reshaped to xsize x nslices x slice_width 3-d cube
            yind = (arange(images[0].size).reshape(images[0].shape) // slice_width).astype(float32)*nslices

            #Select cpu/gpu option
            drihizzle_method = gpu_drihizzle.drihizzle
            if (not self._fdb.getGPUMode()):
                drihizzle_method = drihizzle.drihizzle
            #Drizzle together to create final image
            (image2d, head, exp, pix) = drihizzle.drihizzle(images, ytrans=yind, xsh=xsh, ysh=ysh, kernel=drihizzleKernel, dropsize=dropsize)
            #Reshape - use image2d.shape[1] not slice_width because of shifts, slices are wider
            image3d = image2d[:-1,:-1].reshape((xsize, nslices, image2d.shape[1]-1))

#   *** OLD inefficient row by row code ***
#      for z in range(xsize):
#        if (not integerShifts):
#         images = []
#          for j in range(nslices):
#           images.append(data[z, yinit+j*slice_width:yinit+(j+1)*slice_width])
#            images[j] = images[j].reshape((1, len(images[j])))
#          #Drizzle together to create final image
#          (image2d, head, exp, pix) = drihizzle.drihizzle(images, xsh=xsh, ysh=ysh, kernel='turbo', dropsize=1)
#          image2d = image2d[:-1,:] #Remove blank last row
#          image3d.append(image2d)
#       else:
#         image3d.append(zeros((nslices, slice_width), float32))
#         image3d[-1][j,:] = data[z, yinit+j*slice_width:yinit+(j+1)*slice_width]
#      image3d = array(image3d)

            #Tag data
            fdu.tagDataAs("datacube_"+str(islit), image3d)
            #Tag property with start indices (to subpixel accuracy) and slice width
            #This can be re-used when reconstructing 3-d data cubes
            fdu.setProperty("image_slice_coords_"+str(islit), (ys, slice_width))
    #end createDatacubes

    #Helper method to compute non-integer shifts - returns False as first arg if fails
    #Allows to use integer shifts instead in this case
    def computeNonIntegerShifts(self, fdu, nslices, slice_width, clean1d, data1d, hasCleanFrame, yinit, max_shift):
        images = []
        clean_images = []
        xsh = []
        ysh = []
        ys = []
        cen = 0
        #Use all slices to come up with initial guess
        oned = zeros(slice_width, float32)
        for j in range(nslices):
            oned += clean1d[yinit+j*slice_width:yinit+(j+1)*slice_width]
        if (not hasCleanFrame):
            b = where(oned == oned.max())[0][0]
            if (b <= 2 or b >= len(oned)-3):
                #Set edges to min value (not zero in case negative)
                oned[:3] = oned.min()
                oned[-3:] = oned.min()
        p = zeros(4, float32)
        p[0] = oned.max()
        p[1] = where(oned == oned.max())[0][0]
        p[2] = 2.0
        p[3] = arraymedian(oned)
        lsq = fitGaussian(oned, guess=p)
        if (lsq[1] == False):
            return (False, images, ys, cen)
        cen0 = lsq[0][1]

        for j in range(nslices):
            images.append(data1d[yinit+j*slice_width:yinit+(j+1)*slice_width])
            clean_images.append(clean1d[yinit+j*slice_width:yinit+(j+1)*slice_width])
            if (not hasCleanFrame):
                b = where(clean_images[j] == clean_images[j].max())[0][0]
                if (b <= 2 or b >= len(oned)-3):
                    #Blank out edges with min value not 0 in case all values are negative
                    clean_images[j][:3] = clean_images[j].min()
                    clean_images[j][-3:] = clean_images[j].min()
            p[0] = clean_images[j][int(p[1])]
            p[3] = arraymedian(clean_images[j])
            maskNeg = False
            if (p[3] > 0):
                maskNeg = True
            lsq = fitGaussian(clean_images[j], guess=p, maskNeg=maskNeg)
            if (lsq[1] == False):
                return (False, images, ys)
            if (j == 0):
                cen = lsq[0][1]
            xcen = cen-lsq[0][1]
            if (abs(xcen) > max_shift):
                return (False, images, ys, cen)
            xsh.append(xcen)
            ysh.append(float(j))
            ys.append(yinit+j*slice_width-xsh[j])
            images[j] = images[j].reshape((1, len(images[j])))
        #Drizzle together to create final image
        (image2d, head, exp, pix) = drihizzle.drihizzle(images, xsh=xsh, ysh=ysh, kernel='turbo', dropsize=1)
        image2d = image2d[:-1,:] #Remove blank last row
        return (True, image2d, ys, cen)
    #end computeNonIntegerShifts



    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("MIRADAS: create 3d datacubes")
        print(fdu._identFull)

        #Check if output exists first and update from disk
        csfile = "3dDatacubes/3d_"+fdu.getFullId()
        nslits = 13 #max number of slits
        if (fdu.hasProperty("nslits")):
            nslits = fdu.getProperty("nslits")
        tag = ["TAGNAME"]*nslits
        if (self.checkOutputExists(fdu, csfile, tag=tag)):
            return True

        #Call get calibs to return dict() of calibration frames.
        #For 3dDatacubes, this dict can have a noisemap if this is not a property
        #of the FDU at this point.  It may also have an text file listing the
        #slitlet indices of each row in the RSS file if this is not in the header already.
        calibs = self.getCalibs(fdu, prevProc)

        if (not 'slitmask' in calibs):
            print("miradasCreate3dDatacubesProcess::execute> ERROR: Could not find slitmask so could not collapse spaxels.")
            self._log.writeLog(__name__, "Could not find slitmask so could not collapse spaxels.", type=fatboyLog.ERROR)
            return False
        if (not 'masterLamp' in calibs and not 'cleanSky' in calibs):
            print("miradasCreate3dDatacubesProcess::execute> ERROR: Could not find master arclamp or clean sky frame.  Could not collapse spaxels.")
            self._log.writeLog(__name__, "Could not find master arclamp or clean sky frame.  Could not collapse spaxels.", type=fatboyLog.ERROR)
            return False

        #call createDatacubes helper function to do actual processing into 2d images and data tagging
        self.createDatacubes(fdu, calibs)
        return True
    #end execute

    ## OVERRIDE getCalibs
    def getCalibs(self, fdu, prevProc = None):
        calibs = dict()
        #Look for each calib passed from XML
        smfilename = self.getCalib("slitmask", fdu.getTag())
        if (smfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(smfilename, os.F_OK)):
                print("miradasCreate3dDatacubesProcess::getCalibs> Using slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Using slitmask "+smfilename+"...")
                calibs['slitmask'] = fatboySpecCalib(self._pname, "slitmask", fdu, filename=smfilename, log=self._log)
            else:
                print("miradasCreate3dDatacubesProcess::getCalibs> Warning: Could not find slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Could not find slitmask "+smfilename+"...", type=fatboyLog.WARNING)

        #Look for each master calib passed from XML
        mlfilename = self.getCalib("master_arclamp", fdu.getTag())
        if (mlfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(mlfilename, os.F_OK)):
                print("miradasCreate3dDatacubesProcess::getCalibs> Using master arclamp frame "+mlfilename+"...")
                self._log.writeLog(__name__, "Using master arclamp frame "+mlfilename+"...")
                calibs['masterLamp'] = fatboySpecCalib(self._pname, "master_arclamp", fdu, filename=mlfilename, log=self._log)
            else:
                print("miradasCreate3dDatacubesProcess::getCalibs> Warning: Could not find master arclamp frame "+mlfilename+"...")
                self._log.writeLog(__name__, "Could not find master arclamp frame "+mlfilename+"...", type=fatboyLog.WARNING)

        #Look for matching grism_keyword, specmode, and dispersion
        headerVals = dict()
        headerVals['grism_keyword'] = fdu.grism

        properties = dict()
        properties['specmode'] = fdu.getProperty("specmode")
        properties['dispersion'] = fdu.getProperty("dispersion")

        skyShape = fdu.getShape()

        if (not 'masterLamp' in calibs):
            #1) Check for an already created master arclamp frame matching specmode/filter/grism and TAGGED for this object
            masterLamp = self._fdb.getTaggedMasterCalib(ident=fdu._id, obstype="master_arclamp", filter=fdu.filter, properties=properties, headerVals=headerVals)
            if (masterLamp is None):
                #2) Check for an already created master arclamp frame frame matching specmode/filter/grism
                masterLamp = self._fdb.getMasterCalib(obstype="master_arclamp", filter=fdu.filter, properties=properties, headerVals=headerVals, tag=fdu.getTag())
            if (masterLamp is not None):
                #add to calibs for rectification below
                calibs['masterLamp'] = masterLamp
                #update skyShape for finding slitmask
                skyShape = masterLamp.getShape()

        if (not 'masterLamp' in calibs):
            #Master arclamp not found, try cleanSky instead
            if (not 'cleanSky' in calibs):
                #Check for an already created clean sky frame frame matching specmode/filter/grism/ident
                cleanSky = self._fdb.getMasterCalib(ident = "cleanSky_ds_"+fdu._id, filter=fdu.filter, obstype="master_clean_sky", properties=properties, headerVals=headerVals, tag=fdu.getTag())
                if (cleanSky is not None):
                    #add to calibs
                    calibs['cleanSky'] = cleanSky
                    #Update skyShape for finding slitmask
                    skyShape = cleanSky.getShape()

        if (not 'slitmask' in calibs):
            #Use new fdu.getSlitmask method
            slitmask = fdu.getSlitmask(pname=None, shape=skyShape, properties=properties, headerVals=headerVals)
            if (slitmask is not None):
                #Found slitmask
                calibs['slitmask'] = slitmask
        return calibs
    #end getCalibs

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('apply_dar_correction', 'yes')
        self._optioninfo.setdefault('apply_dar_correction', 'Whether to apply a DAR correction if it exists')
        self._options.setdefault('centroid_method', 'fit_2d_gaussian')
        self._optioninfo.setdefault('centroid_method', 'fit_2d_gaussian | use_derivatives')
        self._options.setdefault('do_centroids', 'yes')
        self._optioninfo.setdefault('do_centroids', 'Whether to centroid images')

        self._options.setdefault('drihizzle_dropsize', '1')
        self._options.setdefault('drihizzle_kernel', 'turbo')
        self._optioninfo.setdefault('drihizzle_kernel', 'turbo | point | point_replace | tophat | gaussian | fastgauss | lanczos')

        self._options.setdefault('max_shift_between_slices', '4')
        self._optioninfo.setdefault('max_shift_between_slices', 'max shift between any slice and the mean center.')
        self._options.setdefault('nslices', '3')
        self._optioninfo.setdefault('nslices', 'Number of slices per slitlet')
        self._options.setdefault('slice_width', 'auto')
        self._optioninfo.setdefault('slice_width', 'Slice width in pixels or auto to auto-detect.')
        self._options.setdefault('slitlet_number', 'all')
        self._optioninfo.setdefault('slitlet_number', 'Set to all (default) to collapse spaxels for all slitlets.\nSet to a number 1-13 to only select one slitlet.')
        self._options.setdefault('use_integer_shifts', 'no')
        self._optioninfo.setdefault('use_integer_shifts', 'If set to yes, integer shifts will be used and data\nwill not be resampled.')
    #end setDefaultOptions

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/3dDatacubes", os.F_OK)):
            os.mkdir(outdir+"/3dDatacubes",0o755)
        nslits = fdu.getProperty("nslits")
        if (nslits is None):
            nslits = 13
        #Set up lists for tag and headerExt
        tags = []
        headerExt = []
        for j in range(1, nslits+1):
            key = "datacube_"+str(j)
            if (fdu.hasProperty(key)):
                tags.append(key)
                #Create new header dict for this extension
                imhead = dict()
                #Add values
                imhead['TAGNAME'] = key
                imhead['SLIT_NUM'] = j
                headerExt.append(imhead)
        #Create output filename
        csfile = outdir+"/3dDatacubes/3d_"+fdu.getFullId()
        #Check to see if it exists
        if (os.access(csfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(csfile)
        if (not os.access(csfile, os.F_OK)):
            #Use fatboyDataUnit writePropertiesToMEF method to write
            fdu.writePropertiesToMEF(csfile, tag=tags, headerExt=headerExt)
    #end writeOutput
