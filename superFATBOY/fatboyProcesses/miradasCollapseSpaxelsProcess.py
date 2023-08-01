from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY import drihizzle
import os, time

usePlot = True
try:
    import matplotlib.pyplot as plt
except Exception as ex:
    print("Warning: Could not import matplotlib!")
    usePlot = False

class miradasCollapseSpaxelsProcess(fatboyProcess):
    """ Collapse each slice to obtain a 3 spaxel x n image for each slitlet.  This can then be used for finding PSFs.
          Also centroiding of 3xn images can be done here.  """
    _modeTags = ["miradas"]

    def centroidImages(self, fdu, calibs):
        #read options
        centroid_method = self.getOption("centroid_method", fdu.getTag())
        nslits = fdu.getProperty("nslits")
        for j in range(1, nslits+1):
            if (fdu.hasProperty("image_slice_"+str(j))):
                data = fdu.getData(tag="image_slice_"+str(j))
                if (usePlot and self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                    plt.plot(data.sum(0))
                    plt.plot(data.sum(1))
                if (centroid_method == "fit_2d_gaussian"):
                    p = zeros(5, float32)
                    p[0] = data.max()
                    b = where(data == p[0])
                    if (fdu.hasProperty("xcen_guess_"+str(j))):
                        p[1] = fdu.getProperty("xcen_guess_"+str(j))
                    else:
                        p[1] = b[1][0]
                    p[2] = b[0][0]
                    p[3] = 2#convert from FWHM in pixels
                    p[4] = arraymedian(data)
                    maskNeg = False
                    if (p[4] > 0):
                        maskNeg = True
                    lsq = fitGaussian2d(data, guess=p, maskNeg=maskNeg)
                    if (lsq[1] == False):
                        print("miradasCollapseSpaxelsProcess::centroidImages> ERROR: Could not centroid slitlet "+str(j))
                        self._log.writeLog(__name__, "Could not centroid slitlet "+str(j), type=fatboyLog.ERROR)
                        continue
                    xcen = lsq[0][1]
                    ycen = lsq[0][2]
                    fwhm = lsq[0][3]*2.3548
                else:
                    #use_derivatives
                    #need to pad data if either dimension < 9
                    padx = max(9-data.shape[1], 0)//2
                    pady = max(9-data.shape[0], 0)//2
                    padded = zeros((data.shape[0]+pady*2, data.shape[1]+padx*2), dtype=float32)
                    padded[pady:padded.shape[0]-pady, padx:padded.shape[1]-padx] = data
                    b = where(padded == padded.max())
                    mx = b[1][0]
                    my = b[0][0]
                    (fwhm, sig, fwhm1ds, bg) = fwhm2d(padded)
                    (xcen, ycen) = getCentroid(padded, mx, my, fwhm)
                    xcen -= padx
                    ycen -= pady
                print("miradasCollapseSpaxelsProcess::centroidImages> Slitlet "+str(j)+" image centroid with "+centroid_method+": (x="+formatNum(xcen)+"; y="+formatNum(ycen)+") fwhm="+formatNum(fwhm))
                self._log.writeLog(__name__, "Slitlet "+str(j)+" image centroid with "+centroid_method+": (x="+formatNum(xcen)+"; y="+formatNum(ycen)+") fwhm="+formatNum(fwhm))
                if (usePlot and self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                    plt.show()
                fdu.setProperty("image_centroid_"+str(j), (xcen, ycen, fwhm))
    #end centroidImages

    def collapseSpaxels(self, fdu, calibs):
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
        pad = False
        if (self.getOption("pad_images_to_same_size", fdu.getTag()).lower() == "yes"):
            pad = True
        box_lo = 0
        max_width = 0

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
            box_hi = data1d.shape[0]+1
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            data1d = fdu.getData().sum(0)
            clean1d = fdu.getData(tag="cleanFrame").sum(0)
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
            max_edge = len(z)//(nslices*2)+1
            left_edge = b[0][0]
            if (left_edge > max_edge):
                left_edge = max_edge
            right_edge = b[0][-1]
            if (right_edge < len(z)-max_edge+1):
                right_edge = len(z)-max_edge+1
            z[:left_edge] = 0
            z[right_edge:] = 0
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
                print("miradasCollapseSpaxelsProcess::collapseSpaxels> Warning: Could not identify spaxels in slit "+str(islit)+". Guessing at slice width.")
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
            if (slice_width < 10):
                slice_width = 23
            yinit -= slice_width * ((nslices-1)//2)
            if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                print("YINIT = "+str(yinit)+"; SLICE WIDTH = "+str(slice_width))

            #Loop over slices and fit each with Gaussian to find shifts to subpixel accuracy
            s = 'Slitlet '+str(islit)+': Slices of width '+str(slice_width)+' at ['
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
                    print("miradasCollapseSpaxelsProcess::collapseSpaxels> Warning: Failed to compute non-integer shifts for slit "+str(islit)+". Using integer shifts instead.")
                    self._log.writeLog(__name__, "Failed to compute non-integer shifts for slit "+str(islit)+". Using integer shifts instead.", type=fatboyLog.WARNING)
            for j in range(len(ys)):
                if (j != 0):
                    s += ', '
                s += formatNum(ys[j])

            max_width = max(max_width, image2d.shape[1])
            #Print info to screen and log
            s += ']'
            print("miradasCollapseSpaxelsProcess::collapseSpaxels> "+s)
            self._log.writeLog(__name__, s)

            if (usePlot and self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                plt.plot(z)
                plt.show()
            #Tag data
            fdu.tagDataAs("image_slice_"+str(islit), image2d)
            #Tag property with start indices (to subpixel accuracy) and slice width
            #This can be re-used when reconstructing 3-d data cubes
            fdu.setProperty("image_slice_coords_"+str(islit), (ys, slice_width))
            if (cen != 0):
                fdu.setProperty("xcen_guess_"+str(islit), cen)
        if (pad):
            #Loop over slitlets (could be one pass or nslits passes)
            for islit in slitlets:
                data = fdu.getData(tag="image_slice_"+str(islit))
                if (data.shape[1] < max_width):
                    image2d = zeros((data.shape[0], max_width), float32)
                    image2d[:,:data.shape[1]] = data
                    fdu.tagDataAs("image_slice_"+str(islit), image2d)
    #end collapseSpaxels

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
        maskNeg = False
        if (p[3] >= 0):
            maskNeg = True
        lsq = fitGaussian(oned, guess=p, maskNeg=maskNeg)
        if (lsq[1] == False):
            return (False, images, ys, cen)
        cen0 = lsq[0][1]
        if (usePlot and self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
            print(lsq)
            plt.plot(oned)

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
            if (p[3] >= 0):
                maskNeg = True
            lsq = fitGaussian(clean_images[j], guess=p, maskNeg=maskNeg)
            if (lsq[1] == False):
                return (False, images, ys)
            if (usePlot and self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                plt.plot(clean_images[j])
                print(j, lsq)
            if (j == 0):
                cen = lsq[0][1]
            xcen = cen-lsq[0][1]
            if (abs(xcen) > max_shift):
                return (False, images, ys, cen)
            xsh.append(xcen)
            ysh.append(float(j))
            ys.append(yinit+j*slice_width-xsh[j])
            images[j] = images[j].reshape((1, len(images[j])))
        if (usePlot and self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
            plt.show()

        #Drizzle together to create final image
        (image2d, head, exp, pix) = drihizzle.drihizzle(images, xsh=xsh, ysh=ysh, kernel='turbo', dropsize=1)
        image2d = image2d[:-1,:] #Remove blank last row
        return (True, image2d, ys, cen)
    #end computeNonIntegerShifts

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("MIRADAS: collapse spaxels")
        print(fdu._identFull)

        #Check if output exists first and update from disk
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        csfile = outdir+"/collapsedSpaxels/cs_"+fdu.getFullId()
        nslits = 13 #max number of slits
        if (fdu.hasProperty("nslits")):
            nslits = fdu.getProperty("nslits")
        tag = ["TAGNAME"]*nslits
        if (self.checkOutputExists(fdu, csfile, tag=tag)):
            #This will update properties listed in FITS header extensions
            #with keyword TAGNAME, e.g. image_slice_n
            #Must manually update image_centroid_n from headers
            temp = pyfits.open(csfile)
            fdu.setProperty("nslits", len(temp)-1)
            for j in range(1, len(temp)):
                if ('XCEN' in temp[j].header):
                    xcen = temp[j].header['XCEN']
                    ycen = temp[j].header['YCEN']
                    fwhm = temp[j].header['FWHM']
                    fdu.setProperty("image_centroid_"+str(j), (xcen, ycen, fwhm))
            temp.close()
            return True

        #Call get calibs to return dict() of calibration frames.
        #For collapsedSpaxels, this dict can have a noisemap if this is not a property
        #of the FDU at this point.  It may also have an text file listing the
        #slitlet indices of each row in the RSS file if this is not in the header already.
        calibs = self.getCalibs(fdu, prevProc)

        if (not 'slitmask' in calibs):
            print("miradasCollapseSpaxelsProcess::execute> ERROR: Could not find slitmask so could not collapse spaxels.")
            self._log.writeLog(__name__, "Could not find slitmask so could not collapse spaxels.", type=fatboyLog.ERROR)
            return False
        if (not 'masterLamp' in calibs and not 'cleanSky' in calibs):
            print("miradasCollapseSpaxelsProcess::execute> ERROR: Could not find master arclamp or clean sky frame.  Could not collapse spaxels.")
            self._log.writeLog(__name__, "Could not find master arclamp or clean sky frame.  Could not collapse spaxels.", type=fatboyLog.ERROR)
            return False

        #call collapseSpaxels helper function to do actual processing into 2d images and data tagging
        self.collapseSpaxels(fdu, calibs)
        if (self.getOption("do_centroids", fdu.getTag()).lower() == "yes"):
            self.centroidImages(fdu, calibs)
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
                print("miradasCollapseSpaxelsProcess::getCalibs> Using slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Using slitmask "+smfilename+"...")
                calibs['slitmask'] = fatboySpecCalib(self._pname, "slitmask", fdu, filename=smfilename, log=self._log)
            else:
                print("miradasCollapseSpaxelsProcess::getCalibs> Warning: Could not find slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Could not find slitmask "+smfilename+"...", type=fatboyLog.WARNING)

        #Look for each master calib passed from XML
        mlfilename = self.getCalib("master_arclamp", fdu.getTag())
        if (mlfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(mlfilename, os.F_OK)):
                print("miradasCollapseSpaxelsProcess::getCalibs> Using master arclamp frame "+mlfilename+"...")
                self._log.writeLog(__name__, "Using master arclamp frame "+mlfilename+"...")
                calibs['masterLamp'] = fatboySpecCalib(self._pname, "master_arclamp", fdu, filename=mlfilename, log=self._log)
            else:
                print("miradasCollapseSpaxelsProcess::getCalibs> Warning: Could not find master arclamp frame "+mlfilename+"...")
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
                #add to calibs
                calibs['masterLamp'] = masterLamp
                #Update skyShape for finding slitmask
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
        self._options.setdefault('box_lo', None)
        self._optioninfo.setdefault('box_lo', 'Bounds of extraction box in dispersion axis\nwhere data exists in arclamp.')
        self._options.setdefault('box_hi', None)
        self._optioninfo.setdefault('box_hi', 'Bounds of extraction box in dispersion axis\nwhere data exists in arclamp.')
        self._options.setdefault('centroid_method', 'fit_2d_gaussian')
        self._optioninfo.setdefault('centroid_method', 'fit_2d_gaussian | use_derivatives')
        self._options.setdefault('debug_mode', 'no')
        self._options.setdefault('do_centroids', 'yes')
        self._optioninfo.setdefault('do_centroids', 'Whether to centroid images')
        self._options.setdefault('max_shift_between_slices', '4')
        self._optioninfo.setdefault('max_shift_between_slices', 'max shift between any slice and the mean center.')
        self._options.setdefault('nslices', '3')
        self._optioninfo.setdefault('nslices', 'Number of slices per slitlet')
        self._options.setdefault('pad_images_to_same_size', 'yes')
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
        if (not os.access(outdir+"/collapsedSpaxels", os.F_OK)):
            os.mkdir(outdir+"/collapsedSpaxels",0o755)
        nslits = fdu.getProperty("nslits")
        #Set up lists for tag and headerExt
        tags = []
        headerExt = []
        for j in range(1, nslits+1):
            key = "image_slice_"+str(j)
            if (fdu.hasProperty(key)):
                tags.append(key)
                #Create new header dict for this extension
                imhead = dict()
                #Add values
                imhead['TAGNAME'] = key
                imhead['SLIT_NUM'] = j
                (xcen, ycen, fwhm) = fdu.getProperty("image_centroid_"+str(j))
                imhead['XCEN'] = xcen
                imhead['YCEN'] = ycen
                imhead['FWHM'] = fwhm
                headerExt.append(imhead)
        #Create output filename
        csfile = outdir+"/collapsedSpaxels/cs_"+fdu.getFullId()
        #Check to see if it exists
        if (os.access(csfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(csfile)
        if (not os.access(csfile, os.F_OK)):
            #Use fatboyDataUnit writePropertiesToMEF method to write
            fdu.writePropertiesToMEF(csfile, tag=tags, headerExt=headerExt)
    #end writeOutput
