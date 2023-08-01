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

class sinfoniCollapseSlitletsProcess(fatboyProcess):
    """ Collapse all slitlets - use clean sky or arclamp to find shifts between
          slitlets and create 2D image.  """
    _modeTags = ["sinfoni"]

    def calculateShiftsBetweenSlitlets(self, fdu, calibs, lampkey):
        refslit = int(self.getOption("reference_slit", fdu.getTag()))

        integerShifts = False
        if (self.getOption("use_integer_shifts", fdu.getTag()).lower() == "yes"):
            integerShifts = True

        debug = False
        if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
            debug = True
        writePlots = False
        if (self.getOption("write_plots", fdu.getTag()).lower() == "yes"):
            writePlots = True

        if (writePlots or self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            #make directory if necessary
            outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
            if (not os.access(outdir+"/collapsedSlitlets", os.F_OK)):
                os.mkdir(outdir+"/collapsedSlitlets",0o755)

        searchbox_lo = int(self.getOption("line_search_box_lo", fdu.getTag()).lower())
        searchbox_hi = int(self.getOption("line_search_box_hi", fdu.getTag()).lower())

        #Get info from slitmask
        slitmask = calibs['slitmask']
        if (slitmask.hasProperty("nslits")):
            nslits = slitmask.getProperty("nslits")
        else:
            nslits = slitmask.getData().max()
            slitmask.setProperty("nslits", nslits)
        if (slitmask.hasProperty("regions")):
            (ylos, yhis, slitx, slitw) = slitmask.getProperty("regions")
        else:
            #Use helper method to all ylo, yhi for each slit in each frame
            (ylos, yhis, slitx, slitw) = findRegions(slitmask.getData(), nslits, fdu, gpu=self._fdb.getGPUMode(), log=self._log)
            slitmask.setProperty("regions", (ylos, yhis, slitx, slitw))

        lamp2d = calibs[lampkey].getData()
        if (not calibs[lampkey].hasProperty("is_resampled") and calibs[lampkey].hasProperty("resampled")):
            lamp2d = calibs[lampkey].getData(tag="resampled")
            slitmask = fdu.getSlitmask(shape=lamp2d.shape, tagname="resampled_slitmask")

        #Find brightest line in search box
        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            lamp1d = lamp2d.sum(0)
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            lamp1d = lamp2d.sum(1)
        maxLoc = where(lamp1d == lamp1d[searchbox_lo:searchbox_hi].max())[0][0]
        lsq = fitGaussian(lamp1d[maxLoc-10:maxLoc+11])
        xlo = int(maxLoc-10+lsq[0][1]-lsq[0][2]*2.348*2)
        xhi = int(maxLoc-10+lsq[0][1]+lsq[0][2]*2.348*2+1)
        if (usePlot and (debug or writePlots)):
            plt.plot(arange(len(lamp1d[searchbox_lo:searchbox_hi]))+searchbox_lo, lamp1d[searchbox_lo:searchbox_hi])
            plt.title('1-d Cut of '+lampkey)
            plt.xlabel("xlo = "+str(xlo)+"; xhi = "+str(xhi))
            if (writePlots):
                plt.savefig(outdir+"/collapsedSlitlets/"+lampkey+"_1dcut.png", dpi=200)
            if (debug):
                plt.show()
            plt.close()

        maxWidth = (yhis-ylos).max()+1
        image2d = zeros((nslits, maxWidth), float32)
        #Loop over slitlets (could be one pass or nslits passes)
        for j in range(nslits):
            #Take 1-d cut of arclamp in each slitlet
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                lamp1d = (lamp2d[ylos[j]:yhis[j]+1,:]*(slitmask.getData()[ylos[j]:yhis[j]+1,:] == j+1)).sum(1)
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                lamp1d = (lamp2d[xlo:xhi+1,ylos[j]:yhis[j]+1]*(slitmask.getData()[xlo:xhi+1,ylos[j]:yhis[j]+1] == j+1)).sum(0)
            image2d[nslits-j-1,:lamp1d.size] = lamp1d #Reverse order, slitlet 32 is top row

        #update row
        refrow = nslits-refslit-1
        images = []
        xsh = []
        ysh = []
        halfWidth = maxWidth//2
        refCut1 = image2d[refrow,:halfWidth]
        refCut2 = image2d[refrow,halfWidth:]

        #debug/write plots
        if (usePlot and (debug or writePlots)):
            for j in range(nslits):
                image2d[j,:] *= (image2d[refrow,:].max()/image2d[j,:].max())
                plt.plot(image2d[j,:])
            plt.xlabel('Pixel')
            plt.ylabel('Slitlet collapsed flux')
            plt.legend(arange(nslits)+1)
            if (writePlots):
                plt.savefig(outdir+"/collapsedSlitlets/slits_collapsed_"+calibs[lampkey]._id+".png", dpi=200)
            if (debug):
                plt.show()
            plt.close()

        for j in range(nslits):
            images.append(image2d[j,:])
            images.append(image2d[j,:]) #Append this row twice
            #debug/write plots
            if (usePlot and (debug or writePlots)):
                fig, (ax1, ax2) = plt.subplots(2)
                ax1.plot(image2d[refrow,:])
                ax1.plot(image2d[j,:])
                ax1.legend(['Slit '+str(refrow), 'Slit '+str(j)])
                ax1.set(xlabel='ylo = '+str(ylos[j])+"; yhi = "+str(yhis[j]))

            #FOV in 2D and 3D seems to have double copy of each row
            ysh.append(float(j*2))
            ysh.append(float(j*2+1)) #SINFONI data => double rows
            if (j == refrow):
                xsh.append(0)
                xsh.append(0)
                continue
            slitCut1 = image2d[j,:halfWidth]
            slitCut2 = image2d[j,halfWidth:]
            ccor1 = correlate(refCut1, slitCut1, mode='same')
            ccor2 = correlate(refCut2, slitCut2, mode='same')
            ccor1 = medianfilterCPU(ccor1) #median filter
            ccor2 = medianfilterCPU(ccor2) #median filter
            if (usePlot and (debug or writePlots)):
                c1 = ccor1.copy()
                c1[c1 < 0] = 0
                c1 *= c1
                ccor1 = c1
                c2 = ccor2.copy()
                c2[c2 < 0] = 0
                c2 *= c2
                ccor2 = c2
                if (usePlot and (debug or writePlots)):
                    ax2.plot(c1)
                    ax2.plot(c2)
                    ax2.set(xlabel='Filtered cross correlations')
                    ax2.legend(['Left edge', 'Right edge'])
            mcor1 = where(ccor1 == max(ccor1))[0]
            shift1 = -1*(len(ccor1)//2-mcor1[0])
            mcor2 = where(ccor2 == max(ccor2))[0]
            shift2 = -1*(len(ccor2)//2-mcor2[0])
            shift = (shift1+shift2)/2.0
            if (integerShifts):
                xsh.append(shift)
                xsh.append(shift)
            else:
                p = zeros(4, float64)
                p[0] = ccor1.max()
                p[1] = mcor1[0]
                p[2] = 4.0
                lsq1 = fitGaussian(ccor1, guess=p, maskNeg=False)
                lsq2 = fitGaussian(ccor2, guess=p, maskNeg=False)
                if (lsq1[1] == False or lsq2[1] == False):
                    xsh.append(shift) #use integer shift
                    xsh.append(shift) #SINFONI data => double rows
                else:
                    shift1 = -1*(len(ccor1)//2-lsq1[0][1])
                    shift2 = -1*(len(ccor2)//2-lsq2[0][1])
                    xsh.append((shift1+shift2)/2.0)
                    xsh.append((shift1+shift2)/2.0) #SINFONI data => double rows
            print("\tRow "+str(j)+": integer shift = "+str(shift)+"; actual shift = "+str(xsh[-1]))
            self._log.writeLog(__name__, "Row "+str(j)+": integer shift = "+str(shift)+"; actual shift = "+str(xsh[-1]), printCaller=False, tabLevel=1)
            if (usePlot):
                if (writePlots):
                    plt.savefig(outdir+"/collapsedSlitlets/slit_"+str(j)+"_correlation_"+calibs[lampkey]._id+".png", dpi=200)
                if (debug):
                    plt.show()
                plt.close()
        xsh = array(xsh)
        xsh -= xsh.min() #subtract min
        widths = []
        for j in range(nslits):
            widths.append(yhis[j]-ylos[j]+xsh[2*j])
        widths = array(widths)
        maxWidth = widths.max()+1

        if (integerShifts):
            image2d = zeros((2*nslits, maxWidth), float32)
            for j in range(2*nslits):
                image2d[j, xsh[j]:xsh[j]+images[j].size] = images[j]
        else:
            for j in range(2*nslits):
                images[j] = images[j].reshape((1, len(images[j])))
            #Drizzle together to create final image
            (image2d, head, exp, pix) = drihizzle.drihizzle(images, xsh=xsh, ysh=ysh, kernel='turbo', dropsize=1, weight=1, outunits='counts')
            image2d = image2d[:-1,:] #Remove blank last row

        #Tag data to lamp/clean sky
        calibs[lampkey].tagDataAs("collapsedSlitlets", image2d)
        calibs[lampkey].setProperty("cs_xsh", xsh)
        calibs[lampkey].setProperty("cs_ysh", ysh)
        calibs[lampkey].setProperty("cs_maxWidth", maxWidth)

        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            #Create output filename
            csfile = outdir+"/collapsedSlitlets/cs_"+calibs[lampkey].getFullId()
        #Check to see if it exists
        if (os.access(csfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(csfile)
        if (not os.access(csfile, os.F_OK)):
            calibs[lampkey].writeTo(csfile, tag="collapsedSlitlets")
    #end calculateShiftsBetweenSlitlets

    def centroidImages(self, fdu, calibs):
        #read options
        centroid_method = self.getOption("centroid_method", fdu.getTag())
        debug = False
        if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
            debug = True
        writePlots = False
        if (self.getOption("write_plots", fdu.getTag()).lower() == "yes"):
            writePlots = True

        if (writePlots or self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            #make directory if necessary
            outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
            if (not os.access(outdir+"/collapsedSlitlets", os.F_OK)):
                os.mkdir(outdir+"/collapsedSlitlets",0o755)

        if (fdu.hasProperty("collapsedSlitlets")):
            data = fdu.getData(tag="collapsedSlitlets")
            if (usePlot and (debug or writePlots)):
                plt.plot(data.sum(0))
                plt.plot(data.sum(1))
                plt.legend(['X-cut','Y-cut'])
            if (centroid_method == "fit_2d_gaussian"):
                p = zeros(5, float32)
                p[0] = data.max()
                b = where(data == p[0])
                if (fdu.hasProperty("xcen_guess")):
                    p[1] = fdu.getProperty("xcen_guess")
                else:
                    p[1] = b[1][0]
                p[2] = b[0][0]
                p[3] = 2#convert from FWHM in pixels
                p[4] = gpu_arraymedian(data)
                maskNeg = False
                if (p[4] > 0):
                    maskNeg = True
                lsq = fitGaussian2d(data, guess=p, maskNeg=maskNeg)
                if (lsq[1] == False):
                    print("sinfoniCollapseSlitletsProcess::centroidImages> ERROR: Could not centroid image.")
                    self._log.writeLog(__name__, "Could not centroid image.", type=fatboyLog.ERROR)
                    return False
                xcen = lsq[0][1]
                ycen = lsq[0][2]
                fwhm = lsq[0][3]*2.3548
            else:
                #use_derivatives
                b = where(data == data.max())
                mx = b[1][0]
                my = b[0][0]
                (fwhm, sig, fwhm1ds, bg) = fwhm2d(data)
                (xcen, ycen) = getCentroid(data, mx, my, fwhm)
                xcen -= padx
                ycen -= pady
            print("sinfoniCollapseSlitletsProcess::centroidImages> 2d image centroid with "+centroid_method+": (x="+formatNum(xcen)+"; y="+formatNum(ycen)+") fwhm="+formatNum(fwhm))
            self._log.writeLog(__name__, "2d image centroid with "+centroid_method+": (x="+formatNum(xcen)+"; y="+formatNum(ycen)+") fwhm="+formatNum(fwhm))
            if (usePlot):
                if (writePlots):
                    plt.savefig(outdir+"/collapsedSlitlets/centroid_"+fdu.getFullId()+".png", dpi=200)
                if (debug):
                    plt.show()
                plt.close()
            fdu.setProperty("image_centroid", (xcen, ycen, fwhm))
    #end centroidImages

    def collapseSlitlets(self, fdu, calibs):
        #read options
        useArclamps = False
        if (self.getOption("use_arclamps", fdu.getTag()).lower() == "yes"):
            useArclamps = True
        refslit = int(self.getOption("reference_slit", fdu.getTag()))
        useMedian = False
        if (self.getOption("collapse_method", fdu.getTag()).lower() == "median"):
            useMedian = True

        integerShifts = False
        if (self.getOption("use_integer_shifts", fdu.getTag()).lower() == "yes"):
            integerShifts = True

        slitmask = calibs['slitmask']

        if (slitmask.hasProperty("nslits")):
            nslits = slitmask.getProperty("nslits")
        else:
            nslits = slitmask.getData().max()
            slitmask.setProperty("nslits", nslits)
        if (slitmask.hasProperty("regions")):
            (ylos, yhis, slitx, slitw) = slitmask.getProperty("regions")
        else:
            #Use helper method to all ylo, yhi for each slit in each frame
            (ylos, yhis, slitx, slitw) = findRegions(slitmask.getData(), nslits, fdu, gpu=self._fdb.getGPUMode(), log=self._log)
            slitmask.setProperty("regions", (ylos, yhis, slitx, slitw))

        properties = dict()
        properties['specmode'] = fdu.getProperty("specmode")
        properties['dispersion'] = fdu.getProperty("dispersion")
        properties['SlitletsIdentified'] = True

        lampkey = 'cleanSky'
        if (useArclamps and 'masterLamp' in calibs):
            #Use master arclamp if it exists
            lampkey = 'masterLamp'

        if (not calibs[lampkey].hasProperty('collapsedSlitlets')):
            #Check to see if master lamp / clean sky has already been used
            #to calc shifts between slitlets - if not, do so
            self.calculateShiftsBetweenSlitlets(fdu, calibs, lampkey)

        xsh = calibs[lampkey].getProperty("cs_xsh")
        ysh = calibs[lampkey].getProperty("cs_ysh")
        maxWidth = calibs[lampkey].getProperty("cs_maxWidth")
        maxWidthInit = (yhis-ylos).max()+1

        hasCleanFrame = False
        if (fdu.hasProperty("cleanFrame")):
            hasCleanFrame = True
            clean2d = zeros((nslits, maxWidthInit), float32)
            clean_images = []

        #Get slitmask
        slitmask = fdu.getSlitmask(properties=properties)
        if (slitmask is None):
            print("sinfoniCollapseSlitletsProcess::collapseSlitlets> ERROR: Could not find slitmask so could not collapse slitlets.")
            self._log.writeLog(__name__, "Could not find slitmask so could not collapse slitlets.", type=fatboyLog.ERROR)
            return False
        smdata = slitmask.getData()
        sm2d = zeros((nslits, maxWidthInit), slitmask.getData().dtype)
        sm_images = []

        image2d = zeros((nslits, maxWidthInit), float32)
        images = []
        #Loop over slitlets (could be one pass or nslits passes)
        #Re-order here
        for j in range(nslits):
            #Take 1-d cut in each slitlet
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                if (useMedian):
                    cut1d = gpu_arraymedian(fdu.getData()[ylos[j]:yhis[j]+1,:]*(smdata[ylos[j]:yhis[j]+1,:] == j+1), axis="X")
                    if (hasCleanFrame):
                        clean1d = gpu_arraymedian(fdu.getData(tag="cleanFrame")[ylos[j]:yhis[j]+1,:]*(smdata[ylos[j]:yhis[j]+1,:] == j+1), axis="X")
                else:
                    cut1d = (fdu.getData()[ylos[j]:yhis[j]+1,:]*(smdata[ylos[j]:yhis[j]+1,:] == j+1)).sum(1)
                    if (hasCleanFrame):
                        clean1d = (fdu.getData(tag="cleanFrame")[ylos[j]:yhis[j]+1,:]*(smdata[ylos[j]:yhis[j]+1,:] == j+1)).sum(1)
                sm1d = (slitmask.getData()[ylos[j]:yhis[j]+1,:]*(smdata[ylos[j]:yhis[j]+1,:] == j+1)).max(1)
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                if (useMedian):
                    cut1d = gpu_arraymedian(fdu.getData()[:,ylos[j]:yhis[j]+1]*(smdata[:,ylos[j]:yhis[j]+1] == j+1), axis="Y")
                    if (hasCleanFrame):
                        clean1d = gpu_arraymedian(fdu.getData(tag="cleanFrame")[:,ylos[j]:yhis[j]+1]*(smdata[:,ylos[j]:yhis[j]+1] == j+1), axis="Y")
                else:
                    cut1d = (fdu.getData()[:,ylos[j]:yhis[j]+1]*(smdata[:,ylos[j]:yhis[j]+1] == j+1)).sum(0)
                    if (hasCleanFrame):
                        clean1d = (fdu.getData(tag="cleanFrame")[:,ylos[j]:yhis[j]+1]*(smdata[:,ylos[j]:yhis[j]+1] == j+1)).sum(0)
                sm1d = (slitmask.getData()[:,ylos[j]:yhis[j]+1]*(smdata[:,ylos[j]:yhis[j]+1] == j+1)).max(0)
            image2d[nslits-j-1,:cut1d.size] = cut1d #Reverse order, slitlet 32 is top row
            if (hasCleanFrame):
                clean2d[nslits-j-1,:clean1d.size] = clean1d
            sm2d[nslits-j-1,:sm1d.size] = sm1d

        #Loop over slitlets again to get list of 1-d cuts
        for j in range(nslits):
            images.append(image2d[j,:])
            images.append(image2d[j,:]) #Append this row twice
            if (hasCleanFrame):
                clean_images.append(clean2d[j,:])
                clean_images.append(clean2d[j,:]) #Append this row twice
            sm_images.append(sm2d[j,:])
            sm_images.append(sm2d[j,:]) #Append this row twice

        if (integerShifts):
            image2d = zeros((2*nslits, maxWidth), float32)
            for j in range(2*nslits):
                image2d[j, xsh[j]:xsh[j]+images[j].size] = images[j]
            if (hasCleanFrame):
                clean2d = zeros((2*nslits, maxWidth), float32)
                for j in range(2*nslits):
                    clean2d[j, xsh[j]:xsh[j]+clean_images[j].size] = clean_images[j]
            sm2d = zeros((2*nslits, maxWidth), fdu.getProperty("slitmask").dtype)
            for j in range(2*nslits):
                sm2d[j, xsh[j]:xsh[j]+sm_images[j].size] = sm_images[j]
        else:
            for j in range(2*nslits):
                images[j] = images[j].reshape((1, len(images[j])))
            #Drizzle together to create final image
            (image2d, head, exp, pix) = drihizzle.drihizzle(images, xsh=xsh, ysh=ysh, kernel='turbo', dropsize=1, weight=1, outunits='counts')
            image2d = image2d[:-1,:] #Remove blank last row
            if (hasCleanFrame):
                for j in range(2*nslits):
                    clean_images[j] = clean_images[j].reshape((1, len(clean_images[j])))
                #Drizzle together to create final image
                (clean2d, head, exp, pix) = drihizzle.drihizzle(clean_images, xsh=xsh, ysh=ysh, kernel='turbo', dropsize=1, weight=1, outunits='counts')
                clean2d = clean2d[:-1,:] #Remove blank last row
            for j in range(2*nslits):
                sm_images[j] = sm_images[j].reshape((1, len(sm_images[j])))
            #Drizzle together to create final image
            (sm2d, head, exp, pix) = drihizzle.drihizzle(sm_images, xsh=xsh, ysh=ysh, kernel='uniform', dropsize=1, weight=1, outunits='counts')
            sm2d = sm2d[:-1,:] #Remove blank last row

        #Tag data
        fdu.tagDataAs("collapsedSlitlets", image2d)
        if (hasCleanFrame):
            fdu.tagDataAs("collapsedSlitlets_cleanFrame", clean2d)
        fdu.setSlitmask(sm2d, pname=self._pname, tagname="collapsedSlitlets_slitmask")
        fdu.setProperty("cs_xsh", xsh)
        fdu.setProperty("cs_ysh", ysh)
        fdu.setProperty("cs_maxWidth", maxWidth)
    #end collapseSlitlets

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("SINFONI: collapse slitlets")
        print(fdu._identFull)

        #Check if output exists first and update from disk
        csfile = "collapsedSlitlets/cs_"+fdu.getFullId()
        if (self.checkOutputExists(fdu, csfile, tag="collapsedSlitlets", headerTag="csHeader")):
            #Also check if "cleanFrame" exists
            cleanfile = "collapsedSlitlets/clean_cs_"+fdu.getFullId()
            self.checkOutputExists(fdu, cleanfile, tag="collapsedSlitlets_cleanFrame", headerTag="csHeader")
            #Also check if "noisemap" exists
            nmfile = "collapsedSlitlets/NM_cs_"+fdu.getFullId()
            self.checkOutputExists(fdu, nmfile, tag="noisemap")
            return True

        #Call get calibs to return dict() of calibration frames.
        #For collapsedSlitlets, this dict can have a noisemap if this is not a property
        #of the FDU at this point.  It may also have an text file listing the
        #slitlet indices of each row in the RSS file if this is not in the header already.
        calibs = self.getCalibs(fdu, prevProc)

        if (not 'slitmask' in calibs):
            print("sinfoniCollapseSlitletsProcess::execute> ERROR: Could not find slitmask so could not collapse slitlets.")
            self._log.writeLog(__name__, "Could not find slitmask so could not collapse slitlets.", type=fatboyLog.ERROR)
            return False
        if (not 'masterLamp' in calibs and not 'cleanSky' in calibs):
            print("sinfoniCollapseSlitletsProcess::execute> ERROR: Could not find master arclamp or clean sky frame.  Could not collapse slitlets.")
            self._log.writeLog(__name__, "Could not find master arclamp or clean sky frame.  Could not collapse slitlets.", type=fatboyLog.ERROR)
            return False

        #call collapseSlitlets helper function to do actual processing into 2d images and data tagging
        self.collapseSlitlets(fdu, calibs)
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
                print("sinfoniCollapseSlitletsProcess::getCalibs> Using slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Using slitmask "+smfilename+"...")
                calibs['slitmask'] = fatboySpecCalib(self._pname, "slitmask", fdu, filename=smfilename, log=self._log)
            else:
                print("sinfoniCollapseSlitletsProcess::getCalibs> Warning: Could not find slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Could not find slitmask "+smfilename+"...", type=fatboyLog.WARNING)
        #Look for each master calib passed from XML
        csfilename = self.getCalib("master_clean_sky", fdu.getTag())
        if (csfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(csfilename, os.F_OK)):
                print("sinfoniCollapseSlitletsProcess::getCalibs> Using master clean sky frame "+csfilename+"...")
                self._log.writeLog(__name__, "Using master clean sky frame "+csfilename+"...")
                calibs['cleanSky'] = fatboySpecCalib(self._pname, "master_clean_sky", fdu, filename=csfilename, log=self._log)
            else:
                print("sinfoniCollapseSlitletsProcess::getCalibs> Warning: Could not find master clean sky frame "+csfilename+"...")
                self._log.writeLog(__name__, "Could not find master clean sky frame "+csfilename+"...", type=fatboyLog.WARNING)
        #Look for each master calib passed from XML
        mlfilename = self.getCalib("master_arclamp", fdu.getTag())
        if (mlfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(mlfilename, os.F_OK)):
                print("sinfoniCollapseSlitletsProcess::getCalibs> Using master arclamp frame "+mlfilename+"...")
                self._log.writeLog(__name__, "Using master arclamp frame "+mlfilename+"...")
                calibs['masterLamp'] = fatboySpecCalib(self._pname, "master_arclamp", fdu, filename=mlfilename, log=self._log)
            else:
                print("sinfoniCollapseSlitletsProcess::getCalibs> Warning: Could not find master arclamp frame "+mlfilename+"...")
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
            #Use new fdu.getSlitmask method
            fdu.printAllSlitmasks()
            properties['SlitletsIdentified'] = True
            slitmask = fdu.getSlitmask(pname=None, shape=skyShape, properties=properties, headerVals=headerVals)
            if (slitmask is not None):
                #Found slitmask
                calibs['slitmask'] = slitmask
        return calibs
    #end getCalibs

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('centroid_method', 'fit_2d_gaussian')
        self._optioninfo.setdefault('centroid_method', 'fit_2d_gaussian | use_derivatives')
        self._options.setdefault('collapse_method', 'sum')
        self._optioninfo.setdefault('collapse_method', 'sum | median')
        self._options.setdefault('debug_mode', 'no')
        self._options.setdefault('do_centroids', 'yes')
        self._optioninfo.setdefault('do_centroids', 'Whether to centroid images')
        self._options.setdefault('line_search_box_lo', '100')
        self._optioninfo.setdefault('line_search_box_lo', 'Search box for brightest line in arclamp/sky to use for finding slitlet shifts.')
        self._options.setdefault('line_search_box_hi', '-100')
        self._optioninfo.setdefault('line_search_box_hi', 'Search box for brightest line in arclamp/sky to use for finding slitlet shifts.')
        self._options.setdefault('reference_slit', '2')
        self._optioninfo.setdefault('reference_slit', 'Reference slit when finding offsets between slits')
        self._options.setdefault('use_arclamps', 'no')
        self._optioninfo.setdefault('use_arclamps', 'no = use master "clean sky", yes = use master arclamp')
        self._options.setdefault('use_integer_shifts', 'no')
        self._optioninfo.setdefault('use_integer_shifts', 'If set to yes, integer shifts will be used and data\nwill not be resampled.')
        self._options.setdefault('write_plots', 'no')
    #end setDefaultOptions

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/collapsedSlitlets", os.F_OK)):
            os.mkdir(outdir+"/collapsedSlitlets",0o755)
        #Create output filename
        csfile = outdir+"/collapsedSlitlets/cs_"+fdu.getFullId()
        #Check to see if it exists
        if (os.access(csfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(csfile)
        if (not os.access(csfile, os.F_OK)):
            #Use fatboyDataUnit writePropertiesToMEF method to write
            fdu.writeTo(csfile, tag="collapsedSlitlets")
        #Write out clean frame if it exists
        if (fdu.hasProperty("collapsedSlitlets_cleanFrame")):
            cleanfile = outdir+"/collapsedSlitlets/clean_cs_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(cleanfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(cleanfile)
            if (not os.access(cleanfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.writeTo(cleanfile, tag="collapsedSlitlets_cleanFrame")
        #Write out slitmask if it exists
        if (fdu.hasProperty("collapsedSlitlets_slitmask")):
            smfile = outdir+"/collapsedSlitlets/slitmask_cs_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(smfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(smfile)
            if (not os.access(smfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.getProperty("collapsedSlitlets_slitmask").writeTo(smfile)
                #fdu.getSlitmask(tagname="collapsedSlitlets_slitmask", shape=fdu.getData(tag="collapsedSlitlets").shape).writeTo(smfile, headerExt=resampHeader)
    #end writeOutput
