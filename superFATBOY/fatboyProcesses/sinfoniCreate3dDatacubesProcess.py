from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY import gpu_drihizzle, drihizzle
import os, time

usePlot = True
try:
    import matplotlib.pyplot as plt
except Exception as ex:
    print("Warning: Could not import matplotlib!")
    usePlot = False

class sinfoniCreate3dDatacubesProcess(fatboyProcess):
    """ Create a 3-d datacube where each "cut" is a monochromatic image at a given wavelength """
    _modeTags = ["sinfoni"]

    def calculateShiftsBetweenSlitlets(self, fdu, calibs, lampkey):
        refslit = int(self.getOption("reference_slit", fdu.getTag()))

        integerShifts = False
        if (self.getOption("use_integer_shifts", fdu.getTag()).lower() == "yes"):
            integerShifts = True

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

        maxWidth = (yhis-ylos).max()+1
        image2d = zeros((nslits, maxWidth), float32)
        #Loop over slitlets (could be one pass or nslits passes)
        for j in range(nslits):
            #Take 1-d cut of arclamp in each slitlet
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                lamp1d = (calibs[lampkey].getData()[ylos[j]:yhis[j]+1,:]*(calibs['slitmask'].getData()[ylos[j]:yhis[j]+1,:] == j+1)).sum(1)
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                lamp1d = (calibs[lampkey].getData()[:,ylos[j]:yhis[j]+1]*(calibs['slitmask'].getData()[:,ylos[j]:yhis[j]+1] == j+1)).sum(0)
            image2d[nslits-j-1,:lamp1d.size] = lamp1d #Reverse order, slitlet 32 is top row

        #update row
        refrow = nslits-refslit-1
        images = []
        xsh = []
        ysh = []
        refCut = image2d[refrow,:]

        #debug/write plots
        if (usePlot and (debug or writePlots)):
            for j in range(nslits):
                plt.plot(image2d[j,:])
            plt.xlabel('Pixel')
            plt.ylabel('Slitlet collapsed flux')
            if (writePlots):
                plt.savefig(outdir+"/collapsedSlitlets/slits_collapsed_"+calibs[lampkey]._id+".png", dpi=200)
            if (debug):
                plt.show()
            plt.close()

        for j in range(nslits):
            images.append(image2d[j,:])
            images.append(image2d[j,:]) #Append this row twice
            ysh.append(float(j*2))
            ysh.append(float(j*2+1)) #SINFONI data => double rows
            if (j == refrow):
                xsh.append(0)
                xsh.append(0)
                continue
            slitCut = image2d[j,:]
            ccor = correlate(refCut, slitCut, mode='same')
            ccor = medianfilterCPU(ccor) #median filter
            if (usePlot and (debug or writePlots)):
                plt.plot(ccor)
            mcor = where(ccor == max(ccor))[0]
            shift = -1*(len(ccor)//2-mcor[0])
            if (integerShifts):
                xsh.append(shift)
                xsh.append(shift)
            else:
                p = zeros(4, float64)
                p[0] = ccor.max()
                p[1] = mcor[0]
                p[2] = 2.0
                lsq = fitGaussian(ccor, guess=p, maskNeg=True)
                if (lsq[1] == False):
                    xsh.append(shift) #use integer shift
                    xsh.append(shift) #SINFONI data => double rows
                else:
                    xsh.append(-1*(len(ccor)//2-lsq[0][1]))
                    xsh.append(-1*(len(ccor)//2-lsq[0][1])) #SINFONI data => double rows
            print("\tRow "+str(j)+": integer shift = "+str(shift)+"; actual shift = "+str(xsh[-1]))
            self._log.writeLog(__name__, "Row "+str(j)+": integer shift = "+str(shift)+"; actual shift = "+str(xsh[-1]), printCaller=False, tabLevel=1)
        xsh = array(xsh)
        #debug/write plots
        if (usePlot and (debug or writePlots)):
            plt.xlabel('Pixel')
            plt.ylabel('Filtered cross-correlation with slit '+str(refslit))
            if (writePlots):
                plt.savefig(outdir+"/collapsedSlitlets/slit_correlations_"+calibs[lampkey]._id+".png", dpi=200)
            if (debug):
                plt.show()
            plt.close()
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

    def createDatacube(self, fdu, calibs):
        #read options
        useArclamps = False
        if (self.getOption("use_arclamps", fdu.getTag()).lower() == "yes"):
            useArclamps = True
        refslit = int(self.getOption("reference_slit", fdu.getTag()))

        integerShifts = False
        if (self.getOption("use_integer_shifts", fdu.getTag()).lower() == "yes"):
            integerShifts = True

        drihizzleKernel = self.getOption("drihizzle_kernel", fdu.getTag()).lower()
        dropsize = float(self.getOption("drihizzle_dropsize", fdu.getTag()))

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

        if (not calibs[lampkey].hasProperty("cs_xsh")):
            #Check to see if master lamp / clean sky has already been used
            #to calc shifts between slitlets - if not, do so
            self.calculateShiftsBetweenSlitlets(fdu, calibs, lampkey)

        xsh = calibs[lampkey].getProperty("cs_xsh")
        ysh = calibs[lampkey].getProperty("cs_ysh")
        maxWidth = calibs[lampkey].getProperty("cs_maxWidth")
        maxWidthInit = (yhis-ylos).max()+1
        swidths = (yhis-ylos)

        if (not fdu.hasProperty("resampled")):
            print("sinfoniCreate3dDatacubesProcess::createDatacube> Warning: Could not find resampled frame!  Make sure wavelength calibration is performed before creating 3D datacubes!")
            self._log.writeLog(__name__, "Could not find resampled frame!  Make sure wavelength calibration is performed before creating 3D datacubes!", type=fatboyLog.WARNING)

        print("sinfoniCreate3dDatacubesProcess::createDatacube> Using shifts between slitlets: "+formatList(xsh))
        self._log.writeLog(__name__, "Using shifts between slitlets: "+formatList(xsh))

        image3d = []
        data = fdu.getData(tag="resampled")
        if (fdu.hasProperty("is_resampled") and fdu.getProperty("is_resampled")):
            data = fdu.getData()
        #Get slitmask with same shape using new getSlitmask
        slitmask = fdu.getSlitmask(shape=data.shape, properties=properties)
        if (slitmask is None and fdu.hasProperty("resampled_slitmask")):
            slitmask = fdu.getSlitmask(shape=data.shape, properties=properties, tagname="resampled_slitmask")
        smdata = slitmask.getData()

        images = []
        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            xsize = data.shape[1]
            for j in range(nslits-1,-1,-1):
                slit = zeros((xsize, maxWidthInit), float32)
                slit[:,:swidths[j]+1] = (data[ylos[j]:yhis[j]+1,:]*(smdata[ylos[j]:yhis[j]+1,:] == j+1)).transpose()
                images.append(slit)
                images.append(slit) #Append this row twice
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            xsize = data.shape[0]
            for j in range(nslits-1,-1,-1):
                slit = zeros((xsize, maxWidthInit), float32)
                slit[:,:swidths[j]+1] = data[:, ylos[j]:yhis[j]+1]*(smdata[:, ylos[j]:yhis[j]+1] == j+1)
                images.append(slit)
                images.append(slit) #Append this row twice

        #Create yind array of indices to layer slits in a xsize*nslits x maxWidth 2-d array
        #This will later be reshaped to xsize x nslits x maxWidth 3-d cube
        yind = (arange(images[0].size).reshape(images[0].shape) // maxWidthInit).astype(float32)*nslits*2

        #Select cpu/gpu option
        drihizzle_method = gpu_drihizzle.drihizzle
        if (not self._fdb.getGPUMode()):
            drihizzle_method = drihizzle.drihizzle
        #Drizzle together to create final image
        (image2d, head, exp, pix) = drihizzle.drihizzle(images, ytrans=yind, xsh=xsh, ysh=ysh, kernel=drihizzleKernel, dropsize=dropsize, weight=1, outunits='counts')
        #Reshape - use image2d.shape[1] not slice_width because of shifts, slits are wider
        image3d = image2d[:-1,:-1].reshape((xsize, 2*nslits, image2d.shape[1]-1))

        #Tag data
        fdu.tagDataAs("datacube", image3d)
        fdu.setProperty("cs_xsh", xsh)
        fdu.setProperty("cs_ysh", ysh)
        fdu.setProperty("cs_maxWidth", maxWidth)

        #Update header
        if (not fdu.hasHeaderValue('CRVAL3') and fdu.hasHeaderValue('CRVAL1') and fdu.hasHeaderValue('CDELT1')):
            image3dHeader = dict()
            image3dHeader['CRVAL3'] = fdu.getHeaderValue('CRVAL1')
            image3dHeader['CDELT3'] = fdu.getHeaderValue('CDELT1')
            image3dHeader['CRPIX3'] = fdu.getHeaderValue('CRPIX1')
            image3dHeader['CD1_3'] = 0
            image3dHeader['CD2_3'] = 0
            image3dHeader['CD3_1'] = 0
            image3dHeader['CD3_2'] = 0
            image3dHeader['CD3_3'] = fdu.getHeaderValue('CDELT1')
            image3dHeader['CTYPE3'] = 'WAVE'
            image3dHeader['CUNIT3'] = 'Angstrom'
            fdu.updateHeader(image3dHeader)
    #end createDatacube

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("SINFONI: create 3d datacube")
        print(fdu._identFull)

        #Check if output exists first and update from disk
        csfile = "3dDatacubes/3d_"+fdu.getFullId()
        if (self.checkOutputExists(fdu, csfile, tag="datacube")):
            return True

        #Call get calibs to return dict() of calibration frames.
        #For 3dDatacubes, this dict can have a noisemap if this is not a property
        #of the FDU at this point.  It may also have an text file listing the
        #slitlet indices of each row in the RSS file if this is not in the header already.
        calibs = self.getCalibs(fdu, prevProc)

        if (not 'slitmask' in calibs):
            print("sinfoniCreate3dDatacubesProcess::execute> ERROR: Could not find slitmask so could not create datacube.")
            self._log.writeLog(__name__, "Could not find slitmask so could not create datacube.", type=fatboyLog.ERROR)
            return False
        if (not 'masterLamp' in calibs and not 'cleanSky' in calibs):
            print("sinfoniCreate3dDatacubesProcess::execute> ERROR: Could not find master arclamp or clean sky frame.  Could not create datacube.")
            self._log.writeLog(__name__, "Could not find master arclamp or clean sky frame.  Could not create datacube.", type=fatboyLog.ERROR)
            return False

        #call createDatacube helper function to do actual processing into 2d images and data tagging
        self.createDatacube(fdu, calibs)
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
                print("sinfoniCreate3dDatacubesProcess::getCalibs> Using slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Using slitmask "+smfilename+"...")
                calibs['slitmask'] = fatboySpecCalib(self._pname, "slitmask", fdu, filename=smfilename, log=self._log)
            else:
                print("sinfoniCreate3dDatacubesProcess::getCalibs> Warning: Could not find slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Could not find slitmask "+smfilename+"...", type=fatboyLog.WARNING)

        #Look for each master calib passed from XML
        mlfilename = self.getCalib("master_arclamp", fdu.getTag())
        if (mlfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(mlfilename, os.F_OK)):
                print("sinfoniCreate3dDatacubesProcess::getCalibs> Using master arclamp frame "+mlfilename+"...")
                self._log.writeLog(__name__, "Using master arclamp frame "+mlfilename+"...")
                calibs['masterLamp'] = fatboySpecCalib(self._pname, "master_arclamp", fdu, filename=mlfilename, log=self._log)
            else:
                print("sinfoniCreate3dDatacubesProcess::getCalibs> Warning: Could not find master arclamp frame "+mlfilename+"...")
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
            properties['SlitletsIdentified'] = True
            slitmask = fdu.getSlitmask(pname=None, shape=skyShape, properties=properties, headerVals=headerVals)
            if (slitmask is not None):
                #Found slitmask
                calibs['slitmask'] = slitmask
        return calibs
    #end getCalibs

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('drihizzle_dropsize', '1')
        self._options.setdefault('drihizzle_kernel', 'turbo')
        self._optioninfo.setdefault('drihizzle_kernel', 'turbo | point | point_replace | tophat | gaussian | fastgauss | lanczos')
        self._options.setdefault('debug_mode', 'no')
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
        if (not os.access(outdir+"/3dDatacubes", os.F_OK)):
            os.mkdir(outdir+"/3dDatacubes",0o755)
        #Create output filename
        csfile = outdir+"/3dDatacubes/3d_"+fdu.getFullId()
        #Check to see if it exists
        if (os.access(csfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(csfile)
        if (not os.access(csfile, os.F_OK)):
            fdu.writeTo(csfile, tag="datacube")
    #end writeOutput
