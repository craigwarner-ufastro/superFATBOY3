from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.fatboyProcesses.badPixelMaskSpecProcess import bpm_replace_median_neighbor
from superFATBOY.fatboyProcesses.badPixelMaskSpecProcess import bpm_replace_median_neighbor_gpu
from numpy import *
import os, time
hasDeepCR = True
try:
    from deepCR import deepCR
except Exception as ex:
    hasDeepCR = False
hasSep = True
try:
    import sep
except Exception as ex:
    hasSep = False

block_size = 512

class removeCosmicRaysSpecProcess(fatboyProcess):
    _modeTags = ["spectroscopy", "miradas"]

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("Cosmic Ray Removal")
        print(fdu._identFull)

        #Check if output exists first
        rcrfile = "removedCosmicRays/rcr_"+fdu.getFullId()
        if (self.checkOutputExists(fdu, rcrfile)):
            #Also check if crmask exists
            crmask = "removedCosmicRays/crmask_"+fdu.getFullId()
            self.checkOutputExists(fdu, crmask, tag="crmask")
            return True

        #Call get calibs to return dict() of calibration frames.
        #For removeCosmicRaysSpec, this should contain a slitmask if data is MOS
        calibs = self.getCalibs(fdu, prevProc)
        #Read options
        cr_algorithm = self.getOption('cosmic_ray_algorithm', fdu.getTag()).lower()
        if (cr_algorithm == "deepcr" and not hasDeepCR):
            print("removeCosmicRaysSpecProcess::execute> WARNING: deepCR specified as cosmic ray removal algorithm but not installed.  Using DCR instead.")
            self._log.writeLog(__name__, "deepCR specified as cosmic ray removal algorithm but not installed.  Using DCR instead.", type=fatboyLog.WARNING)

        success = False
        if (cr_algorithm == "dcr"):
            success = self.runDcr(fdu, calibs)
        elif (cr_algorithm == "lacos"):
            if (fdu.gain is None):
                print("removeCosmicRaysSpecProcess::execute> WARNING: GAIN is not specified.  Using 1 but results MAY BE WRONG!")
                self._log.writeLog(__name__, "GAIN is not specified.  Using 1 but results MAY BE WRONG!", type=fatboyLog.WARNING)
                fdu.gain = 1
            success = self.runLacos(fdu, calibs)
        elif (cr_algorithm == "deepcr"):
            success = self.runDeepCR(fdu, calibs)
        else:
            print("removeCosmicRaysSpecProcess::execute> ERROR: invalid cosmic ray removal algorithm "+cr_algorithm+".  Must be dcr or lacos. Discarding Image!")
            self._log.writeLog(__name__, "Invalid cosmic ray removal algorithm "+cr_algorithm+".  Must be dcr or lacos. Discarding Image!", type=fatboyLog.ERROR)
            #disable this FDU
            fdu.disable()
            return False

        if (not success):
            #return here if CR removal failed
            return False

        if (self.getOption('remove_stray_light', fdu.getTag()).lower() == "yes"):
            #remove stray light too
            self.removeStrayLight(fdu, calibs)
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
                print("removeCosmicRaysSpecProcess::getCalibs> Using slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Using slitmask "+smfilename+"...")
                calibs['slitmask'] = fatboySpecCalib(self._pname, "slitmask", fdu, filename=smfilename, log=self._log)
            else:
                print("removeCosmicRaysSpecProcess::getCalibs> Warning: Could not find slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Could not find slitmask "+smfilename+"...", type=fatboyLog.WARNING)

        headerVals = dict()
        headerVals['grism_keyword'] = fdu.grism
        properties = dict()
        properties['specmode'] = fdu.getProperty("specmode")
        properties['dispersion'] = fdu.getProperty("dispersion")

        if (fdu._specmode != fdu.FDU_TYPE_LONGSLIT and not 'slitmask' in calibs):
            #Multi object data, need slitmask
            #Find slitmask associated with this fdu
            #Use new fdu.getSlitmask method
            slitmask = fdu.getSlitmask(pname=None, properties=properties, headerVals=headerVals)
            if (slitmask is None):
                print("removeCosmicRaysSpecProcess::getCalibs> ERROR: Could not find slitmask associated with "+fdu.getFullId()+"!")
                self._log.writeLog(__name__, "Could not find slitmask associated with "+fdu.getFullId()+"!", type=fatboyLog.ERROR)
                return calibs
            calibs['slitmask'] = slitmask

        return calibs
    #end getCalibs

    def removeStrayLight(self, fdu, calibs):
        if (not hasSep):
            print("removeCosmicRaysSpecProcess::getCalibs> Warning: Sep not found.  Cannot remove stray light.")
            self._log.writeLog(__name__, "Sep not found.  Cannot remove stray light.", type=fatboyLog.WARNING)
            return
        sl_method = self.getOption('stray_light_method', fdu.getTag()).lower()
        max_area = int(self.getOption('stray_light_max_area', fdu.getTag()))
        min_area = int(self.getOption('stray_light_min_area', fdu.getTag()))
        sigma = float(self.getOption('stray_light_sigma_threshold', fdu.getTag()))
        sym = float(self.getOption('stray_light_symmetry_threshold', fdu.getTag()))

        data = fdu.getData().astype(float32)
        bkg = sep.Background(data)
        #Subtract background and calculate threshold
        bkg.subfrom(data)
        thresh = sigma*bkg.globalrms
        #Extract objects
        objects = sep.extract(data, thresh, minarea=min_area)
        #Apply max_area and symmetry thresholds
        b = (objects['npix'] <= max_area)*(objects['b']/objects['a'] >= sym)
        objects = objects[b]
        print("removeCosmicRaysSpecProcess::removeStrayLight> Found "+str(len(objects))+" stray light artifacts.  Using method "+str(sl_method))
        self._log.writeLog(__name__, "Found "+str(len(objects))+" stray light artifacts.  Using method "+str(sl_method))

        #Create objmask
        objmask = zeros(data.shape, dtype=bool)
        sep.mask_ellipse(objmask, objects['x'], objects['y'], objects['a'], objects['b'], objects['theta'], r=3.0)
        sldata = data*objmask

        if (sl_method == "mask"):
            #apply to cosmic ray mask, which is good pixel mask
            slmask = (objmask == 0)
            if (fdu.hasProperty("crmask")):
                slmask = ((fdu.getData(tag="crmask")*slmask) == 1)
            fdu.tagDataAs("crmask", slmask)
        else:
            #method = replace - use median neighbor algorithm from badPixelMaskSpecProcess
            #Default is median_neighbor
            bpm_replace_algorithm = bpm_replace_median_neighbor
            if (self._fdb.getGPUMode()):
                bpm_replace_algorithm = bpm_replace_median_neighbor_gpu
            npix = objmask.sum()
            data = fdu.getData().astype(float32) #re-copy data
            print("removeCosmicRaysSpecProcess::removeStrayLight> Replacing "+str(npix)+" stray light pixels with median neighbor algorithm...")
            self._log.writeLog(__name__, "Replacing "+str(npix)+" stray light pixels with median neighbor algorithm...")
            niter = 5
            (data, nreplace) = bpm_replace_algorithm(data, objmask, niter)
            print("removeCosmicRaysSpecProcess::removeStrayLight> Replaced "+str(nreplace)+" of "+str(npix)+" pixels using "+str(niter)+" iterations.")
            self._log.writeLog(__name__, "Replaced "+str(nreplace)+" of "+str(npix)+" pixels using "+str(niter)+" iterations.")
            fdu.updateData(data)
            fdu._header.add_history("Interpolated "+str(nreplace)+" of "+str(npix)+" bad pixels with median neighbor algorithm, niter="+str(niter))
        if (self.getOption('write_calib_output', fdu.getTag()).lower() == "yes"):
            #write out sl data array - 0 where no stray light found, actual value assigned to sl otherwise
            #make directory if necessary
            outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
            if (not os.access(outdir+"/removedCosmicRays", os.F_OK)):
                os.mkdir(outdir+"/removedCosmicRays",0o755)
            #Create output filename
            slfile = outdir+"/removedCosmicRays/sldata_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(slfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(slfile)
            if (not os.access(slfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.tagDataAs("sldata", sldata)
                fdu.writeTo(slfile, tag="sldata")
                fdu.removeProperty("sldata")
        return True
    #end removeStrayLight

    def runDcr(self, fdu, calibs):
        #Read options
        cr_method = self.getOption('cosmic_ray_method', fdu.getTag()).lower()
        diaxis = int(self.getOption("dcr_disp_axis", fdu.getTag()))
        grad = int(self.getOption("dcr_grow_radius", fdu.getTag()))
        lrad = int(self.getOption("dcr_lower_radius", fdu.getTag()))
        npass = int(self.getOption("dcr_npass", fdu.getTag()))
        thresh = float(self.getOption("dcr_threshold", fdu.getTag()))
        urad = int(self.getOption("dcr_upper_radius", fdu.getTag()))
        verbose = int(self.getOption("dcr_verbosity", fdu.getTag()))
        xrad = int(self.getOption("dcr_xradius", fdu.getTag()))
        yrad = int(self.getOption("dcr_yradius", fdu.getTag()))

        useWholeChip = False
        if (self.getOption("mos_use_whole_chip", fdu.getTag()).lower() == "yes"):
            useWholeChip = True

        data = fdu.getData().astype(float32)
        slitmask = None
        if ('slitmask' in calibs and not useWholeChip):
            slitmask = calibs['slitmask'].getData()
        [npix, cleanData, crData] = dcr(data, slitmask=slitmask, thresh=thresh, xrad=xrad, yrad=yrad, npass=npass, diaxis=diaxis, lrad=lrad, urad=urad, grad=grad, verbose=verbose, log=self._log)

        #Put number of cosmic rays in header
        fdu._header.add_history('Cosmic rays removed: '+str(npix))
        #Update crmask and data depending on cosmic_ray_method
        if (cr_method == "mask"):
            crMask = (crData == 0) #crmask is good pixel mask
            fdu.tagDataAs("crmask", crMask)
            fdu._header.add_history('Cosmic ray method: dcr mask')
        else:
            #method = replace
            fdu.updateData(cleanData) #update FDU data with cleaned data
            fdu._header.add_history('Cosmic ray method: dcr replace')
        if (self.getOption('write_calib_output', fdu.getTag()).lower() == "yes"):
            #write out cr data array - 0 where no cosmic rays found, actual value assigned to cr otherwise
            #make directory if necessary
            outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
            if (not os.access(outdir+"/removedCosmicRays", os.F_OK)):
                os.mkdir(outdir+"/removedCosmicRays",0o755)
            #Create output filename
            crfile = outdir+"/removedCosmicRays/crdata_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(crfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(crfile)
            if (not os.access(crfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.tagDataAs("crdata", crData)
                fdu.writeTo(crfile, tag="crdata")
                fdu.removeProperty("crdata")
        return True
    #end runDcr

    def runDeepCR(self, fdu, calibs):
        #Read options
        cr_method = self.getOption('cosmic_ray_method', fdu.getTag()).lower()
        mask_model = self.getOption('deepcr_mask_model', fdu.getTag())
        inpaint_model = self.getOption('deepcr_inpaint_model', fdu.getTag())
        thresh = float(self.getOption('deepcr_threshold', fdu.getTag()))

        useWholeChip = False
        if (self.getOption("mos_use_whole_chip", fdu.getTag()).lower() == "yes"):
            useWholeChip = True

        data = fdu.getData().astype(float32)
        slitmask = None

        #Initialize model
        if (self._fdb.getGPUMode()):
            try:
                mdl = deepCR(mask=mask_model,inpaint=inpaint_model,device="GPU")
            except Exception as ex:
                print("removeCosmicRaysSpecProcess::runDeepCR> WARNING: deepCR failed to initialize GPU - "+str(ex)+"; Using CPU instead.")
                self._log.writeLog(__name__, "deepCR failed to initialize GPU - "+str(ex)+"; Using CPU instead.", type=fatboyLog.WARNING)
                mdl = deepCR(mask=mask_model,inpaint=inpaint_model,device="CPU")
        else:
            mdl = deepCR(mask=mask_model,inpaint=inpaint_model,device="CPU")

        inpaint = False
        if (cr_method == "replace"):
            inpaint = True

        if ('slitmask' not in calibs or useWholeChip):
            #process whole frame at once
            if (inpaint):
                crMask, cleanData = mdl.clean(data, threshold=thresh, inpaint=inpaint, segment=True)
            else:
                crMask = mdl.clean(data, threshold=thresh, inpaint=inpaint, segment=True)
            npix = crMask.sum()
        else:
            slitmask = calibs['slitmask']
            #mos data
            crMask = ones(data.shape, dtype=int16)
            if (inpaint):
                cleanData = zeros(data.shape, dtype=float32)
            nslits = slitmask.getData().max()
            npix = 0
            #Loop over slitlets
            for j in range(nslits):
                slit = where(slitmask.getData() == (j+1))
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    #horizontal dispersion for slits
                    ylo = slit[0].min()
                    yhi = slit[0].max()+1
                    tempMask = slitmask.getData()[ylo:yhi,:] == (j+1)
                    slit = (data[ylo:yhi,:]*tempMask).astype(float32)
                    #Run deepCR on this one slit.
                    if (inpaint):
                        m, c = mdl.clean(slit, threshold=thresh, inpaint=inpaint, segment=True)
                        cleanData[ylo:yhi,:][tempMask] = c[tempMask].astype(float32)
                    else:
                        m = mdl.clean(slit, threshold=thresh, inpaint=inpaint, segment=True)
                    crMask[ylo,yhi,:][tempMask] = m[tempMask]
                else:
                    #Vertical dispersion for slits
                    xlo = slit[1].min()
                    xhi = slit[1].max()+1
                    tempMask = slitmask.getData()[:,xlo:xhi] == (j+1)
                    slit = (data[:,xlo:xhi]*tempMask).astype(float32)
                    #Run deepCR on this one slit.
                    if (inpaint):
                        m, c = mdl.clean(slit, threshold=thresh, inpaint=inpaint, segment=True)
                        cleanData[:,xlo:xhi][tempMask] = c[tempMask].astype(float32)
                    else:
                        m = mdl.clean(slit, threshold=thresh, inpaint=inpaint, segment=True)
                    crMask[:,xlo:xhi][tempMask] = m[tempMask]
                np = m.sum()
                npix += np
                print("\tSlit "+str((j+1))+": cleaned "+str(np)+" pixels.")
                self._log.writeLog(__name__, "Slit "+str((j+1))+": cleaned "+str(np)+" pixels.", printCaller=False, tabLevel=1)

        #clean up GPU context
        if (self._fdb.getGPUMode()):
            superFATBOY.popGPUContext()
            superFATBOY.createGPUContext()
            if (not superFATBOY.threaded()):
                superFATBOY.popGPUContext()
                import pycuda.autoinit

        #Put number of cosmic rays in header
        fdu._header.add_history('Cosmic rays removed: '+str(npix))

        #Update crmask and data depending on cosmic_ray_method
        if (cr_method == "mask"):
            crMask = (crMask == 0) #crmask is good pixel mask
            crMask = crMask.astype(int16)
            fdu.tagDataAs("crmask", crMask)
            fdu._header.add_history('Cosmic ray method: deepCR mask')
        else:
            #Calculate crdata
            crdata = data - cleanData
            #method = replace
            fdu.updateData(cleanData) #update FDU data with cleaned data
            fdu._header.add_history('Cosmic ray method: deepCR replace')
        if (self.getOption('write_calib_output', fdu.getTag()).lower() == "yes"):
            #write out cr data array - 0 where no cosmic rays found, actual value assigned to cr otherwise
            #make directory if necessary
            outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
            if (not os.access(outdir+"/removedCosmicRays", os.F_OK)):
                os.mkdir(outdir+"/removedCosmicRays",0o755)
            #Create output filename
            crfile = outdir+"/removedCosmicRays/crdata_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(crfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(crfile)
            if (not os.access(crfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.tagDataAs("crdata", crdata)
                fdu.writeTo(crfile, tag="crdata")
                fdu.removeProperty("crdata")
        return True
    #end runDcr

    def runLacos(self, fdu, calibs):
        #Read options
        cr_method = self.getOption('cosmic_ray_method', fdu.getTag()).lower()
        npass = int(self.getOption("lacos_cosmic_ray_passes", fdu.getTag()))
        sigma = float(self.getOption("lacos_cosmic_ray_sigma", fdu.getTag()))
        xorder = int(self.getOption("lacos_xorder", fdu.getTag()))
        yorder = int(self.getOption("lacos_yorder", fdu.getTag()))

        useWholeChip = False
        if (self.getOption("mos_use_whole_chip", fdu.getTag()).lower() == "yes"):
            useWholeChip = True

        data = fdu.getData().astype(float32)
        slitmask = None
        if ('slitmask' in calibs):
            slitmask = calibs['slitmask']

        data = fdu.getData().astype(float32)
        if ('slitmask' not in calibs or useWholeChip):
            (npix, crmask, croutImage) = lacos_spec(data, None, None, gain=fdu.gain, readn=fdu.readnoise, sigclip=sigma, niter=npass, log = self._log, xorder=xorder, yorder=yorder, mask=1-fdu.getBadPixelMask().getData())
        else:
            #mos data
            crmask = ones(data.shape, dtype=int16)
            croutImage = zeros(data.shape, dtype=float32)
            nslits = slitmask.getData().max()
            npix = 0
            #Loop over slitlets
            for j in range(nslits):
                slit = where(slitmask.getData() == (j+1))
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    #horizontal dispersion for slits
                    ylo = slit[0].min()
                    yhi = slit[0].max()+1
                    tempMask = slitmask.getData()[ylo:yhi,:] == (j+1)
                    slit = (data[ylo:yhi,:]*tempMask).astype(float32)
                    #Run lacos on this one slit.  Put return value into cr_data array. slit will contain cleaned_data
                    (np, crslit, crdata) = lacos_spec(slit, None, None, gain=fdu.gain, readn=fdu.readnoise, sigclip=sigma, niter=npass, log = self._log, xorder=xorder, yorder=yorder, mask=1-fdu.getBadPixelMask().getData()[ylo:yhi,:])
                    npix += np
                    crmask[ylo:yhi,:][tempMask] = crslit[tempMask]
                    croutImage[ylo:yhi,:][tempMask] = crdata[tempMask].astype(float32)
                else:
                    #Vertical dispersion for slits
                    xlo = slit[1].min()
                    xhi = slit[1].max()+1
                    tempMask = slitmask.getData()[:,xlo:xhi] == (j+1)
                    slit = (data[:,xlo:xhi]*tempMask).astype(float32)
                    #Run lacos on this one slit.  Put return value into cr_data array. slit will contain cleaned_data
                    (np, crslit, crdata) = lacos_spec(slit, None, None, gain=fdu.gain, readn=fdu.readnoise, sigclip=sigma, niter=npass, log = self._log, xorder=xorder, yorder=yorder, mask=1-fdu.getBadPixelMask().getData()[:,xlo:xhi])
                    npix += np
                    crmask[:,xlo:xhi][tempMask] = crslit[tempMask]
                    croutImage[:,xlo:xhi][tempMask] = crdata[tempMask].astype(float32)
                print("\tSlit "+str((j+1))+": cleaned "+str(np)+" pixels.")
                self._log.writeLog(__name__, "Slit "+str((j+1))+": cleaned "+str(np)+" pixels.", printCaller=False, tabLevel=1)

        #Calculate crdata
        crdata = fdu.getData()-croutImage
        #Put number of cosmic rays in header
        fdu._header.add_history('Cosmic rays removed: '+str(npix))
        #Update crmask and data depending on cosmic_ray_method
        if (cr_method == "mask"):
            #crmask is good pixel mask
            fdu.tagDataAs("crmask", crmask)
            fdu._header.add_history('Cosmic ray method: lacos mask')
        else:
            #method = replace
            fdu.updateData(croutImage) #update FDU data with cleaned data
            fdu._header.add_history('Cosmic ray method: lacos replace')
        if (self.getOption('write_calib_output', fdu.getTag()).lower() == "yes"):
            #write out cr data array - 0 where no cosmic rays found, actual value assigned to cr otherwise
            #make directory if necessary
            outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
            if (not os.access(outdir+"/removedCosmicRays", os.F_OK)):
                os.mkdir(outdir+"/removedCosmicRays",0o755)
            #Create output filename
            crfile = outdir+"/removedCosmicRays/crdata_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(crfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(crfile)
            if (not os.access(crfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.tagDataAs("crdata", crdata)
                fdu.writeTo(crfile, tag="crdata")
                fdu.removeProperty("crdata")
        return True
    #end runLacos

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('cosmic_ray_algorithm', 'dcr') #dcr | lacos
        self._optioninfo.setdefault('cosmic_ray_algorithm', 'dcr | lacos | deepcr; default = dcr, uses histograms of postage stamp subimages\nlacos uses laplacian edge detection\ndeepcr uses deep neural net')
        self._options.setdefault('cosmic_ray_method', 'mask') #mask | replace
        self._optioninfo.setdefault('cosmic_ray_method', 'mask | replace')
        self._options.setdefault('dcr_disp_axis', 1)
        self._optioninfo.setdefault('dcr_disp_axis', 'Dispersion axis: 0 - no dispersion, 1 - X, 2 - Y')
        self._options.setdefault('dcr_grow_radius', 1)
        self._optioninfo.setdefault('dcr_grow_radius', 'Growing radius')
        self._options.setdefault('dcr_lower_radius', 1)
        self._optioninfo.setdefault('dcr_lower_radius', 'Lower radius of region for replacement statistics')
        self._options.setdefault('dcr_npass', 5)
        self._optioninfo.setdefault('dcr_npass', 'Maximum number of cleaning passes')
        self._options.setdefault('dcr_threshold', 4.0)
        self._optioninfo.setdefault('dcr_threshold', 'Threshold (in STDDEV)')
        self._options.setdefault('dcr_upper_radius', 3)
        self._optioninfo.setdefault('dcr_upper_radius', 'Upper radius of region for replacement statistics')
        self._options.setdefault('dcr_verbosity', '1')
        self._optioninfo.setdefault('dcr_verbosity', 'Verbose level [0,1,2]')
        self._options.setdefault('dcr_xradius', 9)
        self._optioninfo.setdefault('dcr_xradius', 'x-radius of the box (size = 2 * radius)')
        self._options.setdefault('dcr_yradius', 9)
        self._optioninfo.setdefault('dcr_yradius', 'y-radius of the box (size = 2 * radius)')

        self._options.setdefault('lacos_cosmic_ray_passes', 1)
        self._optioninfo.setdefault('lacos_cosmic_ray_passes', 'Number of passes for lacos algorithm')
        self._options.setdefault('lacos_cosmic_ray_sigma', 10)
        self._optioninfo.setdefault('lacos_cosmic_ray_sigma', 'Sigma threshold for lacos algorithm')
        self._options.setdefault('lacos_xorder', -1)
        self._optioninfo.setdefault('lacos_xorder', 'Fit order in x-direction (-1 = smooth instead of fit and subtract)')
        self._options.setdefault('lacos_yorder', -1)
        self._optioninfo.setdefault('lacos_yorder', 'Fit order in y-direction (-1 = smooth instead of fit and subtract)')

        self._options.setdefault('deepcr_mask_model', 'ACS-WFC-F606W-2-32')
        self._optioninfo.setdefault('deepcr_mask_model', 'Model to use for masking in deepCR.\nDefault was trained on HST imaging data.')
        self._options.setdefault('deepcr_inpaint_model', 'ACS-WFC-F606W-2-32')
        self._optioninfo.setdefault('deepcr_inpaint_model', 'Model to use for inpainting (replacing) in deepCR.\nDefault was trained on HST imaging data.')
        self._options.setdefault('deepcr_threshold', 0.5)
        self._optioninfo.setdefault('deepcr_threshold', 'Threshold to use with deepCR.  Default 0.5\nSet higher to avoid false positives.')

        self._options.setdefault('remove_stray_light','no')
        self._optioninfo.setdefault('remove_stray_light', 'Additionally attempt to remove circular stray light\nartifcats, useful in KAST red data.\nRequires sep.')
        self._options.setdefault('stray_light_method', 'mask') #mask | replace
        self._optioninfo.setdefault('stray_light_method', 'mask | replace')
        self._options.setdefault('stray_light_max_area', 225)
        self._optioninfo.setdefault('stray_light_max_area', 'Maximum number of pixels for a stray light artifact.')
        self._options.setdefault('stray_light_min_area', 25)
        self._optioninfo.setdefault('stray_light_min_area', 'Minimum number of pixels for a stray light artifact.')
        self._options.setdefault('stray_light_sigma_threshold', 100)
        self._optioninfo.setdefault('stray_light_sigma_threshold', 'Minimum threshold times background RMS to be detected.')
        self._options.setdefault('stray_light_symmetry_threshold', 0.9)
        self._optioninfo.setdefault('stray_light_symmetry_threshold', 'Semi-minor axis b / semi-major axis a\nof feature must be this or higher (1 = circle).')

        self._options.setdefault('mos_use_whole_chip', 'no')
        self._optioninfo.setdefault('mos_use_whole_chip', 'Set to yes to run CR algorithms on the whole chip\nrather than each individual slitlet.')
    #end setDefaultOptions

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/removedCosmicRays", os.F_OK)):
            os.mkdir(outdir+"/removedCosmicRays",0o755)
        #Create output filename
        rcrfile = outdir+"/removedCosmicRays/rcr_"+fdu.getFullId()
        #Check to see if it exists
        if (os.access(rcrfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(rcrfile)
        if (not os.access(rcrfile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(rcrfile)
        #Write out crmask if it exists
        if (fdu.hasProperty("crmask")):
            crmask = outdir+"/removedCosmicRays/crmask_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(crmask, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(crmask)
            if (not os.access(crmask, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.writeTo(crmask, tag="crmask")
    #end writeOutput
