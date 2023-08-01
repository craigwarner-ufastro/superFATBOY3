from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY import gpu_drihizzle, drihizzle
from superFATBOY.gpu_arraymedian import *
import os, time

usePlot = True
try:
    import matplotlib.pyplot as plt
except Exception as ex:
    print("Warning: Could not import matplotlib!")
    usePlot = False

_gpufit_installed = False
try:
    import pygpufit.gpufit as gf
    _gpufit_installed = True
except Exception as ex:
    print("superFATBOY::miradasCharacterizePSFProcess> Info: pygpufit not installed")

class miradasCharacterizePSFProcess(fatboyProcess):
    """ Trace and characterize the PSF at each cut through a 3-d datacube """
    _modeTags = ["miradas"]

    slitlets = []

    def characterizePSF(self, fdu, calibs):
        #read options
        centroid_method = self.getOption("centroid_method", fdu.getTag())
        nslits = fdu.getProperty("nslits")
        dthresh = float(self.getOption("detection_threshold", fdu.getTag()))
        #self.slitlets already has indices of slitlets

        for islit in self.slitlets:
            if (fdu.hasProperty("datacube_"+str(islit))):
                t = time.time()
                data = fdu.getData(tag="datacube_"+str(islit))
                ysize = data.shape[0]
                #Reshape n x 3 x m cubes into n x 3*m to vectorize functions
                rdata = data.reshape((ysize, data.shape[1]*data.shape[2]))
                maxVals = rdata.max(1)
                medVals = gpu_arraymedian(rdata, axis="X", even=True, kernel2d=fatboyclib.median2d)
                stdDevs = rdata.std(axis=1)
                b = where(stdDevs == 0)
                stdDevs[b] = 1
                sigmas = (maxVals-medVals)/stdDevs
                stdDevs[b] = 0
                sigmas[b] = 0
                #PSF = n x 8 array with columns wavelength, x_cen, y_cen, fwhm, peak, median background, std dev, sigma
                #Where the peak is below the detection threshold, x_cen, y_cen, and fwhm are set to -1
                psf = zeros((ysize, 8), dtype=float32)
                #Pass islit-1 to getWavelengthSolution, expects 0-11 not 1-12
                psf[:,0] = getWavelengthSolution(fdu, islit-1, ysize)
                psf[:,4] = maxVals
                psf[:,5] = medVals
                psf[:,6] = stdDevs
                psf[:,7] = sigmas
                #Set flag as bool array, True where a centroid should be done
                #Don't centroid where sigma is below dthresh
                flag = sigmas >= dthresh
                #Don't centroid where there are ANY rows with < 3 datapoints
                flag[(data != 0).sum(2).min(1) < 3] = False
                #Don't centroid if total flux in this cut is < 1% of the average flux
                nzero = data.copy()
                nzero[nzero < 0] = 0
                sum1d = nzero.sum(2).sum(1)
                flag[sum1d < gpu_arraymedian(sum1d)/100.] = False
                #Don't centroid if peak value is at (0,0) in this cut
                flag[argmax(rdata, axis=1) == 0] = False
                if (flag.sum() < flag.size/100):
                    #Less than 1% of cuts can be characterized
                    print("miradasCharacterizePSFProcess::characterizePSF> WARNING: Slitlet "+str(islit)+" - could not characterize PSF.  Not enough datapoints.")
                    self._log.writeLog(__name__, "Slitlet "+str(islit)+" - could not characterize PSF.  Not enough datapoints.", type=fatboyLog.WARNING)
                    fdu.setProperty("psf_"+str(islit), psf)
                    continue
                if (centroid_method == "fit_2d_gaussian"):
                    #Find initial guesses
                    guess = zeros(5, dtype=float32)
                    if (fdu.hasProperty("image_centroid_"+str(islit))):
                        #Collapsed spaxels have been centroided already
                        (xcen, ycen, fwhm) = fdu.getProperty("image_centroid_"+str(islit))
                    else:
                        #Collapse 3-d datacube and centroid
                        data2d = data.sum(0)
                        lsq = fitGaussian2d(data2d, True)
                        if (lsq[1] == False):
                            print("miradasCharacterizePSFProcess::characterizePSF> WARNING: Slitlet "+str(islit)+" - could not characterize PSF.  Centroid of collapsed spaxel invalid!")
                            self._log.writeLog(__name__, "Slitlet "+str(islit)+" - could not characterize PSF.  Centroid of collapsed spaxel invalid.", type=fatboyLog.WARNING)
                            fdu.setProperty("psf_"+str(islit), psf)
                            continue
                        xcen = lsq[0][1]
                        ycen = lsq[0][2]
                        fwhm = lsq[0][3]*2.3548
                    if (xcen < 0 or ycen < 0 or fwhm < 0):
                        print("miradasCharacterizePSFProcess::characterizePSF> WARNING: Slitlet "+str(islit)+" - could not characterize PSF.  Centroid of collapsed spaxel invalid!")
                        self._log.writeLog(__name__, "Slitlet "+str(islit)+" - could not characterize PSF.  Centroid of collapsed spaxel invalid.", type=fatboyLog.WARNING)
                        fdu.setProperty("psf_"+str(islit), psf)
                        continue
                    if (self._fdb.getGPUMode() and _gpufit_installed):
                        #GPU mode - find fwhms and centroids in parallel
                        model_id = gf.ModelID.GAUSS_2D
                        estimator_id = gf.EstimatorID.LSE
                        max_number_iterations = 20
                        #gpufit requires xsize == ysize for each cut
                        dshape = data.shape
                        nfits = dshape[0]
                        maxdim = max(dshape[1], dshape[2])
                        padded_data = zeros((nfits, maxdim, maxdim), float32)
                        pady = (maxdim - dshape[1])//2
                        padx = (maxdim - dshape[2])//2
                        padded_data[:, pady:pady+dshape[1], padx:padx+dshape[2]] = data.astype(float32)
                        padded_data = padded_data.reshape((nfits, maxdim*maxdim))
                        #Setup initial guesses
                        init_guess = zeros((nfits, 5), float32)
                        init_guess[:,0] = padded_data.max(1)
                        init_guess[:,1] = xcen+padx
                        init_guess[:,2] = ycen+pady
                        init_guess[:,3] = fwhm/2.3548
                        #Mask zeros
                        padded_data[padded_data < 0] = 0
                        #Do fit
                        parameters, states, chi_squares, number_iterations, execution_time = gf.fit(padded_data, None, model_id, init_guess, None, max_number_iterations, None, estimator_id, None)
                        psf[:,1] = parameters[:,1]-padx #xcen
                        psf[:,2] = parameters[:,2]-pady #ycen
                        psf[:,3] = abs(parameters[:,3]*2.3543) #fwhm
                        psf[states != 0,1:4] = -1 #did not converge
                        psf[flag == False,1:4] = -1 #not flagged to centroid
                    else:
                        guess[1] = xcen
                        guess[2] = ycen
                        guess[3] = fwhm/2.3548
                        for j in range(ysize):
                            if (not flag[j]):
                                psf[j,1] = -1
                                psf[j,2] = -1
                                psf[j,3] = -1
                            else:
                                guess[0] = maxVals[j]
                                lsq = fitGaussian2d(data[j,:,:], maskNeg=True, maxWidth=3, guess=guess)
                                if (lsq[1] == False):
                                    print("miradasCharacterizePSFProcess::characterizePSF>: ERROR fitting Gaussian for cut "+str(j)+": "+str(ex))
                                    self._log.writeLog(__name__, "ERROR fitting Gaussian for cut "+str(j)+": "+str(ex), type=fatboyLog.ERROR)
                                    psf[j,1] = -1
                                    psf[j,2] = -1
                                    psf[j,3] = -1
                                    continue
                                if (lsq[1] == 5):
                                    #Max iterations exceeded
                                    psf[j,1] = -1
                                    psf[j,2] = -1
                                    psf[j,3] = -1
                                    continue
                                psf[j,1] = lsq[0][1] #xcen
                                psf[j,2] = lsq[0][2] #ycen
                                psf[j,3] = abs(lsq[0][3]*2.3548) #fwhm
                elif (centroid_method == "use_derivatives"):
                    #need to pad data if either dimension < 9
                    padx = max(9-data.shape[2], 0)//2
                    pady = max(9-data.shape[1], 0)//2
                    padded = zeros((data.shape[0], data.shape[1]+pady*2, data.shape[2]+padx*2), dtype=float32)
                    padded[:, pady:padded.shape[1]-pady, padx:padded.shape[2]-padx] = data
                    #Find initial guesses
                    if (fdu.hasProperty("image_centroid_"+str(islit))):
                        #Collapsed spaxels have been centroided already
                        (mx, my, fwhm) = fdu.getProperty("image_centroid_"+str(islit))
                        #Add in padded values
                        mx += padx
                        my += pady
                    else:
                        #Collapse 3-d datacube and centroid
                        padded2d = padded.sum(0)
                        b = where(padded2d == padded2d.max())
                        mx = b[1][0]
                        my = b[0][0]
                        (fwhm, sig, fwhm1ds, bg) = fwhm2d(padded2d)
                        (mx, my) = getCentroid(padded2d, mx, my, fwhm)
                    if (mx < 0 or my < 0 or fwhm < 0):
                        print("miradasCharacterizePSFProcess::characterizePSF> WARNING: Slitlet "+str(islit)+" - could not characterize PSF.  Centroid of collapsed spaxel invalid!")
                        self._log.writeLog(__name__, "Slitlet "+str(islit)+" - could not characterize PSF.  Centroid of collapsed spaxel invalid.", type=fatboyLog.WARNING)
                        fdu.setProperty("psf_"+str(islit), psf)
                        continue
                    #Round mx, my to int values
                    mx = int(mx+0.5)
                    my = int(my+0.5)
                    if (self._fdb.getGPUMode()):
                        #GPU mode - find fwhms and centroids in parallel
                        fwhm_vals = fwhm2d_cube_gpu(padded, flag=flag, log=self._log)
                        cens = getCentroid_cube_gpu(padded, mx, my, fwhm, flag=flag, log=self._log)
                        psf[:,1] = cens[:,0]-padx #xcen
                        psf[:,2] = cens[:,1]-pady #ycen
                        psf[:,3] = fwhm_vals[:,0] #fwhm mean
                    else:
                        for j in range(ysize):
                            if (not flag[j]):
                                psf[j,1] = -1
                                psf[j,2] = -1
                                psf[j,3] = -1
                            else:
                                fwhm_vals = fwhm2d(padded[j,:,:])
                                (xcen, ycen) = getCentroid(padded[j,:,:], mx, my, fwhm)
                                psf[j,1] = xcen-padx
                                psf[j,2] = ycen-pady
                                psf[j,3] = fwhm_vals[0]
                #b = (psf[:,2] != -1)
                #xs = arange(ysize)
                #plt.plot(xs[b], psf[b,2])
                #plt.plot(xs[b], psf[b,3])
                #plt.show()
                b = where(psf[:,1] != -1)
                mxcen = psf[b,1].mean()
                mycen = psf[b,2].mean()
                mfwhm = psf[b,3].mean()
                sxcen = psf[b,1].std()
                sycen = psf[b,2].std()
                sfwhm = psf[b,3].std()
                print("miradasCharacterizePSFProcess::characterizePSF> Slitlet "+str(islit)+" characterized PSF with "+centroid_method+": (x="+formatNum(mxcen)+" +/- "+formatNum(sxcen)+"; y="+formatNum(mycen)+" +/- "+formatNum(sycen)+") fwhm="+formatNum(mfwhm)+" +/- "+formatNum(sfwhm)+", "+str(b[0].size) +" of "+str(ysize)+" datapoints in "+formatNum(time.time()-t)+" sec.")
                self._log.writeLog(__name__, "Slitlet "+str(islit)+" characterized PSF with "+centroid_method+": (x="+formatNum(mxcen)+" +/- "+formatNum(sxcen)+"; y="+formatNum(mycen)+" +/- "+formatNum(sycen)+") fwhm="+formatNum(mfwhm)+" +/- "+formatNum(sfwhm)+", "+str(b[0].size) +" of "+str(ysize)+" datapoints in "+formatNum(time.time()-t)+" sec.")
                fdu.setProperty("psf_"+str(islit), psf)
    #end characterizePSF

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("MIRADAS: characterize PSF")
        print(fdu._identFull)

        #Check if output exists first and update from disk
        cpfile = "characterizedPSFs/3d_"+fdu.getFullId()
        nslits = 13 #max number of slits
        if (fdu.hasProperty("nslits")):
            nslits = fdu.getProperty("nslits")
        tag = ["TAGNAME"]*nslits
        if (self.checkOutputExists(fdu, cpfile, tag=tag)):
            return True

        #Call get calibs to return dict() of calibration frames.
        #For CharacterizePSF, this checks that 3d datacubes have been created and
        #that wavelength calibration has been performed.
        calibs = self.getCalibs(fdu, prevProc)

        if (not fdu.hasProperty('nslits')):
            print("miradasCharacterizePSFProcess::execute> ERROR: Could not find nslits so could not collapse spaxels.")
            self._log.writeLog(__name__, "Could not find nslits so could not collapse spaxels.", type=fatboyLog.ERROR)
            return False

        #call characterizePSF helper function to do actual PSF tracing
        self.characterizePSF(fdu, calibs)
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

        #Look for matching grism_keyword, specmode, and dispersion
        headerVals = dict()
        headerVals['grism_keyword'] = fdu.grism

        properties = dict()
        properties['specmode'] = fdu.getProperty("specmode")
        properties['dispersion'] = fdu.getProperty("dispersion")

        #May need a slitmask to determine nslits
        if (not 'slitmask' in calibs):
            #Use new fdu.getSlitmask method
            slitmask = fdu.getSlitmask(pname=None, properties=properties, headerVals=headerVals)
            if (slitmask is not None):
                #Found slitmask
                calibs['slitmask'] = slitmask

        #read options
        slitlet_number = self.getOption("slitlet_number", fdu.getTag())
        doAllSlitlets = False
        if (slitlet_number == 'all'):
            doAllSlitlets = True
        elif (isInt(slitlet_number)):
            slitlet_number = int(slitlet_number)

        if (fdu.hasProperty("nslits")):
            nslits = fdu.getProperty("nslits")
        else:
            if (fdu.hasProperty('slitmask')):
                nslits = fdu.getData(tag="slitmask").max()
                fdu.setProperty("nslits", nslits)
            elif ('slitmask' in calibs):
                nslits = calibs['slitmask'].getData().max()
                fdu.setProperty("nslits", nslits)

        if (doAllSlitlets):
            self.slitlets = arange(1, nslits+1)
        else:
            self.slitlets = [slitlet_number]

        hasDatacubes = True
        #Loop over self.slitlets (could be one pass or nslits passes)
        for islit in self.slitlets:
            if (not fdu.hasProperty("datacube_"+str(islit))):
                hasDatacubes = False

        if (not hasDatacubes):
            #Call miradasCreate3dDatacubes to create them
            #Use method getProcessByName to return instantiated version of process.  Only works if process is included in XML file.
            #Returns None on a failure
            datacube_process = self._fdb.getProcessByName("miradasCreate3dDatacubes")
            if (datacube_process is None or not isinstance(datacube_process, fatboyProcess)):
                print("miradasCharacterizePSFProcess::getCalibs> ERROR: could not find process miradasCreate3dDatacubes!  Check your XML file!")
                self._log.writeLog(__name__, "could not find process miradasCreate3dDatacubes!  Check your XML file!", type=fatboyLog.ERROR)
                return calibs
            #Create datacubes for any/all slitlets requested for characterizing PSF
            datacube_process.setOption("slitlet_number", slitlet_number)
            #Use recursivelyExecute to execute datacube_process on fdu
            self.recursivelyExecute([fdu], [datacube_process])

        #Use helper method to determine if a wavelength solution has been performed
        hasWavelengthSol = hasWavelengthSolution(fdu)
        if (not hasWavelengthSol):
            #Call wavelengthCalibrate to find wavelength solution
            #Use method getProcessByName to return instantiated version of process.  Only works if process is included in XML file.
            #Returns None on a failure
            wc_process = self._fdb.getProcessByName("wavelengthCalibrate")
            if (wc_process is None or not isinstance(wc_process, fatboyProcess)):
                print("miradasCharacterizePSFProcess::getCalibs> ERROR: could not find process wavelengthCalibrate!  Check your XML file!")
                self._log.writeLog(__name__, "could not find process wavelengthCalibrate!  Check your XML file!", type=fatboyLog.ERROR)
                return calibs
            #Use recursivelyExecute to execute wc_process on fdu
            self.recursivelyExecute([fdu], [wc_process])

        return calibs
    #end getCalibs

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('centroid_method', 'fit_2d_gaussian')
        self._optioninfo.setdefault('centroid_method', 'fit_2d_gaussian | use_derivatives')

        self._options.setdefault('detection_threshold', '2')
        self._optioninfo.setdefault('detection_threshold', 'Minimum sigma above median to detect a source')

        self._options.setdefault('slitlet_number', 'all')
        self._optioninfo.setdefault('slitlet_number', 'Set to all (default) to collapse spaxels for all slitlets.\nSet to a number 1-13 to only select one slitlet.')
    #end setDefaultOptions

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/characterizedPSFs", os.F_OK)):
            os.mkdir(outdir+"/characterizedPSFs",0o755)
        nslits = fdu.getProperty("nslits")
        if (nslits is None):
            nslits = 13
        #Set up lists for tag and headerExt
        tags = []
        headerExt = []
        for j in range(1, nslits+1):
            key = "psf_"+str(j)
            if (fdu.hasProperty(key)):
                tags.append(key)
                #Create new header dict for this extension
                imhead = dict()
                #Add values
                imhead['TAGNAME'] = key
                imhead['SLIT_NUM'] = j
                headerExt.append(imhead)
        #Create output filename
        cpfile = outdir+"/characterizedPSFs/psf_"+fdu.getFullId()
        #Check to see if it exists
        if (os.access(cpfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(cpfile)
        if (not os.access(cpfile, os.F_OK)):
            #Use fatboyDataUnit writePropertiesToMEF method to write
            fdu.writePropertiesToMEF(cpfile, tag=tags, headerExt=headerExt)
    #end writeOutput
