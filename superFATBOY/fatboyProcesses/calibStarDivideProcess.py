from superFATBOY.fatboyDataUnit import fatboyDataUnit
from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.datatypeExtensions.fatboySpecCalib import fatboySpecCalib

from superFATBOY import gpu_drihizzle, drihizzle
from numpy import *
from scipy.optimize import leastsq

block_size = 512

class calibStarDivideProcess(fatboyProcess):
    _modeTags = ["spectroscopy", "miradas"]

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("Calib Star Divide")
        print(fdu._identFull)

        #Check if output exists first and update from disk
        csdfile = "calibStarDivided/csd_"+fdu.getFullId()
        if (self.checkOutputExists(fdu, csdfile)):
            #Also check if "cleanFrame" exists
            cleanfile = "calibStarDivided/clean_csd_"+fdu.getFullId()
            self.checkOutputExists(fdu, cleanfile, tag="cleanFrame")
            #Also check if "resampled" exists
            resampfile = "calibStarDivided/resamp_csd_"+fdu.getFullId()
            self.checkOutputExists(fdu, resampfile, tag="resampled")
            return True

        #Call get calibs to return dict() of calibration frames.
        #For calibStarDivided, this dict should have a standard star
        calibs = self.getCalibs(fdu, prevProc)

        if (not 'standard' in calibs):
            #Failed to obtain standard star
            #Issue error message and disable this FDU
            print("calibStarDivideProcess::execute> ERROR: Could not find standard star calibration for "+fdu.getFullId()+" (filter="+str(fdu.filter)+").  Discarding Image!")
            self._log.writeLog(__name__, "Could not find standard star calibration for "+fdu.getFullId()+" (filter="+str(fdu.filter)+").  Discarding Image!", type=fatboyLog.ERROR)
            #disable this FDU
            fdu.disable()
            return False

        #call calibStarDivide helper function to do gpu/cpu calibration
        success = self.calibStarDivide(fdu, calibs)
        if (success):
            #Update history
            fdu._header.add_history('Calibrated with standard star '+calibs['standard'].getFullId())
        return success
    #end execute

    ## OVERRIDE getCalibs
    def getCalibs(self, fdu, prevProc = None):
        calibs = dict()

        if (fdu.hasProperty("standard")):
            #passed from XML as <property> of <object>.  Use fdu as source header
            stdfilename = fdu.getProperty("standard")
            if (os.access(stdfilename, os.F_OK)):
                print("calibStarDivideProcess::getCalibs> Using standard star "+stdfilename+"...")
                self._log.writeLog(__name__, "Using standard star "+stdfilename+"...")
                calibs['standard'] = fatboySpecCalib(self._pname, "standard", fdu, filename=stdfilename, log=self._log)
                return calibs
            else:
                print("calibStarDivideProcess::getCalibs> Warning: Could not find standard "+stdfilename+"...")
                self._log.writeLog(__name__, "Could not find standard "+stdfilename+"...", type=fatboyLog.WARNING)

        stdfilename = self.getCalib("standard", fdu.getTag())
        if (stdfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(stdfilename, os.F_OK)):
                print("calibStarDivideProcess::getCalibs> Using standard star "+stdfilename+"...")
                self._log.writeLog(__name__, "Using standard star "+stdfilename+"...")
                calibs['standard'] = fatboySpecCalib(self._pname, "standard", fdu, filename=stdfilename, log=self._log)
                return calibs
            else:
                print("calibStarDivideProcess::getCalibs> Warning: Could not find standard "+stdfilename+"...")
                self._log.writeLog(__name__, "Could not find standard "+stdfilename+"...", type=fatboyLog.WARNING)

        #Look for matching grism_keyword, specmode, and dispersion
        headerVals = dict()
        headerVals['grism_keyword'] = fdu.grism

        properties = dict()
        properties['dispersion'] = fdu.getProperty("dispersion")

        if (not 'standard' in calibs):
            #1a) check for a standard matching specmode/filter/grism and TAGGED for this object
            standards = self._fdb.getTaggedCalibs(fdu._id, obstype=fdu.FDU_TYPE_STANDARD, filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals)
            if (len(standards) > 0):
                #Found standard stars associated with this fdu. Recursively process
                print("calibStarDivideProcess::getCalibs> Recursively processing standard star for tagged object "+fdu._id+", filter "+str(fdu.filter)+"...")
                self._log.writeLog(__name__, " Recursively processing standard star for tagged object "+fdu._id+", filter "+str(fdu.filter)+"...")
                #recursively process
                self.recursivelyExecute(standards, prevProc)
                #All but first FDU should be disabled after they're processed through spectral extraction
                #Check that extraction has been done though
                if (standards[0].hasProperty("extracted")):
                    calibs['standard'] = standards[0]
                    return calibs
            #2) Check for individual flat frames matching specmode/filter/grism to create master flat
            standards = self._fdb.getCalibs(obstype=fdu.FDU_TYPE_STANDARD, filter=fdu.filter, section=fdu.section, tag=fdu.getTag(), properties=properties, headerVals=headerVals)
            if (len(standards) > 0):
                #Found standard stars associated with this fdu. Recursively process
                print("calibStarDivideProcess::getCalibs> Recursively processing standard star for object "+fdu._id+", filter "+str(fdu.filter)+"...")
                self._log.writeLog(__name__, " Recursively processing standard star for object "+fdu._id+", filter "+str(fdu.filter)+"...")
                #recursively process
                self.recursivelyExecute(standards, prevProc)
                #All but first FDU should be disabled after they're processed through spectral extraction
                #Check that extraction has been done though
                if (standards[0].hasProperty("extracted")):
                    calibs['standard'] = standards[0]
                    return calibs
        return calibs
    #end getCalibs

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('debug_mode', 'no')
        self._optioninfo.setdefault('debug_mode', 'Show plots of each slitlet and print out debugging information.')
        self._options.setdefault('write_fits_table', 'no')
    #end setDefaultOptions

    ## Wavelength Calibrate data
    def calibStarDivide(self, fdu, calibs):
        #Read options
        doWavelength = True
        doFitsTable = False
        if (self.getOption("write_fits_table", fdu.getTag()).lower() == "yes"):
            doFitsTable = True

        #Check that data has 2-d shape even if e.g., 1x2048
        shp = fdu.getData().shape
        if (len(shp) == 1):
            fdu.updateData(fdu.getData().reshape((1, shp[0])))
        shp = calibs['standard'].getData().shape
        if (len(shp) == 1):
            calibs['standard'].updateData(calibs['standard'].getData().reshape((1, shp[0])))

        xsize = fdu.getShape()[1] #xsize is in wavelength direction
        nspec = fdu.getShape()[0]
        csxsize = calibs['standard'].getShape()[1]

        #Create output dir if it doesn't exist
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/calibStarDivided", os.F_OK)):
            os.mkdir(outdir+"/calibStarDivided",0o755)

        #Shape will now be the same regardless of orientation
        rssdata = zeros((nspec, xsize), dtype=float32)
        if (fdu.hasProperty("cleanFrame")):
            rssclean = zeros((nspec, xsize), dtype=float32)
        if (fdu.hasProperty("resampled")):
            resampxsize = fdu.getData(tag="resampled").shape[1]
            resampcsxsize = calibs['standard'].getData(tag="resampled").shape[1]
            rssresamp = zeros((nspec, resampxsize), dtype=float32)

        doIndivSlitlets = False
        #Use new helper methods
        if (hasMultipleWavelengthSolutions(fdu)):
            #Different wavelength solution for each slitlet
            doIndivSlitlets = True #Process each slitlet below
        elif (hasWavelengthSolution(fdu)):
            #One single wavelength solution
            wave = getWavelengthSolution(fdu, 0, xsize)
            if (fdu.hasProperty("resampled")):
                resamp_wave = getWavelengthSolution(fdu.getProperty("resampledHeader"), 0, resampxsize)
        else:
            print("calibStarDivideProcess::calibStarDivide> Warning: Can not find header keyword PORDER in "+fdu.getFullId()+".  Wavlength solution will not be used to resample before dividing!")
            self._log.writeLog(__name__, "Can not find header keyword PORDER in "+fdu.getFullId()+".  Wavlength solution will not be used to resample before dividing!", type=fatboyLog.WARNING)
            wave = arange(xsize, dtype=float32)
            doWavelength = False

        #set up FITS table
        if (doFitsTable):
            columns = []
            if (not doIndivSlitlets):
                #One single wavelength solution
                columns.append(pyfits.Column(name='Wavelength', format='D', array=wave))

        #Calculate discrete wavelength array for calib star
        xs = arange(csxsize, dtype=float32)
        cswave = zeros(csxsize, dtype=float32)
        if (doWavelength and hasWavelengthSolution(calibs['standard'])):
            cswave = getWavelengthSolution(calibs['standard'], 0, csxsize)
            if (fdu.hasProperty("resampled")):
                resamp_cswave = getWavelengthSolution(calibs['standard'].getProperty("resampledHeader"), 0, resampcsxsize)
        elif (csxsize == xsize):
            #Do pixel to pixel division
            doWavelength = False
            print("calibStarDivideProcess::calibStarDivide> Warning: Can not find header keyword PORDER in "+calibs['standard'].getFullId()+".  Wavlength solution will not be used to resample before dividing!")
            self._log.writeLog(__name__, "Can not find header keyword PORDER in "+calibs['standard'].getFullId()+".  Wavlength solution will not be used to resample before dividing!", type=fatboyLog.WARNING)
        else:
            print("calibStarDivideProcess::calibStarDivide> ERROR: Can not find header keyword PORDER in "+calibs['standard'].getFullId()+". Calibration star division cannot be performed!")
            self._log.writeLog(__name__, "Can not find header keyword PORDER in "+calibs['standard'].getFullId()+". Calibration star division cannot be performed!", type=fatboyLog.ERROR)
            return False

        print("calibStarDivideProcess::calibStarDivide> Resampling standard star "+calibs['standard'].getFullId()+" and dividing...")
        self._log.writeLog(__name__, "Resampling standard star "+calibs['standard'].getFullId()+" and dividing...")

        if (doWavelength and not doIndivSlitlets):
            #Transform once to common wavelength scale
            #Use helper function
            (ystar, b, good) = self.resampleStandard(calibs['standard'], cswave, wave)
            if (fdu.hasProperty("cleanFrame")):
                (ystar_clean, b_clean, good_clean) = self.resampleStandard(calibs['standard'], cswave, wave, tag="cleanFrame")
            if (fdu.hasProperty("resampled")):
                (ystar_resamp, b_resamp, good_resamp) = self.resampleStandard(calibs['standard'], resamp_cswave, resamp_wave, tag="resampled")

        #Loop over specList and extract spectra
        for j in range(nspec):
            if (doWavelength and doIndivSlitlets):
                #Calculate wavelength solution for this spectrum
                wave = getWavelengthSolution(fdu, j, xsize)
                if (doFitsTable):
                    columns.append(pyfits.Column(name='Wavelength_'+str(j+1), format='D', array=wave))
                #Transform standard star to commmon wavelength
                #Use helper function
                (ystar, b, good) = self.resampleStandard(calibs['standard'], cswave, wave)
                if (fdu.hasProperty("cleanFrame")):
                    (ystar_clean, b_clean, good_clean) = self.resampleStandard(calibs['standard'], cswave, wave, tag="cleanFrame")
                if (fdu.hasProperty("resampled")):
                    resamp_wave = getWavelengthSolution(fdu.getProperty("resampledHeader"), j, resampxsize)
                    (ystar_resamp, b_resamp, good_resamp) = self.resampleStandard(calibs['standard'], resamp_cswave, resamp_wave, tag="resampled")
            if (doWavelength):
                #Output data will be zero outside of wavelength range used
                rssdata[j, b[good]] = (fdu.getData()[j][b][good]/ystar[good])
                if (fdu.hasProperty("cleanFrame")):
                    rssclean[j, b_clean[good_clean]] = (fdu.getData(tag="cleanFrame")[j][b_clean][good_clean]/ystar_clean[good_clean])
                if (fdu.hasProperty("resampled")):
                    rssresamp[j, b_resamp[good_resamp]] = (fdu.getData(tag="resampled")[j][b_resamp][good_resamp]/ystar_resamp[good_resamp])
            else:
                #Simply divide all values where calib star is nonzero
                b = calibs['standard'].getData()[0,:] != 0
                rssdata[j,b] = fdu.getData()[j,b]/calibs['standard'].getData()[0,b]
                if (fdu.hasProperty("cleanFrame")):
                    rssclean[j,b_clean] = fdu.getData(tag="cleanFrame")[j,b_clean]/calibs['standard'].getData(tag="cleanFrame")[0,b_clean]
                if (fdu.hasProperty("resampled")):
                    rssresamp[j,b_resamp] = fdu.getData(tag="resampled")[j,b_resamp]/calibs['standard'].getData(tag="resampled")[0,b_resamp]

            if (doFitsTable):
                columns.append(pyfits.Column(name='Spectrum_'+str(j+1), format='D', array=rssdata[j,:]))

        #Update data
        if (fdu.hasProperty("cleanFrame")):
            fdu.tagDataAs("cleanFrame", rssclean)
        if (fdu.hasProperty("resampled")):
            fdu.tagDataAs("resampled", rssresamp)
        if (doFitsTable):
            tbhdu = createFitsTable(columns) #Use fatboyLibs wrapper
            fdu.setProperty("csdTable", tbhdu)
        fdu.updateData(rssdata)
        return True
    #end calibStarDivide

    def resampleStandard(self, calib, cswave, wave, tag=None):
        #Transform to common wavelength scale
        calibData = calib.getData()
        if (tag is not None):
            calibData = calib.getData(tag=tag)
        xlo = max(cswave.min(), wave.min())
        xhi = min(cswave.max(), wave.max())
        #Only include wavelengths were spectrum has data
        b = where((wave >= xlo)*(wave <= xhi))[0]
        #Resample calibration star at these wavelengths
        ystar = []
        valid = ones(len(b), bool)
        for i in range(len(b)):
            #Find calib star datapoint closest in wavelength to this
            #datapoint in spectrum
            ref = where(abs(cswave-wave[b][i]) == min(abs(cswave-wave[b][i])))[0][0]
            if (cswave[ref] == wave[b][i]):
                #Special case, exact same wavelength
                ystar.append(calibData[0, ref])
                continue
            elif (cswave[ref] > wave[b][i]):
                #wavelength in calib star > wavelength in spectrum.
                #Use previous value for interpolation
                ref2 = ref-1
            else:
                #wavelength in calib star < wavelength in spectrum.
                #Use next value for interpolation
                ref2 = ref+1
            #Linearly interpolate to get new value for calib star at this
            #wavelength
            if (ref2 < 0 or ref2 >= len(cswave)):
                print("calibStarDivideProcess::resampleStandard> Warning: wavelength "+str(wave[b][i])+" out of range for standard star "+calib.getFullId()+".  Ignoring.")
                self._log.writeLog(__name__, "wavelength "+str(wave[b][i])+" out of range for standard star "+calib.getFullId()+".  Ignoring.", fatboyLog.WARNING)
                valid[i] = False
                continue
            w1 = abs(cswave[ref]-wave[b][i])
            w2 = abs(cswave[ref2]-wave[b][i])
            ystar.append((w2*calibData[0, ref]+w1*calibData[0, ref2])/(w1+w2))
        ystar = array(ystar,dtype=float32)
        #Normalize to 1
        ystar /= gpu_arraymedian(ystar, nonzero=True)
        b = b[valid] #update b to throw out any indices not used
        good = (ystar != 0)
        return (ystar, b, good)
    #end resampleStandard

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/calibStarDivided", os.F_OK)):
            os.mkdir(outdir+"/calibStarDivided",0o755)
        #Create output filename
        csdfile = outdir+"/calibStarDivided/csd_"+fdu.getFullId()
        #Check to see if it exists
        if (os.access(csdfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(csdfile)
        if (not os.access(csdfile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(csdfile, headerExt=fdu.getProperty("wcHeader"))
        #Write out clean frame if it exists
        if (fdu.hasProperty("cleanFrame")):
            cleanfile = outdir+"/calibStarDivided/clean_csd_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(cleanfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(cleanfile)
            if (not os.access(cleanfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.writeTo(cleanfile, tag="cleanFrame", headerExt=fdu.getProperty("wcHeader"))
        #Write out resampled data if it exists
        if (fdu.hasProperty("resampled")):
            resampfile = outdir+"/calibStarDivided/resamp_csd_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(resampfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(resampfile)
            if (not os.access(resampfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                #Write with resampHeader as header extension
                fdu.writeTo(resampfile, tag="resampled", headerExt=fdu.getProperty("resampledHeader"))
        #Write out FITS table if it exists
        if (fdu.hasProperty("csdTable")):
            tabfile = outdir+"/calibStarDivided/csd_table_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(tabfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(tabfile)
            if (not os.access(tabfile, os.F_OK)):
                fdu.getProperty("csdTable").verify('silentfix')
                fdu.getProperty("csdTable").writeto(tabfile, output_verify="silentfix")
    #end writeOutput
