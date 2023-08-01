from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import fatboyLog
from numpy import *
import os, time

class miradasStitchOrdersProcess(fatboyProcess):
    """For SOL and SOS modes, we will now stitch together the individual orders to produce one spectrum.
       As part of this, we will want to correct for relative slitlet illumination by collapsing each of the
       3 slices vertically into a total flux value for each to determine weighting values w.
       We will then combine the three 1-d slices into a 1-d spectrum.  The equation would be:
       (w_0*F_0/sigma_0^2 + w_1*F_1/sigma_1^2 + w_2*F_2/sigma_2^2) / (w_0/sigma_0^2 + w_1/sigma_1^2 + w_2/sigma_2^2)

       Above is for normalize_flux weighting; none = simple sum, flux_weighted is inverse weights with above
       formula, noisemap_only has w_n == 1
    """
    _modeTags = ["miradas"]

    #Actually do the work of stitching orders
    def stitchOrders(self, fdu, calibs):
        weight_mode = self.getOption("order_weighting", fdu.getTag())
        combined_slices = fdu.getData(tag="combined_slices")
        rows = combined_slices.shape[0]
        nslits = rows
        xsize = combined_slices.shape[1]

        if ('slitletList' in calibs):
            nslits = max(calibs['slitletList'])
            islits = array(calibs['slitletList']).astype(int32)
        else:
            #Start with slitlet 1
            islits = arange(nslits, dtype=int32)+1

        #Get noisemap
        doNM = True
        if ('noisemap' in calibs):
            nm_slices = calibs['noisemap'].getData()
        elif (fdu.hasProperty('noisemap_combined_slices')):
            nm_slices = fdu.getData(tag="noisemap_combined_slices")
        else:
            print("miradasStitchOrdersProcess::stitchOrders> Warning: could not find noisemap for "+fdu.getFullId())
            self._log.writeLog(__name__, "could not find noisemap for "+fdu.getFullId(), type=fatboyLog.WARNING)
            doNM = False

        if (fdu.hasProperty("cleanFrame")):
            clean_slices = fdu.getData("clean_combined_slices")

        #Remove header values for SPEC_nn
        for j in range(rows):
            key = "SPEC_"
            if (j+1 < 10):
                key += "0"
            key += str(j+1)
            if (fdu.hasHeaderValue(key)):
                fdu.removeHeaderKeyword(key)
            wave = getWavelengthSolution(fdu, j, xsize)
            if (j == 0):
                minWave = wave.min()
                maxWave = wave.max()
            else:
                minWave = min(minWave, wave.min())
                maxWave = max(maxWave, wave.max())

        scale = (maxWave-minWave)/(rows*xsize)
        outwave = arange(rows*xsize, dtype=float32)*scale+minWave

        #Create new header dict
        soHeader = dict()
        key = 'STITCHED'
        soHeader[key] = str(nslits)+" orders"

        ##Output size should be 1d with ~xsize * nslits
        ##Need flat field as calib for NORMALxx values

        weights = ones(nslits, dtype=float32)
        if (weight_mode == "normalize_flux" or weight_mode == "flux_weighting"):
            if ('masterFlat' in calibs):
                if (calibs['masterFlat'].hasHeaderValue('NORMAL01')):
                    for j in range(nslits):
                        key = 'NORMAL'
                        if (j+1 < 10):
                            key += '0'
                        key += str(j+1)
                        weights[j] = float(calibs['masterFlat'].getHeaderValue(key))
                    if (weight_mode == "normalize_flux"):
                        weights = weights[0]/weights
                    elif (weight_mode == "flux_weighting"):
                        weights = weights/weights[0]
        print("miradasStitchOrders::stitchOrders> Combining "+str(nslits)+" orders with method "+str(weight_mode)+" and weights: "+str(weights))
        self._log.writeLog(__name__, "Combining "+str(nslits)+" orders with method "+str(weight_mode)+" and weights: "+str(weights))

        #setup numerator and denominator arrays
        num = zeros(nslits*xsize, dtype=float32)
        den = zeros(nslits*xsize, dtype=float32)
        if (fdu.hasProperty("cleanFrame")):
            cleanNum = zeros(nslits*xsize, dtype=float32)
            cleanDen = zeros(nslits*xsize, dtype=float32)
        if (doNM):
            nmNum = zeros(nslits*xsize, dtype=float32)
            nmDen = zeros(nslits*xsize, dtype=float32)

        #Loop over slitlets and combine them
        for j in range(nslits):
            wave = getWavelengthSolution(fdu, j, xsize)
            ##Need to resample, use numpy.interp
            currSlice = interp(outwave, wave, combined_slices[j])
            if (fdu.hasProperty("cleanFrame")):
                currCleanSlice = interp(outwave, wave, clean_slices[j])
            if (doNM):
                currNMslice = interp(outwave, wave, nm_slices[j])
                #Handle areas where noisemap is 0
                b = (currNMslice == 0)
                currNMslice[b] = 1
                if (weight_mode == "none"):
                    currNum = currSlice
                    currDen = ones(currSlice.shape)
                else:
                    currNum = weights[j]*currSlice/currNMslice**2
                    currDen = weights[j]/currNMslice**2
                currNum[b] = 0
                currDen[b] = 0
                num += currNum
                den += currDen
                if (fdu.hasProperty("cleanFrame")):
                    if (weight_mode == "none"):
                        currNum = currCleanSlice
                        currDen = ones(currCleanSlice.shape)
                    else:
                        currNum = weights[j]*currCleanSlice/currNMslice**2
                        currDen = weights[j]/currNMslice**2
                    currNum[b] = 0
                    currDen[b] = 0
                    cleanNum += currNum
                    cleanDen += currDen
                currNMslice[b] = 0
                nmNum += (weights[j]*currNMslice)**2
                nmDen += (weights[j])**2
            else:
                num += weights[j]*currSlice
                den += weights[j]
                if (fdu.hasProperty("cleanFrame")):
                    cleanNum += weights[j]*currCleanSlice
                    cleanDen += weights[j]

        #Divide to produce final weighted combined slices
        #Handle zeros
        b = (den == 0)
        den[b] = 1
        stitched_orders = num/den
        stitched_orders[b] = 0
        if (doNM):
            stitched_nm = sqrt(nmNum/nmDen) #sqrt of sum of squares
        if (fdu.hasProperty("cleanFrame")):
            b = (cleanDen == 0)
            cleanDen[b] = 1
            stitched_clean = cleanNum/cleanDen
            stitched_clean[b] = 0

        #Update wavelength calibration
        keysToRemove = []
        for key in fdu._header:
            if (key.startswith('PORDER') or key.startswith('PCF') or key.startswith('NSEG')):
                keysToRemove.append(key)
        for key in keysToRemove:
            fdu.removeHeaderKeyword(key)
        soHeader['CRVAL1'] = minWave
        soHeader['CDELT1'] = scale

        #Update header
        fdu.updateHeader(soHeader)
        if (fdu.hasProperty("cleanFrame")):
            fdu.tagDataAs("clean_stitched_orders", stitched_clean)
        if (doNM):
            fdu.tagDataAs("noisemap_stitched_orders", stitched_nm)
        return stitched_orders
    #end stitchOrders

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("MIRADAS: combine slices")
        print(fdu._identFull)

        #Check if output exists first and update from disk
        sofile = "stitchedOrders/so_"+fdu.getFullId()
        if (self.checkOutputExists(fdu, sofile, headerTag="wcHeader")):
            #Also check if "noisemap" exists
            nmfile = "stitchedOrders/NM_so_"+fdu.getFullId()
            self.checkOutputExists(fdu, nmfile, tag="noisemap")
            #Also check if "clean frame" exists
            cleanfile = "stitchedOrders/clean_so_"+fdu.getFullId()
            self.checkOutputExists(fdu, cleanfile, tag="cleanFrame")
            fdu.setProperty("extracted", True)
            return True

        #Call get calibs to return dict() of calibration frames.
        #For stitchedOrders, this dict can have a noisemap if this is not a property
        #of the FDU at this point.  It may also have an text file listing the
        #slitlet indices of each row in the RSS file if this is not in the header already.
        calibs = self.getCalibs(fdu, prevProc)

        #call stitchOrders helper function to do actual combining
        stitched_orders = self.stitchOrders(fdu, calibs)
        fdu.tagDataAs("stitched_orders", stitched_orders)
        return True
    #end execute

    ## OVERRIDE getCalibs
    def getCalibs(self, fdu, prevProc = None):
        #No calibrations needed other than noisemap
        calibs = dict()
        #Look for each calib passed from XML
        nmfilename = self.getCalib("noisemap", fdu.getTag())
        if (nmfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(nmfilename, os.F_OK)):
                print("miradasStitchOrdersProcess::getCalibs> Using noisemap "+nmfilename+"...")
                self._log.writeLog(__name__, "Using noisenap "+nmfilename+"...")
                calibs['noisemap'] = fatboySpecCalib(self._pname, "noisemap", fdu, filename=nmfilename, log=self._log)
            else:
                print("miradasStitchOrdersProcess::getCalibs> Warning: Could not find noisemap "+nmfilename+"...")
                self._log.writeLog(__name__, "Could not find noisemap "+nmfilename+"...", type=fatboyLog.WARNING)

        #Look for a slitlet_list passed as a calib
        slitletList = self.getCalib("slitlet_list", fdu.getTag())
        if (slitletList is not None):
            #passed from XML with <calib> tag.
            if (isinstance(slitletList, list) or isinstance(slitletList, ndarray)):
                #Passed as list or array
                print("miradasStitchOrdersProcess::getCalibs> Using slitlet list: "+str(slitletList))
                self._log.writeLog(__name__, "Using slitlet list: "+str(slitletList))
                calibs['slitletList'] = array(slitletList)
            elif (os.access(slitletList, os.F_OK)):
                #Passed as a filename
                print("miradasStitchOrdersProcess::getCalibs> Using slitlet indices from "+slitletList+"...")
                self._log.writeLog(__name__, "Using slitlet indices from "+slitletList+"...")
                calibs['slitletList'] = loadtxt(slitletList)
            else:
                print("miradasStitchOrdersProcess::getCalibs> Warning: Could not find slitlet_list "+str(slitletList)+"...")
                self._log.writeLog(__name__, "Could not find slitlet_list "+str(slitletList)+"...")

        #Look for matching grism_keyword, specmode, and flat_method
        headerVals = dict()
        headerVals['grism_keyword'] = fdu.grism
        properties = dict()
        properties['specmode'] = fdu.getProperty("specmode")

        #First find a master flat frame to also apply bad pixel mask to
        #1) Check for an already created master flat frame matching specmode/filter/grism and TAGGED for this object
        masterFlat = self._fdb.getTaggedMasterCalib(pname=None, ident=fdu._id, obstype="master_flat", filter=fdu.filter, properties=properties, headerVals=headerVals)
        if (masterFlat is not None):
            #Found master flat
            calibs['masterFlat'] = masterFlat
        else:
            #2) Check for an already created master flat frame matching specmode/filter/grism
            masterFlat = self._fdb.getMasterCalib(pname=None, obstype="master_flat", filter=fdu.filter, tag=fdu.getTag(), properties=properties, headerVals=headerVals)
            if (masterFlat is not None):
                #Found master flat.
                calibs['masterFlat'] = masterFlat
            else:
                #3) Look at previous master flats to see if any has a history of being used as master flat for
                #this _id and filter combination from step 7 below.
                masterFlats = self._fdb.getMasterCalibs(obstype="master_flat")
                for mflat in masterFlats:
                    if (mflat.hasHistory('master_flat::'+fdu._id+'::'+str(fdu.filter)+'::'+str(fdu.grism)+'::'+str(fdu.getProperty("specmode")))):
                        #Use this master flat
                        print("miradasStitchOrdersProcess::getCalibs> Using master flat "+mflat.getFilename()+" with filter "+str(mflat.filter)+", grism "+str(mflat.grism)+", specmode "+str(mflat.getProperty("specmode")))
                        self._log.writeLog(__name__, "Using master flat "+mflat.getFilename()+" with filter "+str(mflat.filter)+", grism "+str(mflat.grism)+", specmode "+str(mflat.getProperty("specmode")))
                        #Already in _calibs, no need to appendCalib
                        calibs['masterFlat'] = mflat

        return calibs
    #end getCalibs

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('write_noisemaps', 'no')
        self._options.setdefault('order_weighting', 'none')
        self._optioninfo.setdefault('order_weighting', 'none | normalize_flux | flux_weighted | noisemap_only')
    #end setDefaultOptions

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/stitchedOrders", os.F_OK)):
            os.mkdir(outdir+"/stitchedOrders",0o755)
        #Create output filename
        sofile = outdir+"/stitchedOrders/so_"+fdu.getFullId()
        #Check to see if it exists
        if (os.access(sofile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(sofile)
        if (not os.access(sofile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            ##*** Note that combined slices are stored in a property, as main RSS data will be reused in other processes **##
            fdu.writeTo(sofile, tag="stitched_orders")
        #Write noisemap for spectrocsopy data if requested
        if (self.getOption("write_noisemaps", fdu.getTag()).lower() == "yes" and fdu.hasProperty("noisemap")):
            nmfile = outdir+"/stitchedOrders/NM_so_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(nmfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(nmfile)
            if (not os.access(nmfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.writeTo(nmfile, tag="noisemap_stitched_orders")
        #Write out clean frame if it exists
        if (fdu.hasProperty("cleanFrame")):
            cleanfile = outdir+"/stitchedOrders/clean_so_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(cleanfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(cleanfile)
            if (not os.access(cleanfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.writeTo(cleanfile, tag="clean_stitched_orders")
    #end writeOutput
