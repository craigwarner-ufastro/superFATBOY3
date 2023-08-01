from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import fatboyLog
from numpy import *
import os, time

class miradasCombineSlicesProcess(fatboyProcess):
    """As part of this, we will want to correct for relative slice illumination by collapsing each of the
       3 slices vertically into a total flux value for each to determine weighting values w.
       We will then combine the three 1-d slices into a 1-d spectrum.  The equation would be:
       (w_0*F_0/sigma_0^2 + w_1*F_1/sigma_1^2 + w_2*F_2/sigma_2^2) / (w_0/sigma_0^2 + w_1/sigma_1^2 + w_2/sigma_2^2)

       Above is for normalize_flux weighting; none = simple sum, flux_weighted is inverse weights with above
       formula, noisemap_only has w_n == 1
    """
    _modeTags = ["miradas"]

    #Actually do the work of combining slices
    def combineSlices(self, fdu, calibs):
        slices_per_slitlet = int(self.getOption("slices_per_slitlet", fdu.getTag()))
        weight_mode = self.getOption("slice_weighting", fdu.getTag())
        rows = fdu.getShape()[0]
        xsize = fdu.getShape()[1]
        if (slices_per_slitlet > 0):
            nslits = rows//slices_per_slitlet
            islits = arange(rows, dtype=int32)//slices_per_slitlet+1
        else:
            #Option not set, next check for slitlet_list
            if ('slitletList' in calibs):
                nslits = max(calibs['slitletList'])
                islits = array(calibs['slitletList']).astype(int32)
            else:
                #Check header info for SPEC_xx and find max slitlet
                key = "SPEC_"
                if (rows < 10):
                    key += "0"
                key += str(rows)
                if (fdu.hasHeaderValue(key)):
                    value = fdu.getHeaderValue(key)
                    #Format = 'Slitlet n: [x1:x2]' - we want n
                    nslits = int(value.split()[1].replace(':',''))
                    islits = []
                    for j in range(rows):
                        key = "SPEC_"
                        if (j+1 < 10):
                            key += "0"
                        key += str(j+1)
                        if (fdu.hasHeaderValue(key)):
                            value = fdu.getHeaderValue(key)
                            #Format = 'Slitlet n: [x1:x2]' - we want n
                            islits.append(int(value.split()[1].replace(':','')))
                        else:
                            islits.append(0) #Should not happen, missing a header keyword
                    islits = array(islits).astype(int32)
                else:
                    nslits = rows//3
                    islits = arange(rows, dtype=int32)//3+1
                    print("miradasCombineSlicesProcess::combineSlices> Warning: could not find SPEC_nn keywords in header and no slitlet_list or slices_per_slitlet given.  Assuming 3 slices per slitlet...")
                    self._log.writeLog(__name__, "could not find SPEC_nn keywords in header and no slitlet_list or slices_per_slitlet given.  Assuming 3 slices per slitlet...", type=fatboyLog.WARNING)
        #Get noisemap
        doNM = True
        if ('noisemap' in calibs):
            nmData = calibs['noisemap'].getData()
        elif (fdu.hasProperty('noisemap')):
            nmData = fdu.getData(tag="noisemap")
        else:
            print("miradasCombineSlicesProcess::combineSlices> Warning: could not find noisemap for "+fdu.getFullId())
            self._log.writeLog(__name__, "could not find noisemap for "+fdu.getFullId(), type=fatboyLog.WARNING)
            doNM = False

        #Remove header values for SPEC_nn
        for j in range(rows):
            key = "SPEC_"
            if (j+1 < 10):
                key += "0"
            key += str(j+1)
            if (fdu.hasHeaderValue(key)):
                fdu.removeHeaderKeyword(key)

        #Create new header dict
        csHeader = dict()

        combined_slices = zeros((nslits, xsize), dtype=float32)
        if (fdu.hasProperty("cleanFrame")):
            combined_clean = zeros((nslits, xsize), dtype=float32)
        if (fdu.hasProperty("noisemap")):
            combined_nm = zeros((nslits, xsize), dtype=float32)

        #Loop over slitlets and combine them
        for j in range(nslits):
            slices = fdu.getData()[islits == j+1]
            if (doNM):
                nmSlices = nmData[islits == j+1]
            if (fdu.hasProperty("cleanFrame")):
                cleanSlices = fdu.getData(tag="cleanFrame")[islits == j+1]
            nslices = slices.shape[0]

            #Update header
            key = 'SPEC_'
            if (j+1 < 10):
                key += '0'
            key += str(j+1)
            csHeader[key] = "Slitlet "+str(j+1)+": ["+str(nslices)+" slices]"

            #Get total flux of each spectrum
            #Don't include negative values
            nzslices = slices.copy()
            nzslices[nzslices < 0] = 0
            fluxTotals = nzslices.sum(1)
            if (weight_mode == "normalize_flux"):
                weights = fluxTotals[0] / fluxTotals
            elif (weight_mode == "flux_weighting"):
                weights = fluxTotals / fluxTotals[0]
            else:
                weights = ones(fluxTotals.size)
            if (fdu.hasProperty("cleanFrame")):
                fluxTotals = cleanSlices.sum(1)
                cleanWeights = fluxTotals[0] / fluxTotals
            print("miradasCombineSlices::combineSlices> Slit "+str(j+1)+": combining "+str(nslices)+" slices with method "+str(weight_mode)+" and weights: "+str(weights))
            self._log.writeLog(__name__, "Slit "+str(j+1)+": combining "+str(nslices)+" slices with method "+str(weight_mode)+" and weights: "+str(weights))

            #setup numerator and denominator arrays
            num = zeros(xsize, dtype=float32)
            den = zeros(xsize, dtype=float32)
            if (fdu.hasProperty("cleanFrame")):
                cleanNum = zeros(xsize, dtype=float32)
                cleanDen = zeros(xsize, dtype=float32)
            if (doNM):
                nmNum = zeros(xsize, dtype=float32)
                nmDen = zeros(xsize, dtype=float32)
            for i in range(nslices):
                if (doNM):
                    #Handle areas where noisemap is 0
                    b = (nmSlices[i] == 0)
                    nmSlices[i][b] = 1
                    if (weight_mode == "none"):
                        currNum = slices[i]
                        currDen = ones(slices[i].shape)
                    else:
                        currNum = weights[i]*slices[i]/nmSlices[i]**2
                        currDen = weights[i]/nmSlices[i]**2
                    currNum[b] = 0
                    currDen[b] = 0
                    num += currNum
                    den += currDen
                    if (fdu.hasProperty("cleanFrame")):
                        if (weight_mode == "none"):
                            currNum = cleanSlices[i]
                            currDen = ones(slices[i].shape)
                        else:
                            currNum = cleanWeights[i]*cleanSlices[i]/nmSlices[i]**2
                            currDen = cleanWeights[i]/nmSlices[i]**2
                        currNum[b] = 0
                        currDen[b] = 0
                        cleanNum += currNum
                        cleanDen += currDen
                    nmSlices[i][b] = 0
                    nmNum += (weights[i]*nmSlices[i])**2
                    nmDen += (weights[i])**2
                else:
                    num += weights[i]*slices[i]
                    den += weights[i]
                    if (fdu.hasProperty("cleanFrame")):
                        cleanNum += cleanWeights[i]*cleanSlices[i]
                        cleanDen += cleanWeights[i]

            #Divide to produce final weighted combined slices
            #Handle zeros
            b = (den == 0)
            den[b] = 1
            combined_slices[j] = num/den
            combined_slices[j][b] = 0
            if (doNM):
                combined_nm[j] = sqrt(nmNum/nmDen) #sqrt of sum of squares
            if (fdu.hasProperty("cleanFrame")):
                b = (cleanDen == 0)
                cleanDen[b] = 1
                combined_clean[j] = cleanNum/cleanDen
                combined_clean[j][b] = 0

        #Update header
        fdu.updateHeader(csHeader)
        if (fdu.hasProperty("cleanFrame")):
            fdu.tagDataAs("clean_combined_slices", combined_clean)
        if (doNM):
            fdu.tagDataAs("noisemap_combined_slices", combined_nm)
        return combined_slices
    #end combineSlices

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("MIRADAS: combine slices")
        print(fdu._identFull)

        #Check if output exists first and update from disk
        csfile = "combinedSlices/cs_"+fdu.getFullId()
        if (self.checkOutputExists(fdu, csfile, headerTag="wcHeader")):
            #Also check if "noisemap" exists
            nmfile = "combinedSlices/NM_cs_"+fdu.getFullId()
            self.checkOutputExists(fdu, nmfile, tag="noisemap")
            #Also check if "clean frame" exists
            cleanfile = "combinedSlices/clean_cs_"+fdu.getFullId()
            self.checkOutputExists(fdu, cleanfile, tag="cleanFrame")
            fdu.setProperty("extracted", True)
            return True

        #Call get calibs to return dict() of calibration frames.
        #For combinedSlices, this dict can have a noisemap if this is not a property
        #of the FDU at this point.  It may also have an text file listing the
        #slitlet indices of each row in the RSS file if this is not in the header already.
        calibs = self.getCalibs(fdu, prevProc)

        #call combineSlices helper function to do actual combining
        combined_slices = self.combineSlices(fdu, calibs)
        fdu.tagDataAs("combined_slices", combined_slices)
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
                print("miradasCombineSlicesProcess::getCalibs> Using noisemap "+nmfilename+"...")
                self._log.writeLog(__name__, "Using noisenap "+nmfilename+"...")
                calibs['noisemap'] = fatboySpecCalib(self._pname, "noisemap", fdu, filename=nmfilename, log=self._log)
            else:
                print("miradasCombineSlicesProcess::getCalibs> Warning: Could not find noisemap "+nmfilename+"...")
                self._log.writeLog(__name__, "Could not find noisemap "+nmfilename+"...", type=fatboyLog.WARNING)

        #Look for a slitlet_list passed as a calib
        slitletList = self.getCalib("slitlet_list", fdu.getTag())
        if (slitletList is not None):
            #passed from XML with <calib> tag.
            if (isinstance(slitletList, list) or isinstance(slitletList, ndarray)):
                #Passed as list or array
                print("miradasCombineSlicesProcess::getCalibs> Using slitlet list: "+str(slitletList))
                self._log.writeLog(__name__, "Using slitlet list: "+str(slitletList))
                calibs['slitletList'] = array(slitletList)
            elif (os.access(slitletList, os.F_OK)):
                #Passed as a filename
                print("miradasCombineSlicesProcess::getCalibs> Using slitlet indices from "+slitletList+"...")
                self._log.writeLog(__name__, "Using slitlet indices from "+slitletList+"...")
                calibs['slitletList'] = loadtxt(slitletList)
            else:
                print("miradasCombineSlicesProcess::getCalibs> Warning: Could not find slitlet_list "+str(slitletList)+"...")
                self._log.writeLog(__name__, "Could not find slitlet_list "+str(slitletList)+"...")

        return calibs
    #end getCalibs

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('slices_per_slitlet', '0')
        self._optioninfo.setdefault('slices_per_slitlet', 'Set this to a nonzero integer to specify number of slices\nper slitlet. If 0, it will use info in header or\na slitlet_list passed as calib.')
        self._options.setdefault('slice_weighting', 'none')
        self._optioninfo.setdefault('slice_weighting', 'none | normalize_flux | flux_weighted | noisemap_only')
        self._options.setdefault('write_noisemaps', 'no')
    #end setDefaultOptions

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/combinedSlices", os.F_OK)):
            os.mkdir(outdir+"/combinedSlices",0o755)
        #Create output filename
        csfile = outdir+"/combinedSlices/cs_"+fdu.getFullId()
        #Check to see if it exists
        if (os.access(csfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(csfile)
        if (not os.access(csfile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            ##*** Note that combined slices are stored in a property, as main RSS data will be reused in other processes **##
            fdu.writeTo(csfile, tag="combined_slices", headerExt=fdu.getProperty("wcHeader"))
        #Write noisemap for spectrocsopy data if requested
        if (self.getOption("write_noisemaps", fdu.getTag()).lower() == "yes" and fdu.hasProperty("noisemap")):
            nmfile = outdir+"/combinedSlices/NM_cs_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(nmfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(nmfile)
            if (not os.access(nmfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.writeTo(nmfile, tag="noisemap_combined_slices", headerExt=fdu.getProperty("wcHeader"))
        #Write out clean frame if it exists
        if (fdu.hasProperty("cleanFrame")):
            cleanfile = outdir+"/combinedSlices/clean_cs_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(cleanfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(cleanfile)
            if (not os.access(cleanfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.writeTo(cleanfile, tag="clean_combined_slices", headerExt=fdu.getProperty("wcHeader"))
    #end writeOutput
