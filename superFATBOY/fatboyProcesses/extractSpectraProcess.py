from superFATBOY.fatboyDataUnit import fatboyDataUnit
from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.datatypeExtensions.fatboySpecCalib import fatboySpecCalib

from superFATBOY import gpu_drihizzle, drihizzle
from numpy import *
from scipy.optimize import leastsq

usePlot = True
try:
    import matplotlib.pyplot as plt
except Exception as ex:
    print("Warning: Could not import matplotlib!")
    usePlot = False

block_size = 512

class extractSpectraProcess(fatboyProcess):
    _modeTags = ["spectroscopy", "miradas"]

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("Extract Spectra")
        print(fdu._identFull)

        #Check if output exists first and update from disk
        esfile = "extractedSpectra/es_"+fdu.getFullId()
        if (self.checkOutputExists(fdu, esfile)):#, headerTag="wcHeader")):
            #Also check if "cleanFrame" exists
            cleanfile = "extractedSpectra/clean_es_"+fdu.getFullId()
            self.checkOutputExists(fdu, cleanfile, tag="cleanFrame")
            #Also check if "noisemap" exists
            nmfile = "extractedSpectra/NM_es_"+fdu.getFullId()
            self.checkOutputExists(fdu, nmfile, tag="noisemap")
            #Also check if "resampled" exists
            resampfile = "extractedSpectra/resamp_es_"+fdu.getFullId()
            self.checkOutputExists(fdu, resampfile, tag="resampled")
            fdu.setProperty("extracted", True)
            return True

        #Call get calibs to return dict() of calibration frames.
        #For extractedSpectra, this dict should have slitmask if this is not a property of the FDU at this point.
        #These are obtained by tracing slitlets using the master flat
        calibs = self.getCalibs(fdu, prevProc)

        #call extractSpectra helper function to do gpu/cpu calibration
        success = self.extractSpectra(fdu, calibs)
        if (success):
            fdu.setProperty("extracted", True)
        return success
    #end execute

    ## OVERRIDE getCalibs
    def getCalibs(self, fdu, prevProc = None):
        calibs = dict()

        #Look for each calib passed from XML
        smfilename = self.getCalib("slitmask", fdu.getTag())
        if (smfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(smfilename, os.F_OK)):
                print("extractSpectraProcess::getCalibs> Using slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Using slitmask "+smfilename+"...")
                calibs['slitmask'] = fatboySpecCalib(self._pname, "slitmask", fdu, filename=smfilename, log=self._log)
            else:
                print("extractSpectraProcess::getCalibs> Warning: Could not find slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Could not find slitmask "+smfilename+"...", type=fatboyLog.WARNING)

        #Look for a spec_location_list passed as a calib
        specList = self.getCalib("spec_location_list", fdu.getTag())
        if (specList is not None):
            #passed from XML with <calib> tag.
            if (isinstance(specList, pyfits.hdu.hdulist.HDUList)):
                #Passed as FITS HDUList object
                print("extractSpectraProcess::getCalibs> Using FITS predefined spectral locations: "+str(specList[0].data))
                self._log.writeLog(__name__, "Using FITS predefined spectral locations: "+str(specList[0].data))
                calibs['specList'] = array(specList[0].data)
            elif (isinstance(specList, list) or isinstance(specList, ndarray)):
                #Passed as list or array
                print("extractSpectraProcess::getCalibs> Using predefined spectral locations: "+str(specList))
                self._log.writeLog(__name__, "Using predefined spectral locations: "+str(specList))
                calibs['specList'] = array(specList)
            elif (os.access(specList, os.F_OK)):
                #Passed as a filename
                print("extractSpectraProcess::getCalibs> Using spectral locations from "+specList+"...")
                self._log.writeLog(__name__, "Using spectral locations from "+specList+"...")
                calibs['specList'] = loadtxt(specList)
            else:
                print("extractSpectraProcess::getCalibs> Warning: Could not find spec_location_list "+str(specList)+"...")
                self._log.writeLog(__name__, "Could not find spec_location_list "+str(specList)+"...")

        #Look for continuum source frame to use to trace out spectral locations
        csfilename = self.getCalib("continuum_source", fdu.getTag())
        if (csfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(csfilename, os.F_OK)):
                print("extractSpectraProcess::getCalibs> Using continuum source "+csfilename+" to find spectral locations...")
                self._log.writeLog(__name__, "Using continuum source "+csfilename+" to find spectral locations...")
                calibs['continuum_source'] = fatboySpecCalib(self._pname, "continuum_source", fdu, filename=csfilename, log=self._log)
            else:
                print("extractSpectraProcess::getCalibs> Warning: Could not find continuum source "+csfilename+"...")
                self._log.writeLog(__name__, "Could not find continuum source "+csfilename+"...", type=fatboyLog.WARNING)


        #Look for matching grism_keyword, specmode, and dispersion
        headerVals = dict()
        headerVals['grism_keyword'] = fdu.grism

        properties = dict()
        properties['specmode'] = fdu.getProperty("specmode")
        properties['dispersion'] = fdu.getProperty("dispersion")

        if (fdu._specmode != fdu.FDU_TYPE_LONGSLIT and not 'slitmask' in calibs):
            #Use new fdu.getSlitmask method
            slitmask = fdu.getSlitmask(pname=None, properties=properties, headerVals=headerVals)
            if (slitmask is not None):
                #Found slitmask
                calibs['slitmask'] = slitmask
        if (not 'continuum_source' in calibs):
            #1a) check for an already created continuum source master calib matching specmode/filter/grism and TAGGED for this object
            continuum_source = self._fdb.getTaggedMasterCalib(pname=None, ident=fdu._id, obstype="continuum_source", filter=fdu.filter, shape=fdu.getShape(), properties=properties, headerVals=headerVals)
            if (continuum_source is None):
                #2) Look for a continuum source master calib with matching filter/grism/specmode but NOT ident
                continuum_source = self._fdb.getMasterCalib(filter=fdu.filter, obstype="continuum_source", shape=fdu.getShape(), properties=properties, headerVals=headerVals, tag=fdu.getTag())
            if (continuum_source is not None):
                #Found continuum source
                calibs['continuum_source'] = continuum_source

        if ('specList' in calibs):
            #Speclist is already defined, return here
            return calibs

        if (not 'continuum_source' in calibs):
            #1a) check for continuum source frames matching specmode/filter/grism and TAGGED for this object
            csources = self._fdb.getTaggedCalibs(fdu._id, obstype=fdu.FDU_TYPE_CONTINUUM_SOURCE, filter=fdu.filter, properties=properties, headerVals=headerVals)
            if (len(csources) > 0):
                #Found continuum sources associated with this fdu. Recursively process
                print("extractSpectraProcess::getCalibs> Recursively processing continuum sources for tagged object "+fdu._id+", filter "+str(fdu.filter)+"...")
                self._log.writeLog(__name__, " Recursively processing continuum sources for tagged object "+fdu._id+", filter "+str(fdu.filter)+"...")
                #recursively process
                self.recursivelyExecute(csources, prevProc)
                #All but first FDU should be disabled after they're processed through shift and add
                success = True
                for j in range(1, len(csources)):
                    if (csources[j].inUse):
                        success = False
                        print("extractSpectraProcess::getCalibs> Error: More than one continuum source frame remains after recursively processing...")
                        self._log.writeLog(__name__, "More than one continuum source frame remains after recursively processing...", type=fatboyLog.ERROR)
                if (success):
                    calibs['continuum_source'] = csources[0]
                    self._fdb.appendCalib(csources[0]) #add as a master calib frame
            #2) Check for individual continuum sources matching specmode/filter/grism
            csources = self._fdb.getCalibs(obstype=fdu.FDU_TYPE_CONTINUUM_SOURCE, filter=fdu.filter, tag=fdu.getTag(), properties=properties, headerVals=headerVals)
            if (len(csources) > 0):
                #Found continuum sources associated with this fdu. Recursively process
                print("extractSpectraProcess::getCalibs> Recursively processing continuum sources for object "+fdu._id+", filter "+str(fdu.filter)+"...")
                self._log.writeLog(__name__, " Recursively processing continuum sources for object "+fdu._id+", filter "+str(fdu.filter)+"...")
                #recursively process
                self.recursivelyExecute(csources, prevProc)
                #All but first FDU should be disabled after they're processed through shift and add
                success = True
                for j in range(1, len(csources)):
                    if (csources[j].inUse):
                        success = False
                        print("extractSpectraProcess::getCalibs> Error: More than one continuum source frame remains after recursively processing...")
                        self._log.writeLog(__name__, "More than one continuum source frame remains after recursively processing...", type=fatboyLog.ERROR)
                if (success):
                    calibs['continuum_source'] = csources[0]
                    self._fdb.appendCalib(csources[0]) #add as a master calib frame

        #Use fdu itself or continuum source if specified to find spectral locations
        calibs['specList'] = array(self.findSpectra(fdu, calibs))

        return calibs
    #end getCalibs

    #read an XML style region file
    def readExtractMethodFile(self, esfile, nslits, fdu):
        #Set defaults
        def_extract_method = "auto"
        def_extract_sigma = float(self.getOption("extract_sigma", fdu.getTag()))
        def_extract_min_width = int(self.getOption("extract_min_width", fdu.getTag()))
        def_extract_nspec = int(self.getOption("extract_nspec", fdu.getTag()))
        def_extract_xlo = self.getOption("extract_xlo", fdu.getTag())
        def_extract_xhi = self.getOption("extract_xhi", fdu.getTag())
        def_extract_ylo = self.getOption("extract_ylo", fdu.getTag())
        def_extract_yhi = self.getOption("extract_yhi", fdu.getTag())

        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            xsize = fdu.getShape()[1]
            ysize = fdu.getShape()[0]
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            ##xsize should be size across dispersion direction
            xsize = fdu.getShape()[0]
            ysize = fdu.getShape()[1]

        #Set defaults for xlo, xhi, ylo, yhi
        if (def_extract_xlo is None):
            def_extract_xlo = 0
        else:
            def_extract_xlo = int(def_extract_xlo)
        if (def_extract_xhi is None):
            def_extract_xhi = xsize
        else:
            def_extract_xhi = int(def_extract_xhi)
        if (def_extract_ylo is None):
            def_extract_ylo = 0
        else:
            def_extract_ylo = int(def_extract_ylo)
        if (def_extract_yhi is None):
            def_extract_yhi = ysize
        else:
            def_extract_yhi = int(def_extract_yhi)

        esinfo = dict()
        #doc = xml config file
        try:
            doc = xml.dom.minidom.parse(esfile)
        except Exception as ex:
            print("extractSpectraProcess::readExtractMethodFile> Error parsing XML config file "+esfile+": "+str(ex))
            self._log.writeLog(__name__, "Error parsing XML config file "+esfile+": "+str(ex), type=fatboyLog.ERROR)
            return None
        #get top level dataset node (should only be 1)
        datasetNodes = doc.getElementsByTagName('dataset')
        for node in datasetNodes:
            if (node.hasAttribute("extract_method")):
                def_extract_method = node.getAttribute("extract_method")
            if (node.hasAttribute("extract_sigma")):
                try:
                    def_extract_sigma = float(node.getAttribute("extract_sigma"))
                except Exception as ex:
                    print("extractSpectraProcess::readExtractMethodFile> Warning: misformatted line in "+esfile+": error parsing fit_order.")
                    self._log.writeLog(__name__, "misformatted line in "+esfile+": error parsing extract_spectra.", type=fatboyLog.WARNING)
                    def_extract_sigma = 2
            if (node.hasAttribute("extract_min_width")):
                try:
                    def_extract_min_width = int(node.getAttribute("extract_min_width"))
                except Exception as ex:
                    print("extractSpectraProcess::readExtractMethodFile> Warning: misformatted line in "+esfile+": error parsing extract_min_width.")
                    self._log.writeLog(__name__, "misformatted line in "+esfile+": error parsing extract_min_width.", type=fatboyLog.WARNING)
                    def_extract_min_width = 5
            if (node.hasAttribute("extract_nspec")):
                try:
                    def_extract_nspec = int(node.getAttribute("extract_nspec"))
                except Exception as ex:
                    print("extractSpectraProcess::readExtractMethodFile> Warning: misformatted line in "+esfile+": error parsing extract_nspec.")
                    self._log.writeLog(__name__, "misformatted line in "+esfile+": error parsing extract_nspec.", type=fatboyLog.WARNING)
                    def_extract_nspec = 1
            if (node.hasAttribute("extract_xlo")):
                try:
                    def_extract_xlo = int(node.getAttribute("extract_xlo"))
                except Exception as ex:
                    print("extractSpectraProcess::readExtractMethodFile> Warning: misformatted line in "+esfile+": error parsing extract_xlo.")
                    self._log.writeLog(__name__, "misformatted line in "+esfile+": error parsing extract_xlo.", type=fatboyLog.WARNING)
                    def_extract_xlo = 0
            if (node.hasAttribute("extract_xhi")):
                try:
                    def_extract_xhi = int(node.getAttribute("extract_xhi"))
                except Exception as ex:
                    print("extractSpectraProcess::readExtractMethodFile> Warning: misformatted line in "+esfile+": error parsing extract_xhi.")
                    self._log.writeLog(__name__, "misformatted line in "+esfile+": error parsing extract_xhi.", type=fatboyLog.WARNING)
                    def_extract_xhi = xsize
            if (node.hasAttribute("extract_ylo")):
                try:
                    #Special case for manual extraction, allow comma separated list.  Taken care of by nspec for auto and semi
                    if (def_extract_method == "manual" and node.getAttribute("extract_ylo").count(",") > 0):
                        def_extract_ylo = node.getAttribute("extract_ylo").split(",")
                        for j in range(len(def_extract_ylo)):
                            def_extract_ylo[j] = int(def_extract_ylo[j])
                    else:
                        def_extract_ylo = int(node.getAttribute("extract_ylo"))
                except Exception as ex:
                    print("extractSpectraProcess::readExtractMethodFile> Warning: misformatted line in "+esfile+": error parsing extract_ylo.")
                    self._log.writeLog(__name__, "misformatted line in "+esfile+": error parsing extract_ylo.", type=fatboyLog.WARNING)
                    def_extract_ylo = 0
            if (node.hasAttribute("extract_yhi")):
                try:
                    #Special case for manual extraction, allow comma separated list.  Taken care of by nspec for auto and semi
                    if (def_extract_method == "manual" and node.getAttribute("extract_yhi").count(",") > 0):
                        def_extract_yhi = node.getAttribute("extract_yhi").split(",")
                        for j in range(len(def_extract_yhi)):
                            def_extract_yhi[j] = int(def_extract_yhi[j])
                    else:
                        def_extract_yhi = int(node.getAttribute("extract_yhi"))
                except Exception as ex:
                    print("extractSpectraProcess::readExtractMethodFile> Warning: misformatted line in "+esfile+": error parsing extract_yhi.")
                    self._log.writeLog(__name__, "misformatted line in "+esfile+": error parsing extract_yhi.", type=fatboyLog.WARNING)
                    def_extract_yhi = ysize

            #Now create list entries nslits long in esinfo dict
            esinfo['extract_method'] = [def_extract_method]*nslits
            esinfo['extract_sigma'] = [def_extract_sigma]*nslits
            esinfo['extract_min_width'] = [def_extract_min_width]*nslits
            esinfo['extract_nspec'] = [def_extract_nspec]*nslits
            esinfo['extract_xlo'] = [def_extract_xlo]*nslits
            esinfo['extract_xhi'] = [def_extract_xhi]*nslits
            esinfo['extract_ylo'] = [def_extract_ylo]*nslits
            esinfo['extract_yhi'] = [def_extract_yhi]*nslits

            #Now loop over child nodes
            if (not node.hasChildNodes()):
                break
            islit = 0
            for orderNode in node.childNodes:
                #Ignore non element nodes and nodes not named <order>
                if (orderNode.nodeType != Node.ELEMENT_NODE):
                    continue
                if (orderNode.nodeName != 'order'):
                    continue
                #This is an order node look for slitlet attribute
                if (orderNode.hasAttribute("slitlet")):
                    #This node has a slitlet attribute.  Use this as the index.
                    #Subtract 1 from index to make it zero-ordered
                    try:
                        idx = int(orderNode.getAttribute("slitlet"))-1
                    except Exception as ex:
                        print("extractSpectraProcess::readExtractMethodFile> Warning: misformatted line in "+esfile+": error parsing slitlet index")
                        self._log.writeLog(__name__, "misformatted line in "+esfile+": error parsing slitlet index", type=fatboyLog.WARNING)
                        idx = islit
                else:
                    #Use islit as index, assume orders are in order
                    idx = islit
                if (idx >= nslits):
                    print("extractSpectraProcess::readExtractMethodFile> Warning: index "+str(idx+1)+" is greater than the number of slitlets! Skipping this line!")
                    self._log.writeLog(__name__, "index "+str(idx+1)+" is greater than the number of slitlets! Skipping this line!", type=fatboyLog.WARNING)
                    islit += 1
                    continue
                #Now parse options
                if (orderNode.hasAttribute("extract_method")):
                    esinfo['extract_method'][idx] = orderNode.getAttribute("extract_method")
                if (orderNode.hasAttribute("extract_sigma")):
                    try:
                        esinfo['extract_sigma'][idx] = float(orderNode.getAttribute("extract_sigma"))
                    except Exception as ex:
                        print("extractSpectraProcess::readExtractMethodFile> Warning: misformatted line in "+esfile+": error parsing extract_sigma for slitlet "+str(idx+1))
                        self._log.writeLog(__name__, "misformatted line in "+esfile+": error parsing extract_sigma for slitlet "+str(idx+1), type=fatboyLog.WARNING)
                if (orderNode.hasAttribute("extract_min_width")):
                    try:
                        esinfo['extract_min_width'][idx] = int(orderNode.getAttribute("extract_min_width"))
                    except Exception as ex:
                        print("extractSpectraProcess::readExtractMethodFile> Warning: misformatted line in "+esfile+": error parsing extract_min_width for slitlet "+str(idx+1))
                        self._log.writeLog(__name__, "misformatted line in "+esfile+": error parsing extract_min_width for slitlet "+str(idx+1), type=fatboyLog.WARNING)
                if (orderNode.hasAttribute("extract_nspec")):
                    try:
                        esinfo['extract_nspec'][idx] = int(orderNode.getAttribute("extract_nspec"))
                    except Exception as ex:
                        print("extractSpectraProcess::readExtractMethodFile> Warning: misformatted line in "+esfile+": error parsing extract_nspec for slitlet "+str(idx+1))
                        self._log.writeLog(__name__, "misformatted line in "+esfile+": error parsing extract_nspec for slitlet "+str(idx+1), type=fatboyLog.WARNING)
                if (orderNode.hasAttribute("extract_xlo")):
                    try:
                        esinfo['extract_xlo'][idx] = int(orderNode.getAttribute("extract_xlo"))
                    except Exception as ex:
                        print("extractSpectraProcess::readExtractMethodFile> Warning: misformatted line in "+esfile+": error parsing extract_xlo for slitlet "+str(idx+1))
                        self._log.writeLog(__name__, "misformatted line in "+esfile+": error parsing extract_xlo for slitlet "+str(idx+1), type=fatboyLog.WARNING)
                if (orderNode.hasAttribute("extract_xhi")):
                    try:
                        esinfo['extract_xhi'][idx] = int(orderNode.getAttribute("extract_xhi"))
                    except Exception as ex:
                        print("extractSpectraProcess::readExtractMethodFile> Warning: misformatted line in "+esfile+": error parsing extract_xhi for slitlet "+str(idx+1))
                        self._log.writeLog(__name__, "misformatted line in "+esfile+": error parsing extract_xhi for slitlet "+str(idx+1), type=fatboyLog.WARNING)
                if (orderNode.hasAttribute("extract_ylo")):
                    try:
                        #Special case for manual extraction, allow comma separated list.  Taken care of by nspec for auto and semi
                        if (esinfo['extract_method'][idx] == "manual" and orderNode.getAttribute("extract_ylo").count(",") > 0):
                            esinfo['extract_ylo'][idx] = orderNode.getAttribute("extract_ylo").split(",")
                            for j in range(len(esinfo['extract_ylo'][idx])):
                                esinfo['extract_ylo'][idx][j] = int(esinfo['extract_ylo'][idx][j])
                        else:
                            esinfo['extract_ylo'][idx] = int(orderNode.getAttribute("extract_ylo"))
                    except Exception as ex:
                        print("extractSpectraProcess::readExtractMethodFile> Warning: misformatted line in "+esfile+": error parsing extract_ylo for slitlet "+str(idx+1))
                        self._log.writeLog(__name__, "misformatted line in "+esfile+": error parsing extract_ylo for slitlet "+str(idx+1), type=fatboyLog.WARNING)
                if (orderNode.hasAttribute("extract_yhi")):
                    try:
                        esinfo['extract_yhi'][idx] = int(orderNode.getAttribute("extract_yhi"))
                    except Exception as ex:
                        print("extractSpectraProcess::readExtractMethodFile> Warning: misformatted line in "+esfile+": error parsing extract_yhi for slitlet "+str(idx+1))
                        self._log.writeLog(__name__, "misformatted line in "+esfile+": error parsing extract_yhi for slitlet "+str(idx+1), type=fatboyLog.WARNING)
                #Update islit
                islit += 1
            #There should be only one <dataset> tag so break here
            break
        return esinfo
    #end readExtractMethodFile

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('debug_mode', 'no')
        self._optioninfo.setdefault('debug_mode', 'Show plots of each slitlet and print out debugging information.')
        self._options.setdefault('extract_method', 'auto')
        self._optioninfo.setdefault('extract_method', 'auto | full | manual | semi | filename.xml')

        self._options.setdefault('extract_sigma', '2')
        self._optioninfo.setdefault('extract_sigma', 'Minimum sigma threshold above background in 1-d cut')
        self._options.setdefault('extract_min_width', '5')
        self._optioninfo.setdefault('extract_min_width', 'Minimum width to be defined as a spectrum')
        self._options.setdefault('extract_nspec', '1')
        self._optioninfo.setdefault('extract_nspec', 'Maximum number of spectra per slitlet to extract')
        self._options.setdefault('extract_min_flux_pct', '0.001')
        self._optioninfo.setdefault('extract_min_flux_pct', 'If the flux dips below this percent of the peak flux\nthen it will be considered a break between continua\nwhen auto-detecting.')
        self._options.setdefault('extract_xlo', None)
        self._optioninfo.setdefault('extract_xlo', 'Coordinate for extraction box for 1-d cut to auto-detect')
        self._options.setdefault('extract_xhi', None)
        self._optioninfo.setdefault('extract_xhi', 'Coordinate for extraction box for 1-d cut to auto-detect')
        self._options.setdefault('extract_ylo', None)
        self._optioninfo.setdefault('extract_ylo', 'Coordinate for extraction box for 1-d cut to auto-detect')
        self._options.setdefault('extract_yhi', None)
        self._optioninfo.setdefault('extract_yhi', 'Coordinate for extraction box for 1-d cut to auto-detect')
        self._options.setdefault('extract_gauss_width', None)
        self._optioninfo.setdefault('extract_gauss_width', 'Width in sigma of the extraction box based on\nGaussian fit to 1-d cut (default 3)')

        self._options.setdefault('extract_weighting', 'linear')
        self._optioninfo.setdefault('extract_weighting', 'linear | gaussian | median')
        self._options.setdefault('gaussian_box_size', '25')
        self._options.setdefault('write_fits_table', 'no')
        self._options.setdefault('write_noisemaps', 'no')
        self._options.setdefault('write_plots', 'no')
    #end setDefaultOptions

    ## Extract spectra
    def extractSpectra(self, fdu, calibs):
        ###*** For purposes of extractSpectra algorithm, X = dispersion direction and Y = cross-dispersion direction ***###
        #Read options
        extract_weighting = self.getOption("extract_weighting", fdu.getTag()).lower()
        extract_nspec = int(self.getOption("extract_nspec", fdu.getTag()))
        gaussbox = int(self.getOption("gaussian_box_size", fdu.getTag()))
        doFitsTable = False
        if (self.getOption("write_fits_table", fdu.getTag()).lower() == "yes"):
            doFitsTable = True

        debug = False
        if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
            debug = True
        writePlots = False
        if (self.getOption("write_plots", fdu.getTag()).lower() == "yes"):
            writePlots = True

        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            xsize = fdu.getShape()[1]
            ysize = fdu.getShape()[0]
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            ##xsize should be size across dispersion direction
            xsize = fdu.getShape()[0]
            ysize = fdu.getShape()[1]

        specList = calibs['specList']
        properties = dict()
        properties['specmode'] = fdu.getProperty("specmode")
        properties['dispersion'] = fdu.getProperty("dispersion")

        if (len(specList) == 0):
            #Could not find spectrum
            print(("extractSpectraProcess::extractSpectra> ERROR: Could not find any spectra in "+fdu.getFullId()+"!  Discarding this frame!"))
            self._log.writeLog(__name__, "Could not find any spectra in "+fdu.getFullId()+"!  Discarding this frame!", type=fatboyLog.ERROR)
            fdu.disable()
            return False

        #Create output dir if it doesn't exist
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/extractedSpectra", os.F_OK)):
            os.mkdir(outdir+"/extractedSpectra",0o755)

        #Create new header dict
        esHeader = dict()

        slitmask = None
        if (fdu._specmode != fdu.FDU_TYPE_LONGSLIT and (fdu.hasProperty("slitmask") or "slitmask" in calibs)):
            ###MOS/IFU data -- get slitmask
            #Use FDU "slitmask" property, which has been shifted and added, NOT slitmask corresponding to sky/lamp
            #Use new fdu.getSlitmask method
            slitmask = fdu.getSlitmask(pname=None, properties=properties)

        #Now create output row stacked spectrum data and loop over specList
        nspec = len(specList)
        #Shape will now be the same regardless of orientation
        rssdata = zeros((nspec, xsize), dtype=float32)
        if (fdu.hasProperty("cleanFrame")):
            rssclean = zeros((nspec, xsize), dtype=float32)
        if (fdu.hasProperty("noisemap")):
            rssnm = zeros((nspec, xsize), dtype=float32)
        if (fdu.hasProperty("resampled")):
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                resampxsize = fdu.getData(tag="resampled").shape[1]
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                ##xsize should be size across dispersion direction
                resampxsize = fdu.getData(tag="resampled").shape[0]
            rssresamp = zeros((nspec, resampxsize), dtype=float32)

        #set up FITS table
        if (doFitsTable):
            doIndivSlitlets = False
            columns = []
            #Use new helper methods
            if (hasMultipleWavelengthSolutions(fdu)):
                #Different wavelength solution for each slitlet
                doIndivSlitlets = True #Process each slitlet below
            elif (hasWavelengthSolution(fdu)):
                #One single wavelength solution
                wave = getWavelengthSolution(fdu, 0, xsize)
                columns.append(pyfits.Column(name='Wavelength', format='D', array=wave))
            else:
                print("extractSpectraProcess::extractSpectra> Warning: Can not find header keyword PORDER in "+fdu.getFullId()+". Wavelength scale will not be written to FITS table.")
                self._log.writeLog(__name__, "Can not find header keyword PORDER in "+fdu.getFullId()+". Wavelength scale will not be written to FITS table.", type=fatboyLog.WARNING)
                wave = arange(xsize, dtype=float32)
                columns.append(pyfits.Column(name='Wavelength', format='D', array=wave))

        #Loop over specList and extract spectra
        for j in range(nspec):
            ylo = int(specList[j][0])
            yhi = int(specList[j][1])
            islit = int(specList[j][2])
            #Update header
            key = 'SPEC_'
            if (j+1 < 10):
                key += '0'
            key += str(j+1)
            esHeader[key] = "Slitlet "+str(islit)+": ["+str(ylo)+":"+str(yhi)+"]"

            #Extract spectra
            if (extract_weighting == 'linear'):
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    rssdata[j,:] = sum(fdu.getData()[ylo:yhi+1,:],0)
                    if (fdu.hasProperty("cleanFrame")):
                        rssclean[j,:] = sum(fdu.getData(tag="cleanFrame")[ylo:yhi+1,:],0)
                    if (fdu.hasProperty("noisemap")):
                        rssnm[j,:] = sqrt(sum(fdu.getData(tag="noisemap")[ylo:yhi+1,:]**2,0)) #sqrt of sum of squares
                    if (fdu.hasProperty("resampled")):
                        rssresamp[j,:] = sum(fdu.getData(tag="resampled")[ylo:yhi+1,:],0)
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    rssdata[j,:] = sum(fdu.getData()[:,ylo:yhi+1],1)
                    if (fdu.hasProperty("cleanFrame")):
                        rssclean[j,:] = sum(fdu.getData(tag="cleanFrame")[:,ylo:yhi+1],1)
                    if (fdu.hasProperty("noisemap")):
                        rssnm[j,:] = sqrt(sum(fdu.getData(tag="noisemap")[:,ylo:yhi+1]**2,1)) #sqrt of sum of squares
                    if (fdu.hasProperty("resampled")):
                        rssresamp[j,:] = sum(fdu.getData(tag="resampled")[:,ylo:yhi+1],1)
            elif (extract_weighting == 'median'):
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    rssdata[j,:] = gpu_arraymedian(fdu.getData()[ylo:yhi+1,:],axis="Y",nonzero=True)
                    if (fdu.hasProperty("cleanFrame")):
                        rssclean[j,:] = gpu_arraymedian(fdu.getData(tag="cleanFrame")[ylo:yhi+1,:],axis="Y",nonzero=True)
                    if (fdu.hasProperty("noisemap")):
                        rssnm[j,:] = sqrt(sum(fdu.getData(tag="noisemap")[ylo:yhi+1,:]**2,0)) #noisemap is still sqrt of sum of squares
                    if (fdu.hasProperty("resampled")):
                        rssresamp[j,:] = gpu_arraymedian(fdu.getData(tag="resampled")[ylo:yhi+1,:],axis="Y",nonzero=True)
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    rssdata[j,:] = gpu_arraymedian(fdu.getData()[:,ylo:yhi+1],axis="X",nonzero=True)
                    if (fdu.hasProperty("cleanFrame")):
                        rssclean[j,:] = gpu_arraymedian(fdu.getData(tag="cleanFrame")[:,ylo:yhi+1],axis="X",nonzero=True)
                    if (fdu.hasProperty("noisemap")):
                        rssnm[j,:] = sqrt(sum(fdu.getData(tag="noisemap")[:,ylo:yhi+1]**2,1)) #noisemap is still sqrt of sum of squares
                    if (fdu.hasProperty("resampled")):
                        rssresamp[j,:] = gpu_arraymedian(fdu.getData(tag="resampled")[:,ylo:yhi+1],axis="X",nonzero=True)
            elif (extract_weighting == 'gaussian'):
                ymin = max(ylo-gaussbox, 0)
                ymax = min(yhi+gaussbox+1, ysize)
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    slit = fdu.getData()[ymin:ymax,:].copy()
                    if (slitmask is not None):
                        #Apply mask to slit
                        currMask = slitmask.getData()[ymin:ymax,:] == (islit)
                        slit *= currMask
                    #Instead of taking median, sum so we get short spectra but do a
                    #5 pixel boxcar median smoothing to get rid of hot pixels
                    tempCut = mediansmooth1d(sum(slit, 1), 5)
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    slit = fdu.getData()[:,ymin:ymax].copy()
                    if (slitmask is not None):
                        #Apply mask to slit
                        currMask = slitmask.getData()[:,ymin:ymax] == (islit)
                        slit *= currMask
                    #Instead of taking median, sum so we get short spectra but do a
                    #5 pixel boxcar median smoothing to get rid of hot pixels
                    tempCut = mediansmooth1d(sum(slit, 0), 5)
                tempCut[tempCut < 0] = 0.
                p = zeros(4, dtype=float64)
                p[0] = max(tempCut[ylo-ymin:yhi-ymin+1])
                #p[1] = where(tempCut == max(tempCut))[0][0]
                p[1] = where(tempCut == p[0])[0][0]
                p[2] = 3
                p[3] = 0
                lsq = leastsq(gaussResiduals, p, args=(arange(len(tempCut), dtype=float64), tempCut))
                fit = gaussFunction(lsq[0], arange(len(tempCut)))
                cen = lsq[0][1]+ymin
                ntries = 1
                while ((cen < ylo or cen > yhi) and ntries < extract_nspec and lsq[0][0] > p[0]):
                    #Found wrong spectrum, retry
                    tempCut -= fit
                    tempCut[tempCut < 0] = 0.
                    lsq = leastsq(gaussResiduals, p, args=(arange(len(tempCut), dtype=float64), tempCut))
                    fit = gaussFunction(lsq[0], arange(len(tempCut)))
                    cen = lsq[0][1]+ymin
                    ntries += 1

                if (usePlot and (debug or writePlots)):
                    plt.plot(tempCut)
                    plt.plot(fit)
                if (cen >= ylo and cen <= yhi):
                    print("\tSpectrum "+str(j+1)+" (slitlet "+str(islit)+"): Center = "+formatNum(cen)+"; Gaussian = "+formatList(lsq[0]))
                    self._log.writeLog(__name__, "Spectrum "+str(j+1)+" (slitlet "+str(islit)+"): Center = "+formatNum(cen)+"; Gaussian = "+formatList(lsq[0]), printCaller=False, tabLevel=1)
                else:
                    lsq[0][0] = max(tempCut[ylo-ymin:yhi-ymin+1])
                    lsq[0][1] = (ylo+yhi)//2-ymin
                    lsq[0][2] = (yhi-ylo)/(4*sqrt(2*log(2)))
                    lsq[0][3] = 0
                    print("\tWarning: Spectrum "+str(j+1)+" (slitlet "+str(islit)+"): Could not properly fit Gaussian.  Using approximation instead: "+str(lsq[0]))
                    self._log.writeLog(__name__, "Spectrum "+str(j+1)+" (slitlet "+str(islit)+"): Could not properly fit Gaussian.  Using approximation instead: "+str(lsq[0]), type=fatboyLog.WARNING, printCaller=False, tabLevel=1)
                    if (usePlot and (debug or writePlots)):
                        plt.plot(gaussFunction(lsq[0], arange(len(tempCut))))
                if (usePlot and (debug or writePlots)):
                    plt.title("Spectrum "+str(j+1)+" (slitlet "+str(islit)+"): Center = "+formatNum(cen))
                    plt.xlabel("Gaussian = "+formatList(lsq[0]))
                    if (writePlots):
                        plt.savefig(outdir+"/extractedSpectra/qa_"+fdu.getFullId()+"_spec_"+str(j)+".png", dpi=200)
                    if (debug):
                        plt.show()
                    plt.close()
                #Calculate Gaussian
                cen = lsq[0][1]+ymin
                #Update center to match xs array below
                lsq[0][1] = cen
                xs = arange(yhi-ylo+1, dtype=float32)+ylo
                f = gaussFunction(lsq[0], xs)
                #Normalize
                f /= gpu_arraymedian(f, nonzero=True)
                #Multiply image by Gaussian
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    f = f.reshape(len(f),1)
                    rssdata[j,:] = sum(fdu.getData()[ylo:yhi+1,:]*f,0)
                    if (fdu.hasProperty("cleanFrame")):
                        rssclean[j,:] = sum(fdu.getData(tag="cleanFrame")[ylo:yhi+1,:]*f,0)
                    if (fdu.hasProperty("noisemap")):
                        rssnm[j,:] = sqrt(sum((fdu.getData(tag="noisemap")[ylo:yhi+1,:]*f)**2,0)) #sqrt of sum of squares
                    if (fdu.hasProperty("resampled")):
                        rssresamp[j,:] = sum(fdu.getData(tag="resampled")[ylo:yhi+1,:]*f,0)
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    rssdata[j,:] = sum(fdu.getData()[:,ylo:yhi+1]*f,1)
                    if (fdu.hasProperty("cleanFrame")):
                        rssclean[j,:] = sum(fdu.getData(tag="cleanFrame")[:,ylo:yhi+1]*f,1)
                    if (fdu.hasProperty("noisemap")):
                        rssnm[j,:] = sqrt(sum((fdu.getData(tag="noisemap")[:,ylo:yhi+1]*f)**2,1)) #sqrt of sum of squares
                    if (fdu.hasProperty("resampled")):
                        rssresamp[j,:] = sum(fdu.getData(tag="resampled")[:,ylo:yhi+1]*f,1)
            if (doFitsTable):
                if (doIndivSlitlets):
                    #Append unique wavelength solution here
                    #Calculate wavelength solution
                    wave = getWavelengthSolution(fdu, islit-1, xsize)
                    columns.append(pyfits.Column(name='Wavelength_'+str(j+1), format='D', array=wave))
                columns.append(pyfits.Column(name='Spectrum_'+str(j+1), format='D', array=rssdata[j,:]))
        #Update header
        fdu.updateHeader(esHeader)
        #Update data
        if (fdu.hasProperty("cleanFrame")):
            fdu.tagDataAs("cleanFrame", rssclean)
        if (fdu.hasProperty("noisemap")):
            fdu.tagDataAs("noisemap", rssnm)
        if (fdu.hasProperty("resampled")):
            fdu.tagDataAs("resampled", rssresamp)
        if (doFitsTable):
            tbhdu = createFitsTable(columns) #Use fatboyLibs wrapper
            fdu.setProperty("esTable", tbhdu)
        fdu.updateData(rssdata)
        return True
    #end extractSpectra

    ## Find spectral locations
    def findSpectra(self, fdu, calibs):
        ###*** For purposes of extractSpectra algorithm, X = dispersion direction and Y = cross-dispersion direction ***###
        #Read options
        extract_method = self.getOption("extract_method", fdu.getTag())
        extract_sigma = float(self.getOption("extract_sigma", fdu.getTag()))
        extract_min_width = int(self.getOption("extract_min_width", fdu.getTag()))
        extract_nspec = int(self.getOption("extract_nspec", fdu.getTag()))
        extract_xlo = self.getOption("extract_xlo", fdu.getTag())
        extract_xhi = self.getOption("extract_xhi", fdu.getTag())
        extract_ylo = self.getOption("extract_ylo", fdu.getTag())
        extract_yhi = self.getOption("extract_yhi", fdu.getTag())
        extract_min_flux_pct = float(self.getOption("extract_min_flux_pct", fdu.getTag()))
        extract_nsigma = 3
        if (self.getOption("extract_gauss_width", fdu.getTag()) is not None):
            extract_nsigma = float(self.getOption("extract_gauss_width", fdu.getTag()))

        debug = False
        if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
            debug = True
        writePlots = False
        if (self.getOption("write_plots", fdu.getTag()).lower() == "yes"):
            writePlots = True

        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            xsize = fdu.getShape()[1]
            ysize = fdu.getShape()[0]
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            ##xsize should be size across dispersion direction
            xsize = fdu.getShape()[0]
            ysize = fdu.getShape()[1]

        properties = dict()
        properties['specmode'] = fdu.getProperty("specmode")
        properties['dispersion'] = fdu.getProperty("dispersion")

        #Set defaults for xlo, xhi, ylo, yhi
        if (extract_xlo is None):
            extract_xlo = 0
        else:
            extract_xlo = int(extract_xlo)
        if (extract_xhi is None):
            extract_xhi = xsize
        else:
            extract_xhi = int(extract_xhi)
        if (extract_ylo is None):
            extract_ylo = 0
        else:
            extract_ylo = int(extract_ylo)
        if (extract_yhi is None):
            extract_yhi = ysize
        else:
            extract_yhi = int(extract_yhi)

        #Defaults for longslit - treat whole image as 1 slit
        nslits = 1
        ylos = [0]
        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            yhis = [fdu.getShape()[0]]
        else:
            yhis = [fdu.getShape()[1]]
        slitmask = None
        if (fdu._specmode != fdu.FDU_TYPE_LONGSLIT and (fdu.hasProperty("slitmask") or "slitmask" in calibs)):
            ###MOS/IFU data -- get slitmask
            #Use FDU "slitmask" property, which has been shifted and added, NOT slitmask corresponding to sky/lamp
            #Use new fdu.getSlitmask method
            slitmask = fdu.getSlitmask(pname=None, properties=properties)
            if (not fdu.hasProperty("nslits")):
                fdu.setProperty("nslits", slitmask.getData().max())
            nslits = fdu.getProperty("nslits")
            if (fdu.hasProperty("regions")):
                (ylos, yhis, slitx, slitw) = fdu.getProperty("regions")
            else:
                #Use helper method to all ylo, yhi for each slit in each frame
                (ylos, yhis, slitx, slitw) = findRegions(slitmask.getData(), nslits, fdu, gpu=self._fdb.getGPUMode(), log=self._log)
                fdu.setProperty("regions", (ylos, yhis, slitx, slitw))

        useESfile = False
        esfile = None
        if (os.access(extract_method, os.F_OK)):
            esfile = extract_method
        if (fdu.hasProperty("extract_method_file")):
            #If this individual fdu has a property extract_method_file, it overrides option
            esfile = fdu.getProperty("extract_method_file")
        if (esfile is not None):
            if (os.access(esfile, os.F_OK)):
                esinfo = self.readExtractMethodFile(esfile, nslits, fdu)
                if (esinfo is not None):
                    useESfile = True
            else:
                print("extractSpectraProcess::findSpectra> Warning: Could not find extract_method_file "+esfile+" for "+fdu.getFullId()+"!")
                self._log.writeLog(__name__, "Could not find extract_method_file "+esfile+" for "+fdu.getFullId()+"!", type=fatboyLog.WARNING)

        #Create specList to store locations of spectra
        specList = []

        #Select kernel for 2d median
        kernel2d = fatboyclib.median2d
        if (self._fdb.getGPUMode()):
            #Use GPU for medians
            kernel2d=gpumedian2d

        #Mask negatives and zeros in 2d image before looping over slitlets
        #Use cleanFrame if available
        #Use continuum_source if specified
        if ('continuum_source' in calibs):
            if (self._fdb.getGPUMode()):
                data = maskNegativesAndZeros(calibs['continuum_source'].getData(tag="cleanFrame"), zeroRep=1.e-6, negRep=0.0)
            else:
                data = maskNegativesAndZerosCPU(calibs['continuum_source'].getData(tag="cleanFrame"), zeroRep=1.e-6, negRep=0.0)
        else:
            if (self._fdb.getGPUMode()):
                data = maskNegativesAndZeros(fdu.getData(tag="cleanFrame"), zeroRep=1.e-6, negRep=0.0)
            else:
                data = maskNegativesAndZerosCPU(fdu.getData(tag="cleanFrame"), zeroRep=1.e-6, negRep=0.0)

        #Loop over nslits
        for j in range(nslits):
            if (useESfile):
                #Update options for this slit (or entire image) if using wc file
                extract_sigma = esinfo['extract_sigma'][j]
                extract_min_width = esinfo['extract_min_width'][j]
                extract_nspec = esinfo['extract_nspec'][j]
                extract_method = esinfo['extract_method'][j]
                extract_xlo = esinfo['extract_xlo'][j]
                extract_xhi = esinfo['extract_xhi'][j]
                extract_ylo = esinfo['extract_ylo'][j]
                extract_yhi = esinfo['extract_yhi'][j]
            if (nslits > 1):
                print("extractSpectraProcess::findSpectra> Finding spectra in slitlet "+str(j+1)+"...")
                self._log.writeLog(__name__, "Finding spectra in slitlet "+str(j+1)+"...")

            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                slit = data[ylos[j]:yhis[j]+1,:].copy()
                if (slitmask is not None):
                    #Apply mask to slit
                    currMask = slitmask.getData()[ylos[j]:yhis[j]+1,:] == (j+1)
                    slit *= currMask
                #Instead of taking median, sum so we get short spectra but do a
                #5 pixel boxcar median smoothing to get rid of hot pixels
                oned = mediansmooth1d(sum(slit[:,extract_xlo:extract_xhi], 1), 5)
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                slit = data[:,ylos[j]:yhis[j]+1].copy()
                if (slitmask is not None):
                    #Apply mask to slit
                    currMask = slitmask.getData()[:,ylos[j]:yhis[j]+1] == (j+1)
                    slit *= currMask
                #Instead of taking median, sum so we get short spectra but do a
                #5 pixel boxcar median smoothing to get rid of hot pixels
                oned = mediansmooth1d(sum(slit[extract_xlo:extract_xhi,:], 0), 5)

            if (extract_method == "full"):
                specList.append(array([0, ysize, j+1]))
            elif (extract_method == "manual"):
                if (isinstance(extract_ylo, list) and isinstance(extract_yhi, list)):
                    #Case of extract_ylo, extract_yhi comma separated list
                    n = min(len(extract_ylo), len(extract_yhi))
                    for j in range(n):
                        specList.append(array([extract_ylo[j], extract_yhi[j], j+1]))
                else:
                    #Normal case
                    specList.append(array([extract_ylo, extract_yhi, j+1]))
            elif (extract_method == "auto" or extract_method == "semi"):
                if (extract_method == "semi"):
                    oned = oned[extract_ylo:extract_yhi]
                #4/15/20 add sort=True for longslit or other cases with multiple
                #spectra per slitlet so that we ensure we get the brightest in the
                #case of any overlaps
                y = extractSpectra(oned, extract_sigma, extract_min_width, extract_nspec, minFluxPct=extract_min_flux_pct, sort=True)
                if (usePlot and (debug or writePlots)):
                    plt.plot(oned)
                    plt.title("1D cut of slitlet "+str(j+1))
                    if (writePlots):
                        #make directory if necessary
                        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
                        if (not os.access(outdir+"/extractedSpectra", os.F_OK)):
                            os.mkdir(outdir+"/extractedSpectra",0o755)
                        plt.savefig(outdir+"/extractedSpectra/qa_"+fdu.getFullId()+"_slit_"+str(j+1)+".png", dpi=200)
                    if (debug):
                        print("Slit "+str(j), ylos[j], yhis[j], extract_xlo, extract_xhi)
                        print(y)
                        plt.show()
                    plt.close()
                if (y is None):
                    print("extractSpectraProcess::findSpectra> Warning: Could not find spectrum in slitlet "+str(j+1))
                    self._log.writeLog(__name__, "Could not find spectrum in slitlet "+str(j+1), type=fatboyLog.WARNING)
                    continue
                specAppended = []
                specListSlit = [] #list for this slit
                yloSlit = [] #list of ylos for spectra found to use to sort
                for i in range(len(y)):
                    if (y[i][0] == -1):
                        print("extractSpectraProcess::findSpectra> Warning: Could not find spectrum in slitlet "+str(j+1))
                        self._log.writeLog(__name__, "Could not find spectrum in slitlet "+str(j+1), type=fatboyLog.WARNING)
                        break
                    #Found spectrum.  Use full x-range
                    width = (y[i][1]-y[i][0])//2+1 #should be int
                    ycen = (y[i][1]+y[i][0])//2 #should be int
                    ylo = max(0, int(y[i][0]-width))
                    yhi = min(len(oned), int(y[i][1]+width)+1) #add 1 to hi for indexing
                    #Loop over previous spectra (presumably stronger signal) to ensure no overlap
                    for k in range(i):
                        if (not specAppended[k]):
                            #Spectrum not used
                            continue
                        #Note previous spectra have had ylos[j] appended to y-values
                        if (ylo < y[k][0]-ylos[j] and yhi > y[k][0]-ylos[j] and ycen < y[k][0]-ylos[j]):
                            #This spectrum is to left of previous one and overlaps at high end
                            yhi = y[k][0]-ylos[j]
                        elif (yhi > y[k][1]-ylos[j] and ylo < y[k][1]-ylos[j] and ycen > y[k][1]-ylos[j]):
                            #This spectrum is to the right of previous one and overlaps at low end
                            ylo = y[k][1]-ylos[j]
                    p = zeros(4, dtype=float64)
                    p[0] = max(oned[ylo:yhi])
                    p[1] = ycen-ylo
                    p[2] = width/(2*sqrt(2*log(2)))
                    p[3] = gpu_arraymedian(oned[ylo:yhi])
                    #lsq = leastsq(gaussResiduals, p, args=(arange(len(oned[ylo:yhi]), dtype=float64), oned[ylo:yhi]))
                    #Use helper method now 8/1/18
                    lsq = fitGaussian(oned[ylo:yhi], guess=p, maxWidth=extract_min_width*2)
                    if (lsq[1] == False):
                        print("extractSpectraProcess::findSpectra> Warning: Could not fit spectrum in slitlet "+str(j+1))
                        self._log.writeLog(__name__, "Could not fit spectrum in slitlet "+str(j+1), type=fatboyLog.WARNING)
                        break
                    fwhm = abs(lsq[0][2])*2*sqrt(2*log(2))
                    xwidth = abs(extract_nsigma*lsq[0][2]) #Width for extract box, default = 3sigma.
                    y[i][0] = int(lsq[0][1]-xwidth+ylo)+ylos[j]
                    y[i][1] = int(lsq[0][1]+xwidth+ylo)+ylos[j]
                    if (extract_method == "semi"):
                        #if semi-automatic, add extract_ylo back in here instead of above 8/16/18
                        y[i] += extract_ylo
                    #Check for overlaps with previous spectra
                    doAppend = True
                    if (fwhm > yhi-ylo):
                        #Width is greater than entire fit range
                        doAppend = False
                    if (y[i][0] < 0 or y[i][1] < 0):
                        #Negative = fit failed
                        doAppend = False
                    for spec in specListSlit:
                        if (y[i][0] >= spec[0] and y[i][0] <= spec[1]):
                            doAppend = False
                        if (y[i][1] >= spec[0] and y[i][1] <= spec[1]):
                            doAppend = False
                    if (y[i][1] >= len(oned)+ylos[j]):
                        print("extractSpectraProcess::findSpectra> Warning: Spectrum in slitlet "+str(j+1)+" overlaps with edge of chip.  Truncating.")
                        self._log.writeLog(__name__, "Spectrum in slitlet "+str(j+1)+" overlaps with edge of chip.  Truncating.", type=fatboyLog.WARNING)
                        y[i][1] = len(oned)+ylos[j]-1
                    #Keep track of whether spectrum was appended
                    specAppended.append(doAppend)
                    if (usePlot and self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                        print(p)
                        print(lsq)
                        print(y[i], doAppend)
                    if (doAppend):
                        specListSlit.append(array([y[i][0], y[i][1], j+1]))
                        yloSlit.append(y[i][0])
                #Now sort by yloSlit and add to master specList
                for idx in argsort(yloSlit):
                    specList.append(specListSlit[idx])
                    print("\tFound spectrum: "+str(specList[-1]))
                    self._log.writeLog(__name__, "Found spectrum: "+str(specList[-1]), printCaller=False, tabLevel=1)

        #Write specList to disk if requested
        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            #make directory if necessary
            outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
            if (not os.access(outdir+"/extractedSpectra", os.F_OK)):
                os.mkdir(outdir+"/extractedSpectra",0o755)
            slfile = outdir+"/extractedSpectra/spec_locations_"+fdu._id+".dat"
            #Overwrite if overwrite_files = yes
            if (os.access(slfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(slfile)
            if (not os.access(slfile, os.F_OK)):
                savetxt(slfile, array(specList), fmt='%d', delimiter='\t')
        #return specList
        return specList
    #end findSpectra

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/extractedSpectra", os.F_OK)):
            os.mkdir(outdir+"/extractedSpectra",0o755)
        #Create output filename
        esfile = outdir+"/extractedSpectra/es_"+fdu.getFullId()
        #Check to see if it exists
        if (os.access(esfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(esfile)
        if (not os.access(esfile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(esfile, headerExt=fdu.getProperty("wcHeader"))
        #Write out clean frame if it exists
        if (fdu.hasProperty("cleanFrame")):
            cleanfile = outdir+"/extractedSpectra/clean_es_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(cleanfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(cleanfile)
            if (not os.access(cleanfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.writeTo(cleanfile, tag="cleanFrame", headerExt=fdu.getProperty("wcHeader"))
        #Write noisemap for spectrocsopy data if requested
        if (self.getOption("write_noisemaps", fdu.getTag()).lower() == "yes" and fdu.hasProperty("noisemap")):
            nmfile = outdir+"/extractedSpectra/NM_es_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(nmfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(nmfile)
            if (not os.access(nmfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.writeTo(nmfile, tag="noisemap", headerExt=fdu.getProperty("wcHeader"))
        #Write out resampled data if it exists
        if (fdu.hasProperty("resampled")):
            resampfile = outdir+"/extractedSpectra/resamp_es_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(resampfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(resampfile)
            if (not os.access(resampfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                #Write with resampHeader as header extension
                fdu.writeTo(resampfile, tag="resampled", headerExt=fdu.getProperty("resampledHeader"))
        #Write out FITS table if it exists
        if (fdu.hasProperty("esTable")):
            tabfile = outdir+"/extractedSpectra/es_table_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(tabfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(tabfile)
            if (not os.access(tabfile, os.F_OK)):
                fdu.getProperty("esTable").verify('silentfix')
                fdu.getProperty("esTable").writeto(tabfile, output_verify="silentfix")
    #end writeOutput
