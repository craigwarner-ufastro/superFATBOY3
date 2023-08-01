from superFATBOY.fatboyCalib import fatboyCalib
from superFATBOY.fatboyDataUnit import fatboyDataUnit
from superFATBOY.fatboyImage import fatboyImage
from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY import gpu_imcombine, imcombine

class biasSubtractProcess(fatboyProcess):
    _modeTags = ["imaging"]

    #Convenience method so that code doesn't have to be rewritten several times in getCalib
    def createMasterBias(self, fdu, biases):
        mbfilename = None
        #use biases[0] for exptime in case this is a bias for a different exptime than the fdu
        mbname = "masterBiases/mbias-"+biases[0]._id
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (fdu.getTag(mode="composite") is not None):
            mbname += "-"+fdu.getTag(mode="composite").replace(" ","_")
        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            #Optionally save if write_calib_output = yes
            if (not os.access(outdir+"/masterBiases", os.F_OK)):
                os.mkdir(outdir+"/masterBiases",0o755)
            mbfilename = outdir+"/"+mbname+".fits"
        #Check to see if master bias exists already from a previous run
        prevmbfilename = outdir+"/"+mbname+".fits"
        #Noisemap file
        nmfile = outdir+"/masterBiases/NM_mbias-"+biases[0]._id+".fits"
        if (os.access(prevmbfilename, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(prevmbfilename)
        elif (os.access(prevmbfilename, os.F_OK)):
            #file already exists
            print("biasSubtractProcess::createMasterBias> Master bias "+prevmbfilename+" already exists!  Re-using...")
            self._log.writeLog(__name__, "Master bias "+prevmbfilename+" already exists!  Re-using...")
            masterBias = fatboyCalib(self._pname, "master_bias", biases[0], filename=prevmbfilename, log=self._log)
            #Check to see if a noisemap exists
            if (os.access(nmfile, os.F_OK)):
                nm = pyfits.open(nmfile)
                mef = findMef(nm)
                #Tag noisemap data.  tagDataAs() will handle byteswap
                masterBias.tagDataAs("noisemap", nm[mef].data)
                nm.close()
            #disable these biases as master bias has been created
            for bias in biases:
                bias.disable()
            return masterBias

        #Select cpu/gpu option
        if (self._fdb.getGPUMode()):
            (data, header) = gpu_imcombine.imcombine(biases, outfile=mbfilename, method="median", mef=biases[0]._mef, returnHeader=True, log=self._log)
        else:
            (data, header) = imcombine.imcombine(biases, outfile=mbfilename, method="median", mef=biases[0]._mef, returnHeader=True, log=self._log)
        masterBias = fatboyCalib(self._pname, "master_bias", biases[0], data=data, tagname=mbname, headerExt=header, log=self._log)
        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes" and not os.access(mbfilename, os.F_OK)):
            #Optionally save if write_calib_output = yes
            masterBias.writeTo(mbfilename)
        #Create and write out noisemap for spectroscopy data if requested
        if (self.getOption("write_noisemaps", fdu.getTag()).lower() == "yes"):
            if (not os.access(outdir+"/masterBiases", os.F_OK)):
                os.mkdir(outdir+"/masterBiases",0o755)
            if (os.access(nmfile, os.F_OK) and  self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(nmfile)
            if (not os.access(nmfile, os.F_OK)):
                ncomb = float(masterBias.getHeaderValue('NCOMBINE'))
                #Create noisemap
                if (self._fdb.getGPUMode()):
                    nm = createNoisemap(masterBias.getData(), ncomb)
                else:
                    nm = sqrt(masterBias.getData()/ncomb)
                masterBias.tagDataAs("noisemap", nm)
                masterBias.writeTo(nmfile, tag="noisemap")

        #disable these biases as master bias has been created
        for bias in biases:
            bias.disable()
        return masterBias
    #end createMasterBias

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("Bias Subtract")
        print(fdu._identFull)

        #Call get calibs to return dict() of calibration frames.
        #For biasSubtract, this dict should have one entry 'masterBias' which is an fdu
        calibs = self.getCalibs(fdu, prevProc)
        if (not 'masterBias' in calibs):
            #Failed to obtain master bias frame
            #Issue error message and disable this FDU
            print("biasSubtractProcess::execute> ERROR: Bias not subtracted for "+fdu.getFullId()+".  Discarding Image!")
            self._log.writeLog(__name__, "Bias not subtracted for "+fdu.getFullId()+".  Discarding Image!", type=fatboyLog.ERROR)
            #disable this FDU
            fdu.disable()
            return False

        #Check if output exists first
        bsfile = "biasSubtracted/bs_"+fdu.getFullId()
        if (self.checkOutputExists(fdu, bsfile)):
            #Also check if "noisemap" exists
            nmfile = "biasSubtracted/NM_bs_"+fdu.getFullId()
            self.checkOutputExists(fdu, nmfile, tag="noisemap")
            return True

        #get master bias
        masterBias = calibs['masterBias']

        #Propagate noisemap for spectroscopy data
        if (fdu.hasProperty("noisemap")):
            self.updateNoisemap(fdu, masterBias)

        #make sure both are floating point before subtracting
        fdu.updateData(float32(fdu.getData())-float32(masterBias.getData()))
        fdu._header.add_history('Bias subtracted using '+masterBias._id)
        return True
    #end execute

    ## OVERRIDE getCalibs
    def getCalibs(self, fdu, prevProc = None):
        calibs = dict()

        mbfilename = self.getCalib("masterBias", fdu.getTag())
        if (mbfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(mbfilename, os.F_OK)):
                print("biasSubtractProcess::getCalibs> Using master bias "+mbfilename+"...")
                self._log.writeLog(__name__, "Using master bias "+mbfilename+"...")
                calibs['masterBias'] = fatboyCalib(self._pname, "master_bias", fdu, filename=mbfilename, log=self._log)
                return calibs
            else:
                print("biasSubtractProcess::getCalibs> Warning: Could not find master bias "+mbfilename+"...")
                self._log.writeLog(__name__, "Could not find master bias "+mbfilename+"...", type=fatboyLog.WARNING)

        #1) Check for an already created master bias frame matching section and TAGGED for this object
        masterBias = self._fdb.getTaggedMasterCalib(self._pname, fdu._id, obstype="master_bias", section=fdu.section)
        if (masterBias is not None):
            #Found master bias.  Return here
            calibs['masterBias'] = masterBias
            return calibs
        #2) Check for individual bias frames matching section to create master bias and TAGGED for this object
        biases = self._fdb.getTaggedCalibs(fdu._id, obstype=fatboyDataUnit.FDU_TYPE_BIAS, section=fdu.section)
        if (len(biases) > 0):
            #Found biases associated with this fdu.  Create master bias.
            print("biasSubtractProcess::getCalibs> Creating Master Bias for tagged object "+fdu._id+"...")
            #First recursively process (linearity correction probably)
            self.recursivelyExecute(biases, prevProc)
            #convenience method
            masterBias = self.createMasterBias(fdu, biases)
            self._fdb.appendCalib(masterBias)
            calibs['masterBias'] = masterBias
            return calibs
        #3) Check for an already created master bias frame matching section
        masterBias = self._fdb.getMasterCalib(self._pname, obstype="master_bias", section=fdu.section, tag=fdu.getTag())
        if (masterBias is not None):
            #Found master bias.  Return here
            calibs['masterBias'] = masterBias
            return calibs
        #4) Check for individual bias frames matching section to create master bias
        biases = self._fdb.getCalibs(obstype=fatboyDataUnit.FDU_TYPE_BIAS, section=fdu.section, tag=fdu.getTag())
        if (len(biases) > 0):
            #Found biases associated with this fdu.  Create master bias.
            print("biasSubtractProcess::getCalibs> Creating Master Bias...")
            #First recursively process (linearity correction probably)
            self.recursivelyExecute(biases, prevProc)
            #convenience method
            masterBias = self.createMasterBias(fdu, biases)
            self._fdb.appendCalib(masterBias)
            calibs['masterBias'] = masterBias
            return calibs
        print("biasSubtractProcess::getCalibs> Master Bias for section "+str(fdu.section)+" not found!")
        #5) Check default_master_bias for matching exptime/nreads/section
        defaultMasterBiases = []
        if (self.getOption('default_master_bias', fdu.getTag()) is not None):
            dmblist = self.getOption('default_master_bias', fdu.getTag())
            if (dmblist.count(',') > 0):
                #comma separated list
                defaultMasterBiases = dmblist.split(',')
                removeEmpty(defaultMasterBiases)
                for j in range(len(defaultMasterBiases)):
                    defaultMasterBiases[j] = defaultMasterBiases[j].strip()
            elif (dmblist.endswith('.fit') or dmblist.endswith('.fits')):
                #FITS file given
                defaultMasterBiases.append(dmblist)
            elif (dmblist.endswith('.dat') or dmblist.endswith('.list') or dmblist.endswith('.txt')):
                #ASCII file list
                defaultMasterBiases = readFileIntoList(dmblist)
            for mbiasfile in defaultMasterBiases:
                #Loop over list of default master biases
                #masterBias = fatboyImage(mbiasfile, log=self._log)
                masterBias = fatboyCalib(self._pname, "master_bias", fdu, filename=mbiasfile, log=self._log)
                #read header and initialize
                masterBias.readHeader()
                masterBias.initialize()
                if (fdu.section is not None):
                    #check section if applicable
                    section = -1
                    if (masterBias.hasHeaderValue('SECTION')):
                        section = masterBias.getHeaderValue('SECTION')
                    else:
                        idx = masterBias.getFilename().rfind('.fit')
                        if (masterBias.getFilename()[idx-2] == 'S' and isDigit(masterBias.getFilename()[idx-1])):
                            section = int(masterBias.getFilename()[idx-1])
                    if (section != fdu.section):
                        continue
                masterBias.setType("master_bias")
                #Found matching master bias
                print("biasSubtractProcess::getCalibs> Using default master bias "+masterBias.getFilename())
                self._fdb.appendCalib(masterBias)
                calibs['masterBias'] = masterBias
                return calibs
        return calibs
    #end getCalibs

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('default_master_bias', None)
        self._options.setdefault('write_noisemaps', 'no')
    #end setDefaultOptions

    ## update noisemap for spectroscopy data
    def updateNoisemap(self, fdu, masterBias):
        if (not masterBias.hasProperty("noisemap")):
            #create tagged data "noisemap"
            ncomb = 1.0
            if (masterBias.hasHeaderValue('NCOMBINE')):
                ncomb = float(masterBias.getHeaderValue('NCOMBINE'))
            if (self._fdb.getGPUMode()):
                nm = createNoisemap(masterBias.getData(), ncomb)
            else:
                nm = sqrt(masterBias.getData()/ncomb)
            masterBias.tagDataAs("noisemap", nm)
        #Get this FDU's noisemap
        nm = fdu.getData(tag="noisemap")
        #Propagate noisemaps.  For subtraction, dz = sqrt(dx^2 + dy^2)
        if (self._fdb.getGPUMode()):
            nm = noisemaps_ds_gpu(fdu.getData(tag="noisemap"), masterBias.getData("noisemap"))
        else:
            nm = sqrt(fdu.getData(tag="noisemap")**2+masterBias.getData("noisemap")**2)
        fdu.tagDataAs("noisemap", nm)
    #end updateNoisemap

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/biasSubtracted", os.F_OK)):
            os.mkdir(outdir+"/biasSubtracted",0o755)
        #Create output filename
        bsfile = outdir+"/biasSubtracted/bs_"+fdu.getFullId()
        #Check to see if it exists
        if (os.access(bsfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(bsfile)
        if (not os.access(bsfile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(bsfile)
        #Write noisemap for spectrocsopy data if requested
        if (self.getOption("write_noisemaps", fdu.getTag()).lower() == "yes" and fdu.hasProperty("noisemap")):
            nmfile = outdir+"/biasSubtracted/NM_bs_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(nmfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(nmfile)
            if (not os.access(nmfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.writeTo(nmfile, tag="noisemap")
    #end writeOutput
