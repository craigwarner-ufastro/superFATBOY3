from superFATBOY.fatboyCalib import fatboyCalib
from superFATBOY.fatboyDataUnit import fatboyDataUnit
from superFATBOY.fatboyImage import fatboyImage
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.gpu_arraymedian import *
hasCuda = True
try:
    import superFATBOY
    if (not superFATBOY.gpuEnabled()):
        hasCuda = False
    else:
        import pycuda.driver as drv
        if (not superFATBOY.threaded()):
            #If not threaded mode, import autoinit.  Otherwise assume context exists.
            #Code will crash if in threaded mode and context does not exist.
            import pycuda.autoinit
        from pycuda.compiler import SourceModule
except Exception:
    print("badPixelMaskProcess> Warning: PyCUDA not installed")
    hasCuda = False
from numpy import *
import os, time

block_size = 512

class badPixelMaskProcess(fatboyProcess):
    _modeTags = ["imaging", "circe"]

    if (hasCuda):
        bpm_mod = SourceModule("""
        __global__ void gpu_bad_pixel_mask(int *output, float *input, int nx, int ny, float lo, float hi, int edge, float radius) {
          const int i = blockDim.x*blockIdx.x + threadIdx.x;
          if (i >= nx*ny) return;
          float xin = (i%nx);
          float yin = (i/nx);

          output[i] = 0; //false
          if (input[i] < lo || input[i] > hi) {
            output[i] = 1;
            return;
          }
          if (edge > 0) {
            if (xin < edge || yin < edge || xin >= nx-edge || yin >= ny-edge) {
              output[i] = 1;
              return;
            }
          }
          if (radius > 0) {
            xin -= ((nx-1)/2.);
            yin -= ((ny-1)/2.);
            float r = sqrt(xin*xin+yin*yin);
            if (r > radius) output[i] = 1;
          }
          return;
        }
        """)

    #Convenience method so that code doesn't have to be rewritten several times in getCalib
    def createBadPixelMask(self, fdu, sourceFDU):
        badPixelMask = None
        bpmfilename = None
        bpmname = "badPixelMasks/BPM-"+str(fdu.filter)+"-"+str(fdu._id)
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (fdu.getTag(mode="composite") is not None):
            bpmname += "-"+fdu.getTag(mode="composite").replace(" ","_")
        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            #Optionally save if write_calib_output = yes
            if (not os.access(outdir+"/badPixelMasks", os.F_OK)):
                os.mkdir(outdir+"/badPixelMasks",0o755)
        #Check to see if bad pixel mask exists already from a previous run
        bpmfilename = outdir+"/"+bpmname+".fits"
        if (os.access(bpmfilename, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(bpmfilename)
        elif (os.access(bpmfilename, os.F_OK)):
            #file already exists.  #Use fdu as source header
            print("badPixelMaskProcess::createBadPixelMask> Bad pixel mask "+bpmfilename+" already exists!  Re-using...")
            self._log.writeLog(__name__, "Bad pixel mask "+bpmfilename+" already exists!  Re-using...")
            badPixelMask = fatboyCalib(self._pname, fatboyDataUnit.FDU_TYPE_BAD_PIXEL_MASK, fdu, filename=bpmfilename, log=self._log)
            return badPixelMask

        print("badPixelMaskProcess::createBadPixelMask> Using id="+str(sourceFDU.getFullId())+", orig filename="+str(sourceFDU.getFilename())+" to calculate bad pixel mask...")
        self._log.writeLog(__name__, "Using id="+str(sourceFDU.getFullId())+", orig filename="+str(sourceFDU.getFilename())+" to calculate bad pixel mask...")
        clippingMethod = self.getOption('clipping_method', fdu.getTag()).lower()
        try:
            clipping_high = float(self.getOption('clipping_high', fdu.getTag()))
            clipping_low = float(self.getOption('clipping_low', fdu.getTag()))
            clipping_sigma = float(self.getOption('clipping_sigma', fdu.getTag()))
            edge_reject = int(self.getOption('edge_reject', fdu.getTag()))
            radius_reject = float(self.getOption('radius_reject', fdu.getTag()))
        except ValueError as ex:
            print("badPixelMaskProcess::createBadPixelMask> Error: invalid bad pixel mask options: "+str(ex))
            self._log.writeLog(__name__, " invalid bad pixel mask options: "+str(ex), type=fatboyLog.ERROR)
            return None

        #Select cpu/gpu option
        if (self._fdb.getGPUMode()):
            if (clippingMethod == "values"):
                lo = clipping_low
                hi = clipping_high
            elif (clippingMethod == "sigma"):
                sigclip = sigmaFromClipping(sourceFDU.getData(), clipping_sigma, 5)
                med = sigclip[1]
                stddev = sigclip[2]
                lo = med-sig*stddev
                hi = med+sig*stddev
            #Create bad pixel mask
            (ny, nx) = fdu.getShape()
            gpu_bad_pixel_mask = self.bpm_mod.get_function("gpu_bad_pixel_mask")
            data = empty(fdu.getShape(), int32)
            blocks = data.size//block_size
            if (data.size % block_size != 0):
                blocks += 1
            gpu_bad_pixel_mask(drv.Out(data), drv.In(sourceFDU.getData()), int32(nx), int32(ny), float32(lo), float32(hi), int32(edge_reject), float32(radius_reject), grid=(blocks,1), block=(block_size,1,1))
            data = data.astype(bool)
        else:
            if (clippingMethod == "values"):
                lo = clipping_low
                hi = clipping_high
            elif (clippingMethod == "sigma"):
                sigclip = sigmaFromClipping(sourceFDU.getData(), clipping_sigma, 5)
                med = sigclip[1]
                stddev = sigclip[2]
                lo = med-sig*stddev
                hi = med+sig*stddev
            #Create bad pixel mask
            data = logical_or(sourceFDU.getData() < lo, sourceFDU.getData() > hi)
            if (edge_reject > 0):
                data[0:edge_reject,:] = True
                data[:,0:edge_reject] = True
                data[-1*edge_reject:,:] = True
                data[:,-1*edge_reject:] = True
            if (radius_reject > 0):
                (ny, nx) = data.shape
                xs = arange(ny*nx).reshape(ny,nx) % nx - ((nx-1)/2.)
                ys = arange(ny*nx).reshape(ny,nx) // nx - ((ny-1)/2.)
                rs = sqrt(xs**2+ys**2)
                data[rs > radius_reject] = True
        #Column/row reject
        if (self.getOption('column_reject', fdu.getTag()) is not None):
            #Column reject of format "54, 320:384, 947, 1024:1080"
            column_reject = self.getOption('column_reject', fdu.getTag())
            try:
                #Parse out into list
                column_reject = column_reject.split(",")
                for j in range(len(column_reject)):
                    column_reject[j] = column_reject[j].strip().split(":")
                    #Mask out data
                    if (len(column_reject[j]) == 1):
                        data[:,int(column_reject[j][0])] = True
                    elif (len(column_reject[j]) == 2):
                        data[:,int(column_reject[j][0]):int(column_reject[j][1])] = True
            except ValueError as ex:
                print("badPixelMaskProcess::createBadPixelMask> Error: invalid format in column_reject: "+str(ex))
                self._log.writeLog(__name__, " invalid format in column_reject: "+str(ex), type=fatboyLog.ERROR)
                return None
        if (self.getOption('row_reject', fdu.getTag()) is not None):
            #Row reject of format "54, 320:384, 947, 1024:1080"
            row_reject = self.getOption('row_reject', fdu.getTag())
            try:
                #Parse out into list
                row_reject = row_reject.split(",")
                for j in range(len(row_reject)):
                    row_reject[j] = row_reject[j].strip().split(":")
                    #Mask out data
                    if (len(row_reject[j]) == 1):
                        data[int(row_reject[j][0]),:] = True
                    elif (len(row_reject[j]) == 2):
                        data[int(row_reject[j][0]):int(row_reject[j][1]),:] = True
            except ValueError as ex:
                print("badPixelMaskProcess::createBadPixelMask> Error: invalid format in row_reject: "+str(ex))
                self._log.writeLog(__name__, " invalid format in row_reject: "+str(ex), type=fatboyLog.ERROR)
                return None

        print("badPixelMaskProcess::createBadPixelMask> Found "+str(data.sum())+" bad pixels...")
        self._log.writeLog(__name__, " Found "+str(data.sum())+" bad pixels...")
        pct = 100*float(data.sum())/float(data.size)
        if (pct > 25):
            print("badPixelMaskProcess::createBadPixelMask> WARNING: "+str(pct)+"% of pixels found to be bad.  Check your flat field!")
            self._log.writeLog(__name__, str(pct)+"% of pixels found to be bad.  Check your flat field!", type=fatboyLog.WARNING)
        #Use fdu as source - this will copy over filter/exptime/section
        badPixelMask = fatboyCalib(self._pname, fatboyDataUnit.FDU_TYPE_BAD_PIXEL_MASK, fdu, data=data, tagname=bpmname, log=self._log)
        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            if (os.access(bpmfilename, os.F_OK)):
                os.unlink(bpmfilename)
            #Optionally save if write_calib_output = yes
            badPixelMask.writeTo(bpmfilename)
        return badPixelMask
    #end createBadPixelMask

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("Bad Pixel Mask")
        print(fdu._identFull)

        #Call get calibs to return dict() of calibration frames.
        #For badPixelMask, this dict should have one entry 'badPixelMask' which is an fdu
        calibs = self.getCalibs(fdu, prevProc)
        if (not 'badPixelMask' in calibs):
            #Failed to obtain master flat frame
            #Issue error message and disable this FDU
            print("badPixelMaskProcess::execute> ERROR: Bad pixel mask not applied for "+fdu.getFullId()+" (filter="+str(fdu.filter)+").  Discarding Image!")
            self._log.writeLog(__name__, "Bad pixel mask not applied for "+fdu.getFullId()+" (filter="+str(fdu.filter)+").  Discarding Image!", type=fatboyLog.ERROR)
            #disable this FDU
            fdu.disable()
            return False

        #Check if output exists first
        bafile = "badPixelMaskApplied/ba_"+fdu.getFullId()
        if (self.checkOutputExists(fdu, bafile)):
            return True

        #get bad pixel mask
        badPixelMask = calibs['badPixelMask']
        #apply bad pixel mask to master flat and renormalize
        if ('masterFlat' in calibs):
            #get master flat
            masterFlat = calibs['masterFlat']
            #Renormalize master flat to median value of 1
            masterFlat.renormalize()
            if (not masterFlat.hasHistory('renormalized_bpm')):
                masterFlat.renormalize(badPixelMask.getData())
            if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
                #Optionally save renormalized flat if write_calib_output = yes
                outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
                mffilename = outdir+"/badPixelMasks/renorm_"+masterFlat.getFullId()
                if (os.access(mffilename, os.F_OK)):
                    os.unlink(mffilename)
                #Optionally save if write_calib_output = yes
                masterFlat.writeTo(mffilename)
            #Update FDU with new median value of master flat
            scaleFactor = masterFlat.getHistory('renormalized_bpm')
            fdu.updateData(float32(fdu.getData())*scaleFactor)
            fdu.setHistory('rescaled_for_bad_pixels', scaleFactor)
        #Apply bad pixel mask
        fdu.applyBadPixelMask(badPixelMask)
        fdu._header.add_history('Applied bad pixel mask '+badPixelMask.filename)
        return True
    #end execute

    ## OVERRIDE getCalibs
    def getCalibs(self, fdu, prevProc = None):
        calibs = dict()

        #First find a master flat frame to also apply bad pixel mask to
        #1) Check for an already created master flat frame matching filter/section and TAGGED for this object
        masterFlat = self._fdb.getTaggedMasterCalib(pname=None, ident=fdu._id, obstype="master_flat", filter=fdu.filter, section=fdu.section)
        if (masterFlat is not None):
            #Found master flat
            calibs['masterFlat'] = masterFlat
        else:
            #2) Check for an already created master flat frame matching filter/section
            masterFlat = self._fdb.getMasterCalib(pname=None, obstype="master_flat", filter=fdu.filter, section=fdu.section, tag=fdu.getTag())
            if (masterFlat is not None):
                #Found master flat.
                calibs['masterFlat'] = masterFlat
            else:
                #3) Look at previous master flats to see if any has a history of being used as master flat for
                #this _id and filter combination from step 7 below.
                masterFlats = self._fdb.getMasterCalibs(obstype="master_flat")
                for mflat in masterFlats:
                    if (mflat.hasHistory('master_flat::'+fdu._id+'::'+str(fdu.filter))):
                        #Use this master flat
                        print("badPixelMaskProcess::getCalibs> Using master flat "+mflat.getFilename()+" with filter "+str(mflat.filter))
                        self._log.writeLog(__name__, "Using master flat "+mflat.getFilename()+" with filter "+str(mflat.filter))
                        #Already in _calibs, no need to appendCalib
                        calibs['masterFlat'] = mflat

        bpmfilename = self.getCalib("badPixelMask", fdu.getTag())
        if (bpmfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(bpmfilename, os.F_OK)):
                print("badPixelMaskProcess::getCalibs> Using bad pixel mask "+bpmfilename+"...")
                self._log.writeLog(__name__, "Using bad pixel mask "+bpmfilename+"...")
                calibs['badPixelMask'] = fatboyCalib(self._pname, fatboyDataUnit.FDU_TYPE_BAD_PIXEL_MASK, fdu, filename=bpmfilename, log=self._log)
                return calibs
            else:
                print("badPixelMaskProcess::getCalibs> Warning: Could not find bad pixel mask "+bpmfilename+"...")
                self._log.writeLog(__name__, "Could not find bad pixel mask "+bpmfilename+"...", type=fatboyLog.WARNING)

        #Next find or create bad pixel mask
        #1) Check for an already created bad pixel mask matching section and TAGGED for this object.  filter does not need to match.
        badPixelMask = self._fdb.getTaggedMasterCalib(self._pname, fdu._id, fatboyDataUnit.FDU_TYPE_BAD_PIXEL_MASK, section=fdu.section)
        if (badPixelMask is not None):
            #Found bpm.  Return here
            calibs['badPixelMask'] = badPixelMask
            return calibs
        #2) Check for an already created bad pixel mask matching filter/section
        badPixelMask = self._fdb.getMasterCalib(self._pname, obstype=fatboyDataUnit.FDU_TYPE_BAD_PIXEL_MASK, filter=fdu.filter, section=fdu.section, tag=fdu.getTag())
        if (badPixelMask is not None):
            #Found bpm.  Return here
            calibs['badPixelMask'] = badPixelMask
            return calibs
        #3) Check default_bad_pixel_mask for matching filter/section before using source, master_flat, or master_dark
        defaultBPMs = []
        if (self.getOption('default_bad_pixel_mask', fdu.getTag()) is not None):
            dbpmlist = self.getOption('default_bad_pixel_mask', fdu.getTag())
            ignoreHeader = self.getOption('default_bpm_ignore_header', fdu.getTag()).lower()
            if (dbpmlist.count(',') > 0):
                #comma separated list
                defaultBPMs = dbpmlist.split(',')
                removeEmpty(defaultBPMs)
                for j in range(len(defaultBPMs)):
                    defaultBPMs[j] = defaultBPMs[j].strip()
            elif (dbpmlist.endswith('.fit') or dbpmlist.endswith('.fits')):
                #FITS file given
                defaultBPMs.append(dbpmlist)
            elif (dbpmlist.endswith('.dat') or dbpmlist.endswith('.list') or dbpmlist.endswith('.txt')):
                #ASCII file list
                defaultBPMs = readFileIntoList(dbpmlist)
            for bpmfile in defaultBPMs:
                #Loop over list of bad pixel masks
                #badPixelMask = fatboyImage(bpmfile, log=self._log)
                badPixelMask = fatboyCalib(self._pname, "bad_pixel_mask", fdu, filename=bpmfile, log=self._log)
                #read header and initialize
                badPixelMask.readHeader()
                badPixelMask.initialize()
                if (ignoreHeader == "yes"):
                    #set filter, section to match.  One BPM will be appended below for each filter/section combination.
                    #BPM will then be found in #2 getMasterCalib above for other FDUs with same filter/section
                    badPixelMask.filter = fdu.filter
                    badPixelMask.section = fdu.section
                if (badPixelMask.filter is not None and badPixelMask.filter != fdu.filter):
                    #does not match filter
                    continue
                if (fdu.section is not None):
                    #check section if applicable
                    section = -1
                    if (badPixelMask.hasHeaderValue('SECTION')):
                        section = badPixelMask.getHeaderValue('SECTION')
                    else:
                        idx = badPixelMask.getFilename().rfind('.fit')
                        if (badPixelMask.getFilename()[idx-2] == 'S' and isDigit(badPixelMask.getFilename()[idx-1])):
                            section = int(badPixelMask.getFilename()[idx-1])
                    if (section != fdu.section):
                        continue
                badPixelMask.setType("bad_pixel_mask")
                #Found matching master flat
                print("badPixelMaskProcess::getCalibs> Using bad pixel mask "+badPixelMask.getFilename())
                self._log.writeLog(__name__, " Using bad pixel mask "+badPixelMask.getFilename())
                self._fdb.appendCalib(badPixelMask)
                calibs['badPixelMask'] = badPixelMask
                return calibs
        #4) Check bad_pixel_mask source before trying master_flat and master_dark.  Source could be master_flat or master_dark as well
        if (self.getOption('bad_pixel_mask_source', fdu.getTag()) is not None):
            source = self.getOption('bad_pixel_mask_source', fdu.getTag())
            #1) Check for an already created frame of type source matching section and TAGGED for this object
            #No need to check filter here as it should be TAGGED for this specific id
            sourceFDU = self._fdb.getTaggedMasterCalib(pname=None, ident=fdu._id, obstype=source, section=fdu.section)
            if (sourceFDU is None):
                #2) Check for an already created frame of type source matching filter/section
                sourceFDU = self._fdb.getMasterCalib(self._pname, obstype=source, filter=fdu.filter, section=fdu.section, tag=fdu.getTag())
            if (sourceFDU is None):
                #3) Check for an already created frame of type source matching exptime/nreads/section
                sourceFDU = self._fdb.getMasterCalib(self._pname, obstype=source, exptime=fdu.exptime, nreads=fdu.nreads, section=fdu.section, tag=fdu.getTag())
            if (sourceFDU is not None):
                #Found source.  Create bad pixel mask.
                #convenience method
                badPixelMask = self.createBadPixelMask(fdu, sourceFDU)
                if (badPixelMask is None):
                    return calibs
                self._fdb.appendCalib(badPixelMask)
                calibs['badPixelMask'] = badPixelMask
                return calibs
        #5) Try master_flat.  Already should have obtained master_flat above if it exists
        if ('masterFlat' in calibs):
            #Create bad pixel mask
            #convenience method
            print("badPixelMaskProcess::getCalibs> Creating bad pixel mask for filter: "+str(fdu.filter)+" using FLAT id="+str(calibs['masterFlat'].getFullId())+", orig filename="+str(calibs['masterFlat'].getFilename()))
            self._log.writeLog(__name__, " Creating bad pixel mask for filter: "+str(fdu.filter)+" using FLAT id="+str(calibs['masterFlat'].getFullId())+", orig filename="+str(calibs['masterFlat'].getFilename()))
            #Renormalize master flat to median value of 1
            calibs['masterFlat'].renormalize()
            badPixelMask = self.createBadPixelMask(fdu, calibs['masterFlat'])
            if (badPixelMask is None):
                return calibs
            self._fdb.appendCalib(badPixelMask)
            calibs['badPixelMask'] = badPixelMask
            return calibs
        #6) Try master_dark
        #Search for a TAGGED master_dark first
        masterDark = self._fdb.getTaggedMasterCalib(pname=None, ident=fdu._id, obstype="master_dark", exptime=fdu.exptime, nreads=fdu.nreads, section=fdu.section)
        if (masterDark is None):
            masterDark = self._fdb.getMasterCalib(pname=None, obstype="master_dark", exptime=fdu.exptime, nreads=fdu.nreads, section=fdu.section, tag=fdu.getTag())
        if (masterDark is not None):
            #Found source.  Create bad pixel mask.
            #convenience method
            print("badPixelMaskProcess::getCalibs> Creating bad pixel mask for filter: "+str(fdu.filter)+" using DARK id="+str(masterDark.getFullId())+", orig filename="+str(masterDark.getFilename()))
            self._log.writeLog(__name__, " Creating bad pixel mask for filter: "+str(fdu.filter)+" using DARK id="+str(masterDark.getFullId())+", orig filename="+str(masterDark.getFilename()))
            badPixelMask = self.createBadPixelMask(fdu, masterDark)
            if (badPixelMask is None):
                return calibs
            self._fdb.appendCalib(badPixelMask)
            calibs['badPixelMask'] = badPixelMask
            return calibs
        #7) Prompt user for source file
        print("List of master calibration frames, types, filters, exptimes, sections")
        masterCalibs = self._fdb.getMasterCalibs(obstype=fatboyDataUnit.FDU_TYPE_MASTER_CALIB)
        for mcalib in masterCalibs:
            print(mcalib.getFilename(), mcalib.getObsType(), mcalib.filter, mcalib.exptime, mcalib.section)
        tmp = input("Select a filename to use as a source to create bad pixel mask: ")
        calibfilename = tmp
        #Now find if input matches one of these filenames
        for mcalib in masterCalibs:
            if (mcalib.getFilename() == calibfilename):
                #Found matching master calib
                print("badPixelMaskProcess::getCalibs> Using master calib "+mcalib.getFilename())
                self._log.writeLog(__name__, " Using master calib "+mcalib.getFilename())
                #create bad pixel mask
                #convenience method
                badPixelMask = self.createBadPixelMask(fdu, mcalib)
                if (badPixelMask is None):
                    return calibs
                self._fdb.appendCalib(badPixelMask)
                calibs['badPixelMask'] = badPixelMask
                return calibs
            return calibs
    #end getCalibs

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('default_bad_pixel_mask', None)
        self._options.setdefault('default_bpm_ignore_header', 'no')
        self._options.setdefault('bad_pixel_mask_source', None)
        self._options.setdefault('clipping_method','values')
        self._optioninfo.setdefault('clipping_method', 'values | sigma')
        self._options.setdefault('clipping_high','2.0')
        self._options.setdefault('clipping_low','0.5')
        self._options.setdefault('clipping_sigma','5')
        self._options.setdefault('column_reject', None)
        self._optioninfo.setdefault('column_reject', 'supports slicing, e.g. 320:384, 500, 752:768')
        self._options.setdefault('edge_reject','5')
        self._options.setdefault('radius_reject', '0')
        self._options.setdefault('row_reject', None)
        self._optioninfo.setdefault('row_reject', 'supports slicing, e.g. 320:384, 500, 752:768')
    #end setDefaultOptions

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/badPixelMaskApplied", os.F_OK)):
            os.mkdir(outdir+"/badPixelMaskApplied",0o755)
        #Create output filename
        bafile = outdir+"/badPixelMaskApplied/ba_"+fdu.getFullId()
        #Check to see if it exists
        if (os.access(bafile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(bafile)
        if (not os.access(bafile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(bafile)
    #end writeOutput
