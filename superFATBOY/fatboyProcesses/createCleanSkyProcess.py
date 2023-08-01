from superFATBOY.fatboyDataUnit import fatboyDataUnit
from superFATBOY.fatboyImage import fatboyImage
from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.datatypeExtensions.fatboySpecCalib import fatboySpecCalib
from superFATBOY import gpu_imcombine, imcombine

#Create "clean sky" files from median of dark subtracted images for each object set
class createCleanSkyProcess(fatboyProcess):
    _modeTags = ["spectroscopy", "miradas"]

    #Convenience method so that code doesn't have to be rewritten several times in getCalib
    def createCleanSky(self, fdu, ds_fdus):
        cmethod = self.getOption("combine_method", fdu.getTag()).lower()
        #imcombine dark subtracted fdus at different RA and Dec to filter out continua and leave skylines
        csfilename = None
        csname = "cleanSkies/cleanSky_ds_"+fdu._id
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            #Optionally save if write_calib_output = yes
            if (not os.access(outdir+"/cleanSkies", os.F_OK)):
                os.mkdir(outdir+"/cleanSkies",0o755)
            csfilename = outdir+"/"+csname+".fits"
        #Check to see if clean sky frame exists already from a previous run
        prevcsfilename = outdir+"/"+csname+".fits"
        if (os.access(prevcsfilename, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(prevcsfilename)
        elif (os.access(prevcsfilename, os.F_OK)):
            #file already exists
            print("createCleanSkyProcess::createCleanSky> Clean sky "+prevcsfilename+" already exists!  Re-using...")
            self._log.writeLog(__name__, "Clean sky "+prevcsfilename+" already exists!  Re-using...")
            cleanSky = fatboySpecCalib(self._pname, "master_clean_sky", fdu, filename=prevcsfilename, tagname=csname, log=self._log)
            cleanSky.setProperty("specmode", fdu.getProperty("specmode"))
            cleanSky.setProperty("dispersion", fdu.getProperty("dispersion"))
            return cleanSky

        #Use imcombine to create clean sky file
        #Select cpu/gpu option
        if (self._fdb.getGPUMode()):
            if (cmethod == "min"):
                (data, header) = gpu_imcombine.imcombine(ds_fdus, outfile=csfilename, method="median", reject="minmax", nhigh=len(ds_fdus)-1, scale="none", nonzero=False, even=False, mef=fdu._mef, returnHeader=True, log=self._log)
            elif (cmethod == "quartile"):
                (data, header) = gpu_imcombine.imcombine(ds_fdus, outfile=csfilename, method="median", reject="minmax", nhigh=len(ds_fdus)//2, scale="none", nonzero=False, even=False, mef=fdu._mef, returnHeader=True, log=self._log)
            else:
                #median
                (data, header) = gpu_imcombine.imcombine(ds_fdus, outfile=csfilename, method="median", scale="none", nonzero=False, mef=fdu._mef, returnHeader=True, log=self._log)
        else:
            if (cmethod == "min"):
                (data, header) = imcombine.imcombine(ds_fdus, outfile=csfilename, method="median", reject="minmax", nhigh=len(ds_fdus)-1, scale="none", nonzero=False, mef=fdu._mef, returnHeader=True, log=self._log)
            elif (cmethod == "quartile"):
                (data, header) = imcombine.imcombine(ds_fdus, outfile=csfilename, method="median", reject="minmax", nhigh=len(ds_fdus)//2, scale="none", nonzero=False, mef=fdu._mef, returnHeader=True, log=self._log)
            else:
                #median
                (data, header) = imcombine.imcombine(ds_fdus, outfile=csfilename, method="median", scale="none", nonzero=False, mef=fdu._mef, returnHeader=True, log=self._log)
        #cleanSky = fatboySpecCalib(self._pname, "master_clean_sky", fdu, data=data, tagname="cleanSky_"+fdu._id, headerExt=header, log=self._log)
        cleanSky = fatboySpecCalib(self._pname, "master_clean_sky", fdu, data=data, tagname=csname, headerExt=header, log=self._log)
        cleanSky.setProperty("specmode", fdu.getProperty("specmode"))
        cleanSky.setProperty("dispersion", fdu.getProperty("dispersion"))
        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes" and not os.access(csfilename, os.F_OK)):
            #Optionally save if write_calib_output = yes
            cleanSky.writeTo(csfilename)
        return cleanSky
    #end createCleanSky

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        #Only run createCleanSkyProcess on objects, not calibs
        #Also run on standard stars
        if (not fdu.isObject and not fdu.isStandard):
            return True

        print("Create Clean Sky")
        print(fdu._identFull)

        #Call get calibs to return dict() of calibration frames.
        #For createCleanSky, this dict should have one entry 'cleanSky' which is an fdu
        calibs = self.getCalibs(fdu, prevProc)
        if (not 'cleanSky' in calibs):
            #Failed to obtain master clean sky frame
            #Issue error message but do not disable FDUs as they may still be able to be processed with arclamps
            print("createCleanSkyProcess::execute> Warning: Could not create Clean Sky frame for "+fdu.getFullId())
            self._log.writeLog(__name__, "Could not create Clean Sky frame for "+fdu.getFullId(), type=fatboyLog.WARNING)
            return False

        #Nothing else to do here as cleanSky has been added to database!
        return True
    #end execute

    ## OVERRIDE getCalibs
    def getCalibs(self, fdu, prevProc = None):
        calibs = dict()
        maxFrames = int(self.getOption('max_frames_to_combine'))

        csfilename = self.getCalib("cleanSky", fdu.getTag())
        if (csfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(csfilename, os.F_OK)):
                print("createCleanSkyProcess::getCalibs> Using clean sky frame "+csfilename+"...")
                self._log.writeLog(__name__, "Using clean sky frame "+csfilename+"...")
                calibs['cleanSky'] = fatboySpecCalib(self._pname, "master_clean_sky", fdu, filename=csfilename, tagname=csfilename, log=self._log)
                return calibs
            else:
                print("createCleanSkyProcess::getCalibs> Warning: Could not find clean sky frame "+csfilename+"...")
                self._log.writeLog(__name__, "Could not find clean sky frame "+csfilename+"...", type=fatboyLog.WARNING)

        headerVals = dict()
        headerVals['grism_keyword'] = fdu.grism

        properties = dict()
        properties['specmode'] = fdu.getProperty("specmode")

        #1) Check for an already created clean sky frame frame matching specmode/filter/grism/ident
        cleanSky = self._fdb.getMasterCalib(self._pname, ident = "cleanSky_ds_"+fdu._id, filter=fdu.filter, section=fdu.section, obstype="master_clean_sky", properties=properties, headerVals=headerVals, tag=fdu.getTag())
        if (cleanSky is not None):
            #Found clean sky frame.  Return here
            calibs['cleanSky'] = cleanSky
            return calibs
        #2) Check for individual FDUs matching specmode/filter/grism/ident to create clean sky frame
        ds_fdus = self._fdb.getFDUs(ident = fdu._id, filter=fdu.filter, section=fdu.section, properties=properties, headerVals=headerVals, tag=fdu.getTag())
        if (len(ds_fdus) > 0):
            #Found fdus.  Create clean sky frame.
            print("createCleanSkyProcess::getCalibs> Creating Clean Sky frame for "+fdu._id+" ...")
            self._log.writeLog(__name__, "Creating Clean Sky frame for "+fdu._id+" ...")
            if (len(ds_fdus) > maxFrames):
                #Only use maxFrames frames to combine
                print("createCleanSkyProcess::getCalibs> Found "+str(len(ds_fdus))+" frames.  Only using first "+str(maxFrames)+" frames to create Clean Sky frame.")
                self._log.writeLog(__name__, "Found "+str(len(ds_fdus))+" frames.  Only using first "+str(maxFrames)+" frames to create Clean Sky frame.")
                ds_fdus = ds_fdus[:maxFrames]
            #First recursively process (through dark subtraction presumably)
            self.recursivelyExecute(ds_fdus, prevProc)
            #convenience method
            cleanSky = self.createCleanSky(fdu, ds_fdus)
            self._fdb.appendCalib(cleanSky)
            calibs['cleanSky'] = cleanSky
            return calibs
        print("createCleanSkyProcess::getCalibs> Clean Sky frame for filter "+str(fdu.filter)+", grism "+str(fdu.grism)+", and ident "+str(fdu._id)+" not found!")
        self._log.writeLog(__name__, "Clean Sky frame for filter "+str(fdu.filter)+", grism "+str(fdu.grism)+", and ident "+str(fdu._id)+" not found!")
        #3) Check default_master_clean_sky for matching filter/grism
        defaultCleanSkies = []
        if (self.getOption('default_master_clean_sky', fdu.getTag()) is not None):
            dlist = self.getOption('default_master_clean_sky', fdu.getTag())
            if (dlist.count(',') > 0):
                #comma separated list
                defaultCleanSkies = dlist.split(',')
                removeEmpty(defaultCleanSkies)
                for j in range(len(defaultCleanSkies)):
                    defaultCleanSkies[j] = defaultCleanSkies[j].strip()
            elif (dlist.endswith('.fit') or dlist.endswith('.fits')):
                #FITS file given
                defaultCleanSkies.append(dlist)
            elif (dlist.endswith('.dat') or dlist.endswith('.list') or dlist.endswith('.txt')):
                #ASCII file list
                defaultCleanSkies = readFileIntoList(dlist)
            for csfile in defaultCleanSkies:
                #Loop over list of default clean sky frames
                cleanSky = fatboySpecCalib(self._pname, "master_clean_sky", fdu, filename=csfile, tagname=csfile, log=self._log)
                #read header and initialize
                cleanSky.readHeader()
                cleanSky.initialize()
                if (cleanSky.filter != fdu.filter):
                    #does not match filter
                    continue
                if (cleanSky.grism != fdu.grism):
                    #does not match grism
                    continue
                cleanSky.setType("master_clean_sky")
                #Found matching clean sky frame
                print("createCleanSkyProcess::getCalibs> Using default clean sky frame "+cleanSky.getFilename())
                self._fdb.appendCalib(cleanSky)
                calibs['cleanSky'] = cleanSky
                return calibs
        return calibs
    #end getCalibs

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('default_master_clean_sky', None)
        self._options.setdefault('combine_method', 'median')
        self._optioninfo.setdefault('combine_method', 'median | quartile | min')
        self._options.setdefault('max_frames_to_combine', '10')
    #end setDefaultOptions
