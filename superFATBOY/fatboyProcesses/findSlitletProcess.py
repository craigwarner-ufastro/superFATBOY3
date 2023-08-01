from superFATBOY.fatboyDataUnit import fatboyDataUnit
from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.datatypeExtensions.fatboySpecCalib import fatboySpecCalib

from superFATBOY import gpu_imcombine, imcombine
from numpy import *
from scipy.optimize import leastsq

usePlot = True
try:
    import matplotlib.pyplot as plt
except Exception as ex:
    print("Warning: Could not import matplotlib!")
    usePlot = False

block_size = 512

class findSlitletProcess(fatboyProcess):
    _modeTags = ["spectroscopy", "miradas"]

    #Attempt to auto-detect slitlets at a given x-value
    #instead of reading from a region file
    def autoDetectSlitlets(self, fdu, flatData, normal=False):
        #Read options
        boxsize = int(self.getOption("slitlet_autodetect_boxsize", fdu.getTag()))
        halfbox = boxsize//2
        sigma = float(self.getOption("slitlet_autodetect_sigma", fdu.getTag()))
        min_width = int(self.getOption("slitlet_autodetect_min_width", fdu.getTag()))
        x_auto = int(self.getOption("slitlet_autodetect_x", fdu.getTag()))
        use_peak_local_max = False
        if (self.getOption("autodetect_peak_local_max", fdu.getTag()).lower() == "yes"):
            use_peak_local_max = True
        fiber_width = int(self.getOption("fiber_width", fdu.getTag()))
        do_subtract_bkg = False
        if (self.getOption("subtract_background_level", fdu.getTag()).lower() == "yes"):
            do_subtract_bkg = True
        back_boxsize = int(self.getOption("background_boxcar_width", fdu.getTag()))
        back_halfbox = back_boxsize//2
        min_flux_pct = float(self.getOption("slitlet_autodetect_min_flux_pct", fdu.getTag()))
        use_median = False
        if (self.getOption("slitlet_autodetect_use_median", fdu.getTag()).lower() == "yes"):
            use_median = True
        debug = False
        if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
            debug = True
        writePlots = False
        if (self.getOption("write_plots", fdu.getTag()).lower() == "yes"):
            writePlots = True

        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            if (use_median):
                cut1d = gpu_arraymedian(flatData[:,x_auto-halfbox:x_auto+halfbox+1], axis="X").astype(float64)
            else:
                cut1d = flatData[:,x_auto-halfbox:x_auto+halfbox+1].sum(1).astype(float64)
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            if (use_median):
                cut1d = gpu_arraymedian(flatData[x_auto-halfbox:x_auto+halfbox+1,:], axis="Y").astype(float64)
            else:
                cut1d = flatData[x_auto-halfbox:x_auto+halfbox+1,:].sum(0).astype(float64)
        if (do_subtract_bkg):
            #Use running boxcar min function to subtract off background level
            #Create copy of cut1d to measure background as we update cut1d
            c2 = cut1d.copy()
            for j in range(len(cut1d)):
                #Protect against overflows
                x1 = max(0, j-back_halfbox)
                x2 = min(cut1d.size, j+back_halfbox+1)
                cut1d[j] -= c2[x1:x2].min()

        if (usePlot and (debug or writePlots)):
            plt.plot(cut1d)
            plt.xlabel('Pixel')
            plt.ylabel('Flux of 1-D Flat Field Cut')
            if (writePlots):
                #make directory if necessary
                outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
                if (not os.access(outdir+"/findSlitlets", os.F_OK)):
                    os.mkdir(outdir+"/findSlitlets",0o755)
                plt.savefig(outdir+"/findSlitlets/slits_"+fdu._id+".png", dpi=200)
            if (debug):
                print("boxsize", boxsize, "sigma", sigma, "min_width", min_width, "x_auto", x_auto, "normal", normal)
                plt.show()
            plt.close()

        if (use_peak_local_max):
            y = where(cut1d > median(cut1d))
            x = r_[True, cut1d[1:] > cut1d[:-1]] & r_[cut1d[:-1] > cut1d[1:], True] & r_[True, True, cut1d[2:] > cut1d[:-2]] & r_[cut1d[:-2] > cut1d[2:], True, True]
            x[:y[0][0]] = False
            x[y[0][-1]+1:] = False
            z = where(x)[0]
            sylo = z-fiber_width//2
            syhi = z+fiber_width//2
            slitx = array([x_auto]*len(sylo))
            slitw = array([boxsize]*len(sylo))
            return (sylo, syhi, slitx, slitw)

        if (normal):
            #if normalized, slitlets have already been found, easy to detect nonzero points
            slitlets = extractNonzeroRegions(cut1d, min_width)
        else:
            #use extractSpectra to find step function locations
            slitlets = extractSpectra(cut1d, sigma, min_width, minFluxPct=min_flux_pct)


        if (slitlets is None):
            #Return empty lists
            return([], [], [], [])
        sylo = slitlets[:,0]
        syhi = slitlets[:,1]
        slitx = array([x_auto]*len(sylo))
        slitw = array([boxsize]*len(sylo))

        if (self.getOption("slitlet_attempt_autocorrect", fdu.getTag()).lower() == "yes"):
            nslits_ref = int(self.getOption("slitlet_autodetect_nslits", fdu.getTag()))
            if (nslits_ref > 0 and nslits_ref != len(sylo)):
                print("findSlitletProcess::autoDetectSlitlets> Found "+str(len(sylo))+" slitlets instead of "+str(nslits_ref)+".  Attempting to autocorrect...")
                self._log.writeLog(__name__, "Found "+str(len(sylo))+" slitlets instead of "+str(nslits_ref)+".  Attempting to autocorrect...")
                swidth = syhi-sylo
                sgap = slitlets[1:,0]-slitlets[:-1,1]
                mwidth = gpu_arraymedian(swidth)
                mgap = gpu_arraymedian(sgap)
                wsig = abs((swidth-mwidth)/swidth.std())
                gsig = abs((sgap-mgap)/sgap.std())
                bw = where(wsig > 2)[0] #slit width > 2 sigma
                gw = where(wsig <= 2)[0]
                gg = where(gsig <= 2)[0]
                wsigg = swidth[gw].std() #std dev of "good" slitlets widths
                gsigg = sgap[gg].std() #std dev of "good" slitlets gaps
                possibleGapStart = False
                possibleGapEnd = False
                #convert slitlets to list
                slitlets = slitlets.tolist()
                for islit in bw:
                    if (abs(swidth[islit]/2.-mwidth)/wsigg < 2):
                        #Looks like a double slitlet
                        currWidth = int((swidth[islit]-int(mgap))/2)
                        if (islit == 0):
                            #Special case, first slitlet
                            slitlets.append([syhi[0]-currWidth, syhi[0]])
                            slitlets[0][1] = sylo[0]+currWidth
                        elif (islit == len(sylo)-1):
                            #special case, last slitlet
                            slitlets.append([syhi[islit]-currWidth, syhi[islit]])
                            slitlets[islit][1] = sylo[islit]+currWidth
                        elif (gsig[islit-1] < 2 and gsig[islit] < 2):
                            #verify that gaps before and after double slitlet are normal
                            slitlets.append([syhi[islit]-currWidth, syhi[islit]])
                            slitlets[islit][1] = sylo[islit]+currWidth
                    else:
                        if (islit == 0 and swidth[0] > mwidth and abs(sgap[0]-mgap)/gsigg < 3):
                            #First slitlet and gap 1-2 is normal - shrink slitlet
                            possibleGapStart = True
                            slitlets[0][0] = int(syhi[0]-mwidth)
                        elif (islit == len(sylo)-1 and swidth[islit] > mwidth and abs(sgap[islit]-mgap)/gsigg < 3):
                            #Last slitlet and gap n-1 to n is normal - shrink slitlet
                            possibleGapEnd = True
                            slitlets[islit][1] = int(sylo[islit]+mwidth)
                        elif (swidth[islit] > mwidth and abs(sgap[islit-1]-mgap)/gsigg < 3 and abs(sgap[islit]-mgap)/gsigg >= 3):
                            #low gap is normal, slit is too wide, high gap is big
                            slitlets[islit][1] = int(sylo[islit]+mwidth)
                        elif (swidth[islit] > mwidth and abs(sgap[islit]-mgap)/gsigg < 3 and abs(sgap[islit-1]-mgap)/gsigg >= 3):
                            #high gap is normal, slit is too wide, low gap is big
                            slitlets[islit][0] = int(syhi[islit]-mwidth)
                #Sort in case slitlets were added
                slitlets.sort()
                #recalc variables
                slitlets = array(slitlets)
                if (nslits_ref > 0 and nslits_ref > len(sylo)):
                    #look for gaps
                    sylo = slitlets[:,0]
                    syhi = slitlets[:,1]
                    swidth = syhi-sylo
                    sgap = slitlets[1:,0]-slitlets[:-1,1]
                    mwidth = gpu_arraymedian(swidth)
                    mgap = gpu_arraymedian(sgap)
                    wsig = abs((swidth-mwidth)/swidth.std())
                    gsig = abs((sgap-mgap)/sgap.std())
                    gw = where(wsig <= 2)[0]
                    bg = where(gsig > 2)[0]
                    gg = where(gsig <= 2)[0]
                    wsigg = swidth[gw].std() #std dev of "good" slitlets widths
                    gsigg = sgap[gg].std() #std dev of "good" slitlets gaps
                    #convert slitlets to list
                    slitlets = slitlets.tolist()
                    for islit in bg:
                        if (abs(sgap[islit]-mwidth-mgap)/wsigg <= 3):
                            #This gap fits the size of a slitlet
                            slitlets.append([int(syhi[islit]+mgap), int(sylo[islit+1]-mgap)])
                    if (nslits_ref > len(slitlets) and possibleGapStart):
                        slitlets.append([int(sylo[0]-mwidth-mgap), int(sylo[0]-mgap)])
                    if (nslits_ref > len(slitlets) and possibleGapEnd):
                        slitlets.append([int(syhi[-1]+mgap), int(syhi[-1]+mwidth+mgap)])
                    #Sort and convert to array
                    slitlets.sort()
                    slitlets = array(slitlets)
                sylo = slitlets[:,0]
                syhi = slitlets[:,1]
                slitx = array([x_auto]*len(sylo))
                slitw = array([boxsize]*len(sylo))
                print("findSlitletProcess::autoDetectSlitlets> After autocorrect, found "+str(len(sylo))+" slitlets...")
                self._log.writeLog(__name__, "After autocorrect, found "+str(len(sylo))+" slitlets...")

                if (self.getOption("slitlet_autocorrect_gap_size", fdu.getTag()) is not None):
                    #Check gaps
                    gapsize = int(self.getOption("slitlet_autocorrect_gap_size", fdu.getTag()))
                    #gapsize 0 -> sgap = 1
                    sgap = sylo[1:]-syhi[:-1]
                    swidth = syhi-sylo
                    while (sgap.max()-1 > gapsize):
                        for j in range(len(sgap)):
                            if (sgap[j]-1 > gapsize):
                                sgap[j] -= 1
                                if (swidth[j] > swidth[j+1]):
                                    sylo[j+1] -= 1
                                    swidth[j+1] += 1
                                else:
                                    syhi[j] += 1
                                    swidth[j] += 1
                    print("findSlitletProcess::autoDetectSlitlets> Autocorrected gapsize to "+str(gapsize)+"...")
                    self._log.writeLog(__name__, "Autocorrected gapsize to "+str(gapsize)+"...")

        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            #make directory if necessary
            outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
            if (not os.access(outdir+"/findSlitlets", os.F_OK)):
                os.mkdir(outdir+"/findSlitlets",0o755)
            #Create output filename
            regfile = outdir+"/findSlitlets/regions_"+fdu._id+".reg"
            writeRegionFile(regfile, sylo, syhi, slitx, slitw, horizontal=(fdu.dispersion == fdu.DISPERSION_HORIZONTAL))
            regxmlfile = outdir+"/findSlitlets/regions_"+fdu._id+".xml"
            writeRegionFileXML(regxmlfile, sylo, syhi, slitx, slitw, horizontal=(fdu.dispersion == fdu.DISPERSION_HORIZONTAL))

        return (sylo, syhi, slitx, slitw)
    #end autoDetectSlitlets

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        if (fdu._specmode == fdu.FDU_TYPE_LONGSLIT):
            #Skip longslit data
            return True

        print("Find Slitlets")
        print(fdu._identFull)

        #Call get calibs to return dict() of calibration frames.
        #For findSlitlets, this dict should have 3 entries: 'slitmask', 'slitlo', and 'slithi'
        #These are obtained by tracing slitlets using the master flat
        calibs = self.getCalibs(fdu, prevProc)
        #if ('slitmask' in calibs and 'slitlo' in calibs and 'slithi' in calibs):
        if ('slitmask' in calibs):
            #Found exisiting slitmask for this data.  Return here
            #1/8/18, don't need slitlo and slithi too (DFP).  If there, great but never used after this step
            return True

        if (not 'masterFlat' in calibs):
            #Failed to obtain master flat to trace out calibs
            #Issue error message and disable this FDU
            print("findSlitletProcess::execute> ERROR: Slitlets not traced for "+fdu.getFullId()+" (filter="+str(fdu.filter)+").  Discarding Image!")
            self._log.writeLog(__name__, "Slitlets not traced for "+fdu.getFullId()+" (filter="+str(fdu.filter)+").  Discarding Image!", type=fatboyLog.ERROR)
            #disable this FDU
            fdu.disable()
            return False

        if (self.getOption("trace_slitlets_individually", fdu.getTag()).lower() == "yes"):
            #call traceOrders function to trace out individual echelle orders
            calibs = self.traceOrders(fdu, calibs)
        else:
            #call traceSlitlets function to trace out slitlets as a group
            calibs = self.traceSlitlets(fdu, calibs)
        #Append to database
        if ('slitmask' in calibs and 'slitlo' in calibs and 'slithi' in calibs):
            self._fdb.appendCalib(calibs['slitmask'])
            self._fdb.appendCalib(calibs['slitlo'])
            self._fdb.appendCalib(calibs['slithi'])
        else:
            #Failed to obtain all 3 calibration frames
            #Issue error message and disable this FDU
            print("findSlitletProcess::execute> ERROR: Slitlets not traced for "+fdu.getFullId()+" (filter="+str(fdu.filter)+").  Discarding Image!")
            self._log.writeLog(__name__, "Slitlets not traced for "+fdu.getFullId()+" (filter="+str(fdu.filter)+").  Discarding Image!", type=fatboyLog.ERROR)
            #disable this FDU
            fdu.disable()
            return False

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
                print("findSlitletProcess::getCalibs> Using slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Using slitmask "+smfilename+"...")
                calibs['slitmask'] = fatboySpecCalib(self._pname, "slitmask", fdu, filename=smfilename, log=self._log)
            else:
                print("findSlitletProcess::getCalibs> Warning: Could not find slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Could not find slitmask "+smfilename+"...", type=fatboyLog.WARNING)
        slfilename = self.getCalib("slitlo", fdu.getTag())
        if (slfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(slfilename, os.F_OK)):
                print("findSlitletProcess::getCalibs> Using slitlo "+slfilename+"...")
                self._log.writeLog(__name__, "Using slitlo "+slfilename+"...")
                calibs['slitlo'] = fatboySpecCalib(self._pname, "slitlo", fdu, filename=slfilename, log=self._log)
            else:
                print("findSlitletProcess::getCalibs> Warning: Could not find slitlo "+slfilename+"...")
                self._log.writeLog(__name__, "Could not find slitlo "+slfilename+"...", type=fatboyLog.WARNING)
        shfilename = self.getCalib("slithi", fdu.getTag())
        if (shfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(shfilename, os.F_OK)):
                print("findSlitletProcess::getCalibs> Using slithi "+shfilename+"...")
                self._log.writeLog(__name__, "Using slithi "+shfilename+"...")
                calibs['slithi'] = fatboySpecCalib(self._pname, "slithi", fdu, filename=shfilename, log=self._log)
            else:
                print("findSlitletProcess::getCalibs> Warning: Could not find slithi "+shfilename+"...")
                self._log.writeLog(__name__, "Could not find slithi "+shfilename+"...", type=fatboyLog.WARNING)

        if ('slitmask' in calibs and 'slitlo' in calibs and 'slithi' in calibs):
            #All 3 calibs passed in from XML
            return calibs

        #Look for matching grism_keyword, specmode, and dispersion
        headerVals = dict()
        headerVals['grism_keyword'] = fdu.grism

        properties = dict()
        properties['specmode'] = fdu.getProperty("specmode")
        properties['dispersion'] = fdu.getProperty("dispersion")

        #1) Check for already created calibs matching specmode/filter/grism but NOT ident unless its tagged
        if (not 'slitmask' in calibs):
            #Use new fdu.getSlitmask method
            slitmask = fdu.getSlitmask(pname=self._pname, properties=properties, headerVals=headerVals)
            if (slitmask is not None):
                #Found slitmask
                calibs['slitmask'] = slitmask
        if (not 'slitlo' in calibs):
            #1a) check for an already created slitlo matching specmode/filter/grism and TAGGED for this object
            slitlo = self._fdb.getTaggedMasterCalib(self._pname, fdu._id, obstype="slitlo", filter=fdu.filter, properties=properties, headerVals=headerVals)
            if (slitlo is None):
                #1b) check for an already created slitlo matching specmode/filter/grism
                slitlo = self._fdb.getMasterCalib(self._pname, filter=fdu.filter, obstype="slitlo", properties=properties, headerVals=headerVals, tag=fdu.getTag())
            if (slitlo is not None):
                #Found slitlo
                calibs['slitlo'] = slitlo
        if (not 'slithi' in calibs):
            #1a) check for an already created slithi matching specmode/filter/grism and TAGGED for this object
            slithi = self._fdb.getTaggedMasterCalib(self._pname, fdu._id, obstype="slithi", filter=fdu.filter, properties=properties, headerVals=headerVals)
            if (slithi is None):
                #1b) check for an already created slithi matching specmode/filter/grism
                slithi = self._fdb.getMasterCalib(self._pname, filter=fdu.filter, obstype="slithi", properties=properties, headerVals=headerVals, tag=fdu.getTag())
            if (slithi is not None):
                #Found slithi
                calibs['slithi'] = slithi
        #if ('slitmask' in calibs and 'slitlo' in calibs and 'slithi' in calibs):
        if ('slitmask' in calibs):
            #1/8/18, don't need slitlo and slithi too (DFP).  If there, great but never used after this step
            return calibs

        #2) Check for masterFlat, create if necessary, and trace slitlets
        ##First check for calib passed from XML
        mffilename = self.getCalib("masterFlat", fdu.getTag())
        if (mffilename is not None):
            #passed from XML with <calib> tag.  Use as master flat to trace slitmask
            if (os.access(mffilename, os.F_OK)):
                print("findSlitletProcess::getCalibs> Using master flat "+mffilename+" to create slitmask...")
                self._log.writeLog(__name__, "Using master flat "+mffilename+" to create slitmask...")
                calibs['masterFlat'] = fatboySpecCalib(self._pname, "master_flat", fdu, filename=mffilename, log=self._log)
            else:
                print("findSlitletProcess::getCalibs> Warning: Could not find master flat "+mffilename+"...")
                self._log.writeLog(__name__, "Could not find master flat "+mffilename+"...", type=fatboyLog.WARNING)

        if (not 'masterFlat' in calibs):
            #Use flatDivideSpecProcess.getCalibs to get masterFlat and create if necessary
            #Use method getProcessByName to return instantiated version of process.  Only works if process is included in XML file.
            #Returns None on a failure
            fds_process = self._fdb.getProcessByName("flatDivideSpec")
            if (fds_process is None or not isinstance(fds_process, fatboyProcess)):
                print("findSlitletProcess::getCalibs> ERROR: could not find process flatDivideSpec!  Check your XML file!")
                self._log.writeLog(__name__, "could not find process flatDivideSpec!  Check your XML file!", type=fatboyLog.ERROR)
                return calibs
            #Call setDefaultOptions and getCalibs on flatDivideSpecProcess
            fds_process.setDefaultOptions()
            calibs = fds_process.getCalibs(fdu, prevProc)

        if (not 'masterFlat' in calibs):
            #Failed to obtain master flat frame
            #Issue error message.  FDU will be disabled in execute()
            print("findSlitletProcess::execute> ERROR: Master flat not found for "+fdu.getFullId()+" (filter="+str(fdu.filter)+")!")
            self._log.writeLog(__name__, "Master flat not found for "+fdu.getFullId()+" (filter="+str(fdu.filter)+")!", type=fatboyLog.ERROR)
            return calibs

        if (self.getOption("trace_slitlets_individually", fdu.getTag()).lower() == "yes"):
            #call traceOrders function to trace out individual echelle orders
            calibs = self.traceOrders(fdu, calibs)
        elif (self.getOption("trace_peak_local_max", fdu.getTag()).lower() == "yes"):
            #call tracePeakLocalMax function to trace out individual fibers
            calibs = self.tracePeakLocalMax(fdu, calibs)
        else:
            #call traceSlitlets function to trace out slitlets as a group
            calibs = self.traceSlitlets(fdu, calibs)
        #Append to database
        if ('slitmask' in calibs and 'slitlo' in calibs and 'slithi' in calibs):
            self._fdb.appendCalib(calibs['slitmask'])
            self._fdb.appendCalib(calibs['slitlo'])
            self._fdb.appendCalib(calibs['slithi'])
        return calibs
    #end getCalibs

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('debug_mode', 'no')
        self._options.setdefault('autodetect_peak_local_max', 'no')
        self._optioninfo.setdefault('autodetect_peak_local_max', 'For fiber data such as MEGARA,\nuse peak local max to find fiber locations')
        self._options.setdefault('background_boxcar_width', 25)
        self._optioninfo.setdefault('background_boxcar_width', 'Width in pixels of the boxcar used to subtract off background level\nin 1-d cut.  Should be just under 2 x slit width.')
        self._options.setdefault('edge_threshold', 15)
        self._optioninfo.setdefault('edge_threshold', 'Do not attempt to trace out slitlets within this many pixesl of edges')
        self._options.setdefault('fiber_width', '5')
        self._optioninfo.setdefault('fiber_width', 'Width of fibers, used with peak local max')
        self._options.setdefault('fit_order', '2')
        self._optioninfo.setdefault('fit_order', 'Order of polynomial to use to fit slitlet shape.\nRecommended value = 2 for trace_slitlets_individually, 3 for group mode')
        self._options.setdefault('invert_before_correlating', 'no')
        self._optioninfo.setdefault('invert_before_correlating', 'Invert flat field to turn gap trough into a peak for cross correlations')

        self._options.setdefault('n_segments', '1')
        self._optioninfo.setdefault('n_segments', 'Number of piecewise functions to fit.  Should be 2 for MIRADAS, 1 for most other cases.')
        self._options.setdefault('order_step_size', '5')
        self._optioninfo.setdefault('order_step_size', 'Step size in pixels for tracing out orders, default = 5.')

        self._options.setdefault('padding','0')
        self._optioninfo.setdefault('padding', 'Number of pixels to pad slitlets by.  Default=0')
        self._options.setdefault('region_file', None)
        self._optioninfo.setdefault('region_file', '.reg, .xml, or .txt file describing slitlets')
        self._options.setdefault('slitlet_attempt_autocorrect', 'no')
        self._optioninfo.setdefault('slitlet_attempt_autocorrect', 'If slitlets found does not match slitlet_autodetect_nslits\nattempt to auto-correct before failing.')
        self._options.setdefault('slitlet_autocorrect_gap_size', None)
        self._optioninfo.setdefault('slitlet_autocorrect_gap_size', 'Correct auto-detected slitlets to have uniform gaps between\nslitlets of this size.')
        self._options.setdefault('slitlet_autodetect_nslits', '0')
        self._optioninfo.setdefault('slitlet_autodetect_nslits', 'Set this to the number of slitlets if auto-detecting them\nas a check that it found the correct number\nof slitlets (0 = no check)')
        self._options.setdefault('slitlet_autodetect_boxsize', '5')
        self._optioninfo.setdefault('slitlet_autodetect_boxsize', 'Boxsize for auto-detecting slitlets')
        self._options.setdefault('slitlet_autodetect_min_flux_pct', '0.001')
        self._optioninfo.setdefault('slitlet_autodetect_min_flux_pct', 'When flux drops below this percent of max, force break between slitlets')
        self._options.setdefault('slitlet_autodetect_min_width', '10')
        self._optioninfo.setdefault('slitlet_autodetect_min_width', 'Minimum width of a slitlet for auto-detection')
        self._options.setdefault('slitlet_autodetect_sigma', '5')
        self._optioninfo.setdefault('slitlet_autodetect_sigma', 'Minimum sigma vs local noise to be a step\nfor slitlet detection')
        self._options.setdefault('slitlet_autodetect_use_median', 'no')
        self._optioninfo.setdefault('slitlet_autodetect_use_median', 'Set to yes to use median rather than sum for auto detection')
        self._options.setdefault('slitlet_autodetect_x', '1024')
        self._optioninfo.setdefault('slitlet_autodetect_x', 'Central pixel in continuum direction for auto-detecting\nslitlets if no region file.')
        self._options.setdefault('slitlet_trace_boxsize', '21')
        self._optioninfo.setdefault('slitlet_trace_boxsize', 'Boxsize in cross-dispersion direction of 1-d cut for tracing in individual mode')
        self._options.setdefault('slitlet_trace_ylo', '-1')
        self._optioninfo.setdefault('slitlet_trace_ylo', 'Lower bound in cross-dispersion direction of 1-d cut for tracing in group mode (-1 = 1/4 ysize)')
        self._options.setdefault('slitlet_trace_yhi', '-1')
        self._optioninfo.setdefault('slitlet_trace_yhi', 'Upper bound in cross-dispersion direction of 1-d cut for tracing in group mode (-1 = 3/4 ysize)')
        self._options.setdefault('subtract_background_level', 'no')
        self._optioninfo.setdefault('subtract_background_level', 'Subtract a running boxcar min from the 1-d cut\tbefore attempting to find slitlets')
        self._options.setdefault('trace_peak_local_max', 'no')
        self._optioninfo.setdefault('trace_peak_local_max', 'Set to yes for MEGARA or other fiber data where the curvature changes between fibers')
        self._options.setdefault('trace_slitlets_individually', 'yes')
        self._optioninfo.setdefault('trace_slitlets_individually', 'Set to yes for echelle spectra where the curvature changes between slitlets.')
        self._options.setdefault('write_plots', 'no')
    #end setDefaultOptions

    ## Trace out individual echelle orders
    def traceOrders(self, fdu, calibs):
        ###*** For purposes of traceOrders algorithm, X = dispersion direction and Y = cross-dispersion direction ***###
        ###*** It will trace out and fit Y = f(X) ***###
        #Get masterFlat
        masterFlat = calibs['masterFlat']
        #Read options
        boxsize = int(self.getOption("slitlet_trace_boxsize", fdu.getTag()))
        halfbox = boxsize//2
        order = int(self.getOption("fit_order", fdu.getTag()))
        padding = int(self.getOption("padding", fdu.getTag()))
        #Get region file for this FDU
        if (fdu.hasProperty("region_file")):
            regFile = fdu.getProperty("region_file")
        else:
            regFile = self.getCalib("region_file", fdu.getTag())
        do_subtract_bkg = False
        if (self.getOption("subtract_background_level", fdu.getTag()).lower() == "yes"):
            do_subtract_bkg = True
        do_invert = False
        if (self.getOption("invert_before_correlating", fdu.getTag()).lower() == "yes"):
            do_invert = True
        edge_thresh = int(self.getOption("edge_threshold", fdu.getTag()))
        n_segments = int(self.getOption("n_segments", fdu.getTag()))
        step = int(self.getOption('order_step_size', fdu.getTag()))

        #Check that region file exists
        if (regFile is None or not os.access(regFile, os.F_OK)):
            #If not, attempt to auto-detect slitlets!
            print("findSlitletProcess::traceOrders> No region file given.  Attempting to auto-detect slitlets...")
            self._log.writeLog(__name__, "No region file given.  Attempting to auto-detect slitlets...")
            isNormalized = False
            if (masterFlat.hasProperty("normalized") or masterFlat.hasHeaderValue('NORMAL01')):
                #has been normalized already
                isNormalized = True
            (sylo, syhi, slitx, slitw) = self.autoDetectSlitlets(fdu, masterFlat.getData().copy(), normal=isNormalized)

            nslits = len(sylo)
            nslits_ref = int(self.getOption("slitlet_autodetect_nslits", fdu.getTag()))
            print("findSlitletProcess::traceOrders> Found "+str(nslits)+" slitlets: "+str(list(zip(sylo, syhi))))
            self._log.writeLog(__name__, "Found "+str(nslits)+" slitlets: "+str(list(zip(sylo, syhi))))

            if ((nslits_ref > 0 and nslits != nslits_ref) or nslits == 0):
                print("findSlitletProcess::traceOrders> ERROR: Could not find region file associated with "+fdu.getFullId()+" and auto-detect found incorrect number of slitlets! Discarding Image!")
                self._log.writeLog(__name__, "Could not find region file associated with "+fdu.getFullId()+" and auto-detect found incorrect number of slitlets!  Discarding Image!", type=fatboyLog.ERROR)
                #disable this FDU
                fdu.disable()
                return calibs
        else:
            #Read region file
            if (regFile.endswith(".reg")):
                (sylo, syhi, slitx, slitw) = readRegionFile(regFile, horizontal = (fdu.dispersion == fdu.DISPERSION_HORIZONTAL), log=self._log)
            elif (regFile.endswith(".txt")):
                (sylo, syhi, slitx, slitw) = readRegionFileText(regFile, horizontal = (fdu.dispersion == fdu.DISPERSION_HORIZONTAL), log=self._log)
            elif (regFile.endswith(".xml")):
                (sylo, syhi, slitx, slitw) = readRegionFileXML(regFile, horizontal = (fdu.dispersion == fdu.DISPERSION_HORIZONTAL), log=self._log)
            else:
                print("findSlitletProcess::traceOrders> ERROR: Invalid extension for region file "+regFile+"! Discarding Image "+fdu.getFullId()+"!")
                self._log.writeLog(__name__, "Invalid extension for region file "+regFile+"! Discarding Image "+fdu.getFullId()+"!", type=fatboyLog.ERROR)
                #disable this FDU
                fdu.disable()
                return calibs
        #Check nslits
        nslits = len(sylo)
        if (nslits == 0):
            print("findSlitletProcess::traceOrders> ERROR: Could not parse region file "+regFile+"! Discarding Image "+fdu.getFullId()+"!")
            self._log.writeLog(__name__, "Could not parse region file "+regFile+"! Discarding Image "+fdu.getFullId()+"!", type=fatboyLog.ERROR)
            #disable this FDU
            fdu.disable()
            return calibs

        #Check to see if slitmask already exists
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        #Check to see if slitmask / slithi / slitlo exist already from a previous run
        mfsuffix = masterFlat.getFullId()
        if (not os.access(outdir+"/findSlitlets/slitmask_"+mfsuffix, os.F_OK) and os.access(outdir+"/findSlitlets/slitmask_"+masterFlat._id+".fits", os.F_OK)):
            mfsuffix = masterFlat._id+".fits"
        slitfile = outdir+"/findSlitlets/slitmask_"+mfsuffix
        slitlofile = outdir+"/findSlitlets/slitlo_"+mfsuffix
        slithifile = outdir+"/findSlitlets/slithi_"+mfsuffix
        if (self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "no"):
            if (os.access(slitfile, os.F_OK) and os.access(slitlofile, os.F_OK) and os.access(slithifile, os.F_OK)):
                #files already exists
                #Use master flat as source header
                print("findSlitletProcess::traceOrders> Slitmask "+slitfile+" already exists!  Re-using...")
                self._log.writeLog(__name__, "Slitmask "+slitfile+" already exists!  Re-using...")
                slitmask = fatboySpecCalib(self._pname, "slitmask", masterFlat, filename=slitfile, tagname="slitmask_"+masterFlat._id, log=self._log)
                slitmask.setProperty("specmode", fdu.getProperty("specmode"))
                slitmask.setProperty("dispersion", fdu.getProperty("dispersion"))
                slitmask.setProperty("regions", (sylo, syhi, slitx, slitw))
                slitmask.setProperty("nslits", nslits)
                calibs['slitmask'] = slitmask
                print("findSlitletProcess::traceOrders> Slitlo "+slitlofile+" already exists!  Re-using...")
                self._log.writeLog(__name__, "Slitlo "+slitlofile+" already exists!  Re-using...")
                slitlo = fatboySpecCalib(self._pname, "slitlo", masterFlat, filename=slitlofile, tagname="slitlo_"+masterFlat._id, log=self._log)
                slitlo.setProperty("specmode", fdu.getProperty("specmode"))
                slitlo.setProperty("dispersion", fdu.getProperty("dispersion"))
                calibs['slitlo'] = slitlo
                print("findSlitletProcess::traceOrders> Slithi "+slithifile+" already exists!  Re-using...")
                self._log.writeLog(__name__, "Slithi "+slithifile+" already exists!  Re-using...")
                slithi = fatboySpecCalib(self._pname, "slithi", masterFlat, filename=slithifile, tagname="slithi_"+masterFlat._id, log=self._log)
                slithi.setProperty("specmode", fdu.getProperty("specmode"))
                slithi.setProperty("dispersion", fdu.getProperty("dispersion"))
                calibs['slithi'] = slithi
                return calibs

        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            xsize = fdu.getShape()[1]
            ysize = fdu.getShape()[0]
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            ##xsize should be size across dispersion direction
            xsize = fdu.getShape()[0]
            ysize = fdu.getShape()[1]
        #Get xstride
        xstride = xsize//n_segments
        #Get data from master flat
        flatData = masterFlat.getData().copy()
        qaData = flatData.copy()
        #Slits can't extend beyond image top/bottom
        for j in range(nslits):
            sylo[j] = max(sylo[j], edge_thresh)
            syhi[j] = min(syhi[j], ysize-edge_thresh)

        #Set up yloMask and yhiMask arrays to track low and high values of each slitlet
        yloMask = zeros((nslits, xsize))
        yhiMask = zeros((nslits, xsize))

        #If CPU mode, will need to create slitmask as we go
        if (not self._fdb.getGPUMode()):
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                #Generate y index array
                slitmask = zeros((ysize,xsize), dtype=int32)
                yind = arange(xsize*ysize, dtype=int32).reshape(ysize,xsize)//xsize
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                #Generate x index array
                xind = arange(xsize*ysize, dtype=int32).reshape(xsize,ysize)%ysize
                slitmask = zeros((xsize,ysize), dtype=int32)

        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/findSlitlets", os.F_OK)):
            os.mkdir(outdir+"/findSlitlets",0o755)
        statsfile = outdir+"/findSlitlets/stats_"+masterFlat._id+".txt"
        f = open(statsfile,'w')
        #Loop over each slitlet and trace out top and bottom of slitlet
        t = time.time()
        for slitidx in range(nslits):
            #Trace out both "lower" and "higher" edges of slitlet
            yvals = [sylo[slitidx], syhi[slitidx]]
            #z1 holds zero point corrected output results of traces
            z1 = []
            #Process ylo and yhi for each slitlet
            #syval = slit y-value

            ##For inidividual slitlets, start at given slitx value instead of in middle
            xinit = int(slitx[slitidx])
            #step = 5
            ##xs = x values (dispersion direction) to cross correlate at
            ##Start at slitx value for each slitlet and trace to end then to beginning
            ##Set this up individually for each slitlet
            xs = list(range(xinit, xsize-10, step))+list(range(xinit-step, 10, -1*step))

            if (n_segments > 0):
                #Create xs piecewise if multiple segments
                bndry = 10
                xs = list(range(xinit,xstride*(xinit//xstride+1)-bndry, step))+list(range(xinit-step, xstride*(xinit//xstride)+bndry, -1*step))
                first_seg = xinit//xstride
                #Piece together in consecutively higher then consecutively lower segments
                for seg in range(first_seg+1, n_segments):
                    #Higher x vals
                    xs += list(range(xstride*seg+bndry, xstride*(seg+1)-bndry, step))
                for seg in range(first_seg-1, -1, -1):
                    #Lower x vals
                    xs += list(range(xstride*(seg+1)-bndry, xstride*seg+bndry, -1*step))

            for syval in yvals:
                #1-d cut of central 11 pixels of flat in cross-dispersion direction
                #Only look at 21 pixel box in dispersion direction => 21x11 box => 21 pixel 1-d line
                ylo_slit = syval-halfbox
                yoff_slit = 0
                if (ylo_slit < 0):
                    yoff_slit = ylo_slit
                    ylo_slit = 0
                elif (ylo_slit > ysize-boxsize):
                    yoff_slit = ylo_slit-(ysize-boxsize)
                    ylo_slit = ysize-boxsize
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    #islit = flatData[syval-halfbox:syval+halfbox+1, xinit-5:xinit+6].sum(1).astype(float64)
                    islit = flatData[ylo_slit:ylo_slit+boxsize, xinit-5:xinit+6].sum(1).astype(float64)
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    #islit = flatData[xinit-5:xinit+6, syval-halfbox:syval+halfbox+1].sum(0).astype(float64)
                    islit = flatData[xinit-5:xinit+6, ylo_slit:ylo_slit+boxsize].sum(0).astype(float64)
                if (do_subtract_bkg):
                    islit -= islit.min()
                if (do_invert):
                    islit = (islit.max()-islit)**2
                    islit = medianfilterCPU(islit)
                    islit[islit < 0] = 0

                #Find shifts between segments
                seg_shifts = []
                first_seg = xinit//xstride
                for seg in range(n_segments):
                    if (seg == first_seg):
                        seg_shifts.append(0)
                    else:
                        y1 = max(0, ylo_slit-boxsize)
                        y2 = min(ysize, ylo_slit+2*boxsize)
                        if (seg > first_seg):
                            x1 = xstride*seg+bndry
                            x2 = xstride*seg+bndry+50
                            x3 = xstride*seg-bndry-50
                            x4 = xstride*seg-bndry
                        else:
                            x1 = xstride*(seg+1)-bndry-50
                            x2 = xstride*(seg+1)-bndry
                            x3 = xstride*(seg+1)+bndry
                            x4 = xstride*(seg+1)+bndry+50
                        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                            oned_seg1 = flatData[y1:y2, x1:x2].sum(1).astype(float64)
                            oned_seg0 = flatData[y1:y2, x3:x4].sum(1).astype(float64)
                        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                            oned_seg1 = flatData[x1:x2, y1:y2].sum(0).astype(float64)
                            oned_seg0 = flatData[x3:x4, y1:y2].sum(0).astype(float64)
                        ccor = correlate(oned_seg0, oned_seg1, mode='same')
                        mcor = where(ccor == max(ccor))[0]
                        seg_shifts.append(len(ccor)//2-mcor[0])


                #Setup lists and arrays for within each loop
                #xcoords and ycoords contain lists of fit (x,y) points
                xcoords = []
                ycoords = []
                #median value of 1-d cuts and max values of cross correlations are kept and used as rejection criteria later
                meds = []
                maxcors = []
                #Up to last 10 (x,y) pairs are kept and used in various rejection criteria
                lastXs = []
                lastYs = []
                currX = xs[0] #current X value
                currY = syval #shift in cross-dispersion direction at currX relative to Y at X=xinit

                lastSeg = first_seg
                #Loop over xs every 5 pixels and cross correlate 1-d cut with islit
                for j in range(len(xs)):
                    currSeg = xs[j]//xstride
                    if (xs[j] == xinit-step):
                        #We have finished tracing to the end, starting back at middle to trace in other direction
                        #Reset currY, lastYs, lastXs
                        currY = syval
                        lastYs = [syval]
                        lastXs = [xinit]
                    elif (currSeg != lastSeg):
                        lastIdx = where(abs(array(xcoords)-xs[j]) == min(abs(array(xcoords)-xs[j])))[0][0]
                        currY = ycoords[lastIdx]+seg_shifts[currSeg]
                        lastYs = [ycoords[lastIdx]+seg_shifts[currSeg]]
                        lastXs = [xcoords[lastIdx]]
                    if (currY < edge_thresh):
                        #This slitlet is nearing the edge of the chip.  Don't try to fit anymore values.
                        #Use values that have been fit already to trace it out
                        continue

                    intY = int(round(currY, 3))
                    ylo_slit = intY-halfbox
                    if (ylo_slit < 0):
                        ylo_slit = 0
                    elif (ylo_slit > ysize-boxsize):
                        ylo_slit = ysize-boxsize
                    #1-d cut of flat in cross-dispersion direction, sum of 11 pixels in dispersion direction centered at current X
                    #Only look at 21 pixel box in dispersion direction centered at currY  => 21x11 box => 21 pixel 1-d line
                    if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                        #cut1d = flatData[intY-halfbox:intY+halfbox+1, int(xs[j]-5):int(xs[j]+6)].sum(1).astype(float64)
                        cut1d = flatData[ylo_slit:ylo_slit+boxsize, int(xs[j]-5):int(xs[j]+6)].sum(1).astype(float64)
                    elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                        #cut1d = flatData[int(xs[j]-5):int(xs[j]+6), intY-halfbox:intY+halfbox+1].sum(0).astype(float64)
                        cut1d = flatData[int(xs[j]-5):int(xs[j]+6), ylo_slit:ylo_slit+boxsize].sum(0).astype(float64)
                    if (do_subtract_bkg):
                        cut1d -= cut1d.min()
                    if (do_invert):
                        cut1d = (cut1d.max()-cut1d)**2  #square
                        cut1d = medianfilterCPU(cut1d)
                        cut1d[cut1d < 0] = 0
                    #Check that there is data in this cut
                    if (cut1d.sum()/islit.sum() < 0.05):
                        f.write(str(slitidx)+'\t'+str(syval)+'\t'+str(xs[j])+'\t'+str(currY)+'\t6\n')
                        #Flux in cut1 is less than 5% of that in islit reference cut
                        continue
                    q1 = gpu_arraymedian(cut1d, nhigh=len(cut1d)//2) #quartile
                    if (cut1d.max()/q1 < 3):
                        f.write(str(slitidx)+'\t'+str(syval)+'\t'+str(xs[j])+'\t'+str(currY)+'\t7\n')
                        #Flux in cut1 is less than 5% of that in islit reference cut
                        continue
                    #Cross correlate cut1d with islit
                    #Use numpy correlate since 1d cut -- not enough pixels to benefit from GPU
                    ccor = correlate(cut1d, islit, mode='same')
                    #Median filter with 51 pixel boxcar and set negative values to 0 before fitting
                    ccor = medianfilterCPU(ccor)
                    ccor[ccor < 0] = 0
                    #Use leastsq to fit Gaussian to cross-correlation function
                    p = zeros(4, float64)
                    #p[1] = round(currY,3) #center = currY
                    p[1] = round(currY, 3)-ylo_slit
                    p[2] = 3. #FWHM = 3
                    p[3] = 0.
                    p[0] = max(ccor)
                    #lsq argument should be centered at currY
                    #Use int(round(currY, 3)) to get around floating point bug
                    lsq = fitGaussian(ccor, maskNeg=True, guess=p)
                    if (lsq[1] == False):
                    #try:
                    #  lsq = leastsq(gaussResiduals, p, args=(arange(len(ccor))+intY-halfbox, ccor))
                    #except Exception as ex:
                        print("findSlitletProcess::traceOrders> Warning: Order "+str(slitidx)+", syval="+str(syval)+": Leastsq FAILED at "+str(xs[j])+" with "+str(ex))
                        self._log.writeLog(__name__, "Order "+str(slitidx)+", syval="+str(syval)+": Leastsq FAILED at "+str(xs[j])+" with "+str(ex), type=fatboyLog.WARNING)
                        continue
                    lsq[0][1] += ylo_slit+yoff_slit
                    #Error checking results of leastsq call
                    if (lsq[1] == 5):
                        f.write(str(slitidx)+'\t'+str(syval)+'\t'+str(xs[j])+'\t'+str(lsq[0][1])+'\t1\n')
                        #exceeded max number of calls = ignore
                        continue
                    if (lsq[0][0]+lsq[0][3] < 0):
                        f.write(str(slitidx)+'\t'+str(syval)+'\t'+str(xs[j])+'\t'+str(lsq[0][1])+'\t2\n')
                        #flux less than zero = ignore
                        continue
                    if (lsq[0][2] < 0 and j != 0):
                        f.write(str(slitidx)+'\t'+str(syval)+'\t'+str(xs[j])+'\t'+str(lsq[0][1])+'\t3\n')
                        #negative boxsize = ignore unless first datapoint
                        continue
                    if (j == 0):
                        #First datapoint -- update currX, currY, append to all lists
                        f.write(str(slitidx)+'\t'+str(syval)+'\t'+str(xs[j])+'\t'+str(lsq[0][1])+'\t0\n')
                        currY = lsq[0][1]
                        currX = xs[0]
                        meds.append(arraymedian(cut1d))
                        maxcors.append(max(ccor))
                        xcoords.append(xs[j])
                        ycoords.append(lsq[0][1])
                        lastXs.append(xs[0])
                        lastYs.append(lsq[0][1])
                        lastSeg = currSeg
                    else:
                        #Sanity check
                        #Calculate predicted "ref" value of Y based on slope of previous
                        #fit datapoints
                        wavg = 0.
                        wavgx = 0.
                        wavgDivisor = 0.
                        #Compute weighted avg of previously fitted values
                        #Weight by 1 over sqrt of delta-x
                        #Compare current y fit value to weighted avg instead of just
                        #previous value.
                        for i in range(len(lastYs)):
                            wavg += lastYs[i]/sqrt(abs(lastXs[i]-xs[j]))
                            wavgx += lastXs[i]/sqrt(abs(lastXs[i]-xs[j]))
                            wavgDivisor += 1./sqrt(abs(lastXs[i]-xs[j]))
                        if (wavgDivisor != 0):
                            wavg = wavg/wavgDivisor
                            wavgx = wavgx/wavgDivisor
                        else:
                            #We seem to have no datapoints in lastYs.  Simply use previous value
                            wavg = currY
                            wavgx = currX
                        #More than 50 pixels in deltaX between weight average of last 10
                        #datapoints and current X
                        #And not the discontinuity in middle of xs where we jump from end back to center
                        #because abs(xs[j]-xs[j-1]) == step
                        if (abs(xs[j]-xs[j-1]) == step and abs(wavgx-xs[j]) > 50):
                            if (len(lastYs) > 1):
                                #Fit slope to lastYs
                                lin = leastsq(linResiduals, [0.,0.], args=(array(lastXs),array(lastYs)))
                                slope = lin[0][1]
                            else:
                                #Only 1 datapoint, use -0.04 as slope
                                slope = -0.04
                            #Calculate guess for refY and max acceptable error
                            #err = 1+0.04*deltaX, with a max value of 3.
                            refY = wavg+slope*(xs[j]-wavgx)
                            maxerr = min(1+int(abs(xs[j]-wavgx)*.04),3)
                        else:
                            if (len(lastYs) > 3):
                                #Fit slope to lastYs
                                lin = leastsq(linResiduals, [0.,0.], args=(array(lastXs),array(lastYs)))
                                slope = lin[0][1]
                            else:
                                #Less than 4 datapoints, use -0.04 as slope
                                slope = -0.04
                            #Calculate guess for refY and max acceptable error
                            #0.5 <= maxerr <= 2 in this case.  Use slope*50 if it falls in that range
                            refY = wavg+slope*(xs[j]-wavgx)
                            maxerr = max(min(abs(slope*50),2),0.5)
                        #Discontinuity point in xs. Keep if within +/-1.
                        if (xs[j] == xinit-step and abs(lsq[0][1]-currY) < 1):
                            #update currX, currY, append to all lists
                            f.write(str(slitidx)+'\t'+str(syval)+'\t'+str(xs[j])+'\t'+str(lsq[0][1])+'\t0\n')
                            currY = lsq[0][1]
                            currX = xs[j]
                            meds.append(arraymedian(cut1d))
                            maxcors.append(max(ccor))
                            xcoords.append(xs[j])
                            ycoords.append(lsq[0][1])
                            lastXs.append(xs[j])
                            lastYs.append(lsq[0][1])
                            lastSeg = currSeg
                        elif (lastSeg != currSeg):
                            #update currX, currY, append to all lists
                            f.write(str(slitidx)+'\t'+str(syval)+'\t'+str(xs[j])+'\t'+str(lsq[0][1])+'\t0\n')
                            currY = lsq[0][1]
                            currX = xs[j]
                            meds.append(arraymedian(cut1d))
                            maxcors.append(max(ccor))
                            xcoords.append(xs[j])
                            ycoords.append(lsq[0][1])
                            lastXs.append(xs[j])
                            lastYs.append(lsq[0][1])
                            lastSeg = currSeg
                            continue
                        elif (abs(lsq[0][1] - refY) < maxerr):
                            #Regular datapoint.  Apply sanity check rejection criteria here
                            #Discard if farther than maxerr away from refY
                            if (abs(xs[j]-currX) < 4*step and maxerr > 1 and abs(lsq[0][1]-currY) > maxerr):
                                #Also discard if < 20 pixels in X from last fit datapoint, and deltaY > 1
                                f.write(str(slitidx)+'\t'+str(syval)+'\t'+str(xs[j])+'\t'+str(lsq[0][1])+'\t4\n')
                                continue
                            #update currX, currY, append to all lists
                            f.write(str(slitidx)+'\t'+str(syval)+'\t'+str(xs[j])+'\t'+str(lsq[0][1])+'\t0\n')
                            currY = lsq[0][1]
                            currX = xs[j]
                            meds.append(arraymedian(cut1d))
                            maxcors.append(max(ccor))
                            xcoords.append(xs[j])
                            ycoords.append(lsq[0][1])
                            lastXs.append(xs[j])
                            lastYs.append(lsq[0][1])
                            lastSeg = currSeg
                            #keep lastXs and lastYs at 10 elements or less
                            if (len(lastYs) > 10):
                                lastXs.pop(0)
                                lastYs.pop(0)
                        else:
                            f.write(str(slitidx)+'\t'+str(syval)+'\t'+str(xs[j])+'\t'+str(lsq[0][1])+'\t5\n')
                    #print xs[j], p[1], len(maxcors), lsq[0][1], arraymedian(cut1d), max(ccor)
                print("findSlitletProcess::traceOrders> Order "+str(slitidx)+", syval="+str(syval)+": found "+str(len(ycoords))+" datapoints.")
                self._log.writeLog(__name__, "Order "+str(slitidx)+", syval="+str(syval)+": found "+str(len(ycoords))+" datapoints.")
                #Phase 2 of rejection criteria after slitlets have been traced
                #Find outliers > 2.5 sigma in median value of 1-d cuts
                #and max values of cross correlations and remove them
                meds = array(meds)
                maxcors = array(maxcors)
                #b = (meds > arraymedian(meds)-2.5*meds.std())*(maxcors > arraymedian(maxcors)-2.5*maxcors.std())
                xcoords = array(xcoords)
                ycoords = array(ycoords)
                xc_keep = [] #Create new lists for xcoords and ycoords that will be kept
                yc_keep = []
                iseg_keep = [] #And for segment number of those kept datapoints

                for seg in range(n_segments):
                    seg_order = order
                    xstride = xsize//n_segments
                    sxlo = xstride*seg
                    sxhi = xstride*(seg+1)
                    segmask = (xcoords >= sxlo)*(xcoords < sxhi)
                    if (segmask.sum() < 5):
                        if (seg == 0):
                            z1.append(zeros(xstride))
                            yf0 = 0
                        else:
                            z1[-1] = concatenate([z1[-1], zeros(xstride)-yf0])
                        continue

                    b = (meds[segmask] > arraymedian(meds[segmask])-2.5*meds[segmask].std())*(maxcors[segmask] > arraymedian(maxcors[segmask])-2.5*maxcors[segmask].std())
                    seg_xcoords = xcoords[segmask][b]
                    seg_ycoords = ycoords[segmask][b]

                    if (len(seg_xcoords) < 10):
                        seg_order = 1
                    elif (len(seg_xcoords) < 25 or (seg_xcoords.min() > (sxlo+sxhi)/2) or (seg_xcoords.max() < (sxlo+sxhi)/2)):
                        seg_order = min(2, seg_order)
                    elif (len(seg_xcoords) < 50):
                        seg_order = min(3, seg_order)

                    #xcoords = array(xcoords)[b]
                    #ycoords = array(ycoords)[b]
                    if (n_segments > 1):
                        print("\tSegment "+str(seg)+": rejecting outliers (phase 2) - kept "+str(len(seg_ycoords))+" of "+str(len(ycoords[segmask]))+" datapoints.")
                        self._log.writeLog(__name__, "Segment "+str(seg)+": rejecting outliers (phase 2) - kept "+str(len(ycoords))+" of "+str(len(ycoords[segmask]))+" datapoints.", printCaller=False, tabLevel=1)
                    else:
                        print("\trejecting outliers (phase 2) - kept "+str(len(seg_ycoords))+" datapoints.")
                        self._log.writeLog(__name__, "rejecting outliers (phase 2) - kept "+str(len(seg_ycoords))+" datapoints.", printCaller=False, tabLevel=1)
                    #xin = 1-d array of x indices
                    xin = arange(xstride, dtype=float32)+sxlo
                    #Fit nth order (recommended 2nd) order polynomial to datapoints, Y = f(X)
                    p = zeros(seg_order+1, float64)
                    p[0] = ycoords[0]
                    try:
                        lsq = leastsq(polyResiduals, p, args=(seg_xcoords,seg_ycoords,seg_order))
                    except Exception as ex:
                        print("findSlitletProcess::traceOrders> ERROR: Could not trace Order "+str(slitidx)+" for "+fdu.getFullId()+"! Discarding Image!")
                        self._log.writeLog(__name__, "Could not trace Order "+str(slitidx)+" for "+fdu.getFullId()+"! Discarding Image!", type=fatboyLog.ERROR)
                        #disable this FDU
                        fdu.disable()
                        return calibs

                    #Compute output offsets and residuals from actual datapoints
                    yoffset = polyFunction(lsq[0], xin, seg_order)
                    yresid = yoffset[seg_xcoords-sxlo]-seg_ycoords
                    #Remove outliers and refit
                    b = abs(yresid) < yresid.mean()+2.5*yresid.std()
                    seg_xcoords = seg_xcoords[b]
                    seg_ycoords = seg_ycoords[b]

                    if (n_segments > 1):
                        print("\tSegment "+str(seg)+": rejecting outliers (phase 3). Sigma = "+formatNum(yresid.std())+". Using "+str(len(seg_ycoords))+" datapoints to fit slitlets.")
                        self._log.writeLog(__name__, "Segment "+str(seg)+": rejecting outliers (phase 3). Sigma = "+formatNum(yresid.std())+". Using "+str(len(seg_ycoords))+" datapoints to fit slitlets.", printCaller=False, tabLevel=1)
                    else:
                        print("\trejecting outliers (phase 3). Sigma = "+formatNum(yresid.std())+". Using "+str(len(seg_ycoords))+" datapoints to fit slitlets.")
                        self._log.writeLog(__name__, "rejecting outliers (phase 3). Sigma = "+formatNum(yresid.std())+". Using "+str(len(seg_ycoords))+" datapoints to fit slitlets.", printCaller=False, tabLevel=1)
                    #Use previous guess
                    p = lsq[0].astype(float64)
                    #p = zeros(seg_order+1, float64)
                    #p[0] = ycoords[0]
                    try:
                        lsq = leastsq(polyResiduals, p, args=(seg_xcoords,seg_ycoords,seg_order))
                    except Exception as ex:
                        print("findSlitletProcess::traceOrders> ERROR: Could not trace Order "+str(slitidx)+" for "+fdu.getFullId()+"! Discarding Image!")
                        self._log.writeLog(__name__, "Could not trace Order "+str(slitidx)+" for "+fdu.getFullId()+"! Discarding Image!", type=fatboyLog.ERROR)
                        #disable this FDU
                        fdu.disable()
                        return calibs

                    print("\tFit = "+formatList(lsq[0]))
                    self._log.writeLog(__name__, "Fit = "+formatList(lsq[0]), printCaller=False, tabLevel=1)
                    #Create new yoffset at every integer x
                    yoffset = polyFunction(lsq[0], xin, seg_order)
                    if (seg == 0):
                        #Subtract zero point
                        z1.append(yoffset - yoffset[0])
                        yf0 = yoffset[0]
                    else:
                        z1[-1] = concatenate([z1[-1], (yoffset - yf0)])
                #Generate qa data
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    for i in range(len(xcoords)):
                        yval = int(ycoords[i]+.5)
                        xval = int(xcoords[i]+.5)
                        for yi in range(yval-1,yval+2):
                            for xi in range(xval-1,xval+2):
                                dist = sqrt((ycoords[i]-yi)**2+(xcoords[i]-xi)**2)
                                qaData[yi,xi] = -50000/((1+dist)**2)
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    for i in range(len(xcoords)):
                        yval = int(ycoords[i]+.5)
                        xval = int(xcoords[i]+.5)
                        for yi in range(yval-1,yval+2):
                            for xi in range(xval-1,xval+2):
                                dist = sqrt((ycoords[i]-yi)**2+(xcoords[i]-xi)**2)
                                qaData[xi,yi] = -50000/((1+dist)**2)
            #end for syval
            #Update slitmask
            ylo = sylo[slitidx]-z1[0][int(slitx[slitidx])]-1-padding
            yhi = syhi[slitidx]-z1[1][int(slitx[slitidx])]+padding
            yloMask[slitidx,:] = ylo+z1[0]
            yhiMask[slitidx,:] = yhi+z1[1]
            if (not self._fdb.getGPUMode()):
                #Update slitmask piece by piece here for CPU
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    currMask = (yind >= (ylo+z1[0]).astype("int32"))*(yind <= (yhi+z1[1]).astype("int32"))
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    z1[0] = z1[0].reshape(xsize,1)
                    z1[1] = z1[1].reshape(xsize,1)
                    currMask = (xind >= (ylo+z1[0]).astype("int32"))*(xind <= (yhi+z1[1]).astype("int32"))
                b = where(currMask)
                slitmask[b] = (slitidx+1)
        #end for slitidx
        #print time.time()-t
        f.close()

        #GPU mode - create slitmask at once
        if (self._fdb.getGPUMode()):
            #Use GPU
            slitmask = createSlitmask(flatData.shape, yhiMask, yloMask, nslits, horizontal = (fdu.dispersion == fdu.DISPERSION_HORIZONTAL))

        if (slitmask.max() < 256):
            #Only convert to UInt8 if less than 256 slits
            slitmask = slitmask.astype(uint8)

        #create fatboySpecCalibs and add to calibs dict
        #Use masterFlat as source header
        slitmask = fatboySpecCalib(self._pname, "slitmask", masterFlat, data=slitmask, tagname="slitmask_"+masterFlat._id, log=self._log)
        slitmask.setProperty("specmode", fdu.getProperty("specmode"))
        slitmask.setProperty("dispersion", fdu.getProperty("dispersion"))
        slitmask.setProperty("regions", (sylo, syhi, slitx, slitw))
        slitmask.setProperty("nslits", nslits)
        calibs['slitmask'] = slitmask

        slitlo = fatboySpecCalib(self._pname, "slitlo", masterFlat, data=yloMask, tagname="slitlo_"+masterFlat._id, log=self._log)
        slitlo.setProperty("specmode", fdu.getProperty("specmode"))
        slitlo.setProperty("dispersion", fdu.getProperty("dispersion"))
        calibs['slitlo'] = slitlo

        slithi = fatboySpecCalib(self._pname, "slithi", masterFlat, data=yhiMask, tagname="slithi_"+masterFlat._id, log=self._log)
        slithi.setProperty("specmode", fdu.getProperty("specmode"))
        slithi.setProperty("dispersion", fdu.getProperty("dispersion"))
        calibs['slithi'] = slithi

        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            #make directory if necessary
            outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
            if (not os.access(outdir+"/findSlitlets", os.F_OK)):
                os.mkdir(outdir+"/findSlitlets",0o755)
            #Create output filename
            slitfile = outdir+"/findSlitlets/"+slitmask.getFullId()
            slitlofile = outdir+"/findSlitlets/"+slitlo.getFullId()
            slithifile = outdir+"/findSlitlets/"+slithi.getFullId()
            qafile = outdir+"/findSlitlets/qa_"+slitmask.getFullId()

            #Remove existing files if overwrite = yes
            if (self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                calibfiles = [slitfile, slitlofile, slithifile, qafile]
                for filename in calibfiles:
                    if (os.access(filename, os.F_OK)):
                        os.unlink(filename)

            #Write out slitmask
            if (not os.access(slitfile, os.F_OK)):
                slitmask.writeTo(slitfile)

            #Write out slitlo
            if (not os.access(slitlofile, os.F_OK)):
                slitlo.writeTo(slitlofile)

            #Write out slithi
            if (not os.access(slithifile, os.F_OK)):
                slithi.writeTo(slithifile)

            #Write out qa file
            if (not os.access(qafile, os.F_OK)):
                #TODO - GPU for qa data?
                #Generate qa data
                #if (self._fdb.getGPUMode()):
                #  #Use GPU
                #  flatData = generateQAData(flatData, xcoords, ycoords, sylo, syhi, horizontal = (fdu.dispersion == fdu.DISPERSION_HORIZONTAL))
                masterFlat.tagDataAs("slitqa", qaData)
                masterFlat.writeTo(qafile, tag="slitqa")
                masterFlat.removeProperty("slitqa")
                del qaData
        return calibs
    #end traceOrders

    ## Trace out using peak local max
    def tracePeakLocalMax(self, fdu, calibs):
        ###*** For purposes of tracePeakLocalMax algorithm, X = dispersion direction and Y = cross-dispersion direction ***###
        ###*** It will trace out and fit Y = f(X) ***###
        #Get masterFlat
        masterFlat = calibs['masterFlat']
        #Read options
        boxsize = int(self.getOption("slitlet_trace_boxsize", fdu.getTag()))
        halfbox = boxsize//2
        fiber_width = int(self.getOption("fiber_width", fdu.getTag()))
        #Get region file for this FDU
        if (fdu.hasProperty("region_file")):
            regFile = fdu.getProperty("region_file")
        else:
            regFile = self.getCalib("region_file", fdu.getTag())
        edge_thresh = int(self.getOption("edge_threshold", fdu.getTag()))

        #Check that region file exists
        if (regFile is None or not os.access(regFile, os.F_OK)):
            #If not, attempt to auto-detect slitlets!
            print("findSlitletProcess::tracePeakLocalMax> No region file given.  Attempting to auto-detect slitlets...")
            self._log.writeLog(__name__, "No region file given.  Attempting to auto-detect slitlets...")
            isNormalized = False
            if (masterFlat.hasProperty("normalized") or masterFlat.hasHeaderValue('NORMAL01')):
                #has been normalized already
                isNormalized = True
            (sylo, syhi, slitx, slitw) = self.autoDetectSlitlets(fdu, masterFlat.getData().copy(), normal=isNormalized)

            nslits = len(sylo)
            nslits_ref = int(self.getOption("slitlet_autodetect_nslits", fdu.getTag()))
            print("findSlitletProcess::tracePeakLocalMax> Found "+str(nslits)+" slitlets.")
            self._log.writeLog(__name__, "Found "+str(nslits)+" slitlets.")

            if ((nslits_ref > 0 and nslits != nslits_ref) or nslits == 0):
                print("findSlitletProcess::tracePeakLocalMax> ERROR: Could not find region file associated with "+fdu.getFullId()+"! Discarding Image!")
                self._log.writeLog(__name__, "Could not find region file associated with "+fdu.getFullId()+"!  Discarding Image!", type=fatboyLog.ERROR)
                #disable this FDU
                fdu.disable()
                return calibs
        else:
            #Read region file
            if (regFile.endswith(".reg")):
                (sylo, syhi, slitx, slitw) = readRegionFile(regFile, horizontal = (fdu.dispersion == fdu.DISPERSION_HORIZONTAL), log=self._log)
            elif (regFile.endswith(".txt")):
                (sylo, syhi, slitx, slitw) = readRegionFileText(regFile, horizontal = (fdu.dispersion == fdu.DISPERSION_HORIZONTAL), log=self._log)
            elif (regFile.endswith(".xml")):
                (sylo, syhi, slitx, slitw) = readRegionFileXML(regFile, horizontal = (fdu.dispersion == fdu.DISPERSION_HORIZONTAL), log=self._log)
            else:
                print("findSlitletProcess::tracePeakLocalMax> ERROR: Invalid extension for region file "+regFile+"! Discarding Image "+fdu.getFullId()+"!")
                self._log.writeLog(__name__, "Invalid extension for region file "+regFile+"! Discarding Image "+fdu.getFullId()+"!", type=fatboyLog.ERROR)
                #disable this FDU
                fdu.disable()
                return calibs

        #Check nslits
        nslits = len(sylo)
        if (nslits == 0):
            print("findSlitletProcess::tracePeakLocalMax> ERROR: Could not parse region file "+regFile+"! Discarding Image "+fdu.getFullId()+"!")
            self._log.writeLog(__name__, "Could not parse region file "+regFile+"! Discarding Image "+fdu.getFullId()+"!", type=fatboyLog.ERROR)
            #disable this FDU
            fdu.disable()
            return calibs

        #Check to see if slitmask already exists
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        #Check to see if slitmask / slithi / slitlo exist already from a previous run
        mfsuffix = masterFlat.getFullId()
        if (not os.access(outdir+"/findSlitlets/slitmask_"+mfsuffix, os.F_OK) and os.access(outdir+"/findSlitlets/slitmask_"+masterFlat._id+".fits", os.F_OK)):
            mfsuffix = masterFlat._id+".fits"
        slitfile = outdir+"/findSlitlets/slitmask_"+mfsuffix
        slitlofile = outdir+"/findSlitlets/slitlo_"+mfsuffix
        slithifile = outdir+"/findSlitlets/slithi_"+mfsuffix
        if (self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "no"):
            if (os.access(slitfile, os.F_OK) and os.access(slitlofile, os.F_OK) and os.access(slithifile, os.F_OK)):
                #files already exists
                #Use masterFlat as source header
                print("findSlitletProcess::tracePeakLocalMax> Slitmask "+slitfile+" already exists!  Re-using...")
                self._log.writeLog(__name__, "Slitmask "+slitfile+" already exists!  Re-using...")
                slitmask = fatboySpecCalib(self._pname, "slitmask", masterFlat, filename=slitfile, tagname="slitmask_"+masterFlat._id, log=self._log)
                slitmask.setProperty("specmode", fdu.getProperty("specmode"))
                slitmask.setProperty("dispersion", fdu.getProperty("dispersion"))
                slitmask.setProperty("regions", (sylo, syhi, slitx, slitw))
                slitmask.setProperty("nslits", nslits)
                calibs['slitmask'] = slitmask
                print("findSlitletProcess::tracePeakLocalMax> Slitlo "+slitlofile+" already exists!  Re-using...")
                self._log.writeLog(__name__, "Slitlo "+slitlofile+" already exists!  Re-using...")
                slitlo = fatboySpecCalib(self._pname, "slitlo", masterFlat, filename=slitlofile, tagname="slitlo_"+masterFlat._id, log=self._log)
                slitlo.setProperty("specmode", fdu.getProperty("specmode"))
                slitlo.setProperty("dispersion", fdu.getProperty("dispersion"))
                calibs['slitlo'] = slitlo
                print("findSlitletProcess::tracePeakLocalMax> Slithi "+slithifile+" already exists!  Re-using...")
                self._log.writeLog(__name__, "Slithi "+slithifile+" already exists!  Re-using...")
                slithi = fatboySpecCalib(self._pname, "slithi", masterFlat, filename=slithifile, tagname="slitlo_"+masterFlat._id, log=self._log)
                slithi.setProperty("specmode", fdu.getProperty("specmode"))
                slithi.setProperty("dispersion", fdu.getProperty("dispersion"))
                calibs['slithi'] = slithi
                return calibs

        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            xsize = fdu.getShape()[1]
            ysize = fdu.getShape()[0]
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            ##xsize should be size across dispersion direction
            xsize = fdu.getShape()[0]
            ysize = fdu.getShape()[1]
        #Get data from master flat
        flatData = masterFlat.getData().copy()
        #Slits can't extend beyond image top/bottom
        for j in range(nslits):
            sylo[j] = max(sylo[j], edge_thresh)
            syhi[j] = min(syhi[j], ysize-edge_thresh)
        #Start at given slitx value
        xinit = int(slitx[0])
        step = 5

        #Setup lists and arrays
        #xs = x values (dispersion direction) to cross correlate at
        #Start at slitx value for each slitlet and trace to end then to beginning
        xs = list(range(xinit, xsize-10, step))+list(range(xinit-step, 10, -1*step))

        #xcoords and ycoords contain lists of fit (x,y) points
        xcoords = []
        ycoords = []
        currX = xs[0] #current X value
        currYs = (sylo+syhi)/2 #Y at X=xinit
        lastX = xinit
        #Loop over xs every 5 pixels and cross correlate 1-d cut with islit
        for j in range(len(xs)):
            if (xs[j] == xinit-step):
                #We have finished tracing to the end, starting back at middle to trace in other direction
                #Reset currY, lastYs, lastXs
                currYs = (sylo+syhi)/2
                lastX = xinit

            #1-d cut of flat in cross-dispersion direction, sum of 11 pixels in dispersion direction centered at current X
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                cut1d = flatData[:,int(xs[j]-halfbox):int(xs[j]+halfbox+1)].sum(1).astype(float64)
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                cut1d = flatData[int(xs[j]-halfbox):int(xs[j]+halfbox+1),:].sum(0).astype(float64)

            y = where(cut1d > median(cut1d))
            x = r_[True, cut1d[1:] > cut1d[:-1]] & r_[cut1d[:-1] > cut1d[1:], True] & r_[True, True, cut1d[2:] > cut1d[:-2]] & r_[cut1d[:-2] > cut1d[2:], True, True]
            x[:y[0][0]] = False
            x[y[0][-1]+1:] = False
            z = where(x)[0]
            if (len(z) != nslits):
                #Number of slits found doesn't match
                #print "ERR1", xs[j]
                continue

            maxerr = 2
            if (abs(xs[j]-lastX) > 50):
                maxerr = abs(xs[j]-lastX)/25
            if (abs(z-currYs).max() > maxerr):
                #Shift from last datapoint is > max error
                #print "ERR2", xs[j], abs(z-currYs).max()
                continue

            currYs = z
            currX = xs[j]
            lastX = xs[j]
            xcoords.append(xs[j])
            ycoords.append(currYs)

        print("findSlitletProcess::tracePeakLocalMax> found "+str(len(ycoords))+" datapoints.")
        self._log.writeLog(__name__, "found "+str(len(ycoords))+" datapoints.")

        xcoords = array(xcoords)
        ycoords = array(ycoords)

        #1d array of indices nearest each x index
        idx = zeros(xsize, int32)
        for xi in range(xsize):
            idx[xi] = abs(xi-xcoords).argmin()
        #new xs = 1-d array of x indices
        xs = arange(xsize, dtype=int32)

        yloMask = zeros((nslits, xsize))
        yhiMask = zeros((nslits, xsize))
        #Create slitmask
        for j in range(nslits):
            z1 = ycoords[idx,j]
            yloMask[j,:] = z1-fiber_width//2
            yhiMask[j,:] = z1+fiber_width//2

        if (self._fdb.getGPUMode()):
            #Use GPU
            slitmask = createSlitmask(flatData.shape, yhiMask, yloMask, nslits, horizontal = (fdu.dispersion == fdu.DISPERSION_HORIZONTAL))
        else:
            #CPU mode
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                #Generate y index array
                yind = arange(xsize*ysize, dtype=int32).reshape(ysize,xsize)//xsize
                slitmask = zeros((ysize,xsize), dtype=int32)
                for j in range(nslits):
                    currMask = (yind >= (yloMask[j,:]).astype(int32))*(yind <= (yhiMask[j,:]).astype(int32))
                    b = where(currMask)
                    slitmask[b] = (j+1)
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                #Generate x index array
                xind = arange(xsize*ysize, dtype=int32).reshape(ysize,xsize)%xsize
                slitmask = zeros((ysize,xsize), dtype=int32)
                for j in range(nslits):
                    currMask = (xind >= (yloMask[j,:]).astype(int32))*(xind <= (yhiMask[j,:]).astype(int32))
                    b = where(currMask)
                    slitmask[b] = (j+1)
        if (slitmask.max() < 256):
            #Only convert to UInt8 if less than 256 slits
            slitmask = slitmask.astype(uint8)

        #create fatboySpecCalibs and add to calibs dict
        #use masterFlat as source header
        slitmask = fatboySpecCalib(self._pname, "slitmask", masterFlat, data=slitmask, tagname="slitmask_"+masterFlat._id, log=self._log)
        slitmask.setProperty("specmode", fdu.getProperty("specmode"))
        slitmask.setProperty("dispersion", fdu.getProperty("dispersion"))
        slitmask.setProperty("regions", (sylo, syhi, slitx, slitw))
        slitmask.setProperty("nslits", nslits)
        calibs['slitmask'] = slitmask

        slitlo = fatboySpecCalib(self._pname, "slitlo", masterFlat, data=yloMask, tagname="slitlo_"+masterFlat._id, log=self._log)
        slitlo.setProperty("specmode", fdu.getProperty("specmode"))
        slitlo.setProperty("dispersion", fdu.getProperty("dispersion"))
        calibs['slitlo'] = slitlo

        slithi = fatboySpecCalib(self._pname, "slithi", masterFlat, data=yhiMask, tagname="slithi_"+masterFlat._id, log=self._log)
        slithi.setProperty("specmode", fdu.getProperty("specmode"))
        slithi.setProperty("dispersion", fdu.getProperty("dispersion"))
        calibs['slithi'] = slithi

        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            #make directory if necessary
            outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
            if (not os.access(outdir+"/findSlitlets", os.F_OK)):
                os.mkdir(outdir+"/findSlitlets",0o755)
            #Create output filename
            slitfile = outdir+"/findSlitlets/"+slitmask.getFullId()
            slitlofile = outdir+"/findSlitlets/"+slitlo.getFullId()
            slithifile = outdir+"/findSlitlets/"+slithi.getFullId()
            qafile = outdir+"/findSlitlets/qa_"+slitmask.getFullId()

            #Remove existing files if overwrite = yes
            if (self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                calibfiles = [slitfile, slitlofile, slithifile, qafile]
                for filename in calibfiles:
                    if (os.access(filename, os.F_OK)):
                        os.unlink(filename)

            #Write out slitmask
            if (not os.access(slitfile, os.F_OK)):
                slitmask.writeTo(slitfile)

            #Write out slitlo
            if (not os.access(slitlofile, os.F_OK)):
                slitlo.writeTo(slitlofile)

            #Write out slithi
            if (not os.access(slithifile, os.F_OK)):
                slithi.writeTo(slithifile)

            #Write out qa file
            if (not os.access(qafile, os.F_OK)):
                #Generate qa data
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    for i in range(len(xcoords)):
                        yval = int32(ycoords[i]+.5)
                        xval = int(xcoords[i]+.5)
                        for yi in range(-1,2):
                            for xi in range(-1,2):
                                dist = sqrt((yi**2)+(xi**2))
                                flatData[yval+yi, xval+xi] = -50000/((1+dist)**2)
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    for i in range(len(xcoords)):
                        yval = int32(ycoords[i]+.5)
                        xval = int(xcoords[i]+.5)
                        for yi in range(-1,2):
                            for xi in range(-1,2):
                                dist = sqrt((yi**2)+(xi**2))
                                flatData[xval+xi, yval+yi] = -50000/((1+dist)**2)
                masterFlat.tagDataAs("slitqa", flatData)
                masterFlat.writeTo(qafile, tag="slitqa")
                masterFlat.removeProperty("slitqa")

        return calibs
    #end tracePeakLocalMax

    ## Trace out slitlets
    def traceSlitlets(self, fdu, calibs):
        ###*** For purposes of traceSlitlets algorithm, X = dispersion direction and Y = cross-dispersion direction ***###
        ###*** It will trace out and fit Y = f(X) ***###
        #Get masterFlat
        masterFlat = calibs['masterFlat']
        #Read options
        slitlet_trace_ylo = int(self.getOption("slitlet_trace_ylo", fdu.getTag()))
        slitlet_trace_yhi = int(self.getOption("slitlet_trace_yhi", fdu.getTag()))
        order = int(self.getOption("fit_order", fdu.getTag()))
        cen = (slitlet_trace_yhi-slitlet_trace_ylo)/2.0 #Center of 1-d cut
        #Get region file for this FDU
        if (fdu.hasProperty("region_file")):
            regFile = fdu.getProperty("region_file")
        else:
            regFile = self.getCalib("region_file", fdu.getTag())
        edge_thresh = int(self.getOption("edge_threshold", fdu.getTag()))

        #Check that region file exists
        if (regFile is None or not os.access(regFile, os.F_OK)):
            #If not, attempt to auto-detect slitlets!
            print("findSlitletProcess::traceSlitlets> No region file given.  Attempting to auto-detect slitlets...")
            self._log.writeLog(__name__, "No region file given.  Attempting to auto-detect slitlets...")
            isNormalized = False
            if (masterFlat.hasProperty("normalized") or masterFlat.hasHeaderValue('NORMAL01')):
                #has been normalized already
                isNormalized = True
            (sylo, syhi, slitx, slitw) = self.autoDetectSlitlets(fdu, masterFlat.getData().copy(), normal=isNormalized)

            nslits = len(sylo)
            nslits_ref = int(self.getOption("slitlet_autodetect_nslits", fdu.getTag()))
            print("findSlitletProcess::traceSlitlets> Found "+str(nslits)+" slitlets.")
            self._log.writeLog(__name__, "Found "+str(nslits)+" slitlets.")

            if ((nslits_ref > 0 and nslits != nslits_ref) or nslits == 0):
                print("findSlitletProcess::traceSlitlets> ERROR: Could not find region file associated with "+fdu.getFullId()+"! Discarding Image!")
                self._log.writeLog(__name__, "Could not find region file associated with "+fdu.getFullId()+"!  Discarding Image!", type=fatboyLog.ERROR)
                #disable this FDU
                fdu.disable()
                return calibs
        else:
            #Read region file
            if (regFile.endswith(".reg")):
                (sylo, syhi, slitx, slitw) = readRegionFile(regFile, horizontal = (fdu.dispersion == fdu.DISPERSION_HORIZONTAL), log=self._log)
            elif (regFile.endswith(".txt")):
                (sylo, syhi, slitx, slitw) = readRegionFileText(regFile, horizontal = (fdu.dispersion == fdu.DISPERSION_HORIZONTAL), log=self._log)
            elif (regFile.endswith(".xml")):
                (sylo, syhi, slitx, slitw) = readRegionFileXML(regFile, horizontal = (fdu.dispersion == fdu.DISPERSION_HORIZONTAL), log=self._log)
            else:
                print("findSlitletProcess::traceSlitlets> ERROR: Invalid extension for region file "+regFile+"! Discarding Image "+fdu.getFullId()+"!")
                self._log.writeLog(__name__, "Invalid extension for region file "+regFile+"! Discarding Image "+fdu.getFullId()+"!", type=fatboyLog.ERROR)
                #disable this FDU
                fdu.disable()
                return calibs

        #Check nslits
        nslits = len(sylo)
        if (nslits == 0):
            print("findSlitletProcess::traceSlitlets> ERROR: Could not parse region file "+regFile+"! Discarding Image "+fdu.getFullId()+"!")
            self._log.writeLog(__name__, "Could not parse region file "+regFile+"! Discarding Image "+fdu.getFullId()+"!", type=fatboyLog.ERROR)
            #disable this FDU
            fdu.disable()
            return calibs

        #Check to see if slitmask already exists
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        #Check to see if slitmask / slithi / slitlo exist already from a previous run
        mfsuffix = masterFlat.getFullId()
        if (not os.access(outdir+"/findSlitlets/slitmask_"+mfsuffix, os.F_OK) and os.access(outdir+"/findSlitlets/slitmask_"+masterFlat._id+".fits", os.F_OK)):
            mfsuffix = masterFlat._id+".fits"
        slitfile = outdir+"/findSlitlets/slitmask_"+mfsuffix
        slitlofile = outdir+"/findSlitlets/slitlo_"+mfsuffix
        slithifile = outdir+"/findSlitlets/slithi_"+mfsuffix
        if (self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "no"):
            if (os.access(slitfile, os.F_OK) and os.access(slitlofile, os.F_OK) and os.access(slithifile, os.F_OK)):
                #files already exists
                #Use masterFlat as source header
                print("findSlitletProcess::traceSlitlets> Slitmask "+slitfile+" already exists!  Re-using...")
                self._log.writeLog(__name__, "Slitmask "+slitfile+" already exists!  Re-using...")
                slitmask = fatboySpecCalib(self._pname, "slitmask", masterFlat, filename=slitfile, tagname="slitmask_"+masterFlat._id, log=self._log)
                slitmask.setProperty("specmode", fdu.getProperty("specmode"))
                slitmask.setProperty("dispersion", fdu.getProperty("dispersion"))
                slitmask.setProperty("regions", (sylo, syhi, slitx, slitw))
                slitmask.setProperty("nslits", nslits)
                calibs['slitmask'] = slitmask
                print("findSlitletProcess::traceSlitlets> Slitlo "+slitlofile+" already exists!  Re-using...")
                self._log.writeLog(__name__, "Slitlo "+slitlofile+" already exists!  Re-using...")
                slitlo = fatboySpecCalib(self._pname, "slitlo", masterFlat, filename=slitlofile, tagname="slitlo_"+masterFlat._id, log=self._log)
                slitlo.setProperty("specmode", fdu.getProperty("specmode"))
                slitlo.setProperty("dispersion", fdu.getProperty("dispersion"))
                calibs['slitlo'] = slitlo
                print("findSlitletProcess::traceSlitlets> Slithi "+slithifile+" already exists!  Re-using...")
                self._log.writeLog(__name__, "Slithi "+slithifile+" already exists!  Re-using...")
                slithi = fatboySpecCalib(self._pname, "slithi", masterFlat, filename=slithifile, tagname="slitlo_"+masterFlat._id, log=self._log)
                slithi.setProperty("specmode", fdu.getProperty("specmode"))
                slithi.setProperty("dispersion", fdu.getProperty("dispersion"))
                calibs['slithi'] = slithi
                return calibs

        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            xsize = fdu.getShape()[1]
            ysize = fdu.getShape()[0]
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            ##xsize should be size across dispersion direction
            xsize = fdu.getShape()[0]
            ysize = fdu.getShape()[1]
        #Get data from master flat
        flatData = masterFlat.getData().copy()
        #Slits can't extend beyond image top/bottom
        for j in range(nslits):
            sylo[j] = max(sylo[j], edge_thresh)
            syhi[j] = min(syhi[j], ysize-edge_thresh)
        # -1 => default => 1/4, 3/4 of chip
        if (slitlet_trace_ylo < 0):
            slitlet_trace_ylo = ysize//4
        if (slitlet_trace_yhi < 0):
            slitlet_trace_yhi = (ysize*3)//4
        if (slitlet_trace_yhi < slitlet_trace_ylo):
            tmp = slitlet_trace_ylo
            slitlet_trace_ylo = slitlet_trace_yhi
            slitlet_trace_yhi = tmp
        cen = (slitlet_trace_yhi-slitlet_trace_ylo)/2.0 #Center of 1-d cut
        #Start in middle and step by 5 pixels
        xinit = xsize//2
        step = 5

        #1-d cut of central 11 pixels of flat in cross-dispersion direction
        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            islit = flatData[slitlet_trace_ylo:slitlet_trace_yhi, xinit-5:xinit+6].sum(1).astype(float64)
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            islit = flatData[xinit-5:xinit+6, slitlet_trace_ylo:slitlet_trace_yhi].sum(0).astype(float64)

        #Setup lists and arrays
        #xs = x values (dispersion direction) to cross correlate at
        #Start at middle and trace to end then to beginning
        xs = list(range(xinit, xsize-100, step))+list(range(xinit-step, 100, -1*step))
        #xcoords and ycoords contain lists of fit (x,y) points
        xcoords = []
        ycoords = []
        #median value of 1-d cuts and max values of cross correlations are kept and used as rejection criteria later
        meds = []
        maxcors = []
        #Up to last 10 (x,y) pairs are kept and used in various rejection criteria
        lastXs = []
        lastYs = []
        currX = xs[0] #current X value
        currY = 0 #shift in cross-dispersion direction at currX relative to Y at X=xinit
        #Loop over xs every 5 pixels and cross correlate 1-d cut with islit
        for j in range(len(xs)):
            if (xs[j] == xinit-step):
                #We have finished tracing to the end, starting back at middle to trace in other direction
                #Reset currY, lastYs, lastXs
                currY = 0
                lastYs = [0]
                lastXs = [xinit]

            #1-d cut of flat in cross-dispersion direction, sum of 11 pixels in dispersion direction centered at current X
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                cut1d = flatData[slitlet_trace_ylo:slitlet_trace_yhi, int(xs[j]-5):int(xs[j]+6)].sum(1).astype(float64)
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                cut1d = flatData[int(xs[j]-5):int(xs[j]+6), slitlet_trace_ylo:slitlet_trace_yhi].sum(0).astype(float64)
            #Cross correlate cut1d with islit
            #Use numpy correlate since 1d cut -- not enough pixels to benefit from GPU
            ccor = correlate(cut1d, islit, mode='same')
            #Median filter with 51 pixel boxcar and set negative values to 0 before fitting
            if (self._fdb.getGPUMode()):
                ccor = gpumedianfilter(ccor)
            else:
                ccor = medianfilterCPU(ccor)
            ccor[ccor < 0] = 0

            #Use leastsq to fit Gaussian to cross-correlation function
            p = zeros(4, float64)
            p[1] = round(currY,3) #center = currY
            p[2] = 3. #FWHM = 3
            p[3] = 0.
            #Only examine up to 51 pixels centered at previous result
            llo = max(0, int(cen+p[1]-25))
            lhi = min(len(ccor), int(cen+p[1]+26))
            #print xs[j], llo, lhi, len(ccor), currY
            #print islit.size, cut1d.size, slitlet_trace_ylo, slitlet_trace_yhi
            p[0] = max(ccor[llo:lhi])
            try:
                lsq = leastsq(gaussResiduals, p, args=(arange(lhi-llo, dtype=float64)+llo-cen, ccor[llo:lhi]))
            except Exception as ex:
                print("findSlitletProcess::traceSlitlets> Warning: Leastsq FAILED at "+str(xs[j])+" with "+str(ex))
                self._log.writeLog(__name__, "Leastsq FAILED at "+str(xs[j])+" with "+str(ex), type=fatboyLog.WARNING)
                continue
            #Error checking results of leastsq call
            if (lsq[1] == 5):
                #exceeded max number of calls = ignore
                continue
            if (lsq[0][0]+lsq[0][3] < 0):
                #flux less than zero = ignore
                continue
            if (lsq[0][2] < 0 and j != 0):
                #negative boxsize = ignore unless first datapoint
                continue
            if (j == 0):
                #First datapoint -- update currX, currY, append to all lists
                currY = lsq[0][1]
                currX = xs[0]
                meds.append(arraymedian(cut1d))
                maxcors.append(max(ccor))
                xcoords.append(xs[j])
                ycoords.append(lsq[0][1])
                lastXs.append(xs[0])
                lastYs.append(lsq[0][1])
            else:
                #Sanity check
                #Calculate predicted "ref" value of Y based on slope of previous
                #fit datapoints
                wavg = 0.
                wavgx = 0.
                wavgDivisor = 0.
                #Compute weighted avg of previously fitted values
                #Weight by 1 over sqrt of delta-x
                #Compare current y fit value to weighted avg instead of just
                #previous value.
                for i in range(len(lastYs)):
                    wavg += lastYs[i]/sqrt(abs(lastXs[i]-xs[j]))
                    wavgx += lastXs[i]/sqrt(abs(lastXs[i]-xs[j]))
                    wavgDivisor += 1./sqrt(abs(lastXs[i]-xs[j]))
                if (wavgDivisor != 0):
                    wavg = wavg/wavgDivisor
                    wavgx = wavgx/wavgDivisor
                else:
                    #We seem to have no datapoints in lastYs.  Simply use previous value
                    wavg = currY
                    wavgx = currX
                #More than 50 pixels in deltaX between weight average of last 10
                #datapoints and current X
                #And not the discontinuity in middle of xs where we jump from end back to center
                #because abs(xs[j]-xs[j-1]) == step
                if (abs(xs[j]-xs[j-1]) == step and abs(wavgx-xs[j]) > 50):
                    if (len(lastYs) > 1):
                        #Fit slope to lastYs
                        lin = leastsq(linResiduals, [0.,0.], args=(array(lastXs),array(lastYs)))
                        slope = lin[0][1]
                    else:
                        #Only 1 datapoint, use -0.04 as slope
                        slope = -0.04
                    #Calculate guess for refY and max acceptable error
                    #err = 1+0.04*deltaX, with a max value of 3.
                    refY = wavg+slope*(xs[j]-wavgx)
                    maxerr = min(1+int(abs(xs[j]-wavgx)*.04),3)
                else:
                    if (len(lastYs) > 3):
                        #Fit slope to lastYs
                        lin = leastsq(linResiduals, [0.,0.], args=(array(lastXs),array(lastYs)))
                        slope = lin[0][1]
                    else:
                        #Less than 4 datapoints, use -0.04 as slope
                        slope = -0.04
                    #Calculate guess for refY and max acceptable error
                    #0.5 <= maxerr <= 2 in this case.  Use slope*50 if it falls in that range
                    refY = wavg+slope*(xs[j]-wavgx)
                    maxerr = max(min(abs(slope*50),2),0.5)
                #Discontinuity point in xs. Keep if within +/-1.
                if (xs[j] == xinit-step and abs(lsq[0][1]-currY) < 1):
                    #update currX, currY, append to all lists
                    currY = lsq[0][1]
                    currX = xs[j]
                    meds.append(arraymedian(cut1d))
                    maxcors.append(max(ccor))
                    xcoords.append(xs[j])
                    ycoords.append(lsq[0][1])
                    lastXs.append(xs[j])
                    lastYs.append(lsq[0][1])
                elif (abs(lsq[0][1] - refY) < maxerr):
                    #Regular datapoint.  Apply sanity check rejection criteria here
                    #Discard if farther than maxerr away from refY
                    if (abs(xs[j]-currX) < 4*step and maxerr > 1 and abs(lsq[0][1]-currY) > maxerr):
                        #Also discard if < 20 pixels in X from last fit datapoint, and deltaY > 1
                        continue
                    #update currX, currY, append to all lists
                    currY = lsq[0][1]
                    currX = xs[j]
                    meds.append(arraymedian(cut1d))
                    maxcors.append(max(ccor))
                    xcoords.append(xs[j])
                    ycoords.append(lsq[0][1])
                    lastXs.append(xs[j])
                    lastYs.append(lsq[0][1])
                    #keep lastXs and lastYs at 10 elements or less
                    if (len(lastYs) > 10):
                        lastXs.pop(0)
                        lastYs.pop(0)
            #print xs[j], p[1], len(maxcors), lsq[0][1], arraymedian(cut1d), max(ccor)
        print("findSlitletProcess::traceSlitlets> found "+str(len(ycoords))+" datapoints.")
        self._log.writeLog(__name__, "found "+str(len(ycoords))+" datapoints.")
        #Phase 2 of rejection criteria after slitlets have been traced
        #Find outliers > 2.5 sigma in median value of 1-d cuts
        #and max values of cross correlations and remove them
        meds = array(meds)
        maxcors = array(maxcors)
        b = (meds >= arraymedian(meds)-2.5*meds.std())*(maxcors >= arraymedian(maxcors)-2.5*maxcors.std())
        xcoords = array(xcoords)[b]
        ycoords = array(ycoords)[b]
        print("\trejecting outliers (phase 2) - kept "+str(len(ycoords))+" datapoints.")
        self._log.writeLog(__name__, "rejecting outliers (phase 2) - kept "+str(len(ycoords))+" datapoints.", printCaller=False, tabLevel=1)
        #new xs = 1-d array of x indices
        xs = arange(xsize, dtype=float32)
        #Fit n-th order (recommended 3rd order) polynomial to datapoints, Y = f(X)
        p = zeros(order+1, float64)
        p[0] = ycoords[-1]
        try:
            lsq = leastsq(polyResiduals, p, args=(xcoords,ycoords,order))
        except Exception as ex:
            print("findSlitletProcess::traceOrders> ERROR: Could not trace slitlets for "+fdu.getFullId()+"! Discarding Image!")
            self._log.writeLog(__name__, "Could not trace slitlets for "+fdu.getFullId()+"! Discarding Image!", type=fatboyLog.ERROR)
            #disable this FDU
            fdu.disable()
            return calibs

        #Compute output offsets and residuals from actual datapoints
        yoffset = polyFunction(lsq[0], xs, order)
        yresid = yoffset[xcoords]-ycoords
        #Remove outliers and refit
        b = abs(yresid) < yresid.mean()+2.5*yresid.std()
        xcoords = xcoords[b]
        ycoords = ycoords[b]
        print("\trejecting outliers (phase 3). Sigma = "+formatNum(yresid.std())+". Using "+str(len(ycoords))+" datapoints to fit slitlets.")
        self._log.writeLog(__name__, "rejecting outliers (phase 3). Sigma = "+formatNum(yresid.std())+". Using "+str(len(ycoords))+" datapoints to fit slitlets.", printCaller=False, tabLevel=1)
        #use previous fit as guess
        p = lsq[0].astype(float64)
        #p = zeros(order+1, float64)
        #p[0] = ycoords[-1]
        try:
            lsq = leastsq(polyResiduals, p, args=(xcoords,ycoords,order))
        except Exception as ex:
            print("findSlitletProcess::traceOrders> ERROR: Could not trace slitlets for "+fdu.getFullId()+"! Discarding Image!")
            self._log.writeLog(__name__, "Could not trace slitlets for "+fdu.getFullId()+"! Discarding Image!", type=fatboyLog.ERROR)
            #disable this FDU
            fdu.disable()
            return calibs

        #Create new yoffset at every integer x
        yoffset = polyFunction(lsq[0], xs, order)
        #Subtract zero point
        z1 = yoffset - yoffset[0]
        yloMask = zeros((nslits, len(z1)))
        yhiMask = zeros((nslits, len(z1)))
        #Create slitmask
        for j in range(nslits):
            #ylo = sylo[j]-z1[int(slitx[j])]-1
            ylo = sylo[j]-z1[int(slitx[j])]
            yhi = syhi[j]-z1[int(slitx[j])]
            yloMask[j,:] = ylo+z1
            yhiMask[j,:] = yhi+z1
        if (self._fdb.getGPUMode()):
            #Use GPU
            slitmask = createSlitmask(flatData.shape, yhiMask, yloMask, nslits, horizontal = (fdu.dispersion == fdu.DISPERSION_HORIZONTAL))
        else:
            #CPU mode
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                #Generate y index array
                yind = arange(xsize*ysize, dtype=int32).reshape(ysize,xsize)//xsize
                slitmask = zeros((ysize,xsize), dtype=int32)
                for j in range(nslits):
                    #ylo = sylo[j]-z1[int(slitx[j])]-1
                    ylo = sylo[j]-z1[int(slitx[j])]
                    yhi = syhi[j]-z1[int(slitx[j])]
                    currMask = (yind >= (ylo+z1).astype("int32"))*(yind <= (yhi+z1).astype("int32"))
                    b = where(currMask)
                    slitmask[b] = (j+1)
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                #Generate x index array
                xind = arange(xsize*ysize, dtype=int32).reshape(ysize,xsize)%xsize
                slitmask = zeros((ysize,xsize), dtype=int32)
                #Need to reshape z1 array
                z1 = z1.reshape((len(z1), 1))
                for j in range(nslits):
                    ylo = sylo[j]-z1[int(slitx[j])]-1
                    yhi = syhi[j]-z1[int(slitx[j])]
                    currMask = (xind >= (ylo+z1).astype("int32"))*(xind <= (yhi+z1).astype("int32"))
                    b = where(currMask)
                    slitmask[b] = (j+1)
        if (slitmask.max() < 256):
            #Only convert to UInt8 if less than 256 slits
            slitmask = slitmask.astype(uint8)

        #create fatboySpecCalibs and add to calibs dict
        #use masterFlat as source header
        slitmask = fatboySpecCalib(self._pname, "slitmask", masterFlat, data=slitmask, tagname="slitmask_"+masterFlat._id, log=self._log)
        slitmask.setProperty("specmode", fdu.getProperty("specmode"))
        slitmask.setProperty("dispersion", fdu.getProperty("dispersion"))
        slitmask.setProperty("regions", (sylo, syhi, slitx, slitw))
        slitmask.setProperty("nslits", nslits)
        calibs['slitmask'] = slitmask

        slitlo = fatboySpecCalib(self._pname, "slitlo", masterFlat, data=yloMask, tagname="slitlo_"+masterFlat._id, log=self._log)
        slitlo.setProperty("specmode", fdu.getProperty("specmode"))
        slitlo.setProperty("dispersion", fdu.getProperty("dispersion"))
        calibs['slitlo'] = slitlo

        slithi = fatboySpecCalib(self._pname, "slithi", masterFlat, data=yhiMask, tagname="slithi_"+masterFlat._id, log=self._log)
        slithi.setProperty("specmode", fdu.getProperty("specmode"))
        slithi.setProperty("dispersion", fdu.getProperty("dispersion"))
        calibs['slithi'] = slithi

        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            #make directory if necessary
            outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
            if (not os.access(outdir+"/findSlitlets", os.F_OK)):
                os.mkdir(outdir+"/findSlitlets",0o755)
            #Create output filename
            slitfile = outdir+"/findSlitlets/"+slitmask.getFullId()
            slitlofile = outdir+"/findSlitlets/"+slitlo.getFullId()
            slithifile = outdir+"/findSlitlets/"+slithi.getFullId()
            qafile = outdir+"/findSlitlets/qa_"+slitmask.getFullId()

            #Remove existing files if overwrite = yes
            if (self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                calibfiles = [slitfile, slitlofile, slithifile, qafile]
                for filename in calibfiles:
                    if (os.access(filename, os.F_OK)):
                        os.unlink(filename)

            #Write out slitmask
            if (not os.access(slitfile, os.F_OK)):
                slitmask.writeTo(slitfile)

            #Write out slitlo
            if (not os.access(slitlofile, os.F_OK)):
                slitlo.writeTo(slitlofile)

            #Write out slithi
            if (not os.access(slithifile, os.F_OK)):
                slithi.writeTo(slithifile)

            #Write out qa file
            if (not os.access(qafile, os.F_OK)):
                #Generate qa data
                if (self._fdb.getGPUMode()):
                    #Use GPU
                    flatData = generateQAData(flatData, xcoords, ycoords, sylo, syhi, horizontal = (fdu.dispersion == fdu.DISPERSION_HORIZONTAL))
                else:
                    #CPU version -- loop over coords first
                    for j in range(len(xcoords)):
                        xval = int(xcoords[j]+.5)
                        qaxs = arange(9, dtype=int32).reshape((3,3))%3+xval-1
                        ys = arange(9, dtype=int32).reshape((3,3))//3
                        #There will be 18 x nslits x ncoords pixels used to show where slitlets were traced out
                        for i in range(nslits):
                            yval = int(ycoords[j]+sylo[i]+0.5)
                            #calculate x and y 3x3 index arrays
                            qays = ys+yval-1
                            dist = sqrt((ycoords[j]+sylo[i]-qays)**2+(xcoords[j]-qaxs)**2)
                            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                                flatData[qays,qaxs] = -50000/((1+dist)**2)
                            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                                flatData[qaxs,qays] = -50000/((1+dist)**2)
                            yval = int(ycoords[j]+syhi[i]+0.5)
                            qays = ys+yval-1
                            dist = sqrt((ycoords[j]+syhi[i]-qays)**2+(xcoords[j]-qaxs)**2)
                            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                                flatData[qays,qaxs] = -50000/((1+dist)**2)
                            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                                flatData[qaxs,qays] = -50000/((1+dist)**2)
                masterFlat.tagDataAs("slitqa", flatData)
                masterFlat.writeTo(qafile, tag="slitqa")
                masterFlat.removeProperty("slitqa")
        return calibs
    #end traceSlitlets
