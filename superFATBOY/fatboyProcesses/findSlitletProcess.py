from superFATBOY.fatboyDataUnit import fatboyDataUnit
from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.datatypeExtensions.fatboySpecCalib import fatboySpecCalib

from superFATBOY import gpu_imcombine, imcombine
import numpy as np
import math
from scipy.optimize import leastsq
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import uniform_filter1d, median_filter, map_coordinates
from scipy.ndimage import shift as ndshift

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
    def autoDetectSlitlets(self, fdu, flatData, normal=False, lampData=None):
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
                cut1d = gpu_arraymedian(flatData[:,x_auto-halfbox:x_auto+halfbox+1], axis="X").astype(np.float64)
            else:
                cut1d = flatData[:,x_auto-halfbox:x_auto+halfbox+1].sum(1).astype(np.float64)
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            if (use_median):
                cut1d = gpu_arraymedian(flatData[x_auto-halfbox:x_auto+halfbox+1,:], axis="Y").astype(np.float64)
            else:
                cut1d = flatData[x_auto-halfbox:x_auto+halfbox+1,:].sum(0).astype(np.float64)
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
            y = np.where(cut1d > np.median(cut1d))
            x = np.r_[True, cut1d[1:] > cut1d[:-1]] & np.r_[cut1d[:-1] > cut1d[1:], True] & np.r_[True, True, cut1d[2:] > cut1d[:-2]] & np.r_[cut1d[:-2] > cut1d[2:], True, True]
            x[:y[0][0]] = False
            x[y[0][-1]+1:] = False
            z = np.where(x)[0]
            sylo = z-fiber_width//2
            syhi = z+fiber_width//2
            slitx = np.array([x_auto]*len(sylo))
            slitw = np.array([boxsize]*len(sylo))
            return (sylo, syhi, slitx, slitw)

        if (normal):
            #if normalized, slitlets have already been found, easy to detect nonzero points
            slitlets = extractNonzeroRegions(cut1d, min_width)
        else:
            #use extractSpectra to find step function locations
            #illumination_profile: flat field slitlets may cover most of the cut
            use_orig = self.getOption("slitlet_autodetect_use_orig_algorithm", fdu.getTag()).lower() == "yes"
            min_trough_depth = float(self.getOption("slitlet_autodetect_min_trough_depth", fdu.getTag()))
            slitlets = extractSpectra(cut1d, sigma, min_width, minFluxPct=min_flux_pct, use_orig_algorithm=use_orig, trough_depth=min_trough_depth, illumination_profile=True)

        source = self.getOption("slitlet_autodetect_source", fdu.getTag()).lower()
        if (source in ["arclamp", "both"] and lampData is not None):
            #Flat slitlets are kept as a fallback and to report flat regions with no arc spectrum
            flatSlitlets = slitlets
            refineCut = None
            if (source == "both"):
                refineCut = cut1d
            slitlets = self.autoDetectSlitletsArclamp(fdu, lampData, min_width, cut1d=refineCut)
            if (slitlets is None):
                print("findSlitletProcess::autoDetectSlitlets> ERROR: Could not find any slitlets in master arclamp.  Falling back to the flat!")
                self._log.writeLog(__name__, "Could not find any slitlets in master arclamp.  Falling back to the flat!", type=fatboyLog.ERROR)
                slitlets = flatSlitlets
                lampData = None
            else:
                print("findSlitletProcess::autoDetectSlitlets> Found "+str(len(slitlets))+" slitlets in master arclamp.")
                self._log.writeLog(__name__, "Found "+str(len(slitlets))+" slitlets in master arclamp.")
                nslits_ref = int(self.getOption("slitlet_autodetect_nslits", fdu.getTag()))
                if (nslits_ref > 0 and flatSlitlets is not None):
                    #Compare counts of valid slitlets, since invalid ones (e.g. a mask ID) are dropped below
                    n_arc = len(slitlets)-len(self.findInvalidSlitlets(fdu, flatData, slitlets[:,0], slitlets[:,1], lampData))
                    #Flat slitlets are judged by flat criteria only -- if the arclamp is a poor fit
                    #for this data, its row correlation can't be trusted to validate them either
                    n_flat = len(flatSlitlets)-len(self.findInvalidSlitlets(fdu, flatData, flatSlitlets[:,0], flatSlitlets[:,1]))
                    if (n_arc != nslits_ref and n_flat == nslits_ref):
                        #Arclamp detection is a poor fit for some data (e.g. strongly curved/tilted slitlets)
                        print("findSlitletProcess::autoDetectSlitlets> ERROR: Master arclamp gave "+str(n_arc)+" valid slitlets but slitlet_autodetect_nslits = "+str(nslits_ref)+", while the flat gave "+str(n_flat)+".  Falling back to the flat!")
                        self._log.writeLog(__name__, "Master arclamp gave "+str(n_arc)+" valid slitlets but slitlet_autodetect_nslits = "+str(nslits_ref)+", while the flat gave "+str(n_flat)+".  Falling back to the flat!", type=fatboyLog.ERROR)
                        slitlets = flatSlitlets
                        flatSlitlets = None
                        #Don't use the arclamp to validate the flat's slitlets below either
                        lampData = None
                if (flatSlitlets is not None):
                    for (flo, fhi) in flatSlitlets:
                        if (not ((slitlets[:,0] <= fhi)*(slitlets[:,1] >= flo)).any()):
                            print("findSlitletProcess::autoDetectSlitlets> ERROR: Dropping illuminated flat region "+str(flo)+"-"+str(fhi)+": no coherent arclamp spectrum, so not a valid slitlet (mask ID or alignment hole?)")
                            self._log.writeLog(__name__, "Dropping illuminated flat region "+str(flo)+"-"+str(fhi)+": no coherent arclamp spectrum, so not a valid slitlet (mask ID or alignment hole?)", type=fatboyLog.ERROR)

        if (slitlets is None):
            #Return np.empty lists
            return([], [], [], [])
        sylo = slitlets[:,0]
        syhi = slitlets[:,1]
        slitx = np.array([x_auto]*len(sylo))
        slitw = np.array([boxsize]*len(sylo))

        if (self.getOption("slitlet_attempt_autocorrect", fdu.getTag()).lower() == "yes"):
            nslits_ref = int(self.getOption("slitlet_autodetect_nslits", fdu.getTag()))
            if (nslits_ref > 0 and nslits_ref != len(sylo)):
                print("findSlitletProcess::autoDetectSlitlets> Found "+str(len(sylo))+" slitlets instead of "+str(nslits_ref)+".  Attempting to autocorrect...")
                self._log.writeLog(__name__, "Found "+str(len(sylo))+" slitlets instead of "+str(nslits_ref)+".  Attempting to autocorrect...")
                swidth = syhi-sylo
                sgap = slitlets[1:,0]-slitlets[:-1,1]
                mwidth = gpu_arraymedian(swidth)
                mgap = gpu_arraymedian(sgap)
                wsig = np.abs((swidth-mwidth)/swidth.std())
                gsig = np.abs((sgap-mgap)/sgap.std())
                bw = np.where(wsig > 2)[0] #slit width > 2 sigma
                gw = np.where(wsig <= 2)[0]
                gg = np.where(gsig <= 2)[0]
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
                slitlets = np.array(slitlets)
                if (nslits_ref > 0 and nslits_ref > len(sylo)):
                    #look for gaps
                    sylo = slitlets[:,0]
                    syhi = slitlets[:,1]
                    swidth = syhi-sylo
                    sgap = slitlets[1:,0]-slitlets[:-1,1]
                    mwidth = gpu_arraymedian(swidth)
                    mgap = gpu_arraymedian(sgap)
                    wsig = np.abs((swidth-mwidth)/swidth.std())
                    gsig = np.abs((sgap-mgap)/sgap.std())
                    gw = np.where(wsig <= 2)[0]
                    bg = np.where(gsig > 2)[0]
                    gg = np.where(gsig <= 2)[0]
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
                    #Sort and convert to np.array
                    slitlets.sort()
                    slitlets = np.array(slitlets)
                sylo = slitlets[:,0]
                syhi = slitlets[:,1]
                slitx = np.array([x_auto]*len(sylo))
                slitw = np.array([boxsize]*len(sylo))
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

        #Drop illuminated regions that are not real slitlets (e.g. a mask ID)
        invalid = self.findInvalidSlitlets(fdu, flatData, sylo, syhi, lampData)
        if (len(invalid) > 0):
            keep = np.ones(len(sylo), dtype=bool)
            for (j, reason) in invalid:
                print("findSlitletProcess::autoDetectSlitlets> ERROR: Dropping auto-detected slitlet "+str(sylo[j])+"-"+str(syhi[j])+": "+reason+".  Not a valid slitlet (mask ID or alignment hole?)")
                self._log.writeLog(__name__, "Dropping auto-detected slitlet "+str(sylo[j])+"-"+str(syhi[j])+": "+reason+".  Not a valid slitlet (mask ID or alignment hole?)", type=fatboyLog.ERROR)
                keep[j] = False
            sylo = sylo[keep]
            syhi = syhi[keep]
            slitx = slitx[keep]
            slitw = slitw[keep]

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

    #Correlation between adjacent cross-dispersion rows of the high-pass filtered master
    #arclamp, over the full dispersion range.  r[y] = corr(row y, row y+1) is ~1 inside a
    #slitlet (same line pattern), ~0 in background, and drops sharply at a boundary between
    #packed slitlets whose arc lines are offset in wavelength.
    def arcRowCorrelation(self, fdu, lampData):
        lamp = np.nan_to_num(np.asarray(lampData, dtype=np.float64))
        if (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            #Put dispersion direction along axis 1
            lamp = lamp.T
        #Remove continuum / illumination so that only the line pattern is correlated
        hp = lamp-uniform_filter1d(lamp, 31, axis=1)
        hp = hp[:,16:-16]
        hp -= hp.mean(1).reshape(-1,1)
        norm = np.sqrt((hp*hp).sum(1))
        norm[norm == 0] = 1
        return (hp[:-1]*hp[1:]).sum(1)/(norm[:-1]*norm[1:])
    #end arcRowCorrelation

    #Auto-detect slitlets as runs of rows whose arclamp spectra correlate (see
    #arcRowCorrelation).  Returns an (n,2) int array of [ylo, yhi] or None.  If the flat
    #field cut1d is given, outer edges (not shared with an adjacent slitlet) are moved to
    #the flat's half-max crossing, which is sharper than the arc's tapered edges.
    def autoDetectSlitletsArclamp(self, fdu, lampData, min_width, cut1d=None):
        min_corr = float(self.getOption("slitlet_autodetect_arc_min_corr", fdu.getTag()))
        r = self.arcRowCorrelation(fdu, lampData)
        good = r > min_corr
        slitlets = []
        y = 0
        while (y < len(good)):
            if (good[y]):
                y0 = y
                while (y < len(good) and good[y]):
                    y += 1
                #good[y0:y] all True => rows y0 through y are in one slitlet
                if (y-y0+1 >= min_width):
                    slitlets.append([y0, y])
            y += 1
        if (len(slitlets) == 0):
            return None
        if (cut1d is not None):
            packed_gap = 3 #Max rows between two slitlets that share a boundary
            max_shift = 8 #Max rows an outer edge may move
            nslits = len(slitlets)
            outerLo = [i == 0 or slitlets[i][0]-slitlets[i-1][1] > packed_gap for i in range(nslits)]
            outerHi = [i == nslits-1 or slitlets[i+1][0]-slitlets[i][1] > packed_gap for i in range(nslits)]
            bkg = np.percentile(cut1d, 5)
            for i in range(nslits):
                (ylo, yhi) = slitlets[i]
                half = bkg+0.5*(np.median(cut1d[ylo:yhi+1])-bkg)
                mid = (ylo+yhi)//2
                if (outerLo[i]):
                    #Never walk into the previous slitlet
                    ymin = max(ylo-max_shift, 0)
                    if (i > 0):
                        ymin = max(ymin, slitlets[i-1][1]+1)
                    y = ylo
                    if (cut1d[y] >= half):
                        while (y > ymin and cut1d[y-1] >= half):
                            y -= 1
                    else:
                        while (y < mid and cut1d[y] < half):
                            y += 1
                    slitlets[i][0] = y
                if (outerHi[i]):
                    ymax = min(yhi+max_shift, len(cut1d)-1)
                    if (i < nslits-1):
                        ymax = min(ymax, slitlets[i+1][0]-1)
                    y = yhi
                    if (cut1d[y] >= half):
                        while (y < ymax and cut1d[y+1] >= half):
                            y += 1
                    else:
                        while (y > mid and cut1d[y] < half):
                            y -= 1
                    slitlets[i][1] = y
        return np.array(slitlets, dtype=np.int32)
    #end autoDetectSlitletsArclamp

    #Find illuminated regions that are not real slitlets (e.g. LUCI's mask ID "digits"):
    #a real slitlet's flat is smooth from row to row and, if an arclamp is given, all of
    #its rows share one line pattern.  Returns a list of (index, reason).
    def findInvalidSlitlets(self, fdu, flatData, sylo, syhi, lampData=None):
        max_rough = float(self.getOption("slitlet_validity_max_flat_roughness", fdu.getTag()))
        min_arc_corr = float(self.getOption("slitlet_validity_min_arc_corr", fdu.getTag()))
        x_auto = int(self.getOption("slitlet_autodetect_x", fdu.getTag()))
        invalid = []
        if (max_rough <= 0 and (lampData is None or min_arc_corr <= 0)):
            return invalid
        flat = np.asarray(flatData)
        if (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            #Put dispersion direction along axis 1
            flat = flat.T
        #Median over 201 columns suppresses noise, dust, and bad pixels
        x1 = max(0, x_auto-100)
        x2 = min(flat.shape[1], x_auto+101)
        prof = np.median(flat[:,x1:x2], axis=1)
        r = None
        if (lampData is not None and min_arc_corr > 0):
            r = self.arcRowCorrelation(fdu, lampData)
        for j in range(len(sylo)):
            #Skip 2 rows at each end so partially illuminated edge rows don't count
            lo = int(sylo[j])+2
            hi = int(syhi[j])-2
            if (lo < 0 or hi >= len(prof) or hi-lo < 4):
                continue
            reasons = []
            level = np.median(prof[lo:hi+1])
            if (max_rough > 0 and level > 0):
                d = np.diff(prof[lo:hi+1])
                rough = 1.4826*np.median(np.abs(d-np.median(d)))/level
                if (rough > max_rough):
                    reasons.append("flat row-to-row roughness "+formatNum(rough)+" > slitlet_validity_max_flat_roughness = "+str(max_rough))
            if (r is not None):
                arc_corr = r[lo:hi].mean()
                if (arc_corr < min_arc_corr):
                    reasons.append("mean arclamp row correlation "+formatNum(arc_corr)+" < slitlet_validity_min_arc_corr = "+str(min_arc_corr))
            if (len(reasons) > 0):
                invalid.append((j, "; ".join(reasons)))
        return invalid
    #end findInvalidSlitlets

    #Region file slitlets are kept as given, but warn loudly about any that look invalid
    def warnInvalidRegionSlitlets(self, fdu, calibs, sylo, syhi, regFile):
        lampData = None
        if ('masterLamp' in calibs):
            lampData = calibs['masterLamp'].getData(force_cpu=True)
        invalid = self.findInvalidSlitlets(fdu, calibs['masterFlat'].getData(force_cpu=True), sylo, syhi, lampData)
        for (j, reason) in invalid:
            print("findSlitletProcess::warnInvalidRegionSlitlets> WARNING: Slitlet "+str(j+1)+" ("+str(sylo[j])+"-"+str(syhi[j])+") in region file "+regFile+" does not look like a valid slitlet: "+reason+".  Keeping it since it is in the region file.")
            self._log.writeLog(__name__, "Slitlet "+str(j+1)+" ("+str(sylo[j])+"-"+str(syhi[j])+") in region file "+regFile+" does not look like a valid slitlet: "+reason+".  Keeping it since it is in the region file.", type=fatboyLog.WARNING)
    #end warnInvalidRegionSlitlets

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
            self.correctFlexure(fdu, calibs, prevProc)
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
            self.correctFlexure(fdu, calibs, prevProc)
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

        #3) If auto-detecting from the arclamp, find or create the master arclamp too
        if (self.getOption("slitlet_autodetect_source", fdu.getTag()).lower() in ["arclamp", "both"] and not 'masterLamp' in calibs):
            lampProperties = dict()
            lampProperties['specmode'] = fdu.getProperty("specmode")
            #3a) check for an already created master arclamp matching specmode/filter/grism and TAGGED for this object
            masterLamp = self._fdb.getTaggedMasterCalib(ident=fdu._id, obstype="master_arclamp", filter=fdu.filter, section=fdu.section, properties=lampProperties, headerVals=headerVals)
            if (masterLamp is None):
                #3b) check for an already created master arclamp matching specmode/filter/grism
                masterLamp = self._fdb.getMasterCalib(obstype="master_arclamp", filter=fdu.filter, section=fdu.section, properties=lampProperties, headerVals=headerVals, tag=fdu.getTag())
            if (masterLamp is None):
                #3c) Use createMasterArclampProcess.getCalibs to create masterLamp from individual arclamps
                #Only works if process is included in XML file.  Returns None on a failure
                cma_process = self._fdb.getProcessByName("createMasterArclamps")
                if (cma_process is None or not isinstance(cma_process, fatboyProcess)):
                    print("findSlitletProcess::getCalibs> WARNING: could not find process createMasterArclamps!  Check your XML file!")
                    self._log.writeLog(__name__, "could not find process createMasterArclamps!  Check your XML file!", type=fatboyLog.WARNING)
                else:
                    cma_process.setDefaultOptions()
                    lampCalibs = cma_process.getCalibs(fdu, prevProc)
                    if ('masterLamp' in lampCalibs):
                        masterLamp = lampCalibs['masterLamp']
            if (masterLamp is not None):
                calibs['masterLamp'] = masterLamp
                print("findSlitletProcess::getCalibs> Using master arclamp "+masterLamp.getFullId()+" for slitlet detection...")
                self._log.writeLog(__name__, "Using master arclamp "+masterLamp.getFullId()+" for slitlet detection...")
            else:
                print("findSlitletProcess::getCalibs> WARNING: slitlet_autodetect_source is "+self.getOption("slitlet_autodetect_source", fdu.getTag())+" but no master arclamp found for "+fdu.getFullId()+".  Auto-detection will use the flat only.")
                self._log.writeLog(__name__, "slitlet_autodetect_source is "+self.getOption("slitlet_autodetect_source", fdu.getTag())+" but no master arclamp found for "+fdu.getFullId()+".  Auto-detection will use the flat only.", type=fatboyLog.WARNING)

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
        self._options.setdefault('boundary', 10)
        self._optioninfo.setdefault('boundary', 'Width in pixels of a boundary to not attempt to fit at the edges of each segment.  Should be 100 for MIRADAS.')
        self._options.setdefault('cut1d_max_threshold', 2)
        self._optioninfo.setdefault('cut1d_max_threshold', 'Reject a trace datapoint if 1d cut max < this factor * quartile of cut.')
        self._options.setdefault('edge_detection_method', 'auto')
        self._optioninfo.setdefault('edge_detection_method', 'Method used at each step to find the slitlet edge position:\ncross_correlation = cross-correlate 1-d cut with a reference cut and fit a\nGaussian to the correlation peak.  Best for slitlets separated by a genuine step edge\n(flux drops to ~0 between them).  Regresses badly on weak local-minimum boundaries (see\nlocal_minimum below) -- typically finds 0 datapoints for that edge.\nlocal_minimum = directly find the local minimum flux value in the 1-d cut (with subpixel\nparabolic refinement) instead of cross-correlating.  Much better for closely-packed\nslitlets where the boundary is only a weak dip in flux rather than a full step down to 0,\nwhich cross_correlation fails on -- but regresses badly on genuine step edges (a step\'s\nminimum sits at the edge of the search window, not at an interior parabolic minimum), so\nit is NOT a safe drop-in replacement for cross_correlation across a whole dataset.\nauto (default) = try cross_correlation first for every edge (matches cross_correlation exactly for\nany edge it can trace); only for an edge where that finds literally 0 datapoints (the\nweak-dip failure mode above) does it retry that same edge with local_minimum instead of\ngiving up.  Recommended over local_minimum whenever a dataset mixes both edge types,\nwhich is the common case (see findSlitletProcess algorithm audit notes).')
        self._options.setdefault('local_min_search_radius', '3')
        self._optioninfo.setdefault('local_min_search_radius', 'For local_minimum edge tracing: once the trace has accepted a datapoint, only search for\nthe minimum within this many pixels of the predicted position, and reject the point if the\nminimum is at the edge of that window (no real dip, e.g. a step between two lit slitlets).\nStops the trace drifting onto a random point of a fainter neighboring slitlet.')
        self._options.setdefault('local_min_depth_threshold', '0.05')
        self._optioninfo.setdefault('local_min_depth_threshold', 'For edge_detection_method=local_minimum only: minimum dip depth required to accept a\ndatapoint, as a fraction of the 1-d cut\'s local median flux.  Rejects steps where no real\ndip is present (e.g. pure noise or a genuine data gap).')
        self._options.setdefault('edge_extend_to_chip', 'no')
        self._optioninfo.setdefault('edge_extend_to_chip', 'If set to yes, and one edge of a slitlet is traced out, the other edge\nif it runs into the chip boundary will not be clipped.')
        self._options.setdefault('edge_threshold', 15)
        self._optioninfo.setdefault('edge_threshold', 'Do not attempt to trace out slitlets within this many pixels of edges')
        self._options.setdefault('flexure_correction', 'none')
        self._optioninfo.setdefault('flexure_correction', 'none | shift | gradient (linear = shift).  Correct for flexure between the\nflat and each object: measure the shift between the master flat and the object\'s\nframes from the slitlet edges (sky-lit), then give that object its own slitmask\nmoved to its frames and a master flat whose slit illumination is moved (pixel\nresponse stays in place).  shift = one shift per object; gradient = shift varying\nlinearly along the cross-dispersion direction.  A slitmask that is already aligned\nwith the object (e.g. from a region file drawn on the data) is not moved, only the flat.\nWrites findSlitlets/flexure_<object>.txt with every edge measurement.')
        self._options.setdefault('flexure_max_shift', '5')
        self._optioninfo.setdefault('flexure_max_shift', 'Largest flexure shift in pixels searched for by flexure_correction')
        self._options.setdefault('fiber_width', '5')
        self._optioninfo.setdefault('fiber_width', 'Width of fibers, used with peak local max')
        self._options.setdefault('fit_order', '2')
        self._optioninfo.setdefault('fit_order', 'Order of polynomial to use to fit slitlet shape.\nRecommended value = 2 for trace_slitlets_individually, 3 for group mode')
        self._options.setdefault('fit_function', 'polynomial')
        self._optioninfo.setdefault('fit_function', 'Function used to fit the traced (x,y) edge/shift datapoints to a smooth curve\nY=f(X):\npolynomial (default) = single global leastsq polynomial fit of fit_order, as before.\nA higher fit_order fits real curvature better locally but its extrapolation past the\nfitted x-range grows increasingly unstable (Runge\'s phenomenon) -- see rectifyProcess\'s\nsimilar spline-vs-polynomial finding.\nspline = smoothing B-spline (scipy UnivariateSpline, degree=min(fit_order,5)) through\nthe same datapoints.  Follows local curvature at least as well and extrapolates far\nmore stably at the fitted range\'s edges/gaps, at the cost of no longer having simple\npolynomial coefficients to log.  Falls back to polynomial automatically if there are\ntoo few datapoints for the requested spline degree.')
        self._options.setdefault('spline_smoothing', '-1')
        self._optioninfo.setdefault('spline_smoothing', 'For fit_function=spline only: smoothing factor (scipy UnivariateSpline\'s s).\n-1 (default) = let scipy pick its own default smoothing.  Larger values smooth more\n(fewer, gentler wiggles); 0 = interpolate every point exactly (no smoothing at all).')
        self._options.setdefault('invert_before_correlating', 'no')
        self._optioninfo.setdefault('invert_before_correlating', 'Invert flat field to turn gap trough into a peak for cross correlations')

        self._options.setdefault('max_residual_error', '2.0')
        self._optioninfo.setdefault('max_residual_error', 'Maximum sigma of residuals to fit to be rejected as an invalid fit, default 1.0')
        self._options.setdefault('min_coverage_fraction', '30')
        self._optioninfo.setdefault('min_coverage_fraction', 'Minimum percentage of a slitlet to trace out to be valid for a fit, default 30%')
        self._options.setdefault('narrow_gaps_between_slitlets', 'no')
        self._optioninfo.setdefault('narrow_gaps_between_slitlets', 'Set to yes for closely packed slitlets whose boundaries are only a dip in flux\nrather than a drop to background.  cross_correlation edge tracing rejects any datapoint\nfailing the cut1d_max_threshold (peak vs lower quartile) check, which assumes one side of\nevery edge is dark background, so every datapoint along a packed boundary is rejected.\nWith yes, such a datapoint is measured with local_minimum instead.  Unlike auto, which\nswitches a whole edge only when it finds 0 datapoints, this switches point by point, so\nit also handles an edge that is a step along part of the slit and packed along the rest.')
        self._options.setdefault('n_segments', '1')
        self._optioninfo.setdefault('n_segments', 'Number of piecewise functions to fit.  Should be 2 for MIRADAS, 1 for most other cases.')
        self._options.setdefault('order_step_size', '5')
        self._optioninfo.setdefault('order_step_size', 'Step size in pixels for tracing out orders, default = 5.')

        self._options.setdefault('padding','0')
        self._optioninfo.setdefault('padding', 'Number of pixels to pad slitlets by on each side, into the empty\nrows between slitlets.  A gap narrower than 2*padding is split between\nits two neighbors so slitlets never overlap.  Applies to all tracing\nmethods.  Default=0')
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
        self._options.setdefault('slitlet_autodetect_min_trough_depth', '0.3')
        self._optioninfo.setdefault('slitlet_autodetect_min_trough_depth', 'Minimum depth of the trough between two adjacent slitlets, as a fraction of\nthe fainter slitlet\'s height above background, to split them when the flux between\nthem does not drop all the way to background.  Lower (e.g. 0.1) for closely packed\nslitlets with very shallow boundaries; too low risks splitting slitlets at dust or\nbad-row dips.')
        self._options.setdefault('slitlet_autodetect_min_width', '10')
        self._optioninfo.setdefault('slitlet_autodetect_min_width', 'Minimum width of a slitlet for auto-detection')
        self._options.setdefault('slitlet_autodetect_use_orig_algorithm', 'no')
        self._optioninfo.setdefault('slitlet_autodetect_use_orig_algorithm', 'Set to yes to auto-detect slitlets with the original\nextractSpectra algorithm (extractSpectra_orig, versions <= 2.3.29): global sigma-clipped\nbackground, no trough splitting.')
        self._options.setdefault('slitlet_autodetect_source', 'flat')
        self._optioninfo.setdefault('slitlet_autodetect_source', 'Calibration frame used to auto-detect slitlets if no region file:\nflat (default) = steps in a 1-d cut of the master flat.\narclamp = correlation between adjacent rows of the master arclamp: rows within one slitlet\nshare the same line pattern, so a boundary between closely packed slitlets shows up even\nwhen the flat barely dips there, as long as adjacent slitlets are offset in wavelength.\nUses the full dispersion range, so is best suited to slitlets that are not strongly tilted.\nboth = slitlets and packed boundaries from the arclamp, outer edges refined to the flat\'s\nhalf-max, and flat regions with no coherent arc spectrum (e.g. a mask ID) reported and dropped.\nThe master arclamp is found or created via createMasterArclamps, which must be in the XML.')
        self._options.setdefault('slitlet_autodetect_arc_min_corr', '0.9')
        self._optioninfo.setdefault('slitlet_autodetect_arc_min_corr', 'For slitlet_autodetect_source = arclamp or both: minimum correlation between\nadjacent rows of the high-pass filtered arclamp for them to be part of the same slitlet.')
        self._options.setdefault('slitlet_validity_max_flat_roughness', '0.045')
        self._optioninfo.setdefault('slitlet_validity_max_flat_roughness', 'Flag a slitlet as invalid (e.g. a mask ID or alignment hole) if the robust row-to-row\nscatter of the flat across it, as a fraction of its flux, exceeds this.  Auto-detected\ninvalid slitlets are dropped; region file slitlets get a warning only.  0 = disable.')
        self._options.setdefault('slitlet_validity_min_arc_corr', '0.9')
        self._optioninfo.setdefault('slitlet_validity_min_arc_corr', 'When a master arclamp is used (slitlet_autodetect_source = arclamp or both), also flag a\nslitlet as invalid if the mean correlation between its adjacent arclamp rows is below this.\n0 = disable.')
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
        self._optioninfo.setdefault('trace_peak_local_max', 'Set to yes for MEGARA or other fiber data np.where the curvature changes between fibers')
        self._options.setdefault('trace_slitlets_individually', 'yes')
        self._optioninfo.setdefault('trace_slitlets_individually', 'Set to yes for echelle spectra np.where the curvature changes between slitlets.')
        self._options.setdefault('write_plots', 'no')
    #end setDefaultOptions

    #Fit Y=f(X) to (xdata,ydata) datapoints and evaluate at xeval, for either
    #traceOrders' or traceSlitlets' trace-curve fit step.  Returns
    #(yeval_at_xeval, yfit_at_xdata, coeffs) -- yfit_at_xdata lets a caller compute
    #residuals directly at the input datapoints without a second evaluation call,
    #and coeffs is the polynomial coefficient array for logging (None for spline,
    #which has no simple coefficient list).
    def fitTraceCurve(self, xdata, ydata, order, fit_function, xeval, spline_smoothing=-1):
        xdata = np.asarray(xdata, dtype=np.float64)
        ydata = np.asarray(ydata, dtype=np.float64)
        if (fit_function == "spline"):
            k = max(1, min(int(order), 5))
            if (len(xdata) > k):
                #UnivariateSpline requires strictly increasing, unique x.  Both trace
                #directions (walking out from xinit each way) can overlap near xinit,
                #so average y at any duplicate x rather than erroring or dropping data.
                srt = np.argsort(xdata)
                xu, uidx, counts = np.unique(xdata[srt], return_inverse=True, return_counts=True)
                if (len(xu) > k):
                    yu = np.zeros(len(xu), np.float64)
                    ysort = ydata[srt]
                    for i in range(len(xu)):
                        yu[i] = ysort[uidx == i].mean()
                    try:
                        s = None if (spline_smoothing < 0) else spline_smoothing
                        spl = UnivariateSpline(xu, yu, k=k, s=s)
                        return spl(xeval), spl(xdata), None
                    except Exception:
                        pass #Fall through to polynomial fallback below
        #polynomial (default, and spline fallback for too few/degenerate datapoints)
        p = np.zeros(order+1, np.float64)
        p[0] = ydata[-1] if (len(ydata) > 0) else 0.
        lsq = leastsq(polyResiduals, p, args=(xdata,ydata,order))
        return polyFunction(lsq[0], xeval, order), polyFunction(lsq[0], xdata, order), lsq[0]
    #end fitTraceCurve

    #Grow each slitlet by up to padding pixels into the empty rows next to it, column by column.
    #A gap narrower than 2*padding is split between the two neighbors (the lower slitlet gets the
    #extra row of an odd gap) so slitlets never overlap and packed slitlets are left with no zeros
    #between them.  Rows are the integer rows createSlitmask uses: int(ylo) to int(yhi).
    #Returns new (yloMask, yhiMask).
    def padSlitletEdges(self, yloMask, yhiMask, padding, ysize):
        ylo = np.floor(yloMask).astype(np.int64)
        yhi = np.floor(yhiMask).astype(np.int64)
        if (padding <= 0 or ylo.shape[0] == 0):
            return (yloMask, yhiMask)
        #Order slitlets bottom to top by their median center (slit numbering need not be sorted)
        order = np.argsort(np.median(ylo+yhi, 1))
        lo = ylo[order]
        hi = yhi[order]
        newlo = lo.copy()
        newhi = hi.copy()
        for k in range(len(order)-1):
            gap = np.maximum(lo[k+1]-hi[k]-1, 0)
            glo = np.minimum(padding, (gap+1)//2)
            ghi = np.minimum(padding, gap-glo)
            newhi[k] = hi[k]+glo
            newlo[k+1] = lo[k+1]-ghi
        newlo[0] = np.maximum(lo[0]-padding, 0)
        newhi[-1] = np.minimum(hi[-1]+padding, ysize-1)
        outlo = np.zeros(yloMask.shape)
        outhi = np.zeros(yhiMask.shape)
        outlo[order] = newlo
        outhi[order] = newhi
        return (outlo, outhi)
    #end padSlitletEdges

    #Sub-pixel shift of profile b relative to a (+ = b higher), from the cross-correlation of their derivatives
    #(slit edges).  NaN if the peak is at the edge of +-maxShift.
    def edgeShift(self, a, b, maxShift):
        da = np.diff(a)
        db = np.diff(b)
        if (np.abs(da).max() == 0 or np.abs(db).max() == 0 or len(da) <= 2*maxShift+2):
            return np.nan
        da = da/np.abs(da).max()
        db = db/np.abs(db).max()
        m = maxShift+1
        lags = np.arange(-maxShift, maxShift+1)
        cc = np.array([np.sum(da[m:-m]*np.roll(db, l)[m:-m]) for l in lags])
        i = int(np.argmax(cc))
        if (i == 0 or i == len(lags)-1):
            return np.nan
        denom = cc[i-1]-2*cc[i]+cc[i+1]
        if (denom == 0):
            return np.nan
        return -(lags[i]+0.5*(cc[i-1]-cc[i+1])/denom)
    #end edgeShift

    #Offsets (mask center - flat half-maximum center) of slitlets whose edges both drop to < 25% of the slit
    #level within 8 pixels (isolated, so the half-max is well defined).  Data are cross-dispersion x dispersion.
    def maskFlatCenterOffsets(self, mask, flat, cols, half):
        ny = mask.shape[0]
        out = []
        for xc in cols:
            p = np.median(flat[:,max(xc-half,0):xc+half+1], 1)
            mcol = mask[:,xc]
            for k in range(1, int(mask.max())+1):
                y = np.where(mcol == k)[0]
                if (y.size < 8 or y.min() < 20 or y.max() > ny-21):
                    continue
                lo = y.min()
                hi = y.max()
                plat = np.median(p[lo+4:hi-3])
                bg = np.min(p[lo-12:hi+13])
                if (plat-bg <= 0):
                    continue
                halfmax = bg+0.5*(plat-bg)
                quarter = bg+0.25*(plat-bg)
                a = (lo+hi)//2
                while (p[a-1] >= halfmax and a > lo-10):
                    a -= 1
                b = (lo+hi)//2
                while (p[b+1] >= halfmax and b < hi+10):
                    b += 1
                if (p[a-1] >= halfmax or p[b+1] >= halfmax):
                    continue
                if (min(p[a-8:a]) > quarter or min(p[b+1:b+9]) > quarter):
                    continue
                elo = a-1+(halfmax-p[a-1])/(p[a]-p[a-1])
                ehi = b+(p[b]-halfmax)/(p[b]-p[b+1])
                out.append((lo+hi)/2.-(elo+ehi)/2.)
        return np.array(out)
    #end maskFlatCenterOffsets

    #Measure flexure between the master flat and an object's frames from their slitlet edges.  Data are
    #cross-dispersion x dispersion.  Returns (coeffs, maskOffset, measurements, nkept) where the flat->object
    #shift at row y is coeffs[0] + coeffs[1]*(y-ny/2)/1000 (coeffs[1] = 0 for mode shift) and maskOffset is
    #(mask center - flat center).  coeffs is None if too few edges could be measured.
    def measureFlexure(self, mask, flat, frames, mode, maxShift):
        (ny, nx) = mask.shape
        half = 24
        pad = maxShift+3
        cols = np.linspace(0.1*nx, 0.9*nx, 9).astype(int)
        meas = []
        for xc in cols:
            pflat = np.median(flat[:,max(xc-half,0):xc+half+1], 1)
            pframes = [np.median(f[:,max(xc-half,0):xc+half+1], 1) for f in frames]
            mcol = mask[:,xc]
            for k in range(1, int(mask.max())+1):
                y = np.where(mcol == k)[0]
                if (y.size < 5 or y.min()-pad < 0 or y.max()+pad >= ny):
                    continue
                lo = y.min()-pad
                hi = y.max()+pad
                for p in pframes:
                    s = self.edgeShift(pflat[lo:hi+1], p[lo:hi+1], maxShift)
                    if (np.isfinite(s)):
                        meas.append((y.mean(), xc, k, s))
        meas = np.array(meas)
        if (len(meas) < 10):
            return (None, 0., meas, 0)
        #Iterative 3-sigma (MAD) clipping: edges distorted by a bright object in the slit are outliers
        s = meas[:,3]
        keep = np.ones(len(s), bool)
        coeffs = np.array([np.median(s), 0.])
        for j in range(5):
            if (mode == "gradient"):
                A = np.c_[np.ones(keep.sum()), (meas[keep,0]-ny/2.)/1000.]
                coeffs = np.linalg.lstsq(A, s[keep], rcond=None)[0]
            else:
                coeffs = np.array([np.median(s[keep]), 0.])
            resid = s-(coeffs[0]+coeffs[1]*(meas[:,0]-ny/2.)/1000.)
            mad = 1.4826*np.median(np.abs(resid[keep]))
            newkeep = np.abs(resid) < 3*max(mad, 0.05)
            if (np.array_equal(newkeep, keep)):
                break
            keep = newkeep
        offsets = self.maskFlatCenterOffsets(mask, flat, cols, half)
        maskOffset = float(np.median(offsets)) if (len(offsets) > 0) else 0.
        return (coeffs, maskOffset, meas, int(keep.sum()))
    #end measureFlexure

    #Per-object flexure correction (flexure_correction = shift | gradient): measure the shift between the
    #master flat and this object's frames from the slitlet edges, then make object-tagged copies of the
    #slitmask (moved to the object's frames) and of the master flat (its slit illumination moved, pixel
    #response left in place).  Later processes pick up the tagged copies for this object only.
    def correctFlexure(self, fdu, calibs, prevProc):
        mode = self.getOption("flexure_correction", fdu.getTag()).lower()
        if (mode == "linear"):
            mode = "shift"
        if (mode not in ["shift", "gradient"] or not 'slitmask' in calibs):
            return
        slitmask = calibs['slitmask']
        if (slitmask.hasProperty("flexure_corrected")):
            #Already corrected for this object
            return
        maxShift = int(self.getOption("flexure_max_shift", fdu.getTag()))
        #Master flat: the one the slitmask was traced from, or the one flatDivideSpec would use
        if ('masterFlat' in calibs):
            masterFlat = calibs['masterFlat']
        else:
            fds_process = self._fdb.getProcessByName("flatDivideSpec")
            if (fds_process is None or not isinstance(fds_process, fatboyProcess)):
                print("findSlitletProcess::correctFlexure> WARNING: could not find process flatDivideSpec - no flexure correction for "+fdu.getFullId())
                self._log.writeLog(__name__, "could not find process flatDivideSpec - no flexure correction for "+fdu.getFullId(), type=fatboyLog.WARNING)
                return
            fds_process.setDefaultOptions()
            masterFlat = fds_process.getCalibs(fdu, prevProc).get('masterFlat')
            if (masterFlat is None):
                print("findSlitletProcess::correctFlexure> WARNING: no master flat found - no flexure correction for "+fdu.getFullId())
                self._log.writeLog(__name__, "no master flat found - no flexure correction for "+fdu.getFullId(), type=fatboyLog.WARNING)
                return
        horizontal = (fdu.dispersion == fdu.DISPERSION_HORIZONTAL)
        #Work in cross-dispersion x dispersion
        def orient(a):
            return a if horizontal else a.transpose()
        mask = orient(np.asarray(slitmask.getData(force_cpu=True)))
        flat = orient(np.asarray(masterFlat.getData(force_cpu=True), dtype=np.float64))
        frames = []
        for frame in self._fdb.getFDUs(ident=fdu._id, filter=fdu.filter, section=fdu.section, tag=fdu.getTag()):
            if (frame.getShape() == fdu.getShape()):
                frames.append(orient(np.asarray(frame.getData(force_cpu=True), dtype=np.float64)))
        (coeffs, maskOffset, meas, nkept) = self.measureFlexure(mask, flat, frames, mode, maxShift)
        if (coeffs is None):
            print("findSlitletProcess::correctFlexure> WARNING: only "+str(len(meas))+" slitlet edges could be measured for "+fdu._id+" - no flexure correction.")
            self._log.writeLog(__name__, "only "+str(len(meas))+" slitlet edges could be measured for "+fdu._id+" - no flexure correction.", type=fatboyLog.WARNING)
            return
        ny = mask.shape[0]
        yrows = np.arange(ny, dtype=np.float64)
        flatShift = coeffs[0]+coeffs[1]*(yrows-ny/2.)/1000.
        #The mask may already sit off the flat (e.g. drawn from a region file on science data)
        maskShift = flatShift-maskOffset
        msg = "Flexure for "+fdu._id+" ("+str(len(frames))+" frames, "+str(nkept)+" of "+str(len(meas))+" edge measurements kept): flat -> object shift = "+formatNum(coeffs[0])
        if (mode == "gradient"):
            msg += " + "+formatNum(coeffs[1])+"*(y-"+str(ny//2)+")/1000"
        msg += " px; mask - flat center offset = "+formatNum(maskOffset)+" px; slitmask moved by "+formatNum(maskShift.min())+" to "+formatNum(maskShift.max())+" px."
        print("findSlitletProcess::correctFlexure> "+msg)
        self._log.writeLog(__name__, msg)

        #Shifted slitmask: row y takes the slitlet at row y - shift (nearest row)
        rows = np.clip(np.rint(yrows-maskShift), 0, ny-1).astype(np.int64)
        newMask = mask[rows,:]
        #Shifted flat: slit illumination (smooth along the dispersion direction) moves; pixel response stays
        illum = median_filter(flat, size=(1,31))
        good = illum > 0.02*np.median(illum[illum > 0])
        pixresp = np.ones(flat.shape)
        pixresp[good] = flat[good]/illum[good]
        if (mode == "gradient"):
            (yy, xx) = np.mgrid[0:flat.shape[0], 0:flat.shape[1]].astype(np.float64)
            illumShifted = map_coordinates(illum, [yy-flatShift.reshape(ny,1), xx], order=1, mode='nearest')
            del yy, xx
        else:
            illumShifted = ndshift(illum, (coeffs[0], 0), order=1, mode="nearest")
        newFlat = (illumShifted*pixresp).astype(np.float32)
        if (not horizontal):
            newMask = newMask.transpose().copy()
            newFlat = newFlat.transpose().copy()

        newSlitmask = self._fdb.addNewSlitmask(slitmask, newMask.astype(slitmask.getData(force_cpu=True).dtype), self._pname, tagname=slitmask._id+"_flexure_"+fdu._id, objectTag=fdu._id)
        newSlitmask.setProperty("nslits", int(newMask.max()))
        newSlitmask.setProperty("flexure_corrected", True)
        if (slitmask.hasProperty("regions")):
            (sylo, syhi, slitx, slitw) = slitmask.getProperty("regions")
            yc = np.clip(((np.asarray(sylo)+np.asarray(syhi))/2.).astype(np.int64), 0, ny-1)
            newSlitmask.setProperty("regions", (np.asarray(sylo)+maskShift[yc], np.asarray(syhi)+maskShift[yc], slitx, slitw))
        calibs['slitmask'] = newSlitmask
        #Object-tagged master flat, created under the flat's own process name so flatDivideSpec finds it
        newMasterFlat = fatboySpecCalib(masterFlat.getCalibProcessName(), "master_flat", masterFlat, data=newFlat, tagname=masterFlat._id+"_flexure_"+fdu._id, log=self._log)
        for key in ["specmode", "dispersion", "flat_method"]:
            if (masterFlat.hasProperty(key)):
                newMasterFlat.setProperty(key, masterFlat.getProperty(key))
        newMasterFlat._objectTags = [fdu._id]
        self._fdb.appendCalib(newMasterFlat)

        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
            if (not os.access(outdir+"/findSlitlets", os.F_OK)):
                os.mkdir(outdir+"/findSlitlets", 0o755)
            smfile = outdir+"/findSlitlets/"+newSlitmask.getFullId()
            if (os.access(smfile, os.F_OK)):
                os.unlink(smfile)
            newSlitmask.writeTo(smfile)
            #Every edge measurement, for QA (flagged 0 if rejected as an outlier)
            resid = meas[:,3]-(coeffs[0]+coeffs[1]*(meas[:,0]-ny/2.)/1000.)
            mad = 1.4826*np.median(np.abs(resid))
            f = open(outdir+"/findSlitlets/flexure_"+fdu._id+".txt", 'w')
            f.write("#"+msg+"\n#y_center\tx\tslitlet\tshift\tkept\n")
            for j in range(len(meas)):
                f.write(formatNum(meas[j,0])+"\t"+str(int(meas[j,1]))+"\t"+str(int(meas[j,2]))+"\t"+formatNum(meas[j,3])+"\t"+str(int(abs(resid[j]) < 3*max(mad, 0.05)))+"\n")
            f.close()
    #end correctFlexure

    #CPU equivalent of fatboyLibs.createSlitmask: rows int(yloMask)..int(yhiMask) of each column
    #belong to that slitlet; a later slitlet overwrites an earlier one.
    def slitmaskFromEdges(self, shape, yloMask, yhiMask, horizontal):
        nslits = yloMask.shape[0]
        slitmask = np.zeros(shape, dtype=np.int32)
        if (horizontal):
            yind = np.arange(shape[0], dtype=np.int32).reshape(shape[0], 1)
            for j in range(nslits):
                currMask = (yind >= yloMask[j,:].astype(np.int32))*(yind <= yhiMask[j,:].astype(np.int32))
                slitmask[currMask] = (j+1)
        else:
            xind = np.arange(shape[1], dtype=np.int32).reshape(1, shape[1])
            for j in range(nslits):
                currMask = (xind >= yloMask[j,:].astype(np.int32).reshape(shape[0], 1))*(xind <= yhiMask[j,:].astype(np.int32).reshape(shape[0], 1))
                slitmask[currMask] = (j+1)
        return slitmask
    #end slitmaskFromEdges

    #Apply the padding option to a traced slitmask: pad the edges and rebuild the slitmask from them.
    def applySlitletPadding(self, fdu, slitmask, yloMask, yhiMask, ysize):
        padding = int(self.getOption("padding", fdu.getTag()))
        if (padding <= 0):
            return (slitmask, yloMask, yhiMask)
        (yloMask, yhiMask) = self.padSlitletEdges(yloMask, yhiMask, padding, ysize)
        horizontal = (fdu.dispersion == fdu.DISPERSION_HORIZONTAL)
        if (self._fdb.getGPUMode()):
            slitmask = createSlitmask(slitmask.shape, yhiMask, yloMask, yloMask.shape[0], horizontal = horizontal)
        else:
            slitmask = self.slitmaskFromEdges(slitmask.shape, yloMask, yhiMask, horizontal)
        print("findSlitletProcess> Padded slitlets by up to "+str(padding)+" pixels into the gaps between them for "+fdu.getFullId())
        self._log.writeLog(__name__, "Padded slitlets by up to "+str(padding)+" pixels into the gaps between them for "+fdu.getFullId())
        return (slitmask, yloMask, yhiMask)
    #end applySlitletPadding

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
        bndry = int(self.getOption('boundary', fdu.getTag()))
        minCovFrac = float(self.getOption("min_coverage_fraction", fdu.getTag()))
        cut1d_max_threshold = float(self.getOption("cut1d_max_threshold", fdu.getTag()))
        narrow_gaps = (self.getOption("narrow_gaps_between_slitlets", fdu.getTag()).lower() == "yes")
        maxResidualError = float(self.getOption("max_residual_error", fdu.getTag()))
        edge_detection_method = self.getOption("edge_detection_method", fdu.getTag()).lower()
        local_min_depth_threshold = float(self.getOption("local_min_depth_threshold", fdu.getTag()))
        local_min_search_radius = int(self.getOption("local_min_search_radius", fdu.getTag()))
        fit_function = self.getOption("fit_function", fdu.getTag()).lower()
        spline_smoothing = float(self.getOption("spline_smoothing", fdu.getTag()))
        do_edge_extend = False
        if (self.getOption("edge_extend_to_chip", fdu.getTag()).lower() == "yes"):
            do_edge_extend = True

        #Check that region file exists
        if (regFile is None or not os.access(regFile, os.F_OK)):
            #If not, attempt to auto-detect slitlets!
            print("findSlitletProcess::traceOrders> No region file given.  Attempting to auto-detect slitlets...")
            self._log.writeLog(__name__, "No region file given.  Attempting to auto-detect slitlets...")
            isNormalized = False
            if (masterFlat.hasProperty("normalized") or masterFlat.hasHeaderValue('NORMAL01')):
                #has been normalized already
                isNormalized = True
            lampData = None
            if ('masterLamp' in calibs):
                lampData = calibs['masterLamp'].getData(force_cpu=True)
            (sylo, syhi, slitx, slitw) = self.autoDetectSlitlets(fdu, masterFlat.getData(force_cpu=True).copy(), normal=isNormalized, lampData=lampData)

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
        if (regFile is not None and os.access(regFile, os.F_OK)):
            #Keep region file slitlets as given but warn about any that look invalid
            self.warnInvalidRegionSlitlets(fdu, calibs, sylo, syhi, regFile)

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
        flatData = masterFlat.getData(force_cpu=True).copy()
        qaData = flatData.copy()
        #Slits can't extend beyond image top/bottom
        for j in range(nslits):
            sylo[j] = max(sylo[j], edge_thresh)
            syhi[j] = min(syhi[j], ysize-edge_thresh)

        #Set up yloMask and yhiMask arrays to track low and high values of each slitlet
        yloMask = np.zeros((nslits, xsize))
        yhiMask = np.zeros((nslits, xsize))

        #If CPU mode, will need to create slitmask as we go
        if (not self._fdb.getGPUMode()):
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                #Generate y index np.array
                slitmask = np.zeros((ysize,xsize), dtype=np.int32)
                yind = np.arange(xsize*ysize, dtype=np.int32).reshape(ysize,xsize)//xsize
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                #Generate x index np.array
                xind = np.arange(xsize*ysize, dtype=np.int32).reshape(xsize,ysize)%ysize
                slitmask = np.zeros((xsize,ysize), dtype=np.int32)

        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/findSlitlets", os.F_OK)):
            os.mkdir(outdir+"/findSlitlets",0o755)
        statsfile = outdir+"/findSlitlets/stats_"+masterFlat._id+".txt"
        f = open(statsfile,'w')
        #Count of slitlets that needed a straight (uncurved) fallback for at least one
        #segment, because their curvature could not be reliably traced.  A few bad
        #slitlets shouldn't cost us all the good ones -- see the check after this loop.
        n_slit_failures = 0
        #Loop over each slitlet and trace out top and bottom of slitlet
        t = time.time()
        for slitidx in range(nslits):
            #Set for this slitlet if any segment needs a straight fallback
            slit_degraded = False
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
                    #islit = flatData[syval-halfbox:syval+halfbox+1, xinit-5:xinit+6].sum(1).astype(np.float64)
                    islit = flatData[ylo_slit:ylo_slit+boxsize, xinit-5:xinit+6].sum(1).astype(np.float64)
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    #islit = flatData[xinit-5:xinit+6, syval-halfbox:syval+halfbox+1].sum(0).astype(np.float64)
                    islit = flatData[xinit-5:xinit+6, ylo_slit:ylo_slit+boxsize].sum(0).astype(np.float64)
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
                            oned_seg1 = flatData[y1:y2, x1:x2].sum(1).astype(np.float64)
                            oned_seg0 = flatData[y1:y2, x3:x4].sum(1).astype(np.float64)
                        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                            oned_seg1 = flatData[x1:x2, y1:y2].sum(0).astype(np.float64)
                            oned_seg0 = flatData[x3:x4, y1:y2].sum(0).astype(np.float64)
                        ccor = np.correlate(oned_seg0, oned_seg1, mode='same')
                        mcor = np.where(ccor == np.max(ccor))[0]
                        seg_shifts.append(len(ccor)//2-mcor[0])

                #Setup lists and arrays for within each loop
                #xcoords and ycoords contain lists of fit (x,y) points
                active_edge_method = edge_detection_method
                if (active_edge_method == "auto"):
                    active_edge_method = "cross_correlation"
                while True:
                    xcoords = []
                    ycoords = []
                    #median value of 1-d cuts and max values of cross correlations are kept and used as rejection criteria later
                    meds = []
                    maxcors = []
                    #Which datapoints were measured with local_minimum (maxcors not comparable across methods)
                    lmflags = []
                    #local_minimum searches only near currY once a datapoint has been accepted since the last reset
                    anchored = False
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
                            anchored = False
                        elif (currSeg != lastSeg):
                            anchored = False
                            if (len(xcoords) == 0):
                                currY = syval + seg_shifts[currSeg]
                                lastYs = [syval + seg_shifts[currSeg]]
                                lastXs = [xinit]
                            else:
                                lastIdx = np.where(np.abs(np.array(xcoords)-xs[j]) == min(np.abs(np.array(xcoords)-xs[j])))[0][0]
                                currY = ycoords[lastIdx]+seg_shifts[currSeg]
                                lastYs = [ycoords[lastIdx]+seg_shifts[currSeg]]
                                lastXs = [xcoords[lastIdx]]
                        if (currY < edge_thresh):
                            #This slitlet is nearing the edge of the chip.  Don't try to fit anymore values.
                            #Use values that have been fit already to trace it out
                            continue

                        intY = int(np.round(currY, 3))
                        ylo_slit = intY-halfbox
                        if (ylo_slit < 0):
                            ylo_slit = 0
                        elif (ylo_slit > ysize-boxsize):
                            ylo_slit = ysize-boxsize
                        #1-d cut of flat in cross-dispersion direction, sum of 11 pixels in dispersion direction centered at current X
                        #Only look at 21 pixel box in dispersion direction centered at currY  => 21x11 box => 21 pixel 1-d line
                        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                            #cut1d = flatData[intY-halfbox:intY+halfbox+1, int(xs[j]-5):int(xs[j]+6)].sum(1).astype(np.float64)
                            cut1d = flatData[ylo_slit:ylo_slit+boxsize, int(xs[j]-5):int(xs[j]+6)].sum(1).astype(np.float64)
                        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                            #cut1d = flatData[int(xs[j]-5):int(xs[j]+6), intY-halfbox:intY+halfbox+1].sum(0).astype(np.float64)
                            cut1d = flatData[int(xs[j]-5):int(xs[j]+6), ylo_slit:ylo_slit+boxsize].sum(0).astype(np.float64)
                        if (do_subtract_bkg):
                            cut1d -= cut1d.min()
                        if (do_invert):
                            cut1d = (cut1d.max()-cut1d)**2  #square
                            medVal = arraymedian(cut1d)
                            cut1d = medianfilterCPU(cut1d)
                            cut1d[cut1d < 0] = 0
                        #Check that there is data in this cut
                        if (cut1d.sum()/islit.sum() < 0.05):
                            f.write(str(slitidx)+'\t'+str(syval)+'\t'+str(xs[j])+'\t'+str(currY)+'\t6\n')
                            #Flux in cut1 is less than 5% of that in islit reference cut
                            continue
                        use_local_min = (active_edge_method == "local_minimum")
                        if (not use_local_min):
                            q1 = gpu_arraymedian(cut1d, nhigh=len(cut1d)//2) #quartile
                            cmax = cut1d.max()
                            if (cmax < 0):
                                f.write(str(slitidx)+'\t'+str(syval)+'\t'+str(xs[j])+'\t'+str(currY)+'\t7\n')
                                #Peak flux in cut1d negative - should be caught by #6 but just in case
                                continue
                            if ((do_subtract_bkg and cmax/q1 < 3) or abs(cmax/q1) < cut1d_max_threshold):
                                if (not narrow_gaps):
                                    f.write(str(slitidx)+'\t'+str(syval)+'\t'+str(xs[j])+'\t'+str(currY)+'\t7\n')
                                    #Peak flux in cut1d < 3*quartile
                                    continue
                                #No background on either side, so this point is on a packed
                                #boundary (a dip, not a step) -- use local_minimum for this point
                                use_local_min = True
                        if (use_local_min):
                            #Directly find the local minimum (weak dip) in cut1d instead of
                            #cross-correlating.  Search the whole cut1d window -- it is already
                            #centered on currY (the running prediction), same as cross_correlation
                            #mode's Gaussian fit guess below.
                            local_med = arraymedian(cut1d)
                            if (anchored):
                                #Search only near the running prediction.  A minimum at the edge of
                                #this small window means there is no dip here (e.g. a step between two
                                #lit slitlets of different brightness), so reject rather than let the
                                #trace drift onto a random point of the fainter plateau.
                                icen = int(np.round(currY, 3))-ylo_slit
                                i0 = max(1, icen-local_min_search_radius)
                                i1 = min(len(cut1d)-1, icen+local_min_search_radius+1)
                                if (i1-i0 < 3):
                                    f.write(str(slitidx)+'\t'+str(syval)+'\t'+str(xs[j])+'\t'+str(currY)+'\t9\n')
                                    continue
                                dip_idx = i0+int(np.argmin(cut1d[i0:i1]))
                                if (dip_idx == i0 or dip_idx == i1-1):
                                    f.write(str(slitidx)+'\t'+str(syval)+'\t'+str(xs[j])+'\t'+str(currY)+'\t9\n')
                                    #No interior minimum near the prediction - reject
                                    continue
                            else:
                                #Not yet anchored (currY may be the region file's value, which can be a
                                #few pixels off the real dip) - search the whole cut
                                dip_idx = int(np.argmin(cut1d))
                            dip_val = cut1d[dip_idx]
                            depth_ratio = (local_med-dip_val)/max(abs(local_med), 1.0)
                            if (depth_ratio < local_min_depth_threshold):
                                f.write(str(slitidx)+'\t'+str(syval)+'\t'+str(xs[j])+'\t'+str(currY)+'\t8\n')
                                #No real dip present (flat/noisy cut1d) - reject
                                continue
                            #Subpixel refine via 3-point parabolic interpolation
                            if (dip_idx > 0 and dip_idx < len(cut1d)-1):
                                y0 = cut1d[dip_idx-1]
                                y1 = cut1d[dip_idx]
                                y2 = cut1d[dip_idx+1]
                                denom = (y0-2*y1+y2)
                                delta = 0.5*(y0-y2)/denom if (denom != 0) else 0.
                            else:
                                delta = 0.
                            #Build an lsq-equivalent tuple so downstream code is unchanged:
                            #[amplitude(depth), center, width(unused), offset(dip value)], ier=1(success)
                            lsq = [np.array([local_med-dip_val, dip_idx+delta, 3., dip_val], np.float64), 1]
                            #Stand-in for maxcors (phase 2 rejection) - dip depth serves the same
                            #"how strong is this signal" role that np.max(ccor) does for cross_correlation
                            maxcor_val = local_med-dip_val
                        else:
                            #Cross correlate cut1d with islit
                            #Use numpy correlate since 1d cut -- not enough pixels to benefit from GPU
                            ccor = np.correlate(cut1d, islit, mode='same')
                            #Median filter with 51 pixel boxcar and set negative values to 0 before fitting
                            ccor = medianfilterCPU(ccor)
                            ccor[ccor < 0] = 0
                            #Use leastsq to fit Gaussian to cross-correlation function
                            p = np.zeros(4, np.float64)
                            #p[1] = round(currY,3) #center = currY
                            p[1] = np.round(currY, 3)-ylo_slit
                            p[2] = 3. #FWHM = 3
                            p[3] = 0.
                            p[0] = np.max(ccor)
                            #lsq argument should be centered at currY
                            #Use int(round(currY, 3)) to get around floating point bug
                            lsq = fitGaussian(ccor, maskNeg=True, guess=p)
                            if (lsq[1] == False):
                            #try:
                            #  lsq = leastsq(gaussResiduals, p, args=(np.arange(len(ccor))+intY-halfbox, ccor))
                            #except Exception as ex:
                                print ("LSQ 1")
                                import traceback
                                traceback.print_exc()
                                print("findSlitletProcess::traceOrders> Warning: Order "+str(slitidx)+", syval="+str(syval)+": Leastsq FAILED at "+str(xs[j]))
                                self._log.writeLog(__name__, "Order "+str(slitidx)+", syval="+str(syval)+": Leastsq FAILED at "+str(xs[j]), type=fatboyLog.WARNING)
                                continue
                            maxcor_val = np.max(ccor)
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
                        if (not do_invert):
                            medVal = arraymedian(cut1d)
                        if (j == 0):
                            #First datapoint -- update currX, currY, append to all lists
                            f.write(str(slitidx)+'\t'+str(syval)+'\t'+str(xs[j])+'\t'+str(lsq[0][1])+'\t0\n')
                            currY = lsq[0][1]
                            currX = xs[0]
                            meds.append(medVal)
                            maxcors.append(maxcor_val)
                            lmflags.append(use_local_min)
                            anchored = True
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
                                wavg += lastYs[i]/math.sqrt(abs(lastXs[i]-xs[j]))
                                wavgx += lastXs[i]/math.sqrt(abs(lastXs[i]-xs[j]))
                                wavgDivisor += 1./math.sqrt(abs(lastXs[i]-xs[j]))
                            if (wavgDivisor != 0):
                                wavg = wavg/wavgDivisor
                                wavgx = wavgx/wavgDivisor
                            else:
                                #We seem to have no datapoints in lastYs.  Simply use previous value
                                wavg = currY
                                wavgx = currX
                            #More than 50 pixels in deltaX between weight average of last 10
                            #datapoints and current X
                            #And not the discontinuity in middle of xs np.where we jump from end back to center
                            #because abs(xs[j]-xs[j-1]) == step
                            if (abs(xs[j]-xs[j-1]) == step and abs(wavgx-xs[j]) > 50):
                                if (len(lastYs) > 1):
                                    #Fit slope to lastYs
                                    lin = leastsq(linResiduals, [0.,0.], args=(np.array(lastXs),np.array(lastYs)))
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
                                    lin = leastsq(linResiduals, [0.,0.], args=(np.array(lastXs),np.array(lastYs)))
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
                                meds.append(medVal)
                                maxcors.append(maxcor_val)
                                lmflags.append(use_local_min)
                                anchored = True
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
                                meds.append(medVal)
                                maxcors.append(maxcor_val)
                                lmflags.append(use_local_min)
                                anchored = True
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
                                meds.append(medVal)
                                maxcors.append(maxcor_val)
                                lmflags.append(use_local_min)
                                anchored = True
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
                    if (active_edge_method == "cross_correlation" and edge_detection_method == "auto" and len(ycoords) == 0):
                        #cross_correlation found nothing for this edge - likely a weak local-minimum
                        #dip rather than a genuine step edge.  Retry this same edge with local_minimum
                        #before giving up on it entirely.
                        print("findSlitletProcess::traceOrders> Order "+str(slitidx)+", syval="+str(syval)+": cross_correlation found 0 datapoints, retrying with local_minimum...")
                        self._log.writeLog(__name__, "Order "+str(slitidx)+", syval="+str(syval)+": cross_correlation found 0 datapoints, retrying with local_minimum...", type=fatboyLog.WARNING)
                        active_edge_method = "local_minimum"
                        continue
                    break
                print("findSlitletProcess::traceOrders> Order "+str(slitidx)+", syval="+str(syval)+": found "+str(len(ycoords))+" datapoints.")
                self._log.writeLog(__name__, "Order "+str(slitidx)+", syval="+str(syval)+": found "+str(len(ycoords))+" datapoints.")
                #Phase 2 of rejection criteria after slitlets have been traced
                #Find outliers > 2.5 sigma in median value of 1-d cuts
                #and max values of cross correlations and remove them
                meds = np.array(meds)
                maxcors = np.array(maxcors)
                lmflags = np.array(lmflags, dtype=bool)
                #b = (meds > arraymedian(meds)-2.5*meds.std())*(maxcors > arraymedian(maxcors)-2.5*maxcors.std())
                xcoords = np.array(xcoords)
                ycoords = np.array(ycoords)
                xc_keep = [] #Create new lists for xcoords and ycoords that will be kept
                yc_keep = []
                iseg_keep = [] #And for segment number of those kept datapoints

                for seg in range(n_segments):
                    seg_name = ""
                    if (n_segments > 1):
                        seg_name = "segment "+str(seg)+" of "
                    seg_order = order
                    xstride = xsize//n_segments
                    sxlo = xstride*seg
                    sxhi = xstride*(seg+1)
                    segmask = (xcoords >= sxlo)*(xcoords < sxhi)
                    if (segmask.sum() < 5):
                        if (seg == 0):
                            z1.append(np.zeros(xstride))
                            yf0 = 0
                        else:
                            z1[-1] = np.concatenate([z1[-1], np.zeros(xstride)-yf0])
                        continue

                    b = (meds[segmask] >= arraymedian(meds[segmask])-2.5*meds[segmask].std())
                    #Cross-correlation peaks and local minimum dip depths are on different
                    #scales, so apply the maxcors criterion within each group separately
                    seg_maxcors = maxcors[segmask]
                    seg_lm = lmflags[segmask]
                    bmax = np.ones(len(seg_maxcors), dtype=bool)
                    for grp in [seg_lm, ~seg_lm]:
                        if (grp.sum() > 0):
                            bmax[grp] = seg_maxcors[grp] >= arraymedian(seg_maxcors[grp])-2.5*seg_maxcors[grp].std()
                    b = b*bmax
                    seg_xcoords = xcoords[segmask][b]
                    seg_ycoords = ycoords[segmask][b]

                    if (len(seg_xcoords) < 10):
                        seg_order = 1
                    elif (len(seg_xcoords) < 25 or (seg_xcoords.min() > (sxlo+sxhi)/2) or (seg_xcoords.max() < (sxlo+sxhi)/2)):
                        seg_order = min(2, seg_order)
                    elif (len(seg_xcoords) < 50):
                        seg_order = min(3, seg_order)

                    #xcoords = np.array(xcoords)[b]
                    #ycoords = np.array(ycoords)[b]
                    if (n_segments > 1):
                        print("\tSegment "+str(seg)+": rejecting outliers (phase 2) - kept "+str(len(seg_ycoords))+" of "+str(len(ycoords[segmask]))+" datapoints.")
                        self._log.writeLog(__name__, "Segment "+str(seg)+": rejecting outliers (phase 2) - kept "+str(len(ycoords))+" of "+str(len(ycoords[segmask]))+" datapoints.", printCaller=False, tabLevel=1)
                    else:
                        print("\trejecting outliers (phase 2) - kept "+str(len(seg_ycoords))+" datapoints.")
                        self._log.writeLog(__name__, "rejecting outliers (phase 2) - kept "+str(len(seg_ycoords))+" datapoints.", printCaller=False, tabLevel=1)
                    #xin = 1-d np.array of x indices
                    xin = np.arange(xstride, dtype=np.float32)+sxlo
                    #Fit trace curve (recommended 2nd order/degree) to datapoints, Y = f(X)
                    try:
                        yoffset, yfit_seg, _coeffs = self.fitTraceCurve(seg_xcoords, seg_ycoords, seg_order, fit_function, xin, spline_smoothing)
                    except Exception as ex:
                        print("findSlitletProcess::traceOrders> ERROR: Could not trace "+seg_name+"Slit "+str(slitidx+1)+" for "+fdu.getFullId()+": "+str(ex)+".  Using a straight (uncurved) fallback for this segment.")
                        self._log.writeLog(__name__, "Could not trace "+seg_name+"Slit "+str(slitidx+1)+" for "+fdu.getFullId()+": "+str(ex)+".  Using a straight (uncurved) fallback for this segment.", type=fatboyLog.ERROR)
                        slit_degraded = True
                        if (seg == 0):
                            z1.append(np.zeros(xstride))
                            yf0 = 0
                        else:
                            z1[-1] = np.concatenate([z1[-1], np.zeros(xstride)-yf0])
                        continue

                    #Compute output offsets and residuals from actual datapoints
                    yresid = yfit_seg-seg_ycoords
                    #Remove outliers and refit
                    b = np.abs(yresid) < yresid.mean()+2.5*yresid.std()
                    seg_xcoords = seg_xcoords[b]
                    seg_ycoords = seg_ycoords[b]

                    #Check coverage fraction
                    covfrac = len(seg_ycoords)*100.0/(len(xs)//n_segments)
                    if (n_segments > 1):
                        print("\tSegment "+str(seg)+": rejecting outliers (phase 3). Sigma = "+formatNum(yresid.std())+". Using "+str(len(seg_ycoords))+" datapoints to fit slitlets (Cov. Frac: "+formatNum(covfrac)+")")
                        self._log.writeLog(__name__, "Segment "+str(seg)+": rejecting outliers (phase 3). Sigma = "+formatNum(yresid.std())+". Using "+str(len(seg_ycoords))+" datapoints to fit slitlets (Cov. Frac: "+formatNum(covfrac)+")", printCaller=False, tabLevel=1)
                    else:
                        print("\trejecting outliers (phase 3). Sigma = "+formatNum(yresid.std())+". Using "+str(len(seg_ycoords))+" datapoints to fit slitlets (Cov. Frac: "+formatNum(covfrac)+")")
                        self._log.writeLog(__name__, "rejecting outliers (phase 3). Sigma = "+formatNum(yresid.std())+". Using "+str(len(seg_ycoords))+" datapoints to fit slitlets (Cov. Frac: "+formatNum(covfrac)+")", printCaller=False, tabLevel=1)
                    #A segment that fails either quality check gets the same straight
                    #(uncurved) fallback as an outright fit exception, rather than
                    #propagating an unreliable curve or discarding the whole image over
                    #one bad segment of one slitlet.
                    segBad = False
                    if (covfrac < minCovFrac):
                        print("findSlitletProcess::traceOrders> "+seg_name+"Slit "+str(slitidx+1)+" for "+fdu.getFullId()+": coverage fraction of "+formatNum(covfrac)+"% is below minimum threshold.  Using a straight (uncurved) fallback for this segment.")
                        self._log.writeLog(__name__, seg_name+"Slit "+str(slitidx+1)+" for "+fdu.getFullId()+": coverage fraction of "+formatNum(covfrac)+"% is below minimum threshold.  Using a straight (uncurved) fallback for this segment.", type=fatboyLog.ERROR)
                        segBad = True
                    if (yresid.std() > maxResidualError):
                        print("findSlitletProcess::traceOrders> "+seg_name+"Slit "+str(slitidx+1)+" for "+fdu.getFullId()+": sigma of "+formatNum(yresid.std())+" is greater than max residual error of "+str(maxResidualError)+".  Using a straight (uncurved) fallback for this segment.")
                        self._log.writeLog(__name__, seg_name+"Slit "+str(slitidx+1)+" for "+fdu.getFullId()+": sigma of "+formatNum(yresid.std())+" is greater than max residual error of "+str(maxResidualError)+".  Using a straight (uncurved) fallback for this segment.", type=fatboyLog.ERROR)
                        segBad = True
                    if (segBad):
                        slit_degraded = True
                        if (seg == 0):
                            z1.append(np.zeros(xstride))
                            yf0 = 0
                        else:
                            z1[-1] = np.concatenate([z1[-1], np.zeros(xstride)-yf0])
                        continue

                    #Refit with outliers removed
                    try:
                        yoffset, _, coeffs = self.fitTraceCurve(seg_xcoords, seg_ycoords, seg_order, fit_function, xin, spline_smoothing)
                    except Exception as ex:
                        print("findSlitletProcess::traceOrders> ERROR: Could not trace "+seg_name+"Slit "+str(slitidx+1)+" for "+fdu.getFullId()+": "+str(ex)+".  Using a straight (uncurved) fallback for this segment.")
                        self._log.writeLog(__name__, "Could not trace "+seg_name+"Slit "+str(slitidx+1)+" for "+fdu.getFullId()+": "+str(ex)+".  Using a straight (uncurved) fallback for this segment.", type=fatboyLog.ERROR)
                        slit_degraded = True
                        if (seg == 0):
                            z1.append(np.zeros(xstride))
                            yf0 = 0
                        else:
                            z1[-1] = np.concatenate([z1[-1], np.zeros(xstride)-yf0])
                        continue

                    if (coeffs is not None):
                        print("\tFit = "+formatList(coeffs))
                        self._log.writeLog(__name__, "Fit = "+formatList(coeffs), printCaller=False, tabLevel=1)
                    else:
                        print("\tFit = spline (k="+str(max(1, min(int(seg_order), 5)))+")")
                        self._log.writeLog(__name__, "Fit = spline (k="+str(max(1, min(int(seg_order), 5)))+")", printCaller=False, tabLevel=1)
                    #Create new yoffset at every integer x
                    if (seg == 0):
                        #Subtract zero point
                        z1.append(yoffset - yoffset[0])
                        yf0 = yoffset[0]
                    else:
                        z1[-1] = np.concatenate([z1[-1], (yoffset - yf0)])
                #Generate qa data
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    for i in range(len(xcoords)):
                        yval = int(ycoords[i]+.5)
                        xval = int(xcoords[i]+.5)
                        if (yval == 0 or yval > qaData.shape[0]-2):
                            continue
                        for yi in range(yval-1,yval+2):
                            for xi in range(xval-1,xval+2):
                                dist = math.sqrt((ycoords[i]-yi)**2+(xcoords[i]-xi)**2)
                                qaData[yi,xi] = -50000/((1+dist)**2)
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    for i in range(len(xcoords)):
                        yval = int(ycoords[i]+.5)
                        xval = int(xcoords[i]+.5)
                        for yi in range(yval-1,yval+2):
                            for xi in range(xval-1,xval+2):
                                dist = math.sqrt((ycoords[i]-yi)**2+(xcoords[i]-xi)**2)
                                qaData[xi,yi] = -50000/((1+dist)**2)
            #end for syval
            if (slit_degraded):
                n_slit_failures += 1
            #Update slitmask
            ylo = sylo[slitidx]-z1[0][int(slitx[slitidx])]-1
            yhi = syhi[slitidx]-z1[1][int(slitx[slitidx])]
            if (do_edge_extend):
                if (ylo <= edge_thresh):
                    ylo = 0
                if (yhi >= ysize-edge_thresh):
                    yhi = ysize-1
            yloMask[slitidx,:] = ylo+z1[0]
            yhiMask[slitidx,:] = yhi+z1[1]
            if (not self._fdb.getGPUMode()):
                #Update slitmask piece by piece here for CPU
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    currMask = (yind >= (ylo+z1[0]).astype(np.int32))*(yind <= (yhi+z1[1]).astype(np.int32))
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    z1[0] = z1[0].reshape(xsize,1)
                    z1[1] = z1[1].reshape(xsize,1)
                    currMask = (xind >= (ylo+z1[0]).astype(np.int32))*(xind <= (yhi+z1[1]).astype(np.int32))
                b = np.where(currMask)
                slitmask[b] = (slitidx+1)
        #end for slitidx
        #print time.time()-t
        f.close()
        #Check for errors AFTER writing QA data.  A few slitlets needing a straight
        #fallback (already logged loudly above) shouldn't cost us all the good ones --
        #only discard the whole image if literally nothing could be traced.
        if (n_slit_failures > 0):
            print("findSlitletProcess::traceOrders> WARNING: "+str(n_slit_failures)+" of "+str(nslits)+" slitlets for "+fdu.getFullId()+" needed a straight (uncurved) fallback for at least one segment.  See errors above.")
            self._log.writeLog(__name__, str(n_slit_failures)+" of "+str(nslits)+" slitlets for "+fdu.getFullId()+" needed a straight (uncurved) fallback for at least one segment.", type=fatboyLog.WARNING)
            qafile = outdir+"/findSlitlets/qa_"+masterFlat.getFullId()
            #Write out qa file so degraded/failed slitlets can be visually inspected
            if (not os.access(qafile, os.F_OK)):
                #TODO - GPU for qa data?
                #Generate qa data
                #if (self._fdb.getGPUMode()):
                #  #Use GPU
                #  flatData = generateQAData(flatData, xcoords, ycoords, sylo, syhi, horizontal = (fdu.dispersion == fdu.DISPERSION_HORIZONTAL))
                masterFlat.tagDataAs("slitqa", qaData)
                masterFlat.writeTo(qafile, tag="slitqa")
                masterFlat.removeProperty("slitqa")
                #Don't del qaData here -- the slitmask qa file below still needs it
            if (n_slit_failures == nslits):
                #Every single slitlet needed a fallback -- this isn't "a few bad
                #slitlets," it's nothing usable at all, so discard the image.
                print("findSlitletProcess::traceOrders> ERROR: Could not trace any slitlets for "+fdu.getFullId()+"! Discarding Image!")
                self._log.writeLog(__name__, "Could not trace any slitlets for "+fdu.getFullId()+"!  Discarding Image!", type=fatboyLog.ERROR)
                #disable this FDU
                fdu.disable()
                return calibs

        #GPU mode - create slitmask at once
        if (self._fdb.getGPUMode()):
            #Use GPU
            slitmask = createSlitmask(flatData.shape, yhiMask, yloMask, nslits, horizontal = (fdu.dispersion == fdu.DISPERSION_HORIZONTAL))
        #Pad slitlets into the gaps between them if requested
        (slitmask, yloMask, yhiMask) = self.applySlitletPadding(fdu, slitmask, yloMask, yhiMask, ysize)

        if (slitmask.max() < 256):
            #Only convert to UInt8 if less than 256 slits
            slitmask = slitmask.astype(np.uint8)

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
            lampData = None
            if ('masterLamp' in calibs):
                lampData = calibs['masterLamp'].getData(force_cpu=True)
            (sylo, syhi, slitx, slitw) = self.autoDetectSlitlets(fdu, masterFlat.getData(force_cpu=True).copy(), normal=isNormalized, lampData=lampData)

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
        flatData = masterFlat.getData(force_cpu=True).copy()
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
                cut1d = flatData[:,int(xs[j]-halfbox):int(xs[j]+halfbox+1)].sum(1).astype(np.float64)
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                cut1d = flatData[int(xs[j]-halfbox):int(xs[j]+halfbox+1),:].sum(0).astype(np.float64)

            y = np.where(cut1d > np.median(cut1d))
            x = np.r_[True, cut1d[1:] > cut1d[:-1]] & np.r_[cut1d[:-1] > cut1d[1:], True] & np.r_[True, True, cut1d[2:] > cut1d[:-2]] & np.r_[cut1d[:-2] > cut1d[2:], True, True]
            x[:y[0][0]] = False
            x[y[0][-1]+1:] = False
            z = np.where(x)[0]
            if (len(z) != nslits):
                #Number of slits found doesn't match
                #print "ERR1", xs[j]
                continue

            maxerr = 2
            if (abs(xs[j]-lastX) > 50):
                maxerr = abs(xs[j]-lastX)/25
            if (np.abs(z-currYs).max() > maxerr):
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

        xcoords = np.array(xcoords)
        ycoords = np.array(ycoords)

        #1d np.array of indices nearest each x index
        idx = np.zeros(xsize, np.int32)
        for xi in range(xsize):
            idx[xi] = np.abs(xi-xcoords).argmin()
        #new xs = 1-d np.array of x indices
        xs = np.arange(xsize, dtype=np.int32)

        yloMask = np.zeros((nslits, xsize))
        yhiMask = np.zeros((nslits, xsize))
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
                #Generate y index np.array
                yind = np.arange(xsize*ysize, dtype=np.int32).reshape(ysize,xsize)//xsize
                slitmask = np.zeros((ysize,xsize), dtype=np.int32)
                for j in range(nslits):
                    currMask = (yind >= (yloMask[j,:]).astype(np.int32))*(yind <= (yhiMask[j,:]).astype(np.int32))
                    b = np.where(currMask)
                    slitmask[b] = (j+1)
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                #Generate x index np.array
                xind = np.arange(xsize*ysize, dtype=np.int32).reshape(ysize,xsize)%xsize
                slitmask = np.zeros((ysize,xsize), dtype=np.int32)
                for j in range(nslits):
                    currMask = (xind >= (yloMask[j,:]).astype(np.int32))*(xind <= (yhiMask[j,:]).astype(np.int32))
                    b = np.where(currMask)
                    slitmask[b] = (j+1)
        #Pad slitlets into the gaps between them if requested
        (slitmask, yloMask, yhiMask) = self.applySlitletPadding(fdu, slitmask, yloMask, yhiMask, ysize)
        if (slitmask.max() < 256):
            #Only convert to UInt8 if less than 256 slits
            slitmask = slitmask.astype(np.uint8)

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
                        #ycoords[i] holds every fiber's position at this x: mark them all
                        yval = (ycoords[i]+.5).astype(np.int32)
                        xval = int(xcoords[i]+.5)
                        for yi in range(-1,2):
                            for xi in range(-1,2):
                                dist = np.sqrt((yi**2)+(xi**2))
                                flatData[np.clip(yval+yi, 0, flatData.shape[0]-1), min(max(xval+xi, 0), flatData.shape[1]-1)] = -50000/((1+dist)**2)
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    for i in range(len(xcoords)):
                        #ycoords[i] holds every fiber's position at this x: mark them all
                        yval = (ycoords[i]+.5).astype(np.int32)
                        xval = int(xcoords[i]+.5)
                        for yi in range(-1,2):
                            for xi in range(-1,2):
                                dist = np.sqrt((yi**2)+(xi**2))
                                flatData[min(max(xval+xi, 0), flatData.shape[0]-1), np.clip(yval+yi, 0, flatData.shape[1]-1)] = -50000/((1+dist)**2)
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
        fit_function = self.getOption("fit_function", fdu.getTag()).lower()
        spline_smoothing = float(self.getOption("spline_smoothing", fdu.getTag()))
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
            lampData = None
            if ('masterLamp' in calibs):
                lampData = calibs['masterLamp'].getData(force_cpu=True)
            (sylo, syhi, slitx, slitw) = self.autoDetectSlitlets(fdu, masterFlat.getData(force_cpu=True).copy(), normal=isNormalized, lampData=lampData)

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
        if (regFile is not None and os.access(regFile, os.F_OK)):
            #Keep region file slitlets as given but warn about any that look invalid
            self.warnInvalidRegionSlitlets(fdu, calibs, sylo, syhi, regFile)

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
        flatData = masterFlat.getData(force_cpu=True).copy()
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
            islit = flatData[slitlet_trace_ylo:slitlet_trace_yhi, xinit-5:xinit+6].sum(1).astype(np.float64)
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            islit = flatData[xinit-5:xinit+6, slitlet_trace_ylo:slitlet_trace_yhi].sum(0).astype(np.float64)

        #Write per-datapoint diagnostic stats, same format/codes as traceOrders'
        #stats_<flatid>.txt (0=kept, 1=maxfev exceeded, 2=negative flux, 3=negative
        #width, 4=jump too large vs trend, 5=too far from predicted trend).  traceSlitlets
        #only fits one combined shift curve (not per-edge), so there is one row per x step
        #rather than per slitlet.
        if (not os.access(outdir+"/findSlitlets", os.F_OK)):
            os.mkdir(outdir+"/findSlitlets",0o755)
        statsfile = outdir+"/findSlitlets/stats_"+masterFlat._id+".txt"
        f = open(statsfile,'w')

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
                cut1d = flatData[slitlet_trace_ylo:slitlet_trace_yhi, int(xs[j]-5):int(xs[j]+6)].sum(1).astype(np.float64)
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                cut1d = flatData[int(xs[j]-5):int(xs[j]+6), slitlet_trace_ylo:slitlet_trace_yhi].sum(0).astype(np.float64)
            #Cross correlate cut1d with islit
            #Use numpy correlate since 1d cut -- not enough pixels to benefit from GPU
            ccor = np.correlate(cut1d, islit, mode='same')
            #Median filter with 51 pixel boxcar and set negative values to 0 before fitting
            if (self._fdb.getGPUMode()):
                ccor = gpumedianfilter(ccor)
            else:
                ccor = medianfilterCPU(ccor)
            ccor[ccor < 0] = 0

            #Use leastsq to fit Gaussian to cross-correlation function
            p = np.zeros(4, np.float64)
            p[1] = np.round(currY,3) #center = currY
            p[2] = 3. #FWHM = 3
            p[3] = 0.
            #Only examine up to 51 pixels centered at previous result
            llo = max(0, int(cen+p[1]-25))
            lhi = min(len(ccor), int(cen+p[1]+26))
            #print xs[j], llo, lhi, len(ccor), currY
            #print islit.size, cut1d.size, slitlet_trace_ylo, slitlet_trace_yhi
            p[0] = np.max(ccor[llo:lhi])
            try:
                lsq = leastsq(gaussResiduals, p, args=(np.arange(lhi-llo, dtype=np.float64)+llo-cen, np.asarray(ccor[llo:lhi])))
            except Exception as ex:
                import traceback
                traceback.print_exc()
                print("findSlitletProcess::traceSlitlets> Warning: Leastsq FAILED at "+str(xs[j])+" with "+str(ex))
                self._log.writeLog(__name__, "Leastsq FAILED at "+str(xs[j])+" with "+str(ex), type=fatboyLog.WARNING)
                continue
            #Error checking results of leastsq call
            if (lsq[1] == 5):
                f.write(str(xs[j])+'\t'+str(lsq[0][1])+'\t1\n')
                #exceeded max number of calls = ignore
                continue
            if (lsq[0][0]+lsq[0][3] < 0):
                f.write(str(xs[j])+'\t'+str(lsq[0][1])+'\t2\n')
                #flux less than zero = ignore
                continue
            if (lsq[0][2] < 0 and j != 0):
                f.write(str(xs[j])+'\t'+str(lsq[0][1])+'\t3\n')
                #negative boxsize = ignore unless first datapoint
                continue
            if (j == 0):
                #First datapoint -- update currX, currY, append to all lists
                f.write(str(xs[j])+'\t'+str(lsq[0][1])+'\t0\n')
                currY = lsq[0][1]
                currX = xs[0]
                meds.append(arraymedian(cut1d))
                maxcors.append(np.max(ccor))
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
                    wavg += lastYs[i]/math.sqrt(abs(lastXs[i]-xs[j]))
                    wavgx += lastXs[i]/math.sqrt(abs(lastXs[i]-xs[j]))
                    wavgDivisor += 1./math.sqrt(abs(lastXs[i]-xs[j]))
                if (wavgDivisor != 0):
                    wavg = wavg/wavgDivisor
                    wavgx = wavgx/wavgDivisor
                else:
                    #We seem to have no datapoints in lastYs.  Simply use previous value
                    wavg = currY
                    wavgx = currX
                #More than 50 pixels in deltaX between weight average of last 10
                #datapoints and current X
                #And not the discontinuity in middle of xs np.where we jump from end back to center
                #because abs(xs[j]-xs[j-1]) == step
                if (abs(xs[j]-xs[j-1]) == step and abs(wavgx-xs[j]) > 50):
                    if (len(lastYs) > 1):
                        #Fit slope to lastYs
                        lin = leastsq(linResiduals, [0.,0.], args=(np.array(lastXs),np.array(lastYs)))
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
                        lin = leastsq(linResiduals, [0.,0.], args=(np.array(lastXs),np.array(lastYs)))
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
                    f.write(str(xs[j])+'\t'+str(lsq[0][1])+'\t0\n')
                    currY = lsq[0][1]
                    currX = xs[j]
                    meds.append(arraymedian(cut1d))
                    maxcors.append(np.max(ccor))
                    xcoords.append(xs[j])
                    ycoords.append(lsq[0][1])
                    lastXs.append(xs[j])
                    lastYs.append(lsq[0][1])
                elif (abs(lsq[0][1] - refY) < maxerr):
                    #Regular datapoint.  Apply sanity check rejection criteria here
                    #Discard if farther than maxerr away from refY
                    if (abs(xs[j]-currX) < 4*step and maxerr > 1 and abs(lsq[0][1]-currY) > maxerr):
                        #Also discard if < 20 pixels in X from last fit datapoint, and deltaY > 1
                        f.write(str(xs[j])+'\t'+str(lsq[0][1])+'\t4\n')
                        continue
                    #update currX, currY, append to all lists
                    f.write(str(xs[j])+'\t'+str(lsq[0][1])+'\t0\n')
                    currY = lsq[0][1]
                    currX = xs[j]
                    meds.append(arraymedian(cut1d))
                    maxcors.append(np.max(ccor))
                    xcoords.append(xs[j])
                    ycoords.append(lsq[0][1])
                    lastXs.append(xs[j])
                    lastYs.append(lsq[0][1])
                    #keep lastXs and lastYs at 10 elements or less
                    if (len(lastYs) > 10):
                        lastXs.pop(0)
                        lastYs.pop(0)
                else:
                    f.write(str(xs[j])+'\t'+str(lsq[0][1])+'\t5\n')
            #print xs[j], p[1], len(maxcors), lsq[0][1], arraymedian(cut1d), max(ccor)
        f.close()
        print("findSlitletProcess::traceSlitlets> found "+str(len(ycoords))+" datapoints.")
        self._log.writeLog(__name__, "found "+str(len(ycoords))+" datapoints.")
        #Phase 2 of rejection criteria after slitlets have been traced
        #Find outliers > 2.5 sigma in median value of 1-d cuts
        #and max values of cross correlations and remove them
        meds = np.array(meds)
        maxcors = np.array(maxcors)
        b = (meds >= arraymedian(meds)-2.5*meds.std())*(maxcors >= arraymedian(maxcors)-2.5*maxcors.std())
        xcoords = np.array(xcoords)[b]
        ycoords = np.array(ycoords)[b]
        print("\trejecting outliers (phase 2) - kept "+str(len(ycoords))+" datapoints.")
        self._log.writeLog(__name__, "rejecting outliers (phase 2) - kept "+str(len(ycoords))+" datapoints.", printCaller=False, tabLevel=1)
        #new xs = 1-d np.array of x indices
        xs = np.arange(xsize, dtype=np.float32)
        #Fit trace curve (recommended 3rd order/degree) to datapoints, Y = f(X)
        try:
            yoffset, yfit, _coeffs = self.fitTraceCurve(xcoords, ycoords, order, fit_function, xs, spline_smoothing)
        except Exception as ex:
            print("findSlitletProcess::traceOrders> ERROR: Could not trace slitlets for "+fdu.getFullId()+"! Discarding Image!")
            self._log.writeLog(__name__, "Could not trace slitlets for "+fdu.getFullId()+"! Discarding Image!", type=fatboyLog.ERROR)
            #disable this FDU
            fdu.disable()
            return calibs

        #Compute output offsets and residuals from actual datapoints
        yresid = yfit-ycoords
        #Remove outliers and refit
        b = np.abs(yresid) < yresid.mean()+2.5*yresid.std()
        xcoords = xcoords[b]
        ycoords = ycoords[b]
        print("\trejecting outliers (phase 3). Sigma = "+formatNum(yresid.std())+". Using "+str(len(ycoords))+" datapoints to fit slitlets.")
        self._log.writeLog(__name__, "rejecting outliers (phase 3). Sigma = "+formatNum(yresid.std())+". Using "+str(len(ycoords))+" datapoints to fit slitlets.", printCaller=False, tabLevel=1)
        #Refit with outliers removed
        try:
            yoffset, _, _coeffs2 = self.fitTraceCurve(xcoords, ycoords, order, fit_function, xs, spline_smoothing)
        except Exception as ex:
            print("findSlitletProcess::traceOrders> ERROR: Could not trace slitlets for "+fdu.getFullId()+"! Discarding Image!")
            self._log.writeLog(__name__, "Could not trace slitlets for "+fdu.getFullId()+"! Discarding Image!", type=fatboyLog.ERROR)
            #disable this FDU
            fdu.disable()
            return calibs

        #Subtract zero point
        z1 = yoffset - yoffset[0]
        yloMask = np.zeros((nslits, len(z1)))
        yhiMask = np.zeros((nslits, len(z1)))
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
                #Generate y index np.array
                yind = np.arange(xsize*ysize, dtype=np.int32).reshape(ysize,xsize)//xsize
                slitmask = np.zeros((ysize,xsize), dtype=np.int32)
                for j in range(nslits):
                    #ylo = sylo[j]-z1[int(slitx[j])]-1
                    ylo = sylo[j]-z1[int(slitx[j])]
                    yhi = syhi[j]-z1[int(slitx[j])]
                    currMask = (yind >= (ylo+z1).astype(np.int32))*(yind <= (yhi+z1).astype(np.int32))
                    b = np.where(currMask)
                    slitmask[b] = (j+1)
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                #Generate x index np.array
                xind = np.arange(xsize*ysize, dtype=np.int32).reshape(ysize,xsize)%xsize
                slitmask = np.zeros((ysize,xsize), dtype=np.int32)
                #Need to reshape z1 np.array
                z1 = z1.reshape((len(z1), 1))
                for j in range(nslits):
                    ylo = sylo[j]-z1[int(slitx[j])]-1
                    yhi = syhi[j]-z1[int(slitx[j])]
                    currMask = (xind >= (ylo+z1).astype(np.int32))*(xind <= (yhi+z1).astype(np.int32))
                    b = np.where(currMask)
                    slitmask[b] = (j+1)
        #Pad slitlets into the gaps between them if requested
        (slitmask, yloMask, yhiMask) = self.applySlitletPadding(fdu, slitmask, yloMask, yhiMask, ysize)
        if (slitmask.max() < 256):
            #Only convert to UInt8 if less than 256 slits
            slitmask = slitmask.astype(np.uint8)

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
                #Generate qa data on the CPU in both modes (cheap; the GPU kernel used float32 positions and
                #overlapping marks raced, so GPU and CPU QA images differed)
                #CPU version -- loop over coords first
                for j in range(len(xcoords)):
                    xval = int(xcoords[j]+.5)
                    qaxs = np.arange(9, dtype=np.int32).reshape((3,3))%3+xval-1
                    ys = np.arange(9, dtype=np.int32).reshape((3,3))//3
                    #There will be 18 x nslits x ncoords pixels used to show np.where slitlets were traced out
                    for i in range(nslits):
                        yval = int(ycoords[j]+sylo[i]+0.5)
                        #calculate x and y 3x3 index arrays
                        qays = ys+yval-1
                        dist = np.sqrt((ycoords[j]+sylo[i]-qays)**2+(xcoords[j]-qaxs)**2)
                        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                            flatData[qays,qaxs] = -50000/((1+dist)**2)
                        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                            flatData[qaxs,qays] = -50000/((1+dist)**2)
                        yval = int(ycoords[j]+syhi[i]+0.5)
                        qays = ys+yval-1
                        dist = np.sqrt((ycoords[j]+syhi[i]-qays)**2+(xcoords[j]-qaxs)**2)
                        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                            flatData[qays,qaxs] = -50000/((1+dist)**2)
                        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                            flatData[qaxs,qays] = -50000/((1+dist)**2)
                masterFlat.tagDataAs("slitqa", flatData)
                masterFlat.writeTo(qafile, tag="slitqa")
                masterFlat.removeProperty("slitqa")
        return calibs
    #end traceSlitlets
