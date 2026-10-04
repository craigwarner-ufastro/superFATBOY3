from superFATBOY.fatboyDataUnit import fatboyDataUnit
from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.datatypeExtensions.fatboySpecCalib import fatboySpecCalib
from superFATBOY.datatypeExtensions.fatboySpectrum import fatboySpectrum
from superFATBOY.fatboyProcesses.wavelengthCalibrateProcess import wavelengthCalibrateProcess
from superFATBOY import gpu_drihizzle, drihizzle
import numpy as np
import math
import traceback
from scipy.optimize import leastsq
import inspect

usePlot = True
try:
    import matplotlib.pyplot as plt
except Exception as ex:
    print("Warning: Could not import matplotlib!")
    usePlot = False

block_size = 512

#Stand-in for fatboyLog when running without a fatboyDatabase: messages are printed by the methods themselves
class quietLog:
    def writeLog(self, *args, **kwargs):
        pass
#end quietLog

## Wavelength calibration of a single 1-d cut (one slitlet/segment) outside the pipeline.  Uses the matching, fitting,
## fallback and QA methods of wavelengthCalibrateProcess, so the results agree with the pipeline's for the same cut.
## Guesses that need other slitlets (neighbor, trend, learned intensities) work when the caller passes earlier results
## in calibs: 'solvedCuts' (list of (slitlet-1, segment-1, 1-d cut, coefficients, order)) and 'lineMeasures'
## ({index in the line list: [(slitlet-1, segment-1, intensity)]}); after a successful fit both are updated and set as
## properties of the FDU, so a loop over slitlets can pass them on.
class wavelengthCalibrateSingleProcess(wavelengthCalibrateProcess):
    gpumode = False

    ## The constructor.
    def __init__(self, fdb=None, gpumode=False):
        #Initialize dicts
        self._calibs = dict()
        self._options = dict()
        self._optioninfo = dict()
        #Set default for write_output and write_calib_output to no
        self._options.setdefault("create_calib_only", "no")
        self._options.setdefault("write_output", "no")
        self._options.setdefault("write_calib_output", "no")
        self._outputdir = './'
        self._log = quietLog()
        self.gpumode = gpumode
    #end __init__

    #Set multiple options
    def setOptions(self, options):
        for name in options:
            self._options[name] = options[name]
    #end setOptions

    def execute(self, fdu, calibs, prevProc=None):
        print("Wavelength Calibrate")
        print(fdu._identFull)

        #call wavelengthCalibrate helper function to do gpu/cpu calibration
        self.wavelengthCalibrate(fdu, calibs)
        return True
    #end execute

    ## Wavelength Calibrate data
    def wavelengthCalibrate(self, fdu, calibs):
        ###*** For purposes of wavelengthCalibrate algorithm, X = dispersion direction and Y = cross-dispersion direction ***###
        #Read options
        n_brightest_lines = int(self.getOption("n_brightest_lines", fdu.getTag()))
        n_brightest_data = int(self.getOption("n_brightest_data", fdu.getTag()))
        min_separation = int(self.getOption("min_bright_line_separation", fdu.getTag()))
        bl_min = int(self.getOption("bright_line_searchbox_min", fdu.getTag()))
        bl_max = int(self.getOption("bright_line_searchbox_max", fdu.getTag()))
        min_intensity_pct = float(self.getOption("min_intensity_percent", fdu.getTag()))/100.0
        use_tolerance = False
        shift_tol = None
        if (self.getOption("max_shift_tolerance", fdu.getTag()) is not None):
            use_tolerance = True
            shift_tol = int(self.getOption("max_shift_tolerance", fdu.getTag()))
        fallbackMethods = [m.strip().lower() for m in str(self.getOption("wavecal_fallback", fdu.getTag())).split(",") if m.strip().lower() not in ["", "none"]]
        retryGrade = str(self.getOption("wavecal_retry_grade", fdu.getTag())).lower()
        gradeRank = {"excellent": 0, "good": 1, "satisfactory": 2, "marginal": 3, "poor": 4}

        fit_order = int(self.getOption("fit_order", fdu.getTag()))
        max_wavelength = float(self.getOption("max_wavelength", fdu.getTag()))
        min_wavelength = float(self.getOption("min_wavelength", fdu.getTag()))
        min_threshold = float(self.getOption("min_threshold", fdu.getTag()))
        min_lines_nonlinear = int(self.getOption("min_lines_to_refine_nonlinear_guess", fdu.getTag()))
        line_list = self.getOption("line_list", fdu.getTag())
        if (line_list is None):
            print("wavelengthCalibrateProcess::wavelengthCalibrate> ERROR: No line_list given for "+fdu.getFullId()+"! Discarding Image!")
            #disable this FDU
            fdu.disable()
            return
        if (not os.access(line_list, os.F_OK) and not os.access(self._datadir+"/linelists/"+line_list, os.F_OK)):
            print("wavelengthCalibrateProcess::wavelengthCalibrate> ERROR: Could not find line_list "+line_list+" for "+fdu.getFullId()+"! Discarding Image!")
            #disable this FDU
            fdu.disable()
            return
        wavelength_scale_guess = self.getOption("wavelength_scale_guess", fdu.getTag())
        if (wavelength_scale_guess is None):
            try:
                lambda1 = float(self.getOption("wavelength_line_1", fdu.getTag()))
                lambda2 = float(self.getOption("wavelength_line_2", fdu.getTag()))
                delta = float(self.getOption("wavelength_line_separation", fdu.getTag()))
                #Allow to be negative
                wavelength_scale_guess = [(lambda1-lambda2)/delta]
            except Exception as ex:
                print("wavelengthCalibrateProcess::wavelengthCalibrate> ERROR: No wavelength_scale_guess or (wavelength_line_1, wavelength_line_2, and wavelength_line_separation) found for "+fdu.getFullId()+"! Discarding Image!")
                #disable this FDU
                fdu.disable()
                return
        else:
            wavelength_scale_guess = [float(v) for v in wavelength_scale_guess.split()]

        skyFDU = fdu
        xsize = fdu.getShape()[0]

        #Create output dir if it doesn't exist
        outdir = "./"
        if (not os.access(outdir+"/wavelengthCalibrated", os.F_OK)):
            os.mkdir(outdir+"/wavelengthCalibrated",0o755)

        #Create new header dict
        wcHeader = dict()
        #Get sky data for qadata
        qadata = skyFDU.getData().copy()
        minLambda = None
        maxLambda = None
        coeffs = []

        j = 0
        seg = 0
        nslits = 1
        n_segments = 1
        if 'slitlet' in calibs:
            j = calibs['slitlet']-1
        if 'segment' in calibs:
            seg = calibs['segment']-1
        if 'nslits' in calibs:
            nslits = calibs['nslits']
        if 'n_segments' in calibs:
            n_segments = calibs['n_segments']
        mult_seg = n_segments > 1
        #Results of other slitlets, for the neighbor, trend and learned-intensity guesses
        solvedCuts = list(calibs.get('solvedCuts', []))
        lineMeasures = calibs.get('lineMeasures', dict())

        scale = wavelength_scale_guess[0]
        nonlinear = False
        if (len(wavelength_scale_guess) > 1):
            nonlinear = True
            #Create coeffs list with 0 constant term
            coeffs = [0]
            coeffs.extend(wavelength_scale_guess)

        pass_name = " for order "+str(j+1)+" of "
        #define output specfile, residfile
        specfile = outdir+"/wavelengthCalibrated/spec_"+fdu._id
        residfile = outdir+"/wavelengthCalibrated/resid_"+fdu._id
        if (nslits > 1):
            #Append slit number
            specfile += "_slitlet_"+str(j+1)
            residfile += "_slitlet_"+str(j+1)
        if (mult_seg):
            pass_name = " for order "+str(j+1)+", segment "+str(seg+1)+" of "
            #Append segment number
            specfile += "_segment_"+str(seg+1)
            residfile += "_segment_"+str(seg+1)
        specfile += ".dat"
        residfile += ".dat"

        qaRow = str(j+1)+"\t"+str(seg+1)+"\t0\t0\t-\t-\t-\t[]\t-\t-\tfailed\t-"
        cand = None
        lsq = None
        #Fail cleanly: an unexpected error is reported and the cut left uncalibrated
        try:
            #Read line list
            (masterWave, masterFlux, masterFlag) = self.readLineList(line_list)
            if (nslits > 1):
                if (mult_seg):
                    print("wavelengthCalibrateProcess::wavelengthCalibrate> Finding wavelength solution to slitlet "+str(j+1)+", segment "+str(seg+1)+"...")
                else:
                    print("wavelengthCalibrateProcess::wavelengthCalibrate> Finding wavelength solution to slitlet "+str(j+1)+"...")

            if 'oned' in calibs:
                oned = np.array(calibs['oned'], dtype=np.float64)
            else:
                oned = np.array(fdu.getData(), dtype=np.float64)

            #Filter the 1-d cut!
            #Use quartile instead of median to get better estimate of background levels!
            #Use 2 passes of quartile filter
            badpix = oned == 0 #First find bad pixels
            #Correct for big negative values
            oned[np.where(oned < -100)] = 1.e-6
            for i in range(2):
                tempcut = np.zeros(len(oned))
                nh = 25-badpix[:51].sum()//2 #Instead of defaulting to 25 for quartile, use median of bottom half of *nonzero* pixels
                for k in range(25):
                    tempcut[k] = oned[k] - gpu_arraymedian(oned[:51],nonzero=True,nhigh=nh)
                for k in range(25,len(oned)-25):
                    nh = 25-badpix[k-25:k+26].sum()//2
                    tempcut[k] = oned[k] - gpu_arraymedian(oned[k-25:k+26],nonzero=True,nhigh=nh)
                nh = 25-badpix[len(oned)-50:].sum()//2
                for k in range(len(oned)-25,len(oned)):
                    tempcut[k] = oned[k] - gpu_arraymedian(oned[len(oned)-50:],nonzero=True,nhigh=nh)
                #Set zero values to small positive number to avoid being flagged
                tempcut[tempcut == 0] = 1.e-6
                #Correct for big negative values
                tempcut[np.where(tempcut < -100)] = 1.e-6
                oned = tempcut
            #Set bad pixels back to 0
            oned[badpix] = 0
            oned[:10] = 0
            oned[-10:] = 0

            #Find brightest line in image
            blref = np.where(oned == np.max(oned))[0][0]
            #Fit a Gaussian to find shape.  Square data first to ensure that
            #bright line dominates fit
            refCut = oned[max(blref-10,0):blref+11]**2
            p = np.zeros(4, dtype=np.float64)
            p[0] = np.max(refCut)
            p[1] = 10
            p[2] = 2
            p[3] = gpu_arraymedian(refCut, nonzero=True)
            try:
                lsq = leastsq(gaussResiduals, p, args=(np.arange(len(refCut), dtype=np.float64), refCut))
                #Multiply by math.sqrt(2) because we fit squared data.  This will
                #give us the width of the emission lines.
                gaussWidth = abs(lsq[0][2]*math.sqrt(2))
            except Exception as ex:
                print("wavelengthCalibrateProcess::wavelengthCalibrate> WARNING: leastsq fit failed at pixel "+str(blref)+"; Using gaussWidth=1.5.")
                gaussWidth = 1.5
            if (gaussWidth > 2.5):
                gaussWidth = 1.5
                #If its a broad line for some reason, use 1.5 as default
            elif (gaussWidth > 2):
                gaussWidth = 1.75

            if (min_wavelength > max_wavelength):
                print("wavelengthCalibrateProcess::wavelengthCalibrate> WARNING: min wavelength "+str(min_wavelength)+" is greater than max wavelength "+str(max_wavelength)+"!  Flipping them and proceeding...")
                tmp = min_wavelength
                min_wavelength = max_wavelength
                max_wavelength = tmp

            #Bright lines in the cut, and this cut and the fit settings for trying other guesses (wavecal_fallback)
            (wclines, wccentroids, lineParams, lineWidths, linePeaks) = self.findDataLines(oned, gaussWidth, n_brightest_data, bl_min, bl_max, min_separation)
            ctx = {"fdu": fdu, "oned": oned, "wclines": wclines, "wccentroids": wccentroids, "lineParams": lineParams, "lineWidths": lineWidths, "linePeaks": linePeaks, "masterWave": masterWave, "masterFlux": masterFlux, "masterFlag": masterFlag, "gaussWidth": gaussWidth, "n_brightest_lines": n_brightest_lines, "fit_order": fit_order, "min_lines_nonlinear": min_lines_nonlinear, "min_threshold": min_threshold, "min_intensity_pct": min_intensity_pct, "use_tolerance": use_tolerance, "shift_tol": shift_tol, "pass_name": pass_name}
            configGuess = (min_wavelength, max_wavelength, scale, nonlinear, list(coeffs))
            learnedFlux = None
            if (len(lineMeasures) > 0):
                learnedFlux = self.learnedLineFlux(masterFlux, lineMeasures)

            #The configured guess: template, 3 brightest lines, then the full match and fit
            (dummySize, dummyFlux, dummyWave, dummyOrder) = self.buildDummySpectrum(min_wavelength, max_wavelength, scale, nonlinear, coeffs, masterFlux, masterWave, gaussWidth)
            if (len(dummyFlux) <= 200 or dummyFlux[100:-100].max() == 0):
                #If this happened, no lines were found in the given wavelength range!
                print("wavelengthCalibrateProcess::wavelengthCalibrate> ERROR: No lines found in wavelength range ["+str(min_wavelength)+":"+str(max_wavelength)+"] "+pass_name+fdu.getFullId()+"!")
                failReason = "no lines in range"
            else:
                failReason = "3 brightest lines not matched"
                #Scale template
                fluxScale = oned[100:-100].max()/dummyFlux[100:-100].max()
                dummyFlux *= fluxScale
                #Set zero values to small positive number so they're not considered flagged
                dummyFlux[dummyFlux == 0] = 1.e-6
                #Correct for big negative values
                dummyFlux[np.where(dummyFlux < -100)] = 1.e-6
                (dlines, dpeak, dwave) = self.findTemplateLines(dummyFlux, dummyWave, masterWave, n_brightest_lines, gaussWidth)
                (success, currLines, dumPeak, idx) = self.match3BrightestLines(coeffs, dlines, dpeak, dwave, dummyFlux, dummyOrder, dummyWave, fdu, oned, nonlinear, scale, usePlot, wccentroids, wclines)
                if (success):
                    cand = self.solveFromMatch(fdu, oned, currLines, dumPeak, idx, wclines, wccentroids, lineParams, lineWidths, linePeaks, dummySize, dummyFlux, dummyWave, dummyOrder, fluxScale, masterWave, masterFlux, masterFlag, gaussWidth, scale, nonlinear, list(coeffs), min_wavelength, max_wavelength, min_lines_nonlinear, min_threshold, min_intensity_pct, use_tolerance, shift_tol, fit_order, pass_name)
                    (cand["rmsWave"], cand["rmsPix"], cand["quality"], cand["coverage"]) = self.wavecalQuality(cand["coeffs"], cand["fit_order"], cand["reflines"], cand["residLines"], len(oned), fdu)
                    cand["label"] = None
                    cand["masterFlux"] = masterFlux

            #Fallbacks: when nothing matched, the best good-enough solution of the first method that gives one;
            #when the solution is graded wavecal_retry_grade or worse, any clearly better one (as the pipeline's second pass)
            retry = (cand is not None and retryGrade in gradeRank and gradeRank[cand["quality"]] >= gradeRank[retryGrade])
            if ((cand is None or retry) and len(fallbackMethods) > 0):
                prev = cand
                best = None
                for alt in self.fallbackCandidates(ctx, fallbackMethods, j, seg, [sc for sc in solvedCuts if not (sc[0] == j and sc[1] == seg)], configGuess, learnedFlux, scale):
                    if (prev is None and best is not None and alt["method"] != best["method"]):
                        break
                    if (not self.acceptableFallback(alt)):
                        print("wavelengthCalibrateProcess::wavelengthCalibrate> Rejecting the solution from "+alt["label"]+pass_name+fdu.getFullId()+" ("+alt["quality"]+", "+str(len(alt["reflines"]))+" lines)")
                        failReason = "fallback solution "+alt["quality"]
                        continue
                    if (self.betterSolution(alt, (best if best is not None else prev))):
                        best = alt
                        if (prev is not None and gradeRank[best["quality"]] <= gradeRank["good"]):
                            break
                if (best is not None):
                    if (prev is not None):
                        print("wavelengthCalibrateProcess::wavelengthCalibrate> Replacing the "+prev["quality"]+" solution ("+formatNum(prev["rmsPix"])+" px, "+str(len(prev["reflines"]))+" lines)"+pass_name+fdu.getFullId()+" with the solution from "+best["label"])
                    cand = best
                elif (prev is not None):
                    print("wavelengthCalibrateProcess::wavelengthCalibrate> No better solution"+pass_name+fdu.getFullId())

            if (cand is None):
                print("wavelengthCalibrateProcess::wavelengthCalibrate> ERROR: No wavelength solution "+pass_name+fdu.getFullId()+" ("+failReason+")!")
                print("wavelengthCalibrateProcess::wavelengthCalibrate> Check "+specfile+" for data.")
                qaRow = str(j+1)+"\t"+str(seg+1)+"\t0\t0\t-\t-\t-\t[]\t-\t-\tfailed: "+failReason+"\t-"
                #Output "spec" file with the configured guess
                xs = np.arange(len(oned), dtype=np.float32)*scale+min_wavelength
                if (scale < 0):
                    #Negative scale, need wavelenghts descending
                    xs = np.arange(len(oned), dtype=np.float32)*scale+max_wavelength
                if (nonlinear):
                    xs = min_wavelength+polyFunction(coeffs, np.arange(len(oned), dtype=np.float32), dummyOrder)
                    if (polyFunction(coeffs, dummySize, dummyOrder) < 0):
                        #Negative scale, need wavelengths descending
                        xs = max_wavelength+polyFunction(coeffs, np.arange(len(oned), dtype=np.float32), dummyOrder)
                f = open(specfile,'w')
                f.write("#lambda\t1-d cut\tref\tfit\n")
                for i in range(len(oned)):
                    f.write(str(xs[i])+'\t'+str(oned[i])+'\t0.0\t0.0\n')
                f.close()
            else:
                reflines = cand["reflines"]
                wlines = cand["wlines"]
                lineParams = cand["lineParams"]
                lsq = [cand["coeffs"]]
                fit_order = cand["fit_order"]
                residLines = cand["residLines"]
                norig = cand["norig"]
                obsSpec = cand["obsSpec"]
                gaussWidth = cand["gaussWidth"]
                fluxScale = cand["fluxScale"]
                scale = cand["scale"]
                (rmsWave, rmsPix, quality, coverage) = (cand["rmsWave"], cand["rmsPix"], cand["quality"], cand["coverage"])
                qamsg = "RMS = "+formatNum(rmsWave)+" (wavelength units) = "+formatNum(rmsPix)+" px: "+quality.upper()+"; "+str(len(reflines))+" of "+str(norig)+" lines used, spanning pixels "+str(int(np.min(reflines)))+"-"+str(int(np.max(reflines)))+" ("+str(int(round(coverage*100)))+"% of the cut)"
                print("\t\t"+qamsg)
                if (coverage < 0.5):
                    print("wavelengthCalibrateProcess::wavelengthCalibrate> WARNING: lines span only "+str(int(round(coverage*100)))+"% of the cut"+pass_name+fdu.getFullId()+"; the solution is extrapolated beyond them.")
                if (cand["label"] is not None):
                    print("wavelengthCalibrateProcess::wavelengthCalibrate> Solution"+pass_name+fdu.getFullId()+" found with "+cand["label"])

                #Output "spec" file
                xs = polyFunction(lsq[0], np.arange(len(oned), dtype=np.float32), fit_order)
                dummyFlux = np.zeros(len(oned), dtype=np.float32)
                #Add gaussians for each line in line list
                dummyFlux = self.populateDummyFlux(cand["masterFlux"], masterWave, xs, dummyFlux, lsq[0][1], gaussWidth)
                #Scale template
                dummyFlux *= fluxScale
                f = open(specfile,'w')
                f.write("#lambda\t1-d cut\tref\tfit\n")
                for i in range(len(oned)):
                    f.write(str(xs[i])+'\t'+str(oned[i])+'\t'+str(dummyFlux[i])+'\t'+str(obsSpec[i])+'\n')
                f.close()
                #Output "resid" file with reference wavelength and residuals
                f = open(residfile,'w')
                f.write("#Ref wavelength\tresidual\n")
                for i in range(len(wlines)):
                    f.write(str(wlines[i])+'\t\t'+str(residLines[i])+'\n')
                f.close()

                if (usePlot and (self.getOption("debug_mode", fdu.getTag()).lower() == "yes" or self.getOption("write_plots", fdu.getTag()).lower() == "yes")):
                    plt.plot(oned, '#1f77b4', linewidth=2.0)
                    plt.plot(dummyFlux, '#ff7f0e', linewidth=2.0)
                    plt.plot(obsSpec, '#2ca02c', linewidth=2.0)
                    plt.legend(['Data', 'Line list', 'Fit'], loc=2)
                    plt.xlabel('Pixel')
                    plt.ylabel('Flux')
                    if (self.getOption("write_plots", fdu.getTag()).lower() == "yes"):
                        pltfile = outdir+"/wavelengthCalibrated/qa_"+skyFDU._id+"_slit_"+str(j+1)
                        if (mult_seg):
                            pltfile += "_seg_"+str(seg+1)
                        pltfile += ".png"
                        plt.savefig(pltfile, dpi=200)
                    if (self.getOption("debug_mode", fdu.getTag()).lower() == "yes"):
                        plt.show()
                    plt.close()

                #Update header info
                wcHeader['CTYPE'] = 'WAVE-PLY'
                if (scale > 0):
                    minLambda = polyFunction(lsq[0], 0, fit_order)
                    maxLambda = polyFunction(lsq[0], xsize-1, fit_order)
                else:
                    maxLambda = polyFunction(lsq[0], 0, fit_order)
                    minLambda = polyFunction(lsq[0], xsize-1, fit_order)
                native = cand.get("native")
                if (nslits == 1 and not mult_seg):
                    #If only one slit, use PORDER and PCOEFF_i
                    wcHeader['PORDER'] = fit_order
                    wcHeader['NSEG_01'] = 1
                    for i in range(fit_order+1):
                        wcHeader['PCOEFF_'+str(i)] = lsq[0][i]
                    #QA: RMS (wavelength units, pixels), grade, number of lines
                    wcHeader['WCRMS'] = rmsWave
                    wcHeader['WCRMSPX'] = rmsPix
                    wcHeader['WCQUAL'] = quality
                    wcHeader['WCNLINES'] = len(reflines)
                    if (native is not None):
                        #legendre/chebyshev: the native coefficients, with pixels 0..WCXMAX mapped to [-1, 1]
                        wcHeader['WCFUNC'] = native[0]
                        wcHeader['WCXMAX'] = len(oned)-1
                        for i in range(len(native[1])):
                            wcHeader['NCOEFF_'+str(i)] = native[1][i]
                else:
                    #Use PORDERxx and PCFi_Sxx
                    slitStr = str(j+1)
                    if (j+1 < 10):
                        slitStr = '0'+slitStr
                    wcHeader['NSEG_'+slitStr] = n_segments
                    prefix = ''
                    if (mult_seg):
                        #Multiple segments, use hierarchical keywords PORDER_xx_SEGy and PCFi_Sxx_SEGy
                        slitStr += '_SEG' + str(seg)
                        prefix = 'HIERARCH '
                    wcHeader[prefix+'PORDER'+slitStr] = fit_order
                    for i in range(fit_order+1):
                        wcHeader[prefix+'PCF'+str(i)+'_S'+slitStr] = lsq[0][i]
                    wcHeader[prefix+'WCRMS'+slitStr] = rmsWave
                    wcHeader[prefix+'WCRPX'+slitStr] = rmsPix
                    wcHeader[prefix+'WCQUL'+slitStr] = quality
                    wcHeader[prefix+'WCNLN'+slitStr] = len(reflines)
                    if (native is not None):
                        wcHeader[prefix+'WCFUN'+slitStr] = native[0]
                        wcHeader[prefix+'WCXMX'+slitStr] = len(oned)-1
                        for i in range(len(native[1])):
                            wcHeader[prefix+'NCF'+str(i)+'_S'+slitStr] = native[1][i]
                qaRow = str(j+1)+"\t"+str(seg+1)+"\t"+str(norig)+"\t"+str(len(reflines))+"\t"+formatNum(residLines.std())+"\t"+formatNum(minLambda, 0)+"\t"+formatNum(maxLambda, 0)+"\t"+formatList(lsq[0])+"\t"+formatNum(rmsWave)+"\t"+formatNum(rmsPix)+"\t"+quality+"\t"+str(int(round(coverage*100)))

                #Pass this solution and its measured line intensities on to the caller (next slitlets' guesses);
                #a poor solution may be a wrong match, so it is not used
                solvedCuts = [sc for sc in solvedCuts if not (sc[0] == j and sc[1] == seg)]
                for i in lineMeasures:
                    lineMeasures[i] = [m for m in lineMeasures[i] if not (m[0] == j and m[1] == seg)]
                if (quality != "poor"):
                    solvedCuts.append((j, seg, oned.copy(), np.array(lsq[0], dtype=np.float64), fit_order))
                    measured = self.measureLineIntensities(oned, lsq[0], fit_order, masterWave, masterFlux, wlines, gaussWidth)
                    for i in measured:
                        lineMeasures.setdefault(i, []).append((j, seg, measured[i]))
                fdu.setProperty("solvedCuts", solvedCuts)
                fdu.setProperty("lineMeasures", lineMeasures)
                fdu.setProperty("wcQuality", (rmsWave, rmsPix, quality, len(reflines)))
        except Exception as ex:
            print("wavelengthCalibrateProcess::wavelengthCalibrate> ERROR: "+type(ex).__name__+": "+str(ex)+pass_name+fdu.getFullId()+"!")
            print(traceback.format_exc())
            cand = None
            minLambda = None

        #Update header
        fdu.setProperty("wcHeader", wcHeader)
        skyFDU.setProperty("wcHeader", wcHeader)
        ##8/6/18 use updateHeader for sky/lamp
        skyFDU.updateHeader(wcHeader)
        #Write out qadata
        qafile = outdir+"/wavelengthCalibrated/qa_"+skyFDU.getFullId()
        if (os.access(qafile, os.F_OK)):
            os.unlink(qafile)
        #Write out qa file
        if (not os.access(qafile, os.F_OK)):
            skyFDU.tagDataAs("wcqa", qadata)
            skyFDU.writeTo(qafile, tag="wcqa")
            skyFDU.removeProperty("wcqa")
        del qadata
        #Output qa file with stats about this slitlet/segment
        qafile = outdir+"/wavelengthCalibrated/qa_"+skyFDU._id+".dat"
        f = open(qafile,'w')
        f.write("Slitlet\tSegment\tn lines\tn used\tsigma\tmin\tmax\tfit\tRMS\tRMS px\tquality\tcoverage %\n")
        f.write(qaRow+"\n")
        f.close()
        #Line intensities measured so far (this cut and any passed in), in the line-list format
        if (cand is not None and len(lineMeasures) > 0):
            mfile = outdir+"/wavelengthCalibrated/measured_lines_"+skyFDU._id+".dat"
            f = open(mfile, 'w')
            f.write("#Line intensities measured in the calibrated cuts, on the scale of "+str(line_list)+"\n")
            f.write("#(median over cuts; near 0 = in range but not seen).  Columns: wavelength, measured intensity, flag, #list intensity, n cuts\n")
            for i in sorted(lineMeasures, key=lambda k: masterWave[k]):
                if (len(lineMeasures[i]) == 0):
                    continue
                f.write(str(masterWave[i])+"\t"+formatNum(np.median([m[2] for m in lineMeasures[i]]), 1)+"\t"+str(masterFlag[i])+"\t#"+formatNum(masterFlux[i], 1)+"\t"+str(len(lineMeasures[i]))+"\n")
            f.close()

        if (minLambda is None or cand is None):
            #This cut was not wavelength calibrated
            print("wavelengthCalibrateProcess::wavelengthCalibrate> ERROR: Could not calibrate "+fdu.getFullId()+"! Discarding Image!")
            #disable this FDU
            fdu.disable()
            return

        #Resample: xout gives each pixel's position on a linear scale
        xout_data = np.zeros(fdu.getShape(), dtype=np.float32)
        xs_data = np.arange(xsize, dtype=np.float32)
        #New header keywords for resampled data
        resampHeader = dict()
        resampHeader['RESAMPLD'] = 'YES'
        resampHeader['CRVAL1'] = minLambda
        if (scale < 0):
            resampHeader['CRVAL1'] = maxLambda
        resampHeader['CDELT1'] = scale
        resampHeader['CRPIX1'] = 1

        #Use min/max wavelength from this input slitlet for all segments
        scale_data = (maxLambda-minLambda)/(float)(xsize)

        #Calculate wavelength input scale
        lambdaIn_data = polyFunction(lsq[0], xs_data, len(lsq[0])-1)

        #Use CRVALSxx and CDELTSxx for header keywords
        slitStr = str(j+1)
        if (j+1 < 10):
            slitStr = '0'+slitStr
        if (not mult_seg):
            resampHeader['CRVALS'+slitStr] = minLambda
            if (scale < 0):
                resampHeader['CRVALS'+slitStr] = maxLambda
            resampHeader['CDELTS'+slitStr] = scale
        else:
            #Multiple segments, use hierarchical keywords CRVALS_xx_SEGy and CDELTS_xx_SEGy
            slitStr += '_SEG' + str(seg)
            resampHeader['HIERARCH CRVALS'+slitStr] = minLambda
            if (scale < 0):
                resampHeader['HIERARCH CRVALS'+slitStr] = maxLambda
            resampHeader['HIERARCH CDELTS'+slitStr] = scale

        xout_data[:] = (lambdaIn_data-minLambda)/scale_data
        fdu.setProperty("resampledHeader", resampHeader)
        fdu.tagDataAs("xout_data", xout_data)
        fdu.writeTo(outdir+"/wavelengthCalibrated/xout_data_"+fdu.getFullId(), tag="xout_data")
    #end wavelengthCalibrate

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = "./" 
        if (not os.access(outdir+"/wavelengthCalibrated", os.F_OK)):
            os.mkdir(outdir+"/wavelengthCalibrated",0o755)
        #Create output filename
        wcfile = outdir+"/wavelengthCalibrated/wc_"+fdu.getFullId()
        #Check to see if it exists
        if (os.access(wcfile, os.F_OK)):
            os.unlink(wcfile)
        if (not os.access(wcfile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(wcfile, headerExt=fdu.getProperty("wcHeader"))
        #Write out resampled data if it exists
        if (fdu.hasProperty("resampled")):
            resampfile = outdir+"/wavelengthCalibrated/resamp_wc_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(resampfile, os.F_OK)):
                os.unlink(resampfile)
            if (not os.access(resampfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                #Write with resampHeader as header extension
                fdu.writeTo(resampfile, tag="resampled", headerExt=fdu.getProperty("resampledHeader"))
    #end writeOutput

def extract2DFromImageWithSlitmask(image, slitmask, slitlet=1, segment=1, n_segments=1, horizontal=True, gpumode=False):
    fdu = fatboySpectrum(image)
    fdu.readHeader()
    fdu.initialize()

    slitmask = fatboySpectrum(slitmask)
    slitmask.readHeader()
    slitmask.initialize()

    #Defaults for longslit - treat whole image as 1 slit
    nslits = 1
    ylos = [0]
    if (horizontal):
        yhis = [fdu.getShape()[0]]
    else:
        yhis = [fdu.getShape()[1]]
    if (not slitmask.hasProperty("nslits")):
        slitmask.setProperty("nslits", int(slitmask.getData().max()))
    nslits = slitmask.getProperty("nslits")
    #Use helper method to all ylo, yhi for each slit in each frame
    (ylos, yhis, slitx, slitw) = findRegions(slitmask.getData(), nslits, fdu, gpu=False)
    slitmask.setProperty("regions", (ylos, yhis, slitx, slitw))

    ylo = int(ylos[slitlet-1])
    yhi = int(yhis[slitlet-1])
    fdu.setProperty('nslits', nslits)
    return extract1DFromImage(fdu, ylo, yhi, slitlet, segment, n_segments, horizontal, slitmask=slitmask, gpumode=gpumode)


def extract1DFromImage(image, ylo=-1, yhi=-1, slitlet=1, segment=1, n_segments=1, horizontal=True, slitmask=None, gpumode=False):
    segment -= 1
    if (type(image) == str):
        skyFDU = fatboySpectrum(image)
        skyFDU.readHeader()
        skyFDU.initialize()
    else:
        skyFDU = image

    #Select kernel for 2d median
    kernel2d = fatboyclib.median2d
    if (gpumode):
        #Use GPU for medians
        kernel2d=gpumedian2d

    if (ylo < 0):
        ylo = 0
        
    #Use xstride to split into equal length segments here
    if (horizontal):
        if (yhi < 0):
            yhi = skyFDU.getShape()[0]
        xstride = skyFDU.getShape()[1]//n_segments
        sxlo = xstride*segment
        sxhi = xstride*(segment+1)
        slit = skyFDU.getData()[ylo:yhi+1,sxlo:sxhi].copy()
        if (slitmask is not None):
            #Apply mask to slit - based on if individual slitlets are being calibrated or not
            currMask = slitmask.getData()[ylo:yhi+1,sxlo:sxhi] == (slitlet)
            slit *= currMask
        if (ylo == yhi):
            oned = slit.ravel()
        elif (gpumode):
            #Use GPU
            oned = gpu_arraymedian(slit, axis="Y", nonzero=True, kernel2d=kernel2d, even=True)
        else:
            #Use CPU
            oned = kernel2d(slit.transpose().copy(), nonzero=True, even=True)
    else:
        if (yhi < 0):
            yhi = skyFDU.getShape()[1]
        xstride = skyFDU.getShape()[0]//n_segments
        sxlo = xstride*segment
        sxhi = xstride*(segment+1)
        slit = skyFDU.getData()[sxlo:sxhi,ylo:yhi+1].copy()
        if (slitmask is not None):
            #Apply mask to slit - based on if individual slitlets are being calibrated or not
            currMask = slitmask.getData()[sxlo:sxhi,ylo:yhi+1] == (slitlet)
            slit *= currMask
        if (ylo == yhi):
            oned = slit.ravel()
        else:
            oned = gpu_arraymedian(slit, axis="X", nonzero=True, kernel2d=kernel2d)
    skyFDU.updateData(oned)
    segment += 1
    skyFDU.setProperty('slitlet', slitlet)
    skyFDU.setProperty('segment', segment)
    skyFDU.setProperty('n_segments', n_segments)
    return skyFDU 

def read1DFromImage(image):
    skyFDU = fatboySpectrum(image)
    skyFDU.readHeader()
    skyFDU.initialize()
    return skyFDU

def executeWavelengthCalibration(fdu, options=dict(), calibs=dict(), gpumode=False):
    process = wavelengthCalibrateSingleProcess(gpumode=gpumode)
    process.setDefaultOptions()
    process.setOptions(options)
    for key in fdu._properties:
        if not key in calibs:
            calibs[key] = fdu.getProperty(key)
    print (calibs)
    process.execute(fdu, calibs)
    process.writeOutput(fdu)
