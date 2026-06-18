from superFATBOY.fatboyDataUnit import fatboyDataUnit
from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.datatypeExtensions.fatboySpecCalib import fatboySpecCalib
from superFATBOY.datatypeExtensions.fatboySpectrum import fatboySpectrum
import numpy as np
import math
from scipy.optimize import leastsq
import scipy.signal
import random
import os

class wavelengthCalibrateRANSAC(fatboyProcess):
    """
    An improved wavelength calibration process using RANSAC (Random Sample Consensus)
    for more robust line identification and polynomial fitting.
    """

    def __init__(self, fdb=None, gpumode=False):
        self._calibs = dict()
        self._options = dict()
        self._optioninfo = dict()
        self._outputdir = './'
        self.gpumode = gpumode
        self._datadir = os.path.dirname(os.path.abspath(__file__)) + "/data/"

    def setDefaultOptions(self):
        self._options.setdefault('fit_order', '3')
        self._options.setdefault('line_list', None)
        self._options.setdefault('min_wavelength', '10000')
        self._options.setdefault('max_wavelength', '18500')
        self._options.setdefault('wavelength_scale_guess', None)
        self._options.setdefault('min_threshold', '3')
        self._options.setdefault('ransac_iterations', '2000')
        self._options.setdefault('ransac_tolerance', '2.0') # pixels

    def find_peaks(self, oned, threshold=3.0):
        """Find peaks in the 1D spectrum above a local noise threshold."""
        # Use local background subtraction for cleaner peak finding
        # (This can be more sophisticated as in the original code's quartile filter)
        peaks, _ = scipy.signal.find_peaks(oned, height=threshold * np.std(oned))
        
        # Sub-pixel centroiding for found peaks
        centroids = []
        for p in peaks:
            if p < 2 or p > len(oned) - 3: continue
            y = oned[p-1:p+2]
            x = np.array([-1, 0, 1])
            coeffs = np.polyfit(x, y, 2)
            if coeffs[0] != 0:
                offset = -coeffs[1] / (2 * coeffs[0])
                if abs(offset) < 1:
                    centroids.append(p + offset)
                else:
                    centroids.append(p)
            else:
                centroids.append(p)
        return np.array(centroids)

    def readLineList(self, line_list):
        if not os.path.exists(line_list):
            line_list = self._datadir + "/linelists/" + line_list
        
        masterWave = []
        masterFlux = []
        masterFlag = []
        
        with open(line_list, 'r') as f:
            for line in f:
                if line.startswith('#'): continue
                parts = line.split()
                if len(parts) < 2: continue
                masterWave.append(float(parts[0]))
                masterFlux.append(float(parts[1]))
                if len(parts) > 2:
                    masterFlag.append(int(parts[2]))
                else:
                    masterFlag.append(0)
                    
        return np.array(masterWave), np.array(masterFlux), np.array(masterFlag)

    def ransac_fit(self, data_pixels, ref_wavelengths, scale_guess, tolerance=2.0, iterations=2000, fit_order=2):
        """
        Match observed pixel positions to reference wavelengths using RANSAC.
        Uses a 2-point model (shift and scale) to find candidates.
        """
        best_inliers = []
        best_model = None
        
        if len(data_pixels) < 2 or len(ref_wavelengths) < 2:
            return None, []

        sign = np.sign(scale_guess)

        for _ in range(iterations):
            # Pick two random peaks and two random reference wavelengths
            p1, p2 = np.random.choice(data_pixels, 2, replace=False)
            w1, w2 = np.random.choice(ref_wavelengths, 2, replace=False)
            
            # Ensure order is consistent with scale sign
            if sign * (w2 - w1) / (p2 - p1) < 0:
                continue
                
            s = (w2 - w1) / (p2 - p1)
            # Basic sanity check on scale
            if abs(s - scale_guess) / abs(scale_guess) > 0.5:
                continue
                
            inter = w1 - s * p1
            
            predicted = inter + s * data_pixels
            current_inliers = []
            for i, pw in enumerate(predicted):
                diffs = np.abs(ref_wavelengths - pw)
                idx = np.argmin(diffs)
                if diffs[idx] < tolerance * abs(s):
                    current_inliers.append((data_pixels[i], ref_wavelengths[idx]))
            
            # Remove duplicate matches
            unique_matches = {}
            for p, w in current_inliers:
                if w not in unique_matches or abs(p - (w-inter)/s) < abs(unique_matches[w] - (w-inter)/s):
                    unique_matches[w] = p
            
            inlier_pairs = [(p, w) for w, p in unique_matches.items()]
            
            if len(inlier_pairs) > len(best_inliers):
                best_inliers = inlier_pairs
                
        if len(best_inliers) < fit_order + 1:
            return None, []

        dp = np.array([x[0] for x in best_inliers])
        rw = np.array([x[1] for x in best_inliers])
        
        # np.polyfit coefficients reversed for fatboy ascending order
        model = np.polyfit(dp, rw, fit_order)[::-1]
            
        return model, best_inliers

    def wavelengthCalibrate(self, fdu, calibs):
        print(f"Improved Wavelength Calibrate (RANSAC): {fdu._id}")
        
        # 1. Setup options
        line_list = self.getOption("line_list", fdu.getTag())
        scale_guess = float(self.getOption("wavelength_scale_guess", fdu.getTag()))
        fit_order = int(self.getOption("fit_order", fdu.getTag()))
        ransac_iter = int(self.getOption("ransac_iterations", fdu.getTag()))
        ransac_tol = float(self.getOption("ransac_tolerance", fdu.getTag()))
        min_threshold = float(self.getOption("min_threshold", fdu.getTag()))
        
        # 2. Load line list and select brightest lines for anchor matching
        masterWave, masterFlux, masterFlag = self.readLineList(line_list)
        bright_idx = np.argsort(masterFlux)[-40:]
        anchor_waves = masterWave[bright_idx]

        # 3. Get 1D spectrum and find peaks
        oned = calibs.get('oned', fdu.getData())
        data_pixels = self.find_peaks(oned, threshold=min_threshold)
        
        if len(data_pixels) < fit_order + 1:
            print(f"Error: Only found {len(data_pixels)} peaks, need at least {fit_order+1}.")
            return False

        # 4. RANSAC matching to find the transformation
        model, inliers = self.ransac_fit(data_pixels, anchor_waves, scale_guess, 
                                        tolerance=ransac_tol, iterations=ransac_iter, 
                                        fit_order=fit_order)
        
        if model is None:
            print("RANSAC anchor matching failed. Retrying with all lines...")
            model, inliers = self.ransac_fit(data_pixels, masterWave, scale_guess, 
                                            tolerance=ransac_tol*2, iterations=ransac_iter*2, 
                                            fit_order=fit_order)
        
        if model is None:
            print("Wavelength calibration failed to find a robust match.")
            return False

        print(f"RANSAC matched {len(inliers)} lines.")

        # 5. Iterative Refinement and Sigma Clipping
        reflines = np.array([x[0] for x in inliers])
        wlines = np.array([x[1] for x in inliers])
        
        # Exclude blended lines (-1 flag) from refined fit if possible
        flags = np.array([masterFlag[np.where(masterWave == w)[0][0]] for w in wlines])
        good_mask = flags != -1
        if good_mask.sum() > fit_order + 1:
            reflines = reflines[good_mask]
            wlines = wlines[good_mask]

        current_p = model
        sigThresh = 2.5
        for _ in range(5):
            fit_waves = polyFunction(current_p, reflines, fit_order)
            resid = fit_waves - wlines
            std = resid.std()
            good = np.abs(resid - resid.mean()) <= sigThresh * std
            
            if good.sum() < fit_order + 1 or good.all():
                break
            
            reflines = reflines[good]
            wlines = wlines[good]
            current_p = np.polyfit(reflines, wlines, fit_order)[::-1]

        print(f"Final solution RMS: {resid[good].std():.4f} using {len(reflines)} lines.")
        print(f"Fit Coefficients (ascending): {current_p}")

        # 6. Update FDU header with custom keywords
        wcHeader = dict()
        wcHeader['CTYPE'] = 'WAVE-PLY'
        wcHeader['PORDER'] = fit_order
        for i, val in enumerate(current_p):
            wcHeader[f'PCOEFF_{i}'] = val
        
        slitidx = calibs.get('slitlet', 1)
        slitStr = f"{slitidx:02d}"
        wcHeader[f'NSEG_{slitStr}'] = 1
        wcHeader[f'PORDER{slitStr}'] = fit_order
        for i, val in enumerate(current_p):
            wcHeader[f'PCF{i}_S{slitStr}'] = val

        fdu.setProperty("wcHeader", wcHeader)
        fdu.updateHeader(wcHeader)
        
        return True

# Example usage pattern
def executeWavelengthCalibrationNew(fdu, options=dict(), calibs=dict(), gpumode=False):
    process = wavelengthCalibrateRANSAC(gpumode=gpumode)
    process.setDefaultOptions()
    process.setOptions(options)
    process.execute(fdu, calibs)
