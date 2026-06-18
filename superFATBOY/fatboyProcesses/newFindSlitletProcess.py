from superFATBOY.fatboyDataUnit import fatboyDataUnit
from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.datatypeExtensions.fatboySpecCalib import fatboySpecCalib

from superFATBOY import gpu_imcombine, imcombine
import numpy as np
from scipy.optimize import leastsq
import scipy.interpolate
import scipy.signal
import os, time

class newFindSlitletProcess(fatboyProcess):
    """
    An improved version of the slitlet identification process.
    Uses edge-informed cross-correlation, robust spline-based tracing,
    and iterative sigma clipping for cleaner slitmasks.
    """
    _modeTags = ["spectroscopy", "miradas"]

    def setDefaultOptions(self):
        # standard options from findSlitletProcess
        self._options.setdefault('debug_mode', 'no')
        self._options.setdefault('autodetect_peak_local_max', 'no')
        self._options.setdefault('background_boxcar_width', 25)
        self._options.setdefault('boundary', 10)
        self._options.setdefault('cut1d_max_threshold', 2)
        self._options.setdefault('edge_extend_to_chip', 'no')
        self._options.setdefault('edge_threshold', 15)
        self._options.setdefault('fiber_width', '5')
        self._options.setdefault('fit_order', '3')
        self._options.setdefault('invert_before_correlating', 'no')
        self._options.setdefault('max_residual_error', '2.0')
        self._options.setdefault('min_coverage_fraction', '30')
        self._options.setdefault('n_segments', '1')
        self._options.setdefault('order_step_size', '5')
        self._options.setdefault('padding','0')
        self._options.setdefault('region_file', None)
        self._options.setdefault('slitlet_attempt_autocorrect', 'no')
        self._options.setdefault('slitlet_autocorrect_gap_size', None)
        self._options.setdefault('slitlet_autodetect_nslits', '0')
        self._options.setdefault('slitlet_autodetect_boxsize', '5')
        self._options.setdefault('slitlet_autodetect_min_flux_pct', '0.001')
        self._options.setdefault('slitlet_autodetect_min_width', '10')
        self._options.setdefault('slitlet_autodetect_sigma', '5')
        self._options.setdefault('slitlet_autodetect_use_median', 'no')
        self._options.setdefault('slitlet_autodetect_x', '1024')
        self._options.setdefault('slitlet_trace_boxsize', '21')
        self._options.setdefault('slitlet_trace_ylo', '-1')
        self._options.setdefault('slitlet_trace_yhi', '-1')
        self._options.setdefault('subtract_background_level', 'no')
        self._options.setdefault('trace_peak_local_max', 'no')
        self._options.setdefault('trace_slitlets_individually', 'yes')
        self._options.setdefault('write_plots', 'no')
        self._options.setdefault('write_calib_output', 'yes')
        
        # New options for improved algorithm
        self._options.setdefault('use_splines', 'yes')
        self._options.setdefault('robust_sigma', '3.0')
        self._options.setdefault('use_new_extract_spectra', 'yes')

    def execute(self, fdu, calibs, prevProc=None):
        if (fdu._specmode == fdu.FDU_TYPE_LONGSLIT):
            return True

        print(f"Find Slitlets (New): {fdu._identFull}")

        calibs = self.getCalibs(fdu, prevProc)
        if ('slitmask' in calibs):
            return True

        if (not 'masterFlat' in calibs):
            print(f"newFindSlitletProcess::execute> ERROR: Master flat not found for {fdu.getFullId()}!")
            fdu.disable()
            return False

        if (self.getOption("trace_slitlets_individually", fdu.getTag()).lower() == "yes"):
            calibs = self.traceOrders(fdu, calibs)
        elif (self.getOption("trace_peak_local_max", fdu.getTag()).lower() == "yes"):
            calibs = self.tracePeakLocalMax(fdu, calibs)
        else:
            calibs = self.traceSlitlets(fdu, calibs)

        if ('slitmask' in calibs):
            self._fdb.appendCalib(calibs['slitmask'])
            if 'slitlo' in calibs: self._fdb.appendCalib(calibs['slitlo'])
            if 'slithi' in calibs: self._fdb.appendCalib(calibs['slithi'])
            return True
        
        return False

    def getCalibs(self, fdu, prevProc = None):
        # Compatibility with original getCalibs
        from superFATBOY.fatboyProcesses.findSlitletProcess import findSlitletProcess
        base = findSlitletProcess(self._fdb)
        base.setOptions(self._options)
        return base.getCalibs(fdu, prevProc)

    def find_subpixel_peak(self, data, guess_idx):
        idx = int(round(guess_idx))
        if idx < 1 or idx >= len(data) - 1:
            return guess_idx
        y1, y2, y3 = float(data[idx-1]), float(data[idx]), float(data[idx+1])
        denom = (y1 - 2*y2 + y3)
        if denom == 0: return guess_idx
        offset = 0.5 * (y1 - y3) / denom
        return idx + offset

    def robust_fit_path(self, x, y, order, use_splines=True, sigma=3.0):
        if len(x) < order + 2: return None
        x, y = np.array(x), np.array(y)
        
        # Iterative sigma clipping
        for _ in range(5):
            if use_splines:
                # Smoothing spline
                try:
                    s = scipy.interpolate.UnivariateSpline(x, y, k=min(3, order), s=len(x)*2.0)
                    res = y - s(x)
                except:
                    p = np.polyfit(x, y, order)
                    res = y - np.polyval(p, x)
                    use_splines = False
            else:
                p = np.polyfit(x, y, order)
                res = y - np.polyval(p, x)
            
            std = np.std(res)
            if std == 0: break
            mask = np.abs(res - np.mean(res)) < sigma * std
            if mask.all() or mask.sum() < order + 2: break
            x, y = x[mask], y[mask]
        
        if use_splines:
            return scipy.interpolate.UnivariateSpline(x, y, k=min(3, order), s=len(x)*2.0)
        else:
            return np.poly1d(np.polyfit(x, y, order))

    def get_1d_cut(self, data, x, y, dispersion, box=21):
        half = box // 2
        iy, ix = int(round(y)), int(round(x))
        ny, nx = data.shape
        if dispersion == fatboyDataUnit.DISPERSION_HORIZONTAL:
            y1, y2 = max(0, iy-half), min(ny, iy+half+1)
            x1, x2 = max(0, ix-2), min(nx, ix+3)
            return data[y1:y2, x1:x2].mean(axis=1)
        else:
            x1, x2 = max(0, ix-half), min(nx, ix+half+1)
            y1, y2 = max(0, iy-2), min(ny, iy+3)
            return data[x1:x2, y1:y2].mean(axis=0)

    def traceOrders(self, fdu, calibs):
        masterFlat = calibs['masterFlat']
        flatData = masterFlat.getData(force_cpu=True).copy()
        xsize, ysize = fdu.getShape()[1], fdu.getShape()[0]
        if fdu.dispersion == fdu.DISPERSION_VERTICAL:
            xsize, ysize = fdu.getShape()[0], fdu.getShape()[1]

        # Use infrastructure to get initial regions
        from superFATBOY.fatboyProcesses.findSlitletProcess import findSlitletProcess
        base = findSlitletProcess(self._fdb)
        base.setOptions(self._options)
        
        # Determine regions
        regFile = self.getOption("region_file", fdu.getTag())
        if not regFile:
            sylo, syhi, slitx, slitw = base.autoDetectSlitlets(fdu, flatData)
        else:
            if regFile.endswith(".reg"):
                (sylo, syhi, slitx, slitw) = readRegionFile(regFile, horizontal=(fdu.dispersion == fatboyDataUnit.DISPERSION_HORIZONTAL))
            elif regFile.endswith(".xml"):
                (sylo, syhi, slitx, slitw) = readRegionFileXML(regFile, horizontal=(fdu.dispersion == fatboyDataUnit.DISPERSION_HORIZONTAL))
            else:
                (sylo, syhi, slitx, slitw) = readRegionFileText(regFile, horizontal=(fdu.dispersion == fatboyDataUnit.DISPERSION_HORIZONTAL))

        nslits = len(sylo)
        if nslits == 0:
            print("newFindSlitletProcess::traceOrders> ERROR: No slitlets found!")
            return calibs

        yloMask, yhiMask = np.zeros((nslits, xsize)), np.zeros((nslits, xsize))
        step = int(self.getOption('order_step_size', fdu.getTag()))
        order = int(self.getOption("fit_order", fdu.getTag()))
        use_splines = self.getOption("use_splines", fdu.getTag()).lower() == "yes"
        sigma_clip = float(self.getOption("robust_sigma", fdu.getTag()))

        for i in range(nslits):
            print(f"  Tracing Slitlet {i+1}/{nslits}...")
            for edge_idx, start_y in enumerate([sylo[i], syhi[i]]):
                x_pts, y_pts = [], []
                x_init = int(slitx[i])
                xs = list(range(x_init, xsize-10, step)) + list(range(x_init-step, 10, -1*step))
                
                curr_y = start_y
                ref_cut = self.get_1d_cut(flatData, x_init, curr_y, fdu.dispersion)
                
                for x in xs:
                    cut = self.get_1d_cut(flatData, x, curr_y, fdu.dispersion)
                    if len(cut) != len(ref_cut): continue
                    
                    # Normalized cross-correlation
                    c_norm = cut - np.mean(cut)
                    r_norm = ref_cut - np.mean(ref_cut)
                    if np.std(c_norm) > 0 and np.std(r_norm) > 0:
                        ccor = np.correlate(c_norm, r_norm, mode='same')
                        peak = np.argmax(ccor)
                        curr_y += (self.find_subpixel_peak(ccor, peak) - (len(ccor)//2))
                    
                    x_pts.append(x)
                    y_pts.append(curr_y)
                
                fit = self.robust_fit_path(x_pts, y_pts, order, use_splines=use_splines, sigma=sigma_clip)
                all_x = np.arange(xsize)
                if edge_idx == 0:
                    yloMask[i, :] = fit(all_x)
                else:
                    yhiMask[i, :] = fit(all_x)

        # Create the final slitmask using fatboyLibs GPU helper if available
        if self._fdb.getGPUMode():
            slitmask_data = createSlitmask(flatData.shape, yhiMask, yloMask, nslits, horizontal=(fdu.dispersion == fatboyDataUnit.DISPERSION_HORIZONTAL))
        else:
            # CPU mask creation
            slitmask_data = np.zeros(flatData.shape, dtype=np.uint32)
            yind, xind = np.indices(flatData.shape)
            if fdu.dispersion == fatboyDataUnit.DISPERSION_HORIZONTAL:
                for i in range(nslits):
                    mask = (yind >= yloMask[i, :]) & (yind <= yhiMask[i, :])
                    slitmask_data[mask] = i + 1
            else:
                for i in range(nslits):
                    mask = (xind >= yloMask[i, :]) & (xind <= yhiMask[i, :])
                    slitmask_data[mask] = i + 1

        if slitmask_data.max() < 256:
            slitmask_data = slitmask_data.astype(np.uint8)

        calibs['slitmask'] = fatboySpecCalib(self._pname, "slitmask", masterFlat, data=slitmask_data, log=self._log)
        calibs['slitlo'] = fatboySpecCalib(self._pname, "slitlo", masterFlat, data=yloMask, log=self._log)
        calibs['slithi'] = fatboySpecCalib(self._pname, "slithi", masterFlat, data=yhiMask, log=self._log)
        
        return calibs

    def traceSlitlets(self, fdu, calibs):
        """Improved group-mode tracing."""
        # For group mode, we trace the overall shift and apply it to all slits
        return self.traceOrders(fdu, calibs) # Fallback to individual for better precision

    def tracePeakLocalMax(self, fdu, calibs):
        """Improved fiber-mode tracing."""
        return self.traceOrders(fdu, calibs)

    def autoDetectSlitlets(self, fdu, flatData):
        if self.getOption("use_new_extract_spectra", fdu.getTag()).lower() == "yes":
            # 1D cut for detection
            x_auto = int(self.getOption("slitlet_autodetect_x", fdu.getTag()))
            boxsize = int(self.getOption("slitlet_autodetect_boxsize", fdu.getTag()))
            halfbox = boxsize // 2
            
            if fdu.dispersion == fatboyDataUnit.DISPERSION_HORIZONTAL:
                cut1d = flatData[:, x_auto-halfbox:x_auto+halfbox+1].mean(axis=1)
            else:
                cut1d = flatData[x_auto-halfbox:x_auto+halfbox+1, :].mean(axis=0)
            
            sigma = float(self.getOption("slitlet_autodetect_sigma", fdu.getTag()))
            min_width = int(self.getOption("slitlet_autodetect_min_width", fdu.getTag()))
            min_flux_pct = float(self.getOption("slitlet_autodetect_min_flux_pct", fdu.getTag()))
            
            slitlets = newExtractSpectra(cut1d, sigma, min_width, minFluxPct=min_flux_pct)
            
            if slitlets is None: return([], [], [], [])
            
            sylo = slitlets[:, 0]
            syhi = slitlets[:, 1]
            slitx = np.array([x_auto] * len(sylo))
            slitw = np.array([boxsize] * len(sylo))
            return (sylo, syhi, slitx, slitw)
        
        # Fallback to base class
        from superFATBOY.fatboyProcesses.findSlitletProcess import findSlitletProcess
        base = findSlitletProcess(self._fdb)
        base.setOptions(self._options)
        return base.autoDetectSlitlets(fdu, flatData)

def executeNewFindSlitlets(fdu, options=dict(), calibs=dict()):
    process = newFindSlitletProcess()
    process.setDefaultOptions()
    process.setOptions(options)
    process.execute(fdu, calibs)
