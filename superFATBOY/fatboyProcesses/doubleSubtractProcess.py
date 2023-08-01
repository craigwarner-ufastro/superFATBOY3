from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY.fatboyLibs import *
from superFATBOY.datatypeExtensions.fatboySpecCalib import fatboySpecCalib

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
    print("doubleSubtractProcess> Warning: PyCUDA not installed")
    hasCuda = False
from numpy import *
import os, time

block_size = 512

class doubleSubtractProcess(fatboyProcess):
    _modeTags = ["spectroscopy", "miradas"]

    def get_dbs_mod(self):
        dbs_mod = None
        if (hasCuda):
            dbs_mod = SourceModule("""
          __global__ void getPosNegForDS_float(float *image, float *pos, float *neg, int size) {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= size) return;
            pos[i] = image[i]+0;
            neg[i] = -1*image[i];
            if (pos[i] < 0) pos[i] = 0;
            if (neg[i] < 0) neg[i] = 0;
          }


          __global__ void doubleSubtractImages(int *slitmaskIn, float *imageIn, float *expmapIn, float *nmIn, float *cleanFrameIn, int *slitmaskOut, float *imageOut, float *expmapOut, float *nmOut, float *cleanFrameOut, int doExpmap, int doNM, int doCleanFrame, int cols, int rows, int negOffset, int posOffset, int horizontal) {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            int shift = posOffset+negOffset;
            if (horizontal == 1) {
              if (i > (rows+shift)*cols) return;
            } else{
              if (i > rows*(cols+shift)) return;
            }
            //if (i >= rows*cols) return;
            int yind = i / cols;
            int col = i % cols;
            if (horizontal == 0) {
              //vertical dispersion
              yind = i / (cols+shift);
              col = i % (cols+shift);
            }
            int tempSlit = 0;
            slitmaskOut[i] = 0;
            if (horizontal == 1) {
              if (yind >= posOffset && yind < posOffset+rows) {
                slitmaskOut[i] = slitmaskIn[i-posOffset*cols];
                tempSlit = slitmaskIn[i-posOffset*cols];
              }
              if (yind >= negOffset && yind < negOffset+rows) {
                slitmaskOut[i] += slitmaskIn[i-negOffset*cols];
                tempSlit -= slitmaskIn[i-negOffset*cols];
              }
            } else {
              //vertical dispersion
              if (col >= posOffset && col < posOffset+cols) {
                slitmaskOut[i] = slitmaskIn[i-posOffset-shift*yind];
                tempSlit = slitmaskIn[i-posOffset-shift*yind];
              }
              if (col >= negOffset && col < negOffset+cols) {
                slitmaskOut[i] += slitmaskIn[i-negOffset-shift*yind];
                tempSlit -= slitmaskIn[i-negOffset-shift*yind];
              }
            }
            //Get rid of odd values -- this is where 2 slitlets overlap after shift
            if (slitmaskOut[i] % 2 == 1) slitmaskOut[i] = 0;
            slitmaskOut[i] /= 2;
            //tempSlit == 0 where same slitlet overlaps self after shift -- keep this part
            if (tempSlit != 0) tempSlit = 1;
            tempSlit = 1-tempSlit;
            slitmaskOut[i] *= tempSlit;

            //Perform double subtraction
            imageOut[i] = 0;
            if (doExpmap) expmapOut[i] = 0;
            if (doNM) nmOut[i] = 0;
            if (doCleanFrame) cleanFrameOut[i] = 0;
            if (slitmaskOut[i] != 0) {
              if (horizontal == 1) {
                if (yind >= posOffset && yind < posOffset+rows) {
                  imageOut[i] = imageIn[i-posOffset*cols];
                  if (doExpmap) expmapOut[i] = expmapIn[i-posOffset*cols];
                  if (doNM) nmOut[i] = nmIn[i-posOffset*cols]*nmIn[i-posOffset*cols];
                  if (doCleanFrame) cleanFrameOut[i] = cleanFrameIn[i-posOffset*cols];
                }
                if (yind >= negOffset && yind < negOffset+rows) {
                  imageOut[i] -= imageIn[i-negOffset*cols];
                  if (doExpmap) expmapOut[i] += expmapIn[i-negOffset*cols];
                  if (doNM) nmOut[i] += nmIn[i-negOffset*cols]*nmIn[i-negOffset*cols];
                  if (doCleanFrame) cleanFrameOut[i] -= cleanFrameIn[i-negOffset*cols];
                }
              } else {
                //vertical dispersion
                if (col >= posOffset && col < posOffset+cols) {
                  imageOut[i] = imageIn[i-posOffset-shift*yind];
                  if (doExpmap) expmapOut[i] = expmapIn[i-posOffset-shift*yind];
                  if (doNM) nmOut[i] = nmIn[i-posOffset-shift*yind]*nmIn[i-posOffset-shift*yind];
                  if (doCleanFrame) cleanFrameOut[i] = cleanFrameIn[i-posOffset-shift*yind];
                }
                if (col >= negOffset && col < negOffset+cols) {
                  imageOut[i] -= imageIn[i-negOffset-shift*yind];
                  if (doExpmap) expmapOut[i] += expmapIn[i-negOffset-shift*yind];
                  if (doNM) nmOut[i] += nmIn[i-negOffset-shift*yind]*nmIn[i-negOffset-shift*yind];
                  if (doCleanFrame) cleanFrameOut[i] -= cleanFrameIn[i-negOffset-shift*yind];
               }
              }
              if (doNM) nmOut[i] = sqrt(nmOut[i]);
            }
          }

          __global__ void doubleSubtractImages_noslit(int *slitmask, float *imageIn, float *expmapIn, float *nmIn, float *cleanFrameIn, float *imageOut, float *expmapOut, float *nmOut, float *cleanFrameOut, int hasSlitmask, int doExpmap, int doNM, int doCleanFrame, int cols, int rows, int negOffset, int posOffset, int horizontal) {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            int shift = posOffset+negOffset;
            if (horizontal == 1) {
              if (i > (rows+shift)*cols) return;
            } else{
              if (i > rows*(cols+shift)) return;
            }
            //if (i >= rows*cols) return;
            int yind = i / cols;
            int col = i % cols;
            if (horizontal == 0) {
              //vertical dispersion
              yind = i / (cols+shift);
              col = i % (cols+shift);
            }
            //Perform double subtraction
            imageOut[i] = 0;
            int sm = 1; //placeholder in case of no slitmask
            if (doExpmap) expmapOut[i] = 0;
            if (doNM) nmOut[i] = 0;
            if (doCleanFrame) cleanFrameOut[i] = 0;
            if (hasSlitmask) sm = slitmask[i];
            if (sm != 0) {
              if (horizontal == 1) {
                if (yind >= posOffset && yind < posOffset+rows) {
                  imageOut[i] = imageIn[i-posOffset*cols];
                  if (doExpmap) expmapOut[i] = expmapIn[i-posOffset*cols];
                  if (doNM) nmOut[i] = nmIn[i-posOffset*cols]*nmIn[i-posOffset*cols];
                  if (doCleanFrame) cleanFrameOut[i] = cleanFrameIn[i-posOffset*cols];
                }
                if (yind >= negOffset && yind < negOffset+rows) {
                  imageOut[i] -= imageIn[i-negOffset*cols];
                  if (doExpmap) expmapOut[i] += expmapIn[i-negOffset*cols];
                  if (doNM) nmOut[i] += nmIn[i-negOffset*cols]*nmIn[i-negOffset*cols];
                  if (doCleanFrame) cleanFrameOut[i] -= cleanFrameIn[i-negOffset*cols];
               }
              } else {
                //vertical dispersion
                if (col >= posOffset && col < posOffset+cols) {
                  imageOut[i] = imageIn[i-posOffset-shift*yind];
                  if (doExpmap) expmapOut[i] = expmapIn[i-posOffset-shift*yind];
                  if (doNM) nmOut[i] = nmIn[i-posOffset-shift*yind]*nmIn[i-posOffset-shift*yind];
                  if (doCleanFrame) cleanFrameOut[i] = cleanFrameIn[i-posOffset-shift*yind];
                }
                if (col >= negOffset && col < negOffset+cols) {
                  imageOut[i] -= imageIn[i-negOffset-shift*yind];
                  if (doExpmap) expmapOut[i] += expmapIn[i-negOffset-shift*yind];
                  if (doNM) nmOut[i] += nmIn[i-negOffset-shift*yind]*nmIn[i-negOffset-shift*yind];
                  if (doCleanFrame) cleanFrameOut[i] -= cleanFrameIn[i-negOffset-shift*yind];
               }
              }
              if (doNM) nmOut[i] = sqrt(nmOut[i]);
            }
          }
        """)
        return dbs_mod
    #end get_dbs_mod

    #def doDoubleSubtractionGPU(self, slitmaskIn, imageIn, expmapIn, shift, nmIn=None, doSlitmask=True):
    def doDoubleSubtractGPU(self, fdu, calibs, shift):
        t = time.time()
        #Get negative and positive offsets
        negOffset = 0-min([0,shift])
        posOffset = shift-min([0,shift])

        inShape = fdu.getShape()
        rows = inShape[0]
        cols = inShape[1]
        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            outShape = (rows+abs(shift), cols)
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            outShape = (rows, cols+abs(shift))

        #Input data
        imageIn = fdu.getData().copy()
        #Form output arrays
        imageOut = empty(outShape, dtype=float32)

        #Figure out slitmask options
        doSlitmask = False
        hasSlitmask = False
        ###Important: as of double subtraction, 'slitmask' will no longer be a calib for the
        ###entire object / data set but each frame will have its own slitmask as a property.
        ###This is because double subtract shifts can vary between frames!
        slitmaskIn = empty(1, dtype=int32)
        if (fdu._specmode != fdu.FDU_TYPE_LONGSLIT and 'slitmask' in calibs):
            hasSlitmask = True
            #slitmaskIn will always be the calib
            slitmaskIn = int32(calibs['slitmask'].getData().copy())
            if (not fdu.hasProperty('slitmask')):
                #Output will be tied to each individual frame
                doSlitmask = True
                slitmaskOut = empty(outShape, dtype=int32)

        #Figure out noisemap options
        nmIn = empty(1, dtype=float32)
        nmOut = empty(1, dtype=float32)
        doNM = False
        if (fdu.hasProperty("noisemap")):
            doNM = True
            nmIn = fdu.getProperty("noisemap").copy()
            nmOut = empty(outShape, dtype=float32)

        #Figure out cleanFrame options
        cleanFrameIn = empty(1, dtype=float32)
        cleanFrameOut = empty(1, dtype=float32)
        doCleanFrame = False
        if (fdu.hasProperty("cleanFrame")):
            doCleanFrame = True
            cleanFrameIn = fdu.getProperty("cleanFrame").copy()
            cleanFrameOut = empty(outShape, dtype=float32)

        #Figure out exposure map options
        expmapIn = empty(1, dtype=float32)
        expmapOut = empty(1, dtype=float32)
        doExpmap = False
        if (fdu.hasProperty("exposure_map")):
            doExpmap = True
            expmapIn = fdu.getProperty("exposure_map").copy()
            expmapOut = empty(outShape, dtype=float32)

        blocks = imageOut.size//512
        if (imageOut.size % 512 != 0):
            blocks += 1
        if (doSlitmask):
            kernel = self.get_dbs_mod().get_function("doubleSubtractImages")
            kernel(drv.In(slitmaskIn), drv.In(imageIn), drv.In(expmapIn), drv.In(nmIn), drv.In(cleanFrameIn), drv.Out(slitmaskOut), drv.Out(imageOut), drv.Out(expmapOut), drv.Out(nmOut), drv.Out(cleanFrameOut), int32(doExpmap), int32(doNM), int32(doCleanFrame), int32(cols), int32(rows), int32(negOffset), int32(posOffset), int32(fdu.dispersion == fdu.DISPERSION_HORIZONTAL), grid=(blocks,1), block=(block_size,1,1))
            #Create new data tag "slitmask"
            #Use new fdu.setSlitmask
            fdu.setSlitmask(slitmaskOut, pname=self._pname)
        else:
            kernel = self.get_dbs_mod().get_function("doubleSubtractImages_noslit")
            kernel(drv.In(slitmaskIn), drv.In(imageIn), drv.In(expmapIn), drv.In(nmIn), drv.In(cleanFrameIn), drv.Out(imageOut), drv.Out(expmapOut), drv.Out(nmOut), drv.Out(cleanFrameOut), int32(hasSlitmask), int32(doExpmap), int32(doNM), int32(doCleanFrame), int32(cols), int32(rows), int32(negOffset), int32(posOffset), int32(fdu.dispersion == fdu.DISPERSION_HORIZONTAL), grid=(blocks,1), block=(block_size,1,1))
        fdu.updateData(imageOut)
        if (doNM):
            #Update "noisemap" data tag
            fdu.tagDataAs("noisemap", data=nmOut)
        if (doCleanFrame):
            #Update "cleanFrame" data tag
            fdu.tagDataAs("cleanFrame", data=cleanFrameOut)
        if (doExpmap):
            #Update "exposure_map" data tag
            fdu.tagDataAs("exposure_map", data=expmapOut)
        print(time.time()-t)
        #No need to return anything as fdu and calibs will be updated in place
    #end doDoubleSubtractionGPU

    # Actually perform double subtraction
    def doubleSubtractImage(self, fdu, calibs):
        t = time.time()
        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            xsize = fdu.getShape()[1]
            ysize = fdu.getShape()[0]
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            ##xsize should be size across dispersion direction
            xsize = fdu.getShape()[0]
            ysize = fdu.getShape()[1]

        shift = calibs['shift']

        #Update RA and DEC based on shift
        if (shift > 0 and fdu.hasProperty("matched_ra") and fdu.hasProperty("matched_dec")):
            #shift is positive, use RA and DEC of matched frame
            fdu.ra = fdu.getProperty("matched_ra")
            fdu.dec = fdu.getProperty("matched_dec")

        #Select cpu/gpu option
        if (self._fdb.getGPUMode()):
            #Use GPU
            self.doDoubleSubtractGPU(fdu, calibs, shift)
        else:
            negOffset = 0-min([0,shift])
            posOffset = shift-min([0,shift])
            ###Important: as of double subtraction, 'slitmask' will no longer be a calib for the
            ###entire object / data set but each frame will have its own slitmask as a property.
            ###This is because double subtract shifts can vary between frames!
            if (fdu._specmode != fdu.FDU_TYPE_LONGSLIT and 'slitmask' in calibs):
                slitmask = calibs['slitmask']
                if (not fdu.hasProperty('slitmask')):
                    pos = slitmask.getData().copy().astype(uint8)
                    if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                        data = zeros((ysize+abs(shift),xsize), uint8)
                        data[posOffset:posOffset+ysize,:] = pos
                        data[negOffset:negOffset+ysize,:] += pos
                        overlap = zeros((ysize+abs(shift),xsize), uint8)
                        overlap[posOffset:posOffset+ysize,:] = pos
                        overlap[negOffset:negOffset+ysize,:] -= pos
                    elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                        data = zeros((xsize,ysize+abs(shift)), uint8)
                        data[:,posOffset:posOffset+ysize] = pos
                        data[:,negOffset:negOffset+ysize] += pos
                        overlap = zeros((xsize,ysize+abs(shift)), uint8)
                        overlap[:,posOffset:posOffset+ysize] = pos
                        overlap[:,negOffset:negOffset+ysize] -= pos
                    #Get rid of odd values -- this is where 2 slitlets overlap after shift
                    data[data % 2 == 1] = 0
                    data //= 2
                    #overlap == 0 => where same slitlet overlaps self after shift -- keep this part
                    overlap[overlap!=0] = 1
                    overlap = 1-overlap
                    data *= overlap
                    #Create new data tag "slitmask"
                    #Use new fdu.setSlitmask
                    fdu.setSlitmask(data, pname=self._pname)
                    del overlap

            #Perform double subtraction
            pos = fdu.getData().copy()
            if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                data = zeros((ysize+abs(shift), xsize), dtype=float32)
                data[posOffset:posOffset+ysize,:] = pos
                data[negOffset:negOffset+ysize,:] -= pos
            elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                data = zeros((xsize,ysize+abs(shift)), dtype=float32)
                data[:,posOffset:posOffset+ysize] = pos
                data[:,negOffset:negOffset+ysize] -= pos
            if (fdu._specmode != fdu.FDU_TYPE_LONGSLIT and fdu.hasProperty('slitmask')):
                #Blank out non-overlap regions for mutli object data
                data[fdu.getData(tag="slitmask") == 0] = 0
            fdu.updateData(data)

            if (fdu.hasProperty("cleanFrame")):
                #Double subtract clean frame
                pos = fdu.getData(tag="cleanFrame").copy()
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    data = zeros((ysize+abs(shift), xsize), dtype=float32)
                    data[posOffset:posOffset+ysize,:] = pos
                    data[negOffset:negOffset+ysize,:] -= pos
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    data = zeros((xsize,ysize+abs(shift)), dtype=float32)
                    data[:,posOffset:posOffset+ysize] = pos
                    data[:,negOffset:negOffset+ysize] -= pos
                if (fdu._specmode != fdu.FDU_TYPE_LONGSLIT and fdu.hasProperty('slitmask')):
                    #Blank out non-overlap regions for mutli object data
                    data[fdu.getData(tag="slitmask") == 0] = 0
                #Update "cleanFrame" data tag
                fdu.tagDataAs("cleanFrame", data=data)

            if (fdu.hasProperty("exposure_map")):
                #Double subtract exposure map
                pos = fdu.getData(tag="exposure_map").copy()
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    data = zeros((ysize+abs(shift), xsize), dtype=float32)
                    data[posOffset:posOffset+ysize,:] = pos
                    data[negOffset:negOffset+ysize,:] += pos
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    data = zeros((xsize,ysize+abs(shift)), dtype=float32)
                    data[:,posOffset:posOffset+ysize] = pos
                    data[:,negOffset:negOffset+ysize] += pos
                if (fdu._specmode != fdu.FDU_TYPE_LONGSLIT and fdu.hasProperty('slitmask')):
                    #Blank out non-overlap regions for mutli object data
                    data[fdu.getData(tag="slitmask") == 0] = 0
                #Update "exposure_map" data tag
                fdu.tagDataAs("exposure_map", data=data)

            if (fdu.hasProperty("noisemap")):
                #Propagate noisemap
                pos = fdu.getData(tag="noisemap").copy()
                if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
                    data = zeros((ysize+abs(shift), xsize), dtype=float32)
                    data[posOffset:posOffset+ysize,:] = pos**2
                    data[negOffset:negOffset+ysize,:] += pos**2
                elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
                    data = zeros((xsize,ysize+abs(shift)), dtype=float32)
                    data[:,posOffset:posOffset+ysize] = pos**2
                    data[:,negOffset:negOffset+ysize] += pos**2
                data = sqrt(data).astype(float32)
                if (fdu._specmode != fdu.FDU_TYPE_LONGSLIT and fdu.hasProperty('slitmask')):
                    #Blank out non-overlap regions for mutli object data
                    data[fdu.getData(tag="slitmask") == 0] = 0
                #Update "noisemap" data tag
                fdu.tagDataAs("noisemap", data=data)
            #DO BELOW fdu.exptime *= 2
        print("Double subtract time: ",time.time()-t)
    #end doubleSubtractImage

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        if (fdu.hasProperty("use_only_positive")):
            #Either step sky subtract method or odd frame.  Skip.
            return True

        print("Double Subtract")
        print(fdu._identFull)

        #Check if output exists first
        fdfile = "doubleSubtracted/dbs_"+fdu.getFullId()
        if (self.checkOutputExists(fdu, fdfile)):
            #Also check if "cleanFrame" exists
            cleanfile = "doubleSubtracted/clean_dbs_"+fdu.getFullId()
            self.checkOutputExists(fdu, cleanfile, tag="cleanFrame")
            #Also check if "exposure map" exists
            expfile = "doubleSubtracted/exp_dbs_"+fdu.getFullId()
            self.checkOutputExists(fdu, expfile, tag="exposure_map")
            #Also check if "slitmask" exists
            smfile = "doubleSubtracted/slitmask_dbs_"+fdu.getFullId()
            self.checkOutputExists(fdu, smfile, tag="slitmask")
            #Also check if "noisemap" exists
            nmfile = "doubleSubtracted/NM_dbs_"+fdu.getFullId()
            self.checkOutputExists(fdu, nmfile, tag="noisemap")
            #Double exptime
            fdu.exptime *= 2
            updateHeaderEntry(fdu._header, fdu._keywords['exptime_keyword'], fdu.exptime) #Use wrapper function to update header
            return True

        #Call get calibs to return dict() of calibration frames.
        #For doubleSubtract, this dict should have one entry 'masterFlat' which is an fdu
        calibs = self.getCalibs(fdu, prevProc)
        if (not 'shift' in calibs):
            #Failed to obtain shift
            #Issue error message and disable this FDU
            print("doubleSubtractProcess::execute> ERROR: Shift for double subtraction not found for "+fdu.getFullId()+" (filter="+str(fdu.filter)+").  Discarding Image!")
            self._log.writeLog(__name__, "Shift for double subtraction not found for "+fdu.getFullId()+" (filter="+str(fdu.filter)+").  Discarding Image!", type=fatboyLog.ERROR)
            #disable this FDU
            fdu.disable()
            return False

        #call doubleSubtractImage helper function to do gpu/cpu double subtraction
        self.doubleSubtractImage(fdu, calibs)
        #Double exptime
        fdu.exptime *= 2
        updateHeaderEntry(fdu._header, fdu._keywords['exptime_keyword'], fdu.exptime) #Use wrapper function to update header
        #call doubleSubtractImage helper function to do gpu/cpu division
        fdu._header.add_history('Double subtracted with shift '+str(calibs['shift']))
        return True
    #end execute

    #find the shift between positive and negative continua
    def findDoubleSubtractShift(self, fdu):
        #Get options
        xlo = int(self.getOption("find_shift_box_xlo", fdu.getTag()))
        xhi = int(self.getOption("find_shift_box_xhi", fdu.getTag()))
        ylo = int(self.getOption("find_shift_box_ylo", fdu.getTag()))
        yhi = int(self.getOption("find_shift_box_yhi", fdu.getTag()))
        constrain_boxsize = self.getOption("find_shift_constrain_boxsize", fdu.getTag())
        if (constrain_boxsize is not None):
            constrain_boxsize = int(constrain_boxsize)
        useHeader = False
        if (self.getOption("use_header", fdu.getTag()).lower() == "yes"):
            useHeader = True

        if (useHeader):
            #Use the RA, DEC, and pixscale - guess has already been calculated in sky subtract spec
            if (fdu.hasProperty("double_subtract_guess")):
                dbs_guess = int(round(fdu.getProperty("double_subtract_guess")))
                print("doubleSubtractProcess::findDoubleSubtractShift> Using "+str(dbs_guess)+" as calculated from header info for "+fdu.getFullId()+"...")
                self._log.writeLog(__name__, "Using "+str(dbs_guess)+" as calculated from header info for "+fdu.getFullId()+"...")
                return dbs_guess
            else:
                print("doubleSubtractProcess::findDoubleSubtractShift> Could not find property double_subtract_guess.  Will attempt to find shift for "+fdu.getFullId()+"...")
                self._log.writeLog(__name__, "Could not find property double_subtract_guess.  Will attempt to find shift for "+fdu.getFullId()+"...")

        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            xsize = fdu.getShape()[1]
            ysize = fdu.getShape()[0]
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            ##xsize should be size across dispersion direction
            xsize = fdu.getShape()[0]
            ysize = fdu.getShape()[1]

        if (self._fdb.getGPUMode()):
            (pos, neg) = self.getPosNegForDS(fdu.getData(tag="cleanFrame"))
        else:
            pos = fdu.getData(tag="cleanFrame").copy()
            neg = -1.0*pos
            pos[pos < 0] = 0
            neg[neg < 0] = 0
        #Cross correlate positives only vs inverse of negatives only
        #Find integer pixel offsets
        #Instead of taking median, sum so we get short spectra but do a
        #5 pixel boxcar median smoothing to get rid of hot pixels
        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            posCut = mediansmooth1d(sum(pos[ylo:yhi,xlo:xhi],1), 5)
            negCut = mediansmooth1d(sum(neg[ylo:yhi,xlo:xhi],1), 5)
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            posCut = mediansmooth1d(sum(pos[xlo:xhi,ylo:yhi],0), 5)
            negCut = mediansmooth1d(sum(neg[xlo:xhi,ylo:yhi],0), 5)
        ccor = correlate(posCut,negCut,mode='same')
        mcor = where(ccor == max(ccor))[0]
        if (constrain_boxsize is not None and fdu.hasProperty("double_subtract_guess")):
            print("doubleSubtractProcess::findDoubleSubtractShift> Using initial guess "+str(fdu.getProperty("double_subtract_guess"))+" pixels and boxsize "+str(constrain_boxsize)+" for "+fdu.getFullId()+"...")
            self._log.writeLog(__name__, "Using initial guess "+str(fdu.getProperty("double_subtract_guess"))+" pixels and boxsize "+str(constrain_boxsize)+" for "+fdu.getFullId()+"...")
            guess1 = int(len(ccor)//2-fdu.getProperty("double_subtract_guess"))
            guess2 = int(len(ccor)//2+fdu.getProperty("double_subtract_guess"))
            maxVal = max(ccor[guess1-constrain_boxsize//2:guess1+constrain_boxsize//2].max(), ccor[guess2-constrain_boxsize//2:guess2+constrain_boxsize//2].max())
            mcor = where(ccor == maxVal)[0]
        shift = len(ccor)//2-mcor[0]
        print("doubleSubtractProcess::findDoubleSubtractShift> Double subtract shift = "+str(shift)+" pixels for "+fdu.getFullId())
        self._log.writeLog(__name__, "Double subtract shift = "+str(shift)+" pixels for "+fdu.getFullId())
        return shift
    #end findDoubleSubtractShift

    ## OVERRIDE getCalibs
    def getCalibs(self, fdu, prevProc = None):
        calibs = dict()

        #Look for each master calib passed from XML
        smfilename = self.getCalib("slitmask", fdu.getTag())
        if (smfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(smfilename, os.F_OK)):
                print("doubleSubtractProcess::getCalibs> Using slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Using slitmask "+smfilename+"...")
                calibs['slitmask'] = fatboySpecCalib(self._pname, "slitmask", fdu, filename=smfilename, log=self._log)
            else:
                print("doubleSubtractProcess::getCalibs> Warning: Could not find slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Could not find slitmask "+smfilename+"...", type=fatboyLog.WARNING)

        headerVals = dict()
        headerVals['grism_keyword'] = fdu.grism
        properties = dict()
        properties['specmode'] = fdu.getProperty("specmode")
        properties['dispersion'] = fdu.getProperty("dispersion")

        if (fdu._specmode != fdu.FDU_TYPE_LONGSLIT and not 'slitmask' in calibs):
            #Multi object data, need slitmask
            #Find slitmask associated with this fdu
            #Use new fdu.getSlitmask method
            slitmask = fdu.getSlitmask(pname=None, properties=properties, headerVals=headerVals)
            if (slitmask is None):
                #Warning not ERROR -- can do double subtract fine without a slitmask
                print("doubleSubtractProcess::getCalibs> Warning: Could not find slitmask associated with "+fdu.getFullId()+"!")
                self._log.writeLog(__name__, "Could not find slitmask associated with "+fdu.getFullId()+"!", type=fatboyLog.WARNING)
            else:
                calibs['slitmask'] = slitmask

        calibs['shift'] = self.findDoubleSubtractShift(fdu)
        return calibs
    #end getCalibs

    def getPosNegForDS(self, image):
        #Returns (pos, neg)
        image = float32(image)
        pos = empty(image.shape, dtype=float32)
        neg = empty(image.shape, dtype=float32)
        blocks = image.size//512
        if (image.size % 512 != 0):
            blocks += 1
        kernel = self.get_dbs_mod().get_function("getPosNegForDS_float")
        kernel(drv.In(image), drv.Out(pos), drv.Out(neg), int32(image.size), grid=(blocks,1), block=(block_size,1,1))
        return (pos, neg)
    #end getPosNegForDS

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('find_shift_box_xlo', '0')
        self._optioninfo.setdefault('find_shift_box_xlo', 'Used to specify a range of the chip in dispersion direction to sum\na 1-d cut across and attemt to find shift.')
        self._options.setdefault('find_shift_box_xhi', '-1')
        self._optioninfo.setdefault('find_shift_box_xhi', 'Used to specify a range of the chip in dispersion direction to sum\na 1-d cut across and attemt to find shift.')
        self._options.setdefault('find_shift_box_ylo', '0')
        self._optioninfo.setdefault('find_shift_box_ylo', 'Used to specify a range of the chip in cross-dispersion\ndirection to sum a 1-d cut across and attemt to find shift.')
        self._options.setdefault('find_shift_box_yhi', '-1')
        self._optioninfo.setdefault('find_shift_box_yhi', 'Used to specify a range of the chip in cross-dispersion\ndirection to sum a 1-d cut across and attemt to find shift.')
        self._options.setdefault('find_shift_constrain_boxsize', None)
        self._optioninfo.setdefault('find_shift_constrain_boxsize', 'Constrain the fit to a box of this size, centered\nat the initial guess based on RA and Dec offsets.')
        self._options.setdefault('use_header', 'no')
        self._optioninfo.setdefault('use_header', 'Use the information in the header -\nRA, DEC, PIXSCALE - instead of\nattempting to find shift.')
        self._options.setdefault('write_noisemaps', 'no')
    #end setDefaultOptions

    ## update noisemap for spectroscopy data
    def updateNoisemap(self, fdu, masterFlat):
        if (not masterFlat.hasProperty("noisemap")):
            #Hopefully we don't get here because this means we are reading a previous masterFlat from disk with no corresponding noisemap on disk
            #If masterFlat is dome on, we're fine but if its on-off, we lose separate on and off data.
            #create tagged data "noisemap"
            #Create noisemap
            ncomb = 1.0
            if (masterFlat.hasHeaderValue('NCOMBINE')):
                ncomb = float(masterFlat.getHeaderValue('NCOMBINE'))
            if (self._fdb.getGPUMode()):
                nm = createNoisemap(masterFlat.getData(), ncomb)
            else:
                nm = sqrt(masterFlat.getData()/ncomb)
            masterFlat.tagDataAs("noisemap", nm)
        #Get this FDU's noisemap
        nm = fdu.getData(tag="noisemap")
        #Propagate noisemaps.  For division dz/z = sqrt((dx/x)^2 + (dy/y)^2)
        if (self._fdb.getGPUMode()):
            #noisemaps_dbs_gpu(dbs_image, pre-dbs_noisemap, pre-dbs_image, mflat noisemap, mflat
            nm = noisemaps_dbs_gpu(fdu.getData(), fdu.getData(tag="noisemap"), fdu.getData("cleanFrame"), masterFlat.getData("noisemap"), masterFlat.getData())
        else:
            nm = abs(fdu.getData())*sqrt(fdu.getData(tag="noisemap")**2/fdu.getData("cleanFrame")**2 + masterFlat.getData("noisemap")**2/masterFlat.getData()**2)
            nm[fdu.getData("cleanFrame") == 0] = 0
            nm[masterFlat.getData() == 0] = 0
        fdu.tagDataAs("noisemap", nm)
    #end updateNoisemap

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/doubleSubtracted", os.F_OK)):
            os.mkdir(outdir+"/doubleSubtracted",0o755)
        #Create output filename
        fdfile = outdir+"/doubleSubtracted/dbs_"+fdu.getFullId()
        #Check to see if it exists
        if (os.access(fdfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(fdfile)
        if (not os.access(fdfile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(fdfile)
        #Write out clean frame if it exists
        if (fdu.hasProperty("cleanFrame")):
            cleanfile = outdir+"/doubleSubtracted/clean_dbs_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(cleanfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(cleanfile)
            if (not os.access(cleanfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.writeTo(cleanfile, tag="cleanFrame")
        #Write out exposure map if it exists
        if (fdu.hasProperty("exposure_map")):
            expfile = outdir+"/doubleSubtracted/exp_dbs_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(expfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(expfile)
            if (not os.access(expfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.writeTo(expfile, tag="exposure_map")
        #Write noisemap for spectrocsopy data if requested
        if (self.getOption("write_noisemaps", fdu.getTag()).lower() == "yes" and fdu.hasProperty("noisemap")):
            nmfile = outdir+"/doubleSubtracted/NM_dbs_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(nmfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(nmfile)
            if (not os.access(nmfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                fdu.writeTo(nmfile, tag="noisemap")
        #Write out slitmask if it exists
        if (fdu.hasProperty("slitmask")):
            smfile = outdir+"/doubleSubtracted/slitmask_dbs_"+fdu.getFullId()
            #Check to see if it exists
            if (os.access(smfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                os.unlink(smfile)
            if (not os.access(smfile, os.F_OK)):
                #Use fatboyDataUnit writeTo method to write
                #Use new fdu.getSlitmask to get FDU and then write
                fdu.getSlitmask().writeTo(smfile)
                #fdu.writeTo(smfile, tag="slitmask")
    #end writeOutput
