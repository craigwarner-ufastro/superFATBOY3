from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.fatboyLog import fatboyLog
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
    print("linearityProcess> Warning: PyCUDA not installed")
    hasCuda = False
from numpy import *
import os, time

block_size = 512

class linearityProcess(fatboyProcess):
    _modeTags = ["imaging", "circe", "spectroscopy", "miradas"]

    def get_linearity_mod(self):
        linearity_mod = None
        if (hasCuda):
            linearity_mod = SourceModule("""
          __global__ void gpu_linearity_int(float *output, int *input, float *coeffs, int ncoeffs, int size) {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= size) return;
            int n = 1;
            output[i] = input[i]*coeffs[0];
            for (int j = 1; j < ncoeffs; j++) {
              n++;
              output[i] += coeffs[j] * pow((float)input[i], n);
            }
          }

          __global__ void gpu_linearity_float(float *output, float *input, float *coeffs, int ncoeffs, int size) {
            const int i = blockDim.x*blockIdx.x + threadIdx.x;
            if (i >= size) return;
            int n = 1;
            output[i] = input[i]*coeffs[0];
            for (int j = 1; j < ncoeffs; j++) {
              n++;
              output[i] += coeffs[j] * pow(input[i], n);
            }
          }
        """)
        return linearity_mod
    #end get_linearity_mod

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("Linearity")
        print(fdu._identFull)

        #Check if output exists first
        linfile = "linearized/lin_"+fdu.getFullId()
        if (self.checkOutputExists(fdu, linfile)):
            return True

        #Select cpu/gpu option
        if (self._fdb.getGPUMode()):
            linearity = self.linearity_gpu
        else:
            linearity = self.linearity_cpu
        #Read options
        doCoadd = False
        if (self.getOption('divide_by_coadds', fdu.getTag()).lower() == "yes"):
            doCoadd = True
        doLin = False
        if (self.getOption('do_linearity', fdu.getTag()).lower() == "yes"):
            doLin = True
        #Read coefficients
        coeffs = self.getOption('linearity_coeffs', fdu.getTag()).split()
        #Check if FDU has superceding property - e.g., SINFONI data
        if (fdu.hasProperty('linearity_coeffs')):
            coeffs = fdu.getProperty('linearity_coeffs')
        for j in range(len(coeffs)):
            coeffs[j] = float(coeffs[j])
        if (fdu._name == "circeImage" or fdu._name == "circeFastImage"):
            #Special case for circe data
            self.performCirceLinearity(fdu, coeffs)
            if (doCoadd and fdu.nreads > 1):
                #Faster to multiply by inverse than to divide
                fdu.updateData(fdu.getData()*(1./(fdu.nreads)))
                fdu._header.add_history('divided by '+str(fdu.nreads)+' reads')
            return True
        if (doCoadd):
            #Special case, 0.05s speedup by not dividing by 1.
            if (fdu.coadds*fdu.nreads != 1):
                #Faster to multiply by inverse than to divide
                fdu.updateData(fdu.getData()*(1./(fdu.coadds*fdu.nreads)))
                fdu._header.add_history('divided by '+str(fdu.coadds)+' coadds')
                fdu._header.add_history('divided by '+str(fdu.nreads)+' reads')
        if (doLin):
            #Special case, skip if unity transformation
            if (len(coeffs) != 1 or coeffs[0] != 1):
                fdu.updateData(linearity(fdu.getData(), coeffs))
            fdu._header.add_history('linearized with coeffs '+str(coeffs))
        return True
    #end execute

    def linearity_cpu(self, data, coeffs):
        data = float32(data)
        output = zeros(shape(data), float32)
        n = 0
        t = time.time()
        for j in coeffs:
            n+=1
            output += j*data**n             #j * data^n
        if (self._fdb._verbosity == fatboyLog.VERBOSE):
            print("CPU Linearize: ",time.time()-t)
        return output
    #end linearity

    def linearity_gpu(self, data, coeffs):
        t = time.time()
        blocks = data.size//block_size
        if (data.size % block_size != 0):
            blocks += 1
        gpu_linearity = self.get_linearity_mod().get_function("gpu_linearity_float")
        if (data.dtype == int32):
            gpu_linearity = self.get_linearity_mod().get_function("gpu_linearity_int")
        else:
            #Cast data
            data = data.astype(float32)
        coeffs = array(coeffs).astype(float32)
        ncoeffs = coeffs.size
        output = empty(data.shape, float32)

        gpu_linearity(drv.Out(output), drv.In(data), drv.In(coeffs), int32(ncoeffs), int32(data.size), grid=(blocks,1), block=(block_size,1,1))
        if (self._fdb._verbosity == fatboyLog.VERBOSE):
            print("GPU linearize: ",time.time()-t)

        return output
    #end linearity_gpu

    ## Special algorithm to process CIRCE data
    def performCirceLinearity(self, fdu, coeffs):
        #Select cpu/gpu option
        if (self._fdb.getGPUMode()):
            linearity = self.linearity_gpu
        else:
            linearity = self.linearity_cpu

        if ((fdu._expmode == fdu.EXPMODE_FS and fdu.nreads == 1) or fdu._expmode == fdu.EXPMODE_URG_BYPASS):
            #Linearity should be the same as normal here, linearity(read_n - read_1)
            fdu.updateData(linearity(fdu.getData(), coeffs))
        elif (fdu._expmode == fdu.EXPMODE_FS):
            #Fowler sampling with multiple reads
            #For each read pair (n,m), calculate linearity(read_n-read_1) - linearity(read_m-read_1)
            #E.g. for nreads=4, read=2, linearity(read_6-read_1) - linearity(read_2-read_1)
            #Data has nramps sets of 2x nreads frames but will end up with nramps total images
            #Differences of read pairs are summed
            #First read pair -- do not need to include read_m-read_1 because m == 1
            data = linearity(fdu.getIndividualRead((fdu.ramp-1)*2*fdu.nreads+fdu.nreads+1) - fdu.getIndividualRead((fdu.ramp-1)*2*fdu.nreads+1), coeffs)
            for read in range(2, fdu.nreads+1):
                #Add linearity(read_n - read_1)
                data += linearity(fdu.getIndividualRead((fdu.ramp-1)*2*fdu.nreads+fdu.nreads+read) - fdu.getIndividualRead((fdu.ramp-1)*2*fdu.nreads+1), coeffs)
                #Subtract linearity(read_m - read_1)
                data -= linearity(fdu.getIndividualRead((fdu.ramp-1)*2*fdu.nreads+read) - fdu.getIndividualRead((fdu.ramp-1)*2*fdu.nreads+1), coeffs)
            fdu.updateData(data)
        elif (fdu._expmode == fdu.EXPMODE_URG):
            #Up the ramp mode
            #CIRCE URG data has nreads = 1, nramps sets of ngroups frames
            #E.g., ngroups=4, nramps = 2 => [1,1], [2,1], [3,1], [4,1], RESET, [1,2], [2,2], [3,2], [4,2]
            #Final output is nramps * (ngroups-1) frames, [2,1]-[1,1]; [3,1]-[2,1]; [4,1]-[3,1]; [2,2]-[1,2]; [3,2]-[2,2], [4,2]-[3,2]
            #For each group pair (n,m), calculate linearity(group_n-group_1) - linearity(group_m-group_1)
            #For example, for ramp 2, group 2, frame [3,2] - [2,2] => linearity([3,2]-[1,2]) - linearity([2,2]-[1,2])
            #     => linearity(read_7 - read_5) - linearity(read_6 - read_5)
            #     => linearity(getIndividualRead((ramp-1)*getNGroups()+group+1) - getIndividualRead((ramp-1)*getNGroups()+1))
            #      - linearity(getIndividualRead((ramp-1)*getNGroups()+group)) - getIndividualRead((ramp-1)*getNGroups()+1))
            data = linearity(fdu.getIndividualRead((fdu.ramp-1)*fdu.getNGroups()+fdu.group+1) - fdu.getIndividualRead((fdu.ramp-1)*fdu.getNGroups()+1), coeffs)
            data -= linearity(fdu.getIndividualRead((fdu.ramp-1)*fdu.getNGroups()+fdu.group) - fdu.getIndividualRead((fdu.ramp-1)*fdu.getNGroups()+1), coeffs)
            fdu.updateData(data)
        else:
            ##getData() call will trigger invalid mode message, disable frame
            fdu.getData()
            return
        fdu._header.add_history('linearized with 1/26/18 CIRCE algorithm and coeffs '+str(coeffs))
    #end performCirceLinearity

    linearity = linearity_gpu

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('divide_by_coadds', 'no')
        self._options.setdefault('do_linearity', 'yes')
        self._options.setdefault('linearity_coeffs', '1')
    #end setDefaultOptions

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/linearized", os.F_OK)):
            os.mkdir(outdir+"/linearized",0o755)
        #Create output filename
        linfile = outdir+"/linearized/lin_"+fdu.getFullId()
        #Check to see if it exists
        if (os.access(linfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(linfile)
        if (not os.access(linfile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(linfile)
    #end writeOutput
