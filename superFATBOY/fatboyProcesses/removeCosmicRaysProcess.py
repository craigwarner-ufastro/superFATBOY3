from superFATBOY.fatboyLog import fatboyLog
from superFATBOY.fatboyProcess import fatboyProcess
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
        from superFATBOY.fatboyLibs import fatboy_mod, get_fatboy_mod
except Exception:
    print("removeCosmicRayProcess> Warning: PyCUDA not installed")
    hasCuda = False
from numpy import *
import os, time

block_size = 512

class removeCosmicRaysProcess(fatboyProcess):
    _modeTags = ["imaging", "circe"]

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("Cosmic Ray Removal")
        print(fdu._identFull)

        #Check if output exists first
        rcrfile = "removedCosmicRays/rcr_"+fdu.getFullId()
        if (self.checkOutputExists(fdu, rcrfile)):
            return True

        #Select cpu/gpu option
        if (self._fdb.getGPUMode()):
            removeCosmicRays = self.removeCosmicRays_gpu
        else:
            removeCosmicRays = self.removeCosmicRays_cpu
        #Read options
        crpasses = int(self.getOption('cosmic_ray_passes', fdu.getTag()))
        fdu.updateData(removeCosmicRays(fdu, crpasses))
        if (fdu.hasHistory('cosmic_rays_removed')):
            fdu._header.add_history('cosmic_rays_removed: '+str(fdu.getHistory('cosmic_rays_removed')))
        return True
    #end execute

    def removeCosmicRays_cpu(self, fdu, npass):
        data = fdu.getData()
        nx = shape(data)[0]
        ny = shape(data)[1]
        totict = 0
        for k in range(npass):
            print('Pass ',k)
            sq = data**2
            cols = data[0:nx-2,:]+data[1:nx-1,:]+data[2:nx,:]
            cols = cols[:,0:ny-2]+cols[:,1:ny-1]+cols[:,2:ny]
            cols = (cols-data[1:nx-1,1:ny-1])/8.
            sqcols = sq[0:nx-2,:]+sq[1:nx-1,:]+sq[2:nx,:]
            sqcols = sqcols[:,0:ny-2]+sqcols[:,1:ny-1]+sqcols[:,2:ny]
            sqcols = (sqcols-sq[1:nx-1,1:ny-1])/8.
            sd = sqrt(sqcols-cols**2)
            b = where(abs(data[1:nx-1,1:ny-1]-cols) > 5*sd)
            newData = data.copy()
            ict = 0
            xs = b[0]+1
            ys = b[1]+1
            for i in range(len(xs)):
                j = xs[i]
                l = ys[i]
                ict+=1
                temp = array([data[j-1,l-1],data[j,l-1],data[j+1,l-1], data[j-1,l],data[j+1,l],data[j-1,l+1],data[j,l+1],data[j+1,l+1]])
                temp.sort()
                newData[j,l] = (temp[3]+temp[4])/2.0
            data = newData
            print(ict,' replaced.')
            self._log.writeLog(__name__, "Image "+fdu.getFullId()+", Pass "+str(k)+": "+str(ict)+" replaced.")
            totict += ict
        fdu.setHistory('cosmic_rays_removed', totict)
        return newData
    #end removeCosmicRays_cpu

    def removeCosmicRays_gpu(self, fdu, npass):
        t = time.time()
        data = fdu.getData()
        if (not superFATBOY.threaded()):
            global fatboy_mod
        else:
            fatboy_mod = get_fatboy_mod()
        cosmicRayRemoval = fatboy_mod.get_function("cosmicRayRemoval_float")
        output = empty(data.shape, float32)
        rows = data.shape[0]
        cols = data.shape[1]
        ict = zeros(1, int32)
        blocks = data.size//512
        if (data.size % 512 != 0):
            blocks += 1
        totict = 0
        for j in range(npass):
            cosmicRayRemoval(drv.In(data), drv.Out(output), int32(rows), int32(cols), drv.InOut(ict), grid=(blocks,1), block=(block_size,1,1))
            print("\tPass "+str(j)+": "+str(ict)+" replaced.")
            self._log.writeLog(__name__, "Image "+fdu.getFullId()+", Pass "+str(j)+": "+str(ict)+" replaced.")
            data = output
            totict += ict[0]
            ict = zeros(1, int32)
        print("Cosmic Ray Removal time: "+str(time.time()-t))
        fdu.setHistory('cosmic_rays_removed', totict)
        return output
    #end removeCosmicRays_gpu

    removeCosmicRays = removeCosmicRays_gpu

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('cosmic_ray_passes', 3)
    #end setDefaultOptions

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/removedCosmicRays", os.F_OK)):
            os.mkdir(outdir+"/removedCosmicRays",0o755)
        #Create output filename
        rcrfile = outdir+"/removedCosmicRays/rcr_"+fdu.getFullId()
        #Check to see if it exists
        if (os.access(rcrfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(rcrfile)
        if (not os.access(rcrfile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(rcrfile)
    #end writeOutput
