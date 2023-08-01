from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY.datatypeExtensions.fatboySpecCalib import fatboySpecCalib
from numpy import *
import os, time

class megaraIdentifyFibersProcess(fatboyProcess):
    _modeTags = ["spectroscopy"]

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("Megara Identify Fibers")
        print(fdu._identFull)

        #Call get calibs to return dict() of calibration frames.
        #For megaraIdentifyFibers, this should get a slitmask if MOS data
        calibs = self.getCalibs(fdu, prevProc)

        #Check if output exists first
        #miffile = "megaraIdentifyFibers/mif_"+fdu.getFullId()
        #if (self.checkOutputExists(fdu, miffile, tag="fibers")):
        #  return True

        self.megaraIdentifyFibers(fdu, calibs)
        fdu._header.add_history('Identified Fibers')
        return True
    #end execute

    ## OVERRIDE getCalibs
    def getCalibs(self, fdu, prevProc = None):
        calibs = dict()

        #Look for each master calib passed from XML
        smfilename = self.getCalib("slitmask", fdu.getTag())
        if (smfilename is not None):
            #passed from XML with <calib> tag.  Use fdu as source header
            if (os.access(smfilename, os.F_OK)):
                print("megaraIdentifyFibersProcess::getCalibs> Using slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Using slitmask "+smfilename+"...")
                calibs['slitmask'] = fatboySpecCalib(self._pname, "slitmask", fdu, filename=smfilename, log=self._log)
            else:
                print("megaraIdentifyFibersProcess::getCalibs> Warning: Could not find slitmask "+smfilename+"...")
                self._log.writeLog(__name__, "Could not find slitmask "+smfilename+"...", type=fatboyLog.WARNING)

        #Find master clean sky and master arclamp associated with this object
        headerVals = dict()
        headerVals['grism_keyword'] = fdu.grism
        properties = dict()
        properties['specmode'] = fdu.getProperty("specmode")
        properties['dispersion'] = fdu.getProperty("dispersion")

        if (fdu._specmode != fdu.FDU_TYPE_LONGSLIT):
            if (not 'slitmask' in calibs):
                #Find slitmask associated with this fdu
                #Use new fdu.getSlitmask method
                slitmask = fdu.getSlitmask(pname=None, properties=properties, headerVals=headerVals)
                if (slitmask is None):
                    print("megaraIdentifyFibersProcess::getCalibs> ERROR: Could not find slitmask associated with "+fdu.getFullId()+"!  Unable to collapse fibers!")
                    self._log.writeLog(__name__, "Could not find slitmask associated with "+fdu.getFullId()+"!  Unable to collapse fibers!", type=fatboyLog.ERROR)
                    return calibs
                calibs['slitmask'] = slitmask

        return calibs
    #end getCalibs

    def megaraIdentifyFibers(self, fdu, calibs):
        missing_list = []
        if (self.getOption('missing_fiber_list', fdu.getTag()) is not None):
            missing_list = self.getOption('missing_fiber_list', fdu.getTag()).split(',')
            for j in range(len(missing_list)):
                missing_list[j] = int(missing_list[j])

        writeCalibs = False
        if (self.getOption("write_calib_output", fdu.getTag()).lower() == "yes"):
            writeCalibs = True
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/megaraIdentifyFibers", os.F_OK)):
            os.mkdir(outdir+"/megaraIdentifyFibers",0o755)

        #MOS/IFU data, loop over slitlets
        if (not 'slitmask' in calibs):
            print("megaraIdentifyFibersProcess::megaraIdentifyFibers> ERROR: Could not find slitmask associated with "+fdu.getFullId()+"!  Unable to identify fibers!")
            self._log.writeLog(__name__, "Could not find slitmask associated with "+fdu.getFullId()+"!  Unable to identify fibers!", type=fatboyLog.ERROR)
            return
        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            xsize = fdu.getShape()[1]
            ysize = fdu.getShape()[0]
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            ##xsize should be size across dispersion direction
            xsize = fdu.getShape()[0]
            ysize = fdu.getShape()[1]
        if (not calibs['slitmask'].hasProperty("nslits")):
            calibs['slitmask'].setProperty("nslits", calibs['slitmask'].getData().max())
        nslits = calibs['slitmask'].getProperty("nslits")
        #Use helper method to all ylo, yhi for each slit in each frame
        (ylos, yhis, slitx, slitw) = findRegions(calibs['slitmask'].getData(), nslits, calibs['slitmask'], gpu=self._fdb.getGPUMode(), log=self._log)

        fid = 0
        #Loop over slits found in slitmask
        #Track fiber id separately
        for slitidx in range(1,nslits+1):
            #Increment fiber id
            fid += 1
            #Skip fiber ids in missing list
            while (fid in missing_list):
                fid += 1
            success = fdu.addFiberFromHeader(fid, slitidx)
            if (not success):
                print("megaraIdentifyFibersProcess::megaraIdentifyFibers> Warning: Could not find header info for fiber "+str(fid)+" in "+fdu.getFullId()+".")
                self._log.writeLog(__name__, "Could not find header info for fiber "+str(fid)+" in "+fdu.getFullId()+".", type=fatboyLog.WARNING)
                continue
            if ((ylos[slitidx-1]+yhis[slitidx-1])/2 > ysize//2):
                #Top half
                fdu.getFiber(slitidx).setSection(1)
            #print fdu.getFiber(slitidx).toString()

        fdu.tagDataAs("fibers", fdu.getFibersAsArray())
        nfibers = fdu.getNFibers()
        nsky = len(fdu.getSkyFiberIndices())
        print("megaraIdentifyFibersProcess::megaraIdentifyFibers> Identified "+str(nfibers)+" fibers including "+str(nsky)+" sky fibers.")
        self._log.writeLog(__name__, "Identified "+str(nfibers)+" fibers including "+str(nsky)+" sky fibers.")
    #end megaraIdentifyFibers

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('missing_fiber_list', None)
        self._optioninfo.setdefault('missing_fiber_list', 'Comma separated list of missing fiber numbers\n(ids start with 1)')
    #end setDefaultOptions

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/megaraIdentifyFibers", os.F_OK)):
            os.mkdir(outdir+"/megaraIdentifyFibers",0o755)
        #Create output filename
        miffile = outdir+"/megaraIdentifyFibers/mif_"+fdu.getFullId()
        #Check to see if it exists
        if (os.access(miffile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(miffile)
        if (not os.access(miffile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(miffile, tag="fibers")
    #end writeOutput
