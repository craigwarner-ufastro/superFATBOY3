from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY import drihizzle
import os, time

class miradasRegisterWCSProcess(fatboyProcess):
    """ Register WCS for the various slitlets based on the pointing of the probe arms.
          """
    _modeTags = ["miradas"]

    #Override checkValidDatatype
    def checkValidDatatype(self, fdu):
        if (fdu.getObsType(True) == fatboyDataUnit.FDU_TYPE_OBJECT or fdu.getObsType(True) == fdu.FDU_TYPE_STANDARD):
            #If sky subtract is done before flat divide, it will attempt to
            #recursively process flats.  Make sure it only tries to sky subtract objects
            return True
        if (fdu.getObsType(True) == fdu.FDU_TYPE_CONTINUUM_SOURCE):
            #Also sky subtract for continuum source calibs
            return True
        return False
    #end checkValidDatatype

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("MIRADAS: register WCS")
        print(fdu._identFull)

        #Check if output exists first and update from disk
        rwrwcsfile = "registeredWCS/rwrwcs_"+fdu.getFullId()
        if (self.checkOutputExists(fdu, rwrwcsfile)):
            return True

        #There are no calibs to get.
        nslits = fdu.getProperty("nslits")
        mode = fdu.getMiradasObsMode()
        if (mode == fdu.MIRADAS_MODE_MOS):
            probe = (int)(self.getOption('mos_slitlet_1_probe_arm', fdu.getTag()))
            for j in range(1, nslits+1):
                key = str(j)
                if (j < 10):
                    key = '0'+key
                probekey = (probe+j)%12
                if (probekey == 0):
                    probekey = 12
                probekey = str(probekey)
                if (len(probekey) == 1):
                    probekey = '0'+probekey
                ra = fdu.getHeaderValue('MXS'+probekey+'RA')
                dec = fdu.getHeaderValue('MXS'+probekey+'DEC')
                updateHeaderEntry(fdu._header, 'RA_S'+key, ra)
                updateHeaderEntry(fdu._header, 'DEC_S'+key, dec)
        else:
            #SOL/SOS
            probe = str(self.getOption('single_object_probe_arm', fdu.getTag()))
            if (len(probe) == 1):
                probe = '0'+probe
            ra = fdu.getHeaderValue('MXS'+probe+'RA')
            dec = fdu.getHeaderValue('MXS'+probe+'DEC')
            for j in range(1, nslits+1):
                key = str(j)
                if (j < 10):
                    key = '0'+key
                updateHeaderEntry(fdu._header, 'RA_S'+key, ra)
                updateHeaderEntry(fdu._header, 'DEC_S'+key, dec)

        return True
    #end execute

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('mos_slitlet_1_probe_arm', '1')
        self._optioninfo.setdefault('mos_slitlet_1_probe_arm', 'The probe arm for slitlet 1 in MOS mode')
        self._options.setdefault('single_object_probe_arm', '5')
        self._optioninfo.setdefault('single_object_probe_arm', 'The probe arm used for SOL and SOS modes')
    #end setDefaultOptions

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/registeredWCS", os.F_OK)):
            os.mkdir(outdir+"/registeredWCS",0o755)
        rwcsfile = outdir+"/registeredWCS/rwcs_"+fdu.getFullId()
        #Check to see if it exists
        if (os.access(rwcsfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(rwcsfile)
        if (not os.access(rwcsfile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(rwcsfile)
    #end writeOutput
