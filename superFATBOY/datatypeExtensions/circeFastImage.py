## @package superFATBOY.datatypeExtensions
from superFATBOY.fatboyImage import *
from superFATBOY.datatypeExtensions.circeImage import *

class circeFastImage(circeImage):
    _name = "circeFastImage"
    _expmode = circeImage.EXPMODE_URG

    ## This method can be overridden in subclasses to support images with multiple data extensions, represented as multiple fatboyDataUnits.
    def hasMultipleExtensions(self):
        t = time.time()
        self.readHeader()
        #Move EXPMODE check to here, need to know mode before determining extensions
        #For circeFastImage, force URG mode unless bypass property is set
        self._expmode = self.EXPMODE_URG
        if (self._expmode == self.EXPMODE_URG and self.hasProperty("expmode")):
            if (self.getProperty("expmode").lower() == "bypass_intermediate_reads"):
                self._expmode = self.EXPMODE_URG_BYPASS
        nramps = self.getNRamps()
        if (nramps == 0):
            print("circeFastImage::hasMultipleExtensions> Warning: Unable to find keyword NRAMPS in "+self.filename+"!  Skipping this frame!")
            self._log.writeLog(__name__, "Unable to find keyword NRAMPS in "+self.filename+"!", type=fatboyLog.WARNING)
            self.disable()
            return False
        if (self._expmode == self.EXPMODE_URG):
            ngroups = self.getNGroups()
            if (ngroups == 0):
                print("circeFastImage::hasMultipleExtensions> Warning: Unable to find keyword NGROUPS in "+self.filename+"!  Skipping this frame!")
                self._log.writeLog(__name__, "Unable to find keyword NGROUPS in "+self.filename+"!", type=fatboyLog.WARNING)
                self.disable()
                return False
            if (nramps*(ngroups-1) > 1):
                return True
        elif (self._expmode == self.EXPMODE_URG_BYPASS):
            if (nramps > 1):
                return True
        return False
    #end hasMultipleExtensions
