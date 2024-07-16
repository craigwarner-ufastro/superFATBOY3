from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY.fatboyLibs import *
from numpy import *
import numpy as np
import os, time

class sinfoniCalcLinearityProcess(fatboyProcess):
    _modeTags = ["sinfoni"]

    ###  This will calculate the linearity coeffs from linearity,lamp frames
    ###  It will not actually execute the linearity correction - that will be
    ###  done in linearityProcess which must be run after this.
    ###  The coeffs will be stored in a property 'linearity_coeffs'.

    #Convenience method so that code doesn't have to be rewritten several times in getCalib
    def calcLinearityCoeffs(self, fdu, lin_frames):
        lampmethod = self.getOption("lamp_method", fdu.getTag())
        fit_order = int(self.getOption("linearity_fit_order", fdu.getTag()))

        exptimes = [] #exposure times
        counts = [] #median frame counts

        if (lampmethod == "lamp_on-off"):
            #dome_on-off method. Find off lamps
            offLamps = []
            onLamps = []
            for lamp in lin_frames:
                if (not lamp.hasProperty("lamp_type")):
                    #Look at XML options to find lamp type and assign it to FDUs
                    self.findLampType(fdu, lin_frames)
                if (lamp.getProperty("lamp_type") == "lamp_off"):
                    #This is an OFF lamp
                    offLamps.append(lamp)
                else:
                    #This is an ON lamp
                    onLamps.append(lamp)
            if (len(onLamps) == 0):
                #All off lamps, no ON lamps!  Error!
                print("sinfoniCalcLinearityProcess::sinfoniCalcLinearity> ERROR: No ON lamps found for "+fdu.getFullId())
                self._log.writeLog(__name__, "No ON lamps found for "+fdu.getFullId(), type=fatboyLog.ERROR)
                return None
            if (len(offLamps) == 0):
                #All on lamps, no OFF lamps!  Error!
                print("sinfoniCalcLinearityProcess::sinfoniCalcLinearity> ERROR: No OFF lamps found for "+fdu.getFullId())
                self._log.writeLog(__name__, "No OFF lamps found for "+fdu.getFullId(), type=fatboyLog.ERROR)
                return None
            for j in range(len(onLamps)):
                #Loop over ON lamps, find matching off lamp with same exptime
                match = False
                for i in range(len(offLamps)):
                    if (offLamps[i].exptime == onLamps[j].exptime):
                        #Found match
                        match = True
                        exptimes.append(float(onLamps[j].exptime))
                        counts.append(onLamps[j].getMedian()-offLamps[i].getMedian()) #Append difference med(ON)-med(OFF)
                        offLamps.pop(i) #pop this off lamp out of list
                        break
                if (not match):
                    print("sinfoniCalcLinearityProcess::calcLinearityCoeffs> WARNING: Could not find OFF lamp match for ON lamp "+onLamps[j].getFullId()+".  This lamp will be skipped.")
        else:
            #ON only - just find median value of each frame
            for j in range(len(lin_frames)):
                exptimes.append(float(lin_frames[j].exptime))
                counts.append(lin_frames[j].getMedian())

        exptimes = array(exptimes, dtype=float32)
        counts = array(counts, dtype=float32)
        slp = (counts/exptimes).mean() #Find mean slope
        yout = slp*exptimes #mean slope * exptime is expected counts
        # "A parabolic fit of the product of DIT (i) × mean, as a function of med_dit(i), is performed."
        # => fit_order = 2 is default
        coeffs = np.polyfit(counts, yout, fit_order)[::-1] #polyfit reverses coeffs - we want 0th order first
        coeffs = coeffs[1:] #throw away constant term
        print("sinfoniCalcLinearityProcess::calcLinearityCoeffs> Fit coefficients "+formatList(coeffs))
        return coeffs

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("Sinfoni Linearity")
        print(fdu._identFull)

        #Check if coeffs exist
        if (fdu.hasProperty("linearity_coeffs")):
            return True

        #Call get calibs to return dict() of calibration frames.
        #For sinfoniLinearity, this dict should have one entry 'linearityCoeffs' which is an FDU with a 3D array N x M x order
        calibs = self.getCalibs(fdu, prevProc)
        if (not 'linearity_coeffs' in calibs or not 'fduList' in calibs):
            #Failed to obtain coeffs
            #Issue warning not error
            print("sinfoniCalcLinearityProcess::execute> Warning: Could not find linearity coefficients for "+fdu.getFullId())
            self._log.writeLog(__name__, "Could not find linearity coefficients for "+fdu.getFullId(), type=fatboyLog.WARNING)
            #Return false but do not disable FDU
            return False

        coeffs = calibs['linearity_coeffs']
        for currFDU in calibs['fduList']:
            if (not currFDU.hasProperty('linearity_coeffs')):
                currFDU.setProperty('linearity_coeffs', coeffs)
        return True
    #end execute

    def findLampType(self, fdu, lamps):
        #properties has lamp_method = lamp_on-off.  Only need to process lamps matching this lamp_method.
        lampoff = self.getOption("lamp_off_files", fdu.getTag())
        if (os.access(lampoff, os.F_OK)):
            #This is an ASCII file listing off lamps
            #Process entire file here
            lampoffList = readFileIntoList(lampoff)
            #loop over lampoffList do a split on each line
            for j in range(len(lampoffList)-1, -1, -1):
                lampoffList[j] = lampoffList[j].split()
                #remove misformatted lines
                if (len(lampoffList[j]) < 3):
                    print("sinfoniCalcLinearityProcess::findLampType> Warning: line "+str(j)+" misformatted in "+lampoff)
                    self._log.writeLog(__name__, " line "+str(j)+" misformatted in "+lampoff, type=fatboyLog.WARNING)
                    lampoffList.pop(j)
                    continue
                lampoffList[j][0] = lampoffList[j][0].lower()
                try:
                    lampoffList[j][1] = int(lampoffList[j][1])
                    lampoffList[j][2] = int(lampoffList[j][2])
                except Exception:
                    print("sinfoniCalcLinearityProcess::findLampType> Warning: line "+str(j)+" misformatted in "+lampoff)
                    self._log.writeLog(__name__, " line "+str(j)+" misformatted in "+lampoff, type=fatboyLog.WARNING)
                    lampoffList.pop(j)
                    continue
            #loop over dataset and assign property to all lamp_on-off lamps that don't already have 'lamp_type' property.
            for lamp in lamps:
                if (lamp.hasProperty("lamp_type")):
                    #this FDU already has lamp_type set
                    continue
                lamp.setProperty("lamp_type", "lamp_on") #set to on by default then loop over lampoffList for matches
                #offLine = [ 'identifier', startIdx, stopIdx ]
                for offLine in lampoffList:
                    if (lamp._id.lower().find(offLine[0]) != -1 and int(lamp._index) >= offLine[1] and int(lamp._index) <= offLine[2]):
                        #Partial match for identifier and index within range given
                        lamp.setProperty("lamp_type", "lamp_off")
        elif (fdu.hasHeaderValue(lampoff)):
            #This is a FITS keyword
            lampoffVal = self.getOption("lamp_off_header_value", fdu.getTag())
            if (len(lamps) > 0):
                for lamp in lamps:
                    if (lamp.hasProperty("lamp_type")):
                        #this FDU already has lamp_type set
                        continue
                    print(lamp.getFullId(), lampoff, lampoffVal, str(lamp.getHeaderValue(lampoff)), "========")
                    if (str(lamp.getHeaderValue(lampoff)) == lampoffVal):
                        lamp.setProperty("lamp_type", "lamp_off")
                    else:
                        lamp.setProperty("lamp_type", "lamp_on") #set to on if it does not match
            else:
                #loop over dataset and assign property to all dome_on-off lamps that don't already have 'lamp_type' property.
                for lamp in lamps:
                    if (lamp.hasProperty("lamp_type")):
                        #this FDU already has lamp_type set
                        continue
                    if (str(lamp.getHeaderValue(lampoff)) == lampoffVal):
                        lamp.setProperty("lamp_type", "lamp_off")
                    else:
                        lamp.setProperty("lamp_type", "lamp_on") #set to on if it does not match
        else:
            #This is a filename fragment.  Find which lamps match
            #loop over dataset and assign property to all lamp_on-off lamps that don't already have 'lamp_type' property.
            for lamp in lamps:
                if (not lamp.hasProperty("lamp_type")):
                    if (lamp._id.lower().find(lampoff.lower()) != -1):
                        #partial match for identifier
                        lamp.setProperty("lamp_type", "lamp_off")
                    else:
                        #no match -- this is a lamp on
                        lamp.setProperty("lamp_type", "lamp_on")
    #end findLampType

    ## OVERRIDE getCalibs
    def getCalibs(self, fdu, prevProc = None):
        calibs = dict()

        #1) Check for individual linearity frames TAGGED for this object
        lin_frames = self._fdb.getTaggedCalibs(fdu._id, obstype="linearity,lamp")
        if (len(lin_frames) > 0):
            #Found lin_frames associated with this fdu.  Calculate linearity coeffs
            print("sinfoniCalcLinearityProcess::getCalibs> Calculating linearity coeffs for tagged object "+fdu._id+"...")
            self._log.writeLog(__name__, "Calculating linearity coeffs for tagged object "+fdu._id+"...")
            #First recursively process
            self.recursivelyExecute(lin_frames, prevProc)
            #convenience method
            coeffs = self.calcLinearityCoeffs(fdu, lin_frames)
            if (coeffs is not None):
                calibs['linearity_coeffs'] = coeffs
                #Disable the lin_frames before getting FDUs to tag
                for linFDU in lin_frames:
                    linFDU.disable()
                #Tag these calibs to frame with this ID
                calibs['linearity_coeffs'] = self._fdb.getFDUs(ident=fdu._id)
            return calibs
        #2) Check for individual linearity frames
        lin_frames = self._fdb.getCalibs(obstype="linearity,lamp", tag=fdu.getTag())
        if (len(lin_frames) > 0):
            #Found lin_frames associated with this fdu.  Calculate linearity coeffs
            print("sinfoniCalcLinearityProcess::getCalibs> Calculating linearity coeffs...")
            self._log.writeLog(__name__, "Calculating linearity coeffs...")
            #First recursively process
            self.recursivelyExecute(lin_frames, prevProc)
            #convenience method
            coeffs = self.calcLinearityCoeffs(fdu, lin_frames)
            if (coeffs is not None):
                calibs['linearity_coeffs'] = coeffs
                #Disable the lin_frames before getting FDUs to tag
                for linFDU in lin_frames:
                    linFDU.disable()
                #Get all FDUs and use this linearity correction for them
                calibs['fduList'] = self._fdb.getFDUs()
            return calibs
        print("sinfoniCalcLinearityProcess::getCalibs> Linearity frames not found!")
        self._log.writeLog(__name__, "Linearity Frame not found!", type=fatboyLog.WARNING)
        return calibs
    #end getCalibs

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('lamp_method', "lamp_on-off")
        self._optioninfo.setdefault('lamp_method', 'lamp_on | lamp_on-off')
        self._options.setdefault('lamp_off_files', 'ESO INS1 LAMP5 ST')
        self._optioninfo.setdefault('lamp_off_files', 'An ASCII text file listing on and off lamps\nor a filename fragment or a FITS header keyword\nfor identifying off lamps')
        self._options.setdefault('lamp_off_header_value', 'False')
        self._optioninfo.setdefault('lamp_off_header_value', 'If lamp_off_files is a FITS keyword, value for off lamps')
        self._options.setdefault('linearity_fit_order', '2')
        self._optioninfo.setdefault('linearity_fit_order', 'Order of the polynomial fit, default=2')
    #end setDefaultOptions

    ## No output to write
