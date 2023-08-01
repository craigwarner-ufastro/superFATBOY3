from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import fatboyLog
from numpy import *
import os, time

class trimWindowProcess(fatboyProcess):
    _modeTags = ["circe"]

    _xmin = None
    _xmax = None
    _ymin = None
    _ymax = None

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("Trim Window")
        print(fdu._identFull)

        #Check if output exists first
        twfile = "trimmedWindow/tw_"+fdu.getFullId()
        if (self.checkOutputExists(fdu, twfile)):
            fitsKeywordFlag = self.getOption('fits_keyword_flag', fdu.getTag())
            if (fitsKeywordFlag is not None and fdu.hasHeaderValue(fitsKeywordFlag)):
                flagTrueValue = self.getOption('flag_true_value', fdu.getTag())
                if (isInt(flagTrueValue)):
                    flagTrueValue = int(flagTrueValue)
                elif (isFloat(flagTrueValue)):
                    flagTrueValue = float(flagTrueValue)
                if (fdu.getHeaderValue(fitsKeywordFlag) == flagTrueValue):
                    #Trimmed already - use this file to update trim section
                    self.trimWindow(fdu)
            return True

        fdu.updateData(self.trimWindow(fdu))
        fdu._header.add_history('trimmed to '+str(fdu.getData().shape))
        return True
    #end execute

    def trimWindow(self, fdu):
        fitsKeywordFlag = self.getOption('fits_keyword_flag', fdu.getTag())

        pythonSlicing = False
        if (self.getOption('python_slicing', fdu.getTag()).lower() == "yes"):
            pythonSlicing = True
        windowXmin = self.getOption('window_xmin', fdu.getTag())
        windowXmax = self.getOption('window_xmax', fdu.getTag())
        windowYmin = self.getOption('window_ymin', fdu.getTag())
        windowYmax = self.getOption('window_ymax', fdu.getTag())

        data = fdu.getData()
        flagStat = "missing"
        if (fitsKeywordFlag is not None and fdu.hasHeaderValue(fitsKeywordFlag)):
            flagTrueValue = self.getOption('flag_true_value', fdu.getTag())
            if (isInt(flagTrueValue)):
                flagTrueValue = int(flagTrueValue)
            elif (isFloat(flagTrueValue)):
                flagTrueValue = float(flagTrueValue)
            if (fdu.getHeaderValue(fitsKeywordFlag) == flagTrueValue):
                #Trimmed already - look at header for trim section
                if (windowXmin is not None and fdu.hasHeaderValue(windowXmin)):
                    self._xmin = fdu.getHeaderValue(windowXmin)
                if (windowXmax is not None and fdu.hasHeaderValue(windowXmax)):
                    self._xmax = fdu.getHeaderValue(windowXmax)
                    if (not pythonSlicing):
                        self._xmax += 1
                if (windowYmin is not None and fdu.hasHeaderValue(windowYmin)):
                    self._ymin = fdu.getHeaderValue(windowYmin)
                if (windowYmax is not None and fdu.hasHeaderValue(windowYmax)):
                    self._ymax = fdu.getHeaderValue(windowYmax)
                    if (not pythonSlicing):
                        self._ymax += 1
                print("trimWindowProcess::trimWindow> Trimed flag is true for "+fdu.getFullId()+" - saving trim window (X,Y) = ("+str(self._xmin)+", "+str(self._ymin)+") to ("+str(self._xmax)+", "+str(self._ymax)+").")
                self._log.writeLog(__name__, "Trimed flag is true for "+fdu.getFullId()+" - saving trim window (X,Y) = ("+str(self._xmin)+", "+str(self._ymin)+") to ("+str(self._xmax)+", "+str(self._ymax)+").")
                #return here without modifying data
                return data
            else:
                flagStat = "set to false" #update flag status for print message

        #Not trimmed.  Trim this data to predetermined window.
        #Default is full image
        x1 = 0
        x2 = data.shape[1]
        y1 = 0
        y2 = data.shape[0]
        if (self._xmin is not None):
            x1 = self._xmin
        elif (windowXmin is not None and isInt(windowXmin)):
            x1 = int(windowXmin)
        if (self._xmax is not None):
            x2 = self._xmax
        elif (windowXmax is not None and isInt(windowXmax)):
            x2 = int(windowXmax)
        if (self._ymin is not None):
            y1 = self._ymin
        elif (windowYmin is not None and isInt(windowYmin)):
            y1 = int(windowYmin)
        if (self._ymax is not None):
            y2 = self._ymax
        elif (windowYmax is not None and isInt(windowYmax)):
            y2 = int(windowxYmax)
        print("trimWindowProcess::trimWindow> Trimed flag is "+flagStat+" for "+fdu.getFullId()+" - TRIMMING to window (X,Y) = ("+str(x1)+", "+str(y1)+") to ("+str(x2)+", "+str(y2)+").")
        self._log.writeLog(__name__, "Trimed flag is "+flagStat+" for "+fdu.getFullId()+" - TRIMMING to window (X,Y) = ("+str(x1)+", "+str(y1)+") to ("+str(x2)+", "+str(y2)+").")
        data = data[y1:y2, x1:x2]
        return ascontiguousarray(data)
    #end trimWindow

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('fits_keyword_flag', None)
        self._optioninfo.setdefault('fits_keyword_flag', 'Set this to a FITS keyword flagging whether images are trimmed')
        self._options.setdefault('flag_true_value', '1')
        self._optioninfo.setdefault('flag_true_value', 'Value indicating image is trimmed.  If not equal\nto this, image WILL be trimmed.')
        self._options.setdefault('python_slicing', 'no')
        self._optioninfo.setdefault('python_slicing', 'Set to yes if min:max should be used.\nDefault is min:max+1.')
        self._options.setdefault('window_xmin', None)
        self._optioninfo.setdefault('window_xmin', 'Number or FITS keyword for min x value')
        self._options.setdefault('window_xmax', None)
        self._optioninfo.setdefault('window_xmax', 'Number or FITS keyword for max x value')
        self._options.setdefault('window_ymin', None)
        self._optioninfo.setdefault('window_ymin', 'Number or FITS keyword for min y value')
        self._options.setdefault('window_ymax', None)
        self._optioninfo.setdefault('window_ymax', 'Number or FITS keyword for max y value')
    #end setDefaultOptions

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/trimmedWindow", os.F_OK)):
            os.mkdir(outdir+"/trimmedWindow",0o755)
        #Create output filename
        twfile = outdir+"/trimmedWindow/tw_"+fdu.getFullId()
        #Check to see if it exists
        if (os.access(twfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(twfile)
        if (not os.access(twfile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(twfile)
    #end writeOutput
