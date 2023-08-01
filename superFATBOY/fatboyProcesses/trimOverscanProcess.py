from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import fatboyLog
from numpy import *
import os, time

class trimOverscanProcess(fatboyProcess):
    _modeTags = ["imaging", "spectroscopy"]

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("Trim Overscan")
        print(fdu._identFull)

        #Check if output exists first
        tofile = "trimmedOverscan/to_"+fdu.getFullId()
        if (self.checkOutputExists(fdu, tofile)):
            return True

        fdu.updateData(self.trimOverscan(fdu))
        fdu._header.add_history('trimmed overscan to new shape '+str(fdu.getData().shape))
        return True
    #end execute

    def trimOverscan(self, fdu):
        fitsKeywordFlag = self.getOption('fits_keyword_flag', fdu.getTag())

        pythonSlicing = False
        if (self.getOption('python_slicing', fdu.getTag()).lower() == "yes"):
            pythonSlicing = True

        data = fdu.getData()
        flagStat = "missing"
        if (fitsKeywordFlag is not None and fdu.hasHeaderValue(fitsKeywordFlag)):
            flagTrueValue = self.getOption('flag_true_value', fdu.getTag())
            if (isInt(flagTrueValue)):
                flagTrueValue = int(flagTrueValue)
            elif (isFloat(flagTrueValue)):
                flagTrueValue = float(flagTrueValue)
            if (fdu.getHeaderValue(fitsKeywordFlag) == flagTrueValue):
                #Trimmed already
                print("trimOverscanProcess::trimOverscan> Trimed flag is already true for "+fdu.getFullId())
                self._log.writeLog(__name__, "Trimed flag is already true for "+fdu.getFullId())
                #return here without modifying data
                return data
            else:
                flagStat = "set to false" #update flag status for print message

        #Not trimmed.  Trim this data to remove overscan regions
        #Create boolean array of indices to keep
        keep = ones(data.shape, bool)

        if (self.getOption('overscan_cols', fdu.getTag()) is not None):
            #Columns of format "54, 320:384, 947, 1024:1080"
            cols = self.getOption('overscan_cols', fdu.getTag())
            try:
                #Parse out into list
                cols = cols.split(",")
                for j in range(len(cols)):
                    cols[j] = cols[j].strip().split(":")
                    #Set keep to False in overscan cols
                    if (len(cols[j]) == 1):
                        keep[:,int(cols[j][0])] = False
                    elif (len(cols[j]) == 2):
                        if (pythonSlicing):
                            keep[:,int(cols[j][0]):int(cols[j][1])] = False
                        else:
                            keep[:,int(cols[j][0]):int(cols[j][1])+1] = False
            except ValueError as ex:
                print("trimOverscanProcess::trimOverscan> Error: invalid format in overscan_cols: "+str(ex))
                self._log.writeLog(__name__, " invalid format in overscan_cols: "+str(ex), type=fatboyLog.ERROR)
                return None
        if (self.getOption('overscan_rows', fdu.getTag()) is not None):
            #Row reject of format "54, 320:384, 947, 1024:1080"
            rows = self.getOption('overscan_rows', fdu.getTag())
            try:
                #Parse out into list
                rows = rows.split(",")
                for j in range(len(rows)):
                    rows[j] = rows[j].strip().split(":")
                    #Set keep to False in overscan rows
                    if (len(rows[j]) == 1):
                        keep[int(rows[j][0]),:] = False
                    elif (len(rows[j]) == 2):
                        if (pythonSlicing):
                            keep[int(rows[j][0]):int(rows[j][1]),:] = False
                        else:
                            keep[int(rows[j][0]):int(rows[j][1])+1,:] = False
            except ValueError as ex:
                print("trimOverscanProcess::trimOverscan> Error: invalid format in overscan_rows: "+str(ex))
                self._log.writeLog(__name__, " invalid format in overscan_rows: "+str(ex), type=fatboyLog.ERROR)
                return None

        #Use keep as index to extract values we will keep
        #Then find new shape and reshape
        oldy = data.shape[0]
        oldx = data.shape[1]
        data = ascontiguousarray(data[keep])
        ysize = keep.sum(0).max()
        xsize = keep.sum(1).max()
        data = data.reshape((ysize, xsize))

        print("trimOverscanProcess::trimOverscan> Trimed flag is "+flagStat+" for "+fdu.getFullId()+" - TRIMMING from original shape = ("+str(oldy)+", "+str(oldx)+") to new shape ("+str(ysize)+", "+str(xsize)+").")
        self._log.writeLog(__name__, "Trimed flag is "+flagStat+" for "+fdu.getFullId()+" - TRIMMING from original shape = ("+str(oldy)+", "+str(oldx)+") to new shape ("+str(ysize)+", "+str(xsize)+").")
        return ascontiguousarray(data)
    #end trimOverscan

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('fits_keyword_flag', 'OTRIMMED')
        self._optioninfo.setdefault('fits_keyword_flag', 'Set this to a FITS keyword which will be added when images are trimmed')
        self._options.setdefault('flag_true_value', '1')
        self._optioninfo.setdefault('flag_true_value', 'Value indicating image is trimmed.  If not equal\nto this, image WILL be trimmed.')
        self._options.setdefault('overscan_cols', None)
        self._optioninfo.setdefault('overscan_cols', 'Columns to be trimmed. Supports slicing, e.g. 320:384, 500, 752:768')
        self._options.setdefault('overscan_rows', None)
        self._optioninfo.setdefault('overscan_rows', 'Rows to be trimmed. Supports slicing, e.g. 320:384, 500, 752:768')
        self._options.setdefault('python_slicing', 'no')
        self._optioninfo.setdefault('python_slicing', 'Set to yes if min:max should be used.\nDefault is min:max+1.')
    #end setDefaultOptions

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/trimmedOverscan", os.F_OK)):
            os.mkdir(outdir+"/trimmedOverscan",0o755)
        #Create output filename
        tofile = outdir+"/trimmedOverscan/to_"+fdu.getFullId()
        #Check to see if it exists
        if (os.access(tofile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
            os.unlink(tofile)
        if (not os.access(tofile, os.F_OK)):
            #Use fatboyDataUnit writeTo method to write
            fdu.writeTo(tofile)
    #end writeOutput
