from superFATBOY.fatboyProcess import fatboyProcess
from superFATBOY.fatboyLibs import *
from superFATBOY.fatboyLog import fatboyLog
from superFATBOY import drihizzle
import os, time

class miradasDARFromConditionsProcess(fatboyProcess):
    """ Estimate differential atmospheric refraction correction as a function
          of wavelength from the site conditions according to Filippenko, 1982
          1) x = 64.328+29498.1/(146-(1/lmbda)**2)+255.4/(41-(1/lmbda)**2)
              where x = (n(lmbda)_15,760 - 1)*10^6
          2) y = (x/1.e+6)*(P*(1+(1.049-0.0157*T)*1.e-6*P))/(720.883*(1+0.003661*T))
              where P in mm HG, T in C, and y = n(lmbda)_T,P - 1
          3) z = y - (0.0624-0.000680/lmbda**2)/(1+0.003661*T)*f*1.e-6
              where z = n(lmbda)_T,P - 1, corrected for water vapor pressure,
              f, in mm HG.
          4) delta_R(lmbda) = R_lmbda - R_5000 = 206265*(n_lmbda - n_5000)*tan z
              where n_5000 is the refractive index at 5000 angstroms and
              z is the zenith angle of the object

          Quantities needed:
          wavelengths - so this should be done after wavelength calibration
          pressure
          temperature
          water vapor pressure
          zenith angle of object
          0.16 arcsec/pixel
          """
    _modeTags = ["miradas"]

    def calculateDAR(self, fdu):
        #Read options
        slitlet_number = self.getOption("slitlet_number", fdu.getTag())
        doAllSlitlets = False
        if (slitlet_number == 'all'):
            doAllSlitlets = True
        elif (isInt(slitlet_number)):
            slitlet_number = int(slitlet_number)

        if (doAllSlitlets):
            slitlets = arange(1, nslits+1)
        else:
            slitlets = [slitlet_number]

        #Get size in dispersion direction
        if (fdu.dispersion == fdu.DISPERSION_HORIZONTAL):
            xsize = fdu.getShape()[1]
        elif (fdu.dispersion == fdu.DISPERSION_VERTICAL):
            xsize = fdu.getShape()[0]

        #Get header info for formulas
        pressure = float(fdu.getHeaderValue("pressure_keyword"))
        temperature = float(fdu.getHeaderValue("temperature_keyword"))
        watervapor = float(fdu.getHeaderValue("water_vapor_keyword"))
        zenith = 90-float(fdu.getHeaderValue("altitude_keyword"))
        pixscale = float(fdu.getHeaderValue("pixscale_keyword"))

        #Calculate n_5000
        lmbda = 0.5
        x_5000 = 64.328+29498.1/(146-(1/lmbda)**2)+255.4/(41-(1/lmbda)**2)
        y_5000 = (x_5000/1.e+6)*(pressure*(1+(1.049-0.0157*temperature)*1.e-6*pressure))/(720.883*(1+0.003661*temperature))
        n_5000 = 1 + (y_5000 - (0.0624-0.000680/lmbda**2)/(1+0.003661*temperature)*watervapor*1.e-6)

        #Loop over slitlets (could be one pass or nslits passes)
        for islit in slitlets:
            #Pass islit-1 to getWavelengthSolution, expects 0-11 not 1-12
            wave = getWavelengthSolution(fdu, islit-1, xsize)
            x_i = 64.328+29498.1/(146-(1/wave)**2)+255.4/(41-(1/wave)**2)
            y_i = (x_i/1.e+6)*(pressure*(1+(1.049-0.0157*temperature)*1.e-6*pressure))/(720.883*(1+0.003661*temperature))
            n_i = 1 + (y_i - (0.0624-0.000680/wave**2)/(1+0.003661*temperature)*watervapor*1.e-6)
            R_i = 206265*(n_i - n_5000)*tan(zenith)
            #R_i in arcsec, we want pixels
            dar_i = R_i/pixscale
            fdu.tagDataAs("dar_slit_"+str(islit), dar_i)
    #end calculateDAR

    #Override checkValidDatatype
    def checkValidDatatype(self, fdu):
        #Should only be done for objects or continuum sources
        if (fdu.getObsType(True) == fatboyDataUnit.FDU_TYPE_OBJECT or fdu.getObsType(True) == fdu.FDU_TYPE_STANDARD):
            return True
        if (fdu.getObsType(True) == fdu.FDU_TYPE_CONTINUUM_SOURCE):
            return True
        return False
    #end checkValidDatatype

    ## OVERRIDE execute
    def execute(self, fdu, prevProc=None):
        print("MIRADAS: compute DAR from site conditions")
        print(fdu._identFull)

        #Check if output exists first and update from disk
        darfile = "DAR/darc_"+fdu.getFullId()
        if (self.checkOutputExists(fdu, darfile)):
            return True

        #There are no calibs to get.  Simply calculate dar.
        calibs = self.getCalibs(fdu, prevProc)

        self.calculateDAR(fdu)

        return True
    #end execute

    ## OVERRRIDE set default options here
    def setDefaultOptions(self):
        self._options.setdefault('slitlet_number', 'all')
        self._optioninfo.setdefault('slitlet_number', 'Set to all (default) to collapse spaxels for all slitlets.\nSet to a number 1-13 to only select one slitlet.')
    #end setDefaultOptions

    ## OVERRRIDE write output here
    def writeOutput(self, fdu):
        #make directory if necessary
        outdir = str(self._fdb.getParam("outputdir", fdu.getTag()))
        if (not os.access(outdir+"/DAR", os.F_OK)):
            os.mkdir(outdir+"/DAR",0o755)
        nslits = fdu.getProperty("nslits")
        for j in range(1, nslits+1):
            if (fdu.hasProperty("dar_slit_"+str(j))):
                #Create output filename
                darfile = outdir+"/DAR/darc_slit_"+str(j)+"_"+fdu.getFullId()
                #Check to see if it exists
                if (os.access(darfile, os.F_OK) and self._fdb.getParam('overwrite_files', fdu.getTag()).lower() == "yes"):
                    os.unlink(darfile)
                if (not os.access(darfile, os.F_OK)):
                    #Use fatboyDataUnit writeTo method to write
                    fdu.writeTo(darfile, tag="dar_slit_"+str(j))
    #end writeOutput
