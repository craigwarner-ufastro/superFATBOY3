from . import *

def getDatatypeDict():
    datatypeDict = dict()
    datatypeDict['spectrum'] = fatboySpectrum.fatboySpectrum
    datatypeDict['specCalib'] = fatboySpecCalib.fatboySpecCalib
    datatypeDict['circeImage'] = circeImage.circeImage
    datatypeDict['fourStarImage'] = fourStarImage.fourStarImage
    datatypeDict['circeFastImage'] = circeFastImage.circeFastImage
    datatypeDict['miradasSpectrum'] = miradasSpectrum.miradasSpectrum
    datatypeDict['megaraSpectrum'] = megaraSpectrum.megaraSpectrum
    datatypeDict['osirisSpectrum'] = osirisSpectrum.osirisSpectrum
    datatypeDict['gmosSpectrum'] = gmosSpectrum.gmosSpectrum
    return datatypeDict
