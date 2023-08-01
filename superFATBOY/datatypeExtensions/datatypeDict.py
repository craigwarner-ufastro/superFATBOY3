from . import *

def getDatatypeDict():
    datatypeDict = dict()
    datatypeDict['spectrum'] = fatboySpectrum.fatboySpectrum
    datatypeDict['specCalib'] = fatboySpecCalib.fatboySpecCalib
    datatypeDict['circeImage'] = circeImage.circeImage
    datatypeDict['circeFastImage'] = circeFastImage.circeFastImage
    datatypeDict['miradasSpectrum'] = miradasSpectrum.miradasSpectrum
    datatypeDict['megaraSpectrum'] = megaraSpectrum.megaraSpectrum
    datatypeDict['osirisSpectrum'] = osirisSpectrum.osirisSpectrum
    return datatypeDict
