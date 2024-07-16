from . import *
#import linearityProcess
#import darkSubtractProcess
#import flatDivideProcess
#import badPixelMaskProcess
#import removeCosmicRaysProcess
#import skySubtractProcess
#import remergeCirceProcess
#import alignStackProcess
#import deboneCirceProcess

def getProcessDict():
    processDict = dict()
    #Imaging
    processDict['linearity'] = linearityProcess.linearityProcess
    processDict['darkSubtract'] = darkSubtractProcess.darkSubtractProcess
    processDict['flatDivide'] = flatDivideProcess.flatDivideProcess
    processDict['badPixelMask'] = badPixelMaskProcess.badPixelMaskProcess
    processDict['cosmicRays'] = removeCosmicRaysProcess.removeCosmicRaysProcess
    processDict['skySubtract'] = skySubtractProcess.skySubtractProcess
    processDict['alignStack'] = alignStackProcess.alignStackProcess
    processDict['mergeObjects'] = mergeObjectsProcess.mergeObjectsProcess

    #CIRCE specific
    processDict['remergeCirce'] = remergeCirceProcess.remergeCirceProcess
    processDict['deboneCirce'] = deboneCirceProcess.deboneCirceProcess
    processDict['trimWindow'] = trimWindowProcess.trimWindowProcess

    #Spectroscopy
    processDict['badPixelMaskSpec'] = badPixelMaskSpecProcess.badPixelMaskSpecProcess
    processDict['calibStarDivide'] = calibStarDivideProcess.calibStarDivideProcess
    processDict['cosmicRaysSpec'] = removeCosmicRaysSpecProcess.removeCosmicRaysSpecProcess
    processDict['createCleanSkies'] = createCleanSkyProcess.createCleanSkyProcess
    processDict['createMasterArclamps'] = createMasterArclampProcess.createMasterArclampProcess
    processDict['doubleSubtract'] = doubleSubtractProcess.doubleSubtractProcess
    processDict['extractSpectra'] = extractSpectraProcess.extractSpectraProcess
    processDict['findSlitlets'] = findSlitletProcess.findSlitletProcess
    processDict['flatDivideSpec'] = flatDivideSpecProcess.flatDivideSpecProcess
    processDict['noisemap'] = noisemapProcess.noisemapProcess
    processDict['rectify'] = rectifyProcess.rectifyProcess
    processDict['resample'] = resampleProcess.resampleProcess
    processDict['shiftAdd'] = shiftAddProcess.shiftAddProcess
    processDict['slitletAlign'] = slitletAlignProcess.slitletAlignProcess
    processDict['skySubtractSpec'] = skySubtractSpecProcess.skySubtractSpecProcess
    processDict['wavelengthCalibrate'] = wavelengthCalibrateProcess.wavelengthCalibrateProcess

    #GMOS specific
    processDict['biasSubtract'] = biasSubtractProcess.biasSubtractProcess

    #MIRADAS specific
    processDict['miradasCombineSlices'] = miradasCombineSlicesProcess.miradasCombineSlicesProcess
    processDict['miradasCollapseSpaxels'] = miradasCollapseSpaxelsProcess.miradasCollapseSpaxelsProcess
    processDict['miradasCreate3dDatacubes'] = miradasCreate3dDatacubesProcess.miradasCreate3dDatacubesProcess
    processDict['miradasCharacterizePSF'] = miradasCharacterizePSFProcess.miradasCharacterizePSFProcess
    processDict['miradasDARFromConditions'] = miradasDARFromConditionsProcess.miradasDARFromConditionsProcess
    processDict['miradasDARFromData'] = miradasDARFromDataProcess.miradasDARFromDataProcess
    processDict['miradasRegisterWCS'] = miradasRegisterWCSProcess.miradasRegisterWCSProcess
    processDict['miradasStitchOrders'] = miradasStitchOrdersProcess.miradasStitchOrdersProcess

    #MEGARA specific
    processDict['trimOverscan'] = trimOverscanProcess.trimOverscanProcess
    processDict['collapseFibers'] = collapseFibersProcess.collapseFibersProcess
    processDict['megaraIdentifyFibers'] = megaraIdentifyFibersProcess.megaraIdentifyFibersProcess
    processDict['megaraSkySubtract'] = megaraSkySubtractProcess.megaraSkySubtractProcess

    #SINFONI specific
    processDict['sinfoniCalcLinearity'] = sinfoniCalcLinearityProcess.sinfoniCalcLinearityProcess
    processDict['sinfoniCharacterizePSF'] = sinfoniCharacterizePSFProcess.sinfoniCharacterizePSFProcess
    processDict['sinfoniCollapseSlitlets'] = sinfoniCollapseSlitletsProcess.sinfoniCollapseSlitletsProcess
    processDict['sinfoniCreate3dDatacubes'] = sinfoniCreate3dDatacubesProcess.sinfoniCreate3dDatacubesProcess
    processDict['sinfoniIdentifySlitlets'] = sinfoniIdentifySlitletsProcess.sinfoniIdentifySlitletsProcess
    processDict['sinfoniRegisterStack'] = sinfoniRegisterStackProcess.sinfoniRegisterStackProcess
    processDict['sinfoniRemoveBadLines'] = sinfoniBadLineRemovalProcess.sinfoniBadLineRemovalProcess

    #EMIR specific
    processDict['emirBiasSubtract'] = emirBiasSubtractProcess.emirBiasSubtractProcess

    return processDict
