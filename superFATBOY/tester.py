from .fatboyImage import *
import glob

t = time.time()

objects = glob.glob('/ssd/warner/FATBOY/oriImages/oribcf5k1*.fits')
objects.sort()
skies = glob.glob('/ssd/warner/FATBOY/oriImages/onsourceSkies/sky_oribcf5k1*.fits')
skies.sort()

params = dict()
params['MIN_FRAME_VALUE'] = 0
params['MAX_FRAME_VALUE'] = 50000
params['DIVIDE_BY_COADD'] = 'no'
params['COSMIC_RAY_PASSES'] = 3

stepsToDo = dict()
stepsToDo['rejection'] = True
stepsToDo['linearity'] = True
stepsToDo['darkSubtract'] = True
stepsToDo['flatDivide'] = True
stepsToDo['badPixelMask'] = True
stepsToDo['skySubtract'] = True
stepsToDo['removeCosmicRays'] = True

linCoeffs = [1.00425, -1.01413e-6, 4.18096e-11]

darkFdu = fatboyImage('/ssd/warner/FATBOY/oriImages/masterDarks/mdark-35s-1rd-dark35s.fits')
flatFdu = fatboyImage('/ssd/warner/FATBOY/oriImages/masterFlats/mflat-dome_on-off-K-10s-flatkon.fits')
temp = pyfits.open('/ssd/warner/FATBOY/oriImages/badPixelMasks/BPM_mflat-dome_on-off-K-10s-flatkon.fits')
bpm = temp[0].data
temp.close()

log = fatboyLog(filename=None)
t2 = time.time()

print(t2-t)
for j in range(len(objects)):
    t = time.time()
    image = fatboyImage(objects[j])
    sky = fatboyImage(skies[j])
    t2 = time.time()
    out = doItAll(image, params, log, linCoeffs, darkFdu, flatFdu, bpm, sky, stepsToDo)
    t3 = time.time()
    image.updateData(out)
    image.writeTo('/ssd/warner/FATBOY/oriImages/superFatboy/out_'+image._identFull)
    t4 = time.time()
    print(t2-t, t3-t2, t4-t3, t4-t)
