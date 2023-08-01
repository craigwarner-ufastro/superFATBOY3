#!/usr/bin/python -u
from superFATBOY.fatboyLibs import *
from numpy import *
import sys, glob

if (len(sys.argv) == 1 or sys.argv[1] == '-h' or sys.argv[1] == '-help'):
    print("Usage: fitsUpdate.py [-h] file/filelist keyword value")
    sys.exit()

file = sys.argv[1]
mef = 0
if (file[-2] == ':'):
    mef = (int)(file[-1])
    file = file[:-2]
if (file.lower().find('.fit') == -1):
    f = open(file,'r')
    filelist = f.read().split('\n')
    f.close()
    filelist.remove('')
elif (file.find('*') != -1):
    filelist = glob.glob(file)
    filelist.sort()
else:
    filelist = [file]

key = sys.argv[2]
val = sys.argv[3]
if (isInt(val)):
    val = int(val)
elif (isFloat(val)):
    val = float(val)

for j in filelist:
    temp = pyfits.open(j,'update')
    updateHeaderEntry(temp[mef].header, key, val)
    temp.verify('silentfix')
    temp.flush()
    temp.close()
