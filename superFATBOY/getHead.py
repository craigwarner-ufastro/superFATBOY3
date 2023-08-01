#!/usr/bin/python -u
from superFATBOY.fatboyLibs import *
from numpy import *
import os, sys, glob

def space(x):
    return ' '*x

if (len(sys.argv) == 1 or sys.argv[1] == '-h' or sys.argv[1] == '-help'):
    print("Usage: getHead.py [-h] file/filelist/pattern keyword1 keyword2 ...")
    sys.exit()

files = sys.argv[1]
if (files.lower().find('.fit') == -1):
    f = open(files,'r')
    x = f.read().split('\n')
    f.close()
    x.remove('')
else:
    x = glob.glob(files)
    x.sort()

keys = []
for j in range(2, len(sys.argv)):
    keys.append(sys.argv[j])

for j in range(len(x)):
    temp = pyfits.open(x[j])
    s = x[j]
    for l in range(len(keys)):
        if (keys[l] in temp[0].header):
            val = str(temp[0].header[keys[l]])
        else:
            val = "--NA--"
        if (len(val) < 8):
            s+='\t'+val
        else:
            z = len(s.expandtabs())%8
            spc = 15-len(val)-z
            if (spc < 1):
                spc = 1
            s+=space(spc)+val
    print(s)
    temp.close()
