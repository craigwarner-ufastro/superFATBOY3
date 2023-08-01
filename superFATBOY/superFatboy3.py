#!/usr/bin/python
## @package superFATBOY
#  Documentation for pipeline.
#
#
import sys, os, inspect, glob
sys.path.append('..') #Append parent directory to sys.path
sfbdir = os.path.dirname(os.path.abspath(__file__))
sfbdir = sfbdir[:sfbdir.rfind('/superFATBOY')]
sys.path.append(sfbdir) #Append absoulute path in case script is run from another dir
#Check for argument specifiying CUDA_DEVICE before anything imports pycuda.autoinit
for j in range(len(sys.argv)):
    if (sys.argv[j] == "-gpu" and len(sys.argv) > j+1):
        os.environ['CUDA_DEVICE'] = sys.argv[j+1]

import superFATBOY
from superFATBOY.fatboyDatabase import *
if (len(sys.argv) < 2 or sys.argv.count("-list") != 0):
    #print params, processes, and options, and exit!
    modeTag = None
    if (sys.argv[1] == "-list" and len(sys.argv) > 2):
        modeTag = sys.argv[2]
    fb = fatboyDatabase(modeTag=modeTag)
    sys.exit(0)
elif (sys.argv.count("-config") != 0):
    #list config files and exit
    print("*** List of superFATBOY included config files ***")
    datadir = os.path.dirname(inspect.getfile(superFATBOY))+"/data/"
    dirnames = glob.glob(datadir+"/*")
    dirnames.sort()
    for dirname in dirnames:
        if (not os.path.isdir(dirname)):
            continue
        print(dirname[dirname.rfind('/')+1:])
        configs = glob.glob(dirname+"/*")
        configs.sort()
        for config in configs:
            if (config.endswith(".py") or config.endswith(".pyc")):
                continue
            print("\t"+config[config.rfind('/')+1:])
    sys.exit(0)
config = sys.argv[1]
fb = fatboyDatabase(config)
fb.execute()
