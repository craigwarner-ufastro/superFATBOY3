## @package superFATBOY
#  Documentation for pipeline.
#
from datetime import *
import os, sys

## Documentation for fatboyLog
#
#
class fatboyLog:
    INFO = 0
    WARNING = 1
    ERROR = 2
    #Verbosity flags
    BRIEF = 0
    NORMAL = 1
    VERBOSE = 2
    SIM = -1
    _name = "FATBOYLog"
    _filename = None
    _verbosity = NORMAL
    _date = None
    _xmlName = None
    _dir = None
    _dashes = "\n\t"+"-"*60+"\n\n"
    _error = "\t"+"*"*20+" ERROR "+"*"*20+"\n"
    _warning = "\t*Warning*\n"
    _shortlog = False

    ## The constructor.
    def __init__(self, filename=None, logdir=None, xmlName="No XML file given!", verbosity=NORMAL, shortLog=False):
        if (verbosity == self.SIM):
            return
        self._date = datetime.now()
        self._dir = os.getcwd()
        self._xmlName = xmlName
        self._verbosity = verbosity
        self._shortlog = shortLog
        if (logdir is None):
            logdir = "flogs"
        if (not os.access(logdir, os.F_OK)):
            os.makedirs(logdir)
        if (filename is None):
            if (shortLog):
                if (xmlName.endswith('.xml')):
                    self._filename = logdir+"/FATBOYShortLog."+xmlName[xmlName.rfind('/')+1:xmlName.rfind('.xml')]+self._date.strftime("%Y-%m-%d.%H:%M:%S")+".log"
                else:
                    self._filename = logdir+"/FATBOYShortLog."+self._date.strftime("%Y-%m-%d.%H:%M:%S")+".log"
            else:
                if (xmlName.endswith('.xml')):
                    self._filename = logdir+"/FATBOYLog."+xmlName[xmlName.rfind('/')+1:xmlName.rfind('.xml')]+self._date.strftime("%Y-%m-%d.%H:%M:%S")+".log"
                    #Write out XML file to flogs directory
                    f = open(xmlName, 'r')
                    x = f.read()
                    f.close()
                    f = open(self._filename+'.xml','w')
                    f.write(x)
                    f.close()
                else:
                    self._filename = logdir+"/FATBOYLog."+self._date.strftime("%Y-%m-%d.%H:%M:%S")+".log"
        else:
            self._filename = filename
        f = open(self._filename, 'a')
        f.write("superFATBOY Log of "+self._xmlName+"\n")
        f.write("\trun at "+self._date.strftime("%Y-%m-%d.%H:%M:%S")+"\n")
        f.write("\tin directory "+self._dir+"\n")
        f.write(self._dashes)
        f.close()
    #end __init__

    def getXMLName(self):
        return self._xmlName
    #end getXMLName

    def writeLog(self, name, message, type=INFO, tabLevel=0, printCaller=True, prependSeperator=False, appendSeperator=False, verbosity=None, callerLevel=1):
        if (verbosity is None):
            if (type == self.WARNING or type == self.ERROR):
                verbosity = self.BRIEF
            else:
                verbosity = self.NORMAL
        if (verbosity > self._verbosity):
            #This message is below the set verbosity level.  Do nothing.
            return
        caller = name+"::"+sys._getframe(callerLevel).f_code.co_name+"> "
        f = open(self._filename, 'a')
        if (prependSeperator):
            f.write(self._dashes)
        if (type == self.ERROR):
            f.write(self._error)
            caller += " ERROR: "
        elif (type == self.WARNING):
            f.write(self._warning)
            caller += " Warning: "
        if (tabLevel > 0):
            f.write("\t"*tabLevel)
        if (printCaller):
            f.write(caller)
        f.write(message)
        if (appendSeperator):
            f.write(self._dashes)
        f.write("\n")
        f.close()
    #end writeLog

    def addSeparator(self):
        f = open(self._filename, 'a')
        f.write(self._dashes)
        f.close()
    #end addSeparator
