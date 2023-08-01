## @package superFATBOY
#  Documentation for pipeline.
#
#

import glob
import xml.dom.minidom
import time
from datetime import *
from xml.dom.minidom import Node
from .fatboyImage import *

## Documentation for fatboyQuery
#
#
class fatboyQuery:
    _name = "fatboyQuery"
    ##Static vars
    FILELIST_LIST = 0
    FILELIST_GROUPING = 1

    _type = None #type of query
    _datatypeDict = dict() #This is a dict mapping actual datatype classes to names used in XML.
    _delim = None #delimiter for this query
    _dir = None #source directory
    _tag = None #tag
    _query = None
    _xmlNode = None
    _validQuery = False
    _log = None
    _fdus = []

    ## The constructor.
    def __init__(self, query, log=None, datatypeDict=None):
        self._xmlNode = query
        self._log = log
        self._datatypeDict = datatypeDict
        self._images = self.parseQuery(query)
        print("fatboyQuery::__init__> Imagelist contains "+str(len(self._images))+" images.")
        if (log is not None):
            self._log.writeLog(__name__, "Imagelist contains "+str(len(self._images))+" images.")
    #end __init__

    #Execute returns list of fatboyImages
    def executeQuery(self):
        #First check if multiple FDUs should be created, e.g. multiple FITS data extensions
        extendedImages = []
        for image in self._images:
            #fatboyDataUnit base class returns false.  Subclasses can return true and implement getMultipleExtensions to return a list
            if (image.hasMultipleExtensions()):
                #Returns a list of additional FDUs.  Overridden in subclasses.  These should have a different fdu.section value
                extendedImages.extend(image.getMultipleExtensions())
        self._images.extend(extendedImages)
        #Check for any duplicates and remove them here.  Do this by creating a list of _identFull
        idents = []
        for image in self._images:
            idents.append(image.getFullId())
        #Remove duplicates
        ndup = 0
        #Loop backwards
        for j in range(len(idents)-1, -1, -1):
            if (idents.count(idents[j]) > 1):
                ndup += 1
                print("fatboyQuery::executeQuery> Warning: removing duplicate "+idents[j])
                self._log.writeLog(__name__, "removing duplicate "+idents[j], type=fatboyLog.WARNING)
                idents.pop(j)
                self._images.pop(j)
        if (ndup > 0):
            print("fatboyQuery::executeQuery> Warning: "+str(ndup)+" duplicate files removed.")
            self._log.writeLog(__name__, str(ndup)+" duplicate files removed.", type=fatboyLog.WARNING)
        return self._images
    #end executeQuery

    #Convenience method to find identifier and index number given filename and delimiter
    def findIdentAndIndex(self, fname, suffix=False):
        if (suffix):
            #Handle suffix case
            if (self._delim is not None):
                #Everything before delimiter
                sfileindex = fname[:fname.find(self._delim)]
                filesuffix = fname[fname.find(self._delim) + len(self._delim):fname.rfind('.')]
            else:
                dpos = fname.rfind('.')
                if (fname.endswith('.fz')):
                    #Handle .fits.fz files
                    dpos = fname[:-3].rfind('.')
                #Start with position 0 and find leftmost non-numerical character
                spos = 0
                cpos = 0
                if (fname.rfind('/') > -1):
                    cpos = fname.rfind('/')+1
                    spos = fname.rfind('/')+1
                while (isDigit(fname[cpos]) and cpos < dpos):
                    cpos += 1
                sfileindex = fname[spos:cpos]
                #Make sure suffix does not start with . - or _
                filesuffix = fname[cpos:dpos]
                while (filesuffix.startswith('.') or filesuffix.startswith('-') or filesuffix.startswith('_')):
                    filesuffix = filesuffix[1:]
            filesuffix = str(filesuffix) #convert from unicode to str!!
            sfileindex = str(sfileindex) #convert fron unicode to str!!
            print((filesuffix, sfileindex))
            return (filesuffix, sfileindex)
        #Normal prefix case
        if (self._delim is not None):
            #Everything before delimiter
            fileprefix = fname[:fname.rfind(self._delim, 0, fname.rfind('.'))]
            sfileindex = fname[fname.rfind(self._delim, 0, fname.rfind('.')) + len(self._delim):fname.rfind('.')]
        else:
            dpos = fname.rfind('.')
            if (fname.endswith('.fz')):
                #Handle .fits.fz files
                dpos = fname[:-3].rfind('.')
            cpos = dpos-1
            #Find rightmost non-numerical character before .fits
            while(isDigit(fname[cpos]) and cpos > 0):
                cpos-=1
            fileprefix = fname[:cpos+1]
            #Make sure prefix does not end with a '.'
            while(fileprefix.endswith('.') or fileprefix.endswith('-') or fileprefix.endswith('_')):
                fileprefix = fileprefix[:-1]
            sfileindex = fname[cpos+1:dpos]
        fileprefix = str(fileprefix) #convert from unicode to str!!
        sfileindex = str(sfileindex) #convert fron unicode to str!!
        return (fileprefix, sfileindex)
    #end findIdentAndIndex

    ### parse individual query
    def parseQuery(self, query):
        images = []
        #Tag name - filelist, object, etc.
        self._type = query.nodeName
        if (query.nodeName == 'filelist'):
            images.extend(self.parseFilelist(query))
        elif (query.nodeName == 'dataset'):
            images.extend(self.parseDataset(query))
        else:
            print("fatboyQuery::parseQuery> Invalid query node: "+query.nodeName)
            self._log.writeLog(__name__, "Invalid node query: "+query.nodeName, type=fatboyLog.ERROR)
            self._type = None
            return None
        return images
    #end parseQuery

    def parseFilelist(self, query):
        _datatype = fatboyImage
        images = []
        listFilename = None
        listType = self.FILELIST_LIST #LIST or GROUPING
        #Parse attributes first
        if (query.hasAttributes()):
            for j in range(query.attributes.length):
                attr = query.attributes.item(j)
                if (attr.nodeName == "type"):
                    val = attr.nodeValue
                    if (val.lower() == "grouping"):
                        listType = self.FILELIST_GROUPING
                elif (attr.nodeName == "tag"):
                    self._tag = str(attr.nodeValue) #convert from unicode to str!!
                elif (attr.nodeName == "datatype"):
                    if (attr.nodeValue in self._datatypeDict):
                        _datatype = self._datatypeDict[attr.nodeValue]
                elif (attr.nodeName == "delim"):
                    self._delim = attr.nodeValue
                elif (attr.nodeName == "dir"):
                    self._dir = attr.nodeValue
                else:
                    print("Warning: node "+query.nodeName+" has invalid attribute: "+attr.nodeName)
                    self._log.writeLog(__name__, "node "+query.nodeName+" has invalid attribute: "+attr.nodeName, type=fatboyLog.WARNING)

        if (query.hasChildNodes()):
            #filelist should contain a text node child only
            if (query.childNodes[0].nodeType == Node.TEXT_NODE):
                listFilename = query.childNodes[0].nodeValue

        if (listFilename is None):
            print("Error: no filename given for filelist.")
            self._log.writeLog(__name__, "node "+query.nodeName+" does not specify a filename!", type=fatboyLog.ERROR)
            return images
        #prepend directory if applicable
        if (self._dir is not None):
            listFilename = self._dir+"/"+listFilename
        #Check for existance of list
        if (not os.access(listFilename, os.R_OK)):
            print("Error: could not find filelist "+listFilename)
            self._log.writeLog(__name__, "node "+query.nodeName+": could not find filelist "+listFilename, type=fatboyLog.ERROR)
            return images
        #read in file
        print("fatboyQuery::parseFileList> Parsing file list: "+str(listFilename))
        self._log.writeLog(__name__, "Parsing file list: "+str(listFilename))
        filelist = readFileIntoList(listFilename)

        if (listType == self.FILELIST_LIST):
            #Create fatboy images out of filenames
            for fname in filelist:
                if (self._dir is not None and not os.access(fname, os.F_OK) and os.access(self._dir+"/"+fname, os.F_OK)):
                    #if dir is specified check source dir if no match in current dir
                    fname = self._dir+"/"+fname
                if (os.access(fname, os.F_OK)):
                    #create a fatboyImage and add to list
                    #_datatype is fatboyImage unless overridden with datatype attribute
                    currImage = _datatype(fname, log=self._log, tag=self._tag)
                    (fileprefix, sfileindex) = self.findIdentAndIndex(fname)
                    currImage.setIdentifier("filename", fileprefix, sfileindex)
                    images.append(currImage)
        elif (listType == self.FILELIST_GROUPING):
            #Process grouping file
            n = 0
            for line in filelist:
                #split each line into tokens prefix start_index end_index [out_prefix] [out_start_index]
                n+=1
                tokens = line.split()
                try:
                    pfix = tokens[0]
                    startIdx = int(tokens[1])
                    endIdx = int(tokens[2])
                    #loop over start to end index numbers and look for matching files
                    for j in range(startIdx, endIdx+1):
                        sindex = str(j)
                        #pad index with left zeros to number of digits in endIdx
                        while (len(sindex) < len(tokens[2])):
                            sindex = '0'+sindex
                        matches = glob.glob(pfix+'*'+sindex+'.fits')
                        matches.extend(glob.glob(pfix+'*'+sindex+'.fits.fz'))
                        if (len(matches) == 0 and self._dir is not None):
                            #in case dir is specified check source dir if no matches in current dir
                            matches = glob.glob(self._dir+"/"+pfix+'*'+sindex+'.fits')
                            matches.extend(glob.glob(self._dir+"/"+pfix+'*'+sindex+'.fits.fz'))
                        if (len(matches) > 0):
                            #add match to filenames
                            matches.sort()
                            fname = matches[0]
                            #create a fatboyImage and add to list
                            #_datatype is fatboyImage unless overridden with datatype attribute
                            currImage = _datatype(fname, log=self._log, tag=self._tag)
                            (fileprefix, sfileindex) = self.findIdentAndIndex(fname)
                            if (len(tokens) > 4):
                                #output prefix and start index both given
                                sfileindex = str(int(tokens[4])-startIdx+int(sfileindex))
                                zeros = '0000'
                                sfileindx = zeros[len(sfileindex):]+sfileindex
                                ident = tokens[3]
                            elif (len(tokens) > 3):
                                #output prefix given
                                ident = tokens[3]
                            else:
                                ident = fileprefix.replace('/','_')
                            currImage.setIdentifier("manual", ident, sfileindex)
                            images.append(currImage)
                except Exception as ex:
                    print(ex)
                    print("Warning: Syntax Error on line "+str(n)+" of "+listFilename)
                    self._log.writeLog(__name__, "Warning: Syntax Error on line "+str(n)+" of "+listFilename, type=fatboyLog.WARNING)
                    continue
        return images

    def parseDataset(self, query):
        _datatype = fatboyImage
        images = []
        #Parse attributes first
        if (query.hasAttributes()):
            for j in range(query.attributes.length):
                attr = query.attributes.item(j)
                if (attr.nodeName == "dir"):
                    self._dir = attr.nodeValue
                elif (attr.nodeName == "tag"):
                    self._tag = str(attr.nodeValue) #convert from unicode to str!!
                elif (attr.nodeName == "datatype"):
                    if (attr.nodeValue in self._datatypeDict):
                        _datatype = self._datatypeDict[attr.nodeValue]
                elif (attr.nodeName == "delim"):
                    self._delim = attr.nodeValue
                else:
                    print("Warning: node "+query.nodeName+" has invalid attribute: "+attr.nodeName)
                    self._log.writeLog(__name__, "node "+query.nodeName+" has invalid attribute: "+attr.nodeName, type=fatboyLog.WARNING)

        print("fatboyQuery::parseDataset> Parsing dataset: dir="+str(self._dir)+"; tag="+str(self._tag))
        self._log.writeLog(__name__, "Parsing dataset: dir="+str(self._dir)+"; tag="+str(self._tag))
        #Now parse child nodes.  Allowable child nodes are:
        ## object - this is classified an object.  name attribute gives output prefix
        ## calib - this is classified a calib and has a type = auto | dark | flat | etc.  default is auto
        if (query.hasChildNodes()):
            for node in query.childNodes:
                #Ignore whitespace text nodes.  Just look at elements.
                if (node.nodeType != Node.ELEMENT_NODE):
                    continue
                if (node.nodeName == "object"):
                    images.extend(self.parseObject(node, _datatype))
                elif (node.nodeName == "calib"):
                    images.extend(self.parseObject(node, _datatype))
                else:
                    print("Warning: node "+query.nodeName+" has invalid child node: "+node.nodeName)
                    self._log.writeLog(__name__, "node "+query.nodeName+" has invalid child node: "+node.nodeName, type=fatboyLog.WARNING)
                    continue
        return images
    #end parseDataset

    def parseObject(self, node, _datatype):
        images = []
        ident = None
        name = None
        pfix = None
        sfix = None
        pattern = None
        subdir = ""
        calibFilename = None
        subtag = None
        thisType = "auto"
        isCalib = False
        attrCount = 0
        if (node.nodeName == "calib"):
            isCalib = True
            objTags = [] #Create list for object tags
        properties = dict() #create properties dictionary
        matchAllInDir = True #if no <value> or <index> subtags, match prefix*.fits, pattern.fits, or for calibs, filename
        #Now parse object or calib node attributes and subnodes
        #First look at attributes
        if (node.hasAttributes()):
            for j in range(node.attributes.length):
                attr = node.attributes.item(j)
                if (attr.nodeName == "name"):
                    name = attr.nodeValue
                elif (attr.nodeName == "prefix"):
                    pfix = attr.nodeValue
                    attrCount += 1
                elif (attr.nodeName == "pattern"):
                    pattern = attr.nodeValue
                    attrCount += 1
                elif (attr.nodeName == "suffix"):
                    sfix = attr.nodeValue
                    attrCount += 1
                elif (attr.nodeName == "type"):
                    thisType = attr.nodeValue
                elif (attr.nodeName == "filename"):
                    #for calib tags only
                    if (isCalib):
                        calibFilename = attr.nodeValue
                elif (attr.nodeName == "tag"):
                    #a sub-tag is defined for this calib/object
                    subtag = str(attr.nodeValue) #convert from unicode to str!!
                elif (attr.nodeName == "subdir"):
                    #subdir specified - used especially with suffix
                    subdir = str(attr.nodeValue)+"/" #add trailing slash
                else:
                    print("Warning: node "+node.nodeName+" has invalid attribute: "+attr.nodeName)
                    self._log.writeLog(__name__, "node "+node.nodeName+" has invalid attribute: "+attr.nodeName, type=fatboyLog.WARNING)

        if (attrCount == 0 and calibFilename is None):
            print("Error: node "+node.nodeName+" does not specify a prefix, suffix, or pattern attribute!")
            self._log.writeLog(__name__, "node "+node.nodeName+" does not specify a prefix, suffix, or pattern attribute!", type=fatboyLog.ERROR)
            return images
        if (attrCount > 1):
            print("Warning: node "+node.nodeName+" specifies more than one of prefix, suffix, or pattern!  Prefix will be used first, then suffix!")
            self._log.writeLog(__name__, "node "+node.nodeName+" specifies more than one of prefix, suffix, or pattern!  Prefix will be used first, then suffix!", type=fatboyLog.WARNING)

        #Now look at subnodes
        ## index, value, property, object, timestamp
        tsMatches = 0 #separate counter for timestamp matches because need to assign unique indices
        if (node.hasChildNodes()):
            for snode in node.childNodes:
                #Ignore whitespace text nodes.  Just look at elements.
                if (snode.nodeType != Node.ELEMENT_NODE):
                    continue
                if (snode.nodeName == "value"):
                    matchAllInDir = False #this node has subtags -- only match these
                    if (snode.hasChildNodes()):
                        #value should contain a text node child only
                        if (snode.childNodes[0].nodeType == Node.TEXT_NODE):
                            sindex = snode.childNodes[0].nodeValue
                            matches = None
                            if (pfix is not None):
                                #Check prefix first
                                if (self._dir is not None):
                                    matches = glob.glob(self._dir+"/"+subdir+pfix+'*'+sindex+'.fits')
                                    matches.extend(glob.glob(self._dir+"/"+subdir+pfix+'*'+sindex+'.fits.fz'))
                                else:
                                    #if no dir specified, assume cwd
                                    matches = glob.glob(subdir+pfix+'*'+sindex+'.fits')
                                    matches.extend(glob.glob(subdir+pfix+'*'+sindex+'.fits.fz'))
                            elif (sfix is not None):
                                #Check suffix next, then pattern
                                if (self._dir is not None):
                                    matches = glob.glob(self._dir+"/"+subdir+"*"+sindex+"*"+sfix+"*.fits")
                                    matches.extend(glob.glob(self._dir+"/"+subdir+"*"+sindex+"*"+sfix+"*.fits.fz"))
                                else:
                                    #if no dir specified, assume cwd
                                    matches = glob.glob(subdir+"*"+sindex+"*"+sfix+"*.fits")
                                    matches.extend(glob.glob(subdir+"*"+sindex+"*"+sfix+"*.fits.fz"))
                            elif (pattern is not None):
                                #Don't add more wildcards to pattern
                                if (self._dir is not None):
                                    matches = glob.glob(self._dir+"/"+subdir+pattern+sindex+'.fits')
                                    matches.extend(glob.glob(self._dir+"/"+subdir+pattern+sindex+'.fits.fz'))
                                    #Allow pattern to be prefix or suffix
                                    matches.extend(glob.glob(self._dir+"/"+subdir+sindex+pattern+'.fits'))
                                    matches.extend(glob.glob(self._dir+"/"+subdir+sindex+pattern+'.fits.fz'))
                                else:
                                    matches = glob.glob(subdir+pattern+sindex+'.fits')
                                    matches.extend(glob.glob(subdir+pattern+sindex+'.fits.fz'))
                                    #Allow pattern to be prefix or suffix
                                    matches.extend(glob.glob(subdir+sindex+pattern+'.fits'))
                                    matches.extend(glob.glob(subdir+sindex+pattern+'.fits.fz'))
                            if (matches is None):
                                print("Warning: could not find file index matching value "+sindex)
                                self._log.writeLog(__name__, "could not find file index matching value "+sindex, type=fatboyLog.WARNING)
                                continue
                            if (len(matches) > 0):
                                #add match to images
                                matches.sort()
                                fname = matches[0]
                                #create a fatboyImage and add to list
                                #_datatype is fatboyImage unless overridden with datatype attribute
                                currImage = _datatype(fname, log=self._log, tag=self._tag)
                                if (subtag is not None):
                                    #set subtag
                                    currImage.setTag(subtag, True)
                                if ((sfix is not None and fname.find(sfix) > 0) or (pattern is not None and fname.find(pattern) > fname.find(sindex))):
                                    currImage.setSuffix(True)
                                    currImage.findIdentifier()
                                (fileprefix, sfileindex) = self.findIdentAndIndex(fname, suffix=currImage._suffix)
                                if (name is None):
                                    #if name attribute not specified, use file prefix as ident
                                    ident = fileprefix.replace('/','_')
                                else:
                                    ident = name
                                currImage.setIdentifier("manual", ident, sfileindex)
                                #set type if specified for a calib
                                #11/16/17 allow type to be set for object too
                                #if (isCalib and thisType != "auto"):
                                if (thisType != "auto"):
                                    currImage.setType(thisType, False)
                                images.append(currImage)
                            else:
                                print("Warning: could not find file index matching value "+sindex)
                                self._log.writeLog(__name__, "could not find file index matching value "+sindex, type=fatboyLog.WARNING)
                elif (snode.nodeName == "index"):
                    matchAllInDir = False #this node has subtags -- only match these
                    #index should have attributes start and stop
                    #index can also have value and except subtags
                    startIdx = None
                    stopIdx = None
                    excepts = []
                    vals = []
                    if (snode.hasAttributes()):
                        for j in range(snode.attributes.length):
                            attr = snode.attributes.item(j)
                            if (attr.nodeName == "start"):
                                startIdx = attr.nodeValue
                            elif (attr.nodeName == "stop"):
                                stopIdx = attr.nodeValue
                            else:
                                print("Warning: node "+node.nodeName+" has a subnode "+snode.nodeName+" with an invalid attribute: "+attr.nodeName)
                                self._log.writeLog(__name__, "node "+node.nodeName+" has a subnode "+snode.nodeName+" with an invalid attribute: "+attr.nodeName, type=fatboyLog.WARNING)
                    if (startIdx is None or stopIdx is None):
                        print("Error: node "+snode.nodeName+" must specify start and stop attributes!")
                        self._log.writeLog(__name__, "node "+snode.nodeName+" must specify start and stop attributes!", type=fatboyLog.ERROR)
                        continue
                    try:
                        #now look for value and except subtags
                        if (snode.hasChildNodes()):
                            for tnode in snode.childNodes:
                                #tnode = tertiary node
                                #Ignore whitespace text nodes.  Just look at elements.
                                if (tnode.nodeType != Node.ELEMENT_NODE):
                                    continue
                                if (tnode.nodeName == "value"):
                                    if (tnode.hasChildNodes()):
                                        #value should contain a text node child only
                                        if (tnode.childNodes[0].nodeType == Node.TEXT_NODE):
                                            #append int value to vals list
                                            vals.append(int(tnode.childNodes[0].nodeValue))
                                elif (tnode.nodeName == "except"):
                                    if (tnode.hasChildNodes()):
                                        #except should contain a text node child only
                                        if (tnode.childNodes[0].nodeType == Node.TEXT_NODE):
                                            #append int value to excepts list
                                            excepts.append(int(tnode.childNodes[0].nodeValue))
                                else:
                                    print("Warning: node "+node.nodeName+" has child node: "+snode.nodeName+" with invalid granchild: "+tnode.nodeName)
                                    self._log.writeLog(__name__, "node "+node.nodeName+" has child node: "+snode.nodeName+" with invalid granchild: "+tnode.nodeName, type=fatboyLog.WARNING)
                                    continue
                        #build list of indices from start, stop index and vals
                        indices = list(range(int(startIdx), int(stopIdx)+1))
                        indices.extend(vals)
                        totalMatches = 0
                        #loop over index numbers and look for matching files
                        for j in indices:
                            #check for excepts here
                            if (j in excepts):
                                continue
                            sindex = str(j)
                            #pad index with left zeros to number of digits in stopIdx
                            while (len(sindex) < len(stopIdx)):
                                sindex = '0'+sindex
                            matches = None
                            if (pfix is not None):
                                #Check prefix first
                                if (self._dir is not None):
                                    matches = glob.glob(self._dir+"/"+subdir+pfix+'*'+sindex+'.fits')
                                    matches.extend(glob.glob(self._dir+"/"+subdir+pfix+'*'+sindex+'.fits.fz'))
                                else:
                                    #if no dir specified, assume cwd
                                    matches = glob.glob(subdir+pfix+'*'+sindex+'.fits')
                                    matches.extend(glob.glob(subdir+pfix+'*'+sindex+'.fits.fz'))
                            elif (sfix is not None):
                                #Check suffix next, then pattern
                                if (self._dir is not None):
                                    matches = glob.glob(self._dir+"/"+subdir+"*"+sindex+"*"+sfix+"*.fits")
                                    matches.extend(glob.glob(self._dir+"/"+subdir+"*"+sindex+"*"+sfix+"*.fits.fz"))
                                else:
                                    #if no dir specified, assume cwd
                                    matches = glob.glob(subdir+"*"+sindex+"*"+sfix+"*.fits")
                                    matches.extend(glob.glob(subdir+"*"+sindex+"*"+sfix+"*.fits.fz"))
                            elif (pattern is not None):
                                #Don't add more wildcards to pattern
                                if (self._dir is not None):
                                    matches = glob.glob(self._dir+"/"+subdir+pattern+sindex+'.fits')
                                    matches.extend(glob.glob(self._dir+"/"+subdir+pattern+sindex+'.fits.fz'))
                                    #Allow pattern to be prefix or suffix
                                    matches.extend(glob.glob(self._dir+"/"+subdir+sindex+pattern+'.fits'))
                                    matches.extend(glob.glob(self._dir+"/"+subdir+sindex+pattern+'.fits.fz'))
                                else:
                                    matches = glob.glob(subdir+pattern+sindex+'.fits')
                                    matches.extend(glob.glob(subdir+pattern+sindex+'.fits.fz'))
                                    #Allow pattern to be prefix or suffix
                                    matches.extend(glob.glob(subdir+sindex+pattern+'.fits'))
                                    matches.extend(glob.glob(subdir+sindex+pattern+'.fits.fz'))
                            if (matches is None):
                                #No matches found.  Don't print out warning message here as some indices may not exist.
                                continue
                            if (len(matches) > 0):
                                totalMatches += 1
                                #add match to filenames
                                matches.sort()
                                fname = matches[0]
                                #create a fatboyImage and add to list
                                #_datatype is fatboyImage unless overridden with datatype attribute
                                currImage = _datatype(fname, log=self._log, tag=self._tag)
                                if (subtag is not None):
                                    #set subtag
                                    currImage.setTag(subtag, True)
                                if ((sfix is not None and fname.find(sfix) > 0) or (pattern is not None and fname.find(pattern) > fname.find(sindex))):
                                    currImage.setSuffix(True)
                                    currImage.findIdentifier()
                                (fileprefix, sfileindex) = self.findIdentAndIndex(fname, suffix=currImage._suffix)
                                if (name is None):
                                    #if name attribute not specified, use file prefix as ident
                                    ident = fileprefix.replace('/','_')
                                else:
                                    ident = name
                                currImage.setIdentifier("manual", ident, sfileindex)
                                #if (isCalib and thisType != "auto"):
                                if (thisType != "auto"):
                                    currImage.setType(thisType, False)
                                images.append(currImage)
                        if (totalMatches == 0):
                            #No matches found for entire index sequence.  Now print out error message
                            pfxStr = " matching "
                            if (pfix is not None):
                                pfxStr += "prefix "+str(pfix)
                            elif (sfix is not None):
                                pfxStr += "suffix "+str(sfix)
                            elif (pattern is not None):
                                pfxStr += "pattern "+str(pattern)
                            print("Warning: could not find any file index between "+startIdx+" and "+stopIdx+pfxStr)
                            self._log.writeLog(__name__, "could not find any file index between "+startIdx+" and "+stopIdx+pfxStr, type=fatboyLog.WARNING)
                    except Exception as ex:
                        print(ex)
                        print("Warning: Syntax Error processing node "+snode.nodeName)
                        self._log.writeLog(__name__, "Warning: Syntax Error processing node "+snode.nodeName, type=fatboyLog.WARNING)
                        continue
                elif (snode.nodeName == "property"):
                    #property should have attributes name and value
                    pname = None
                    pval = None
                    if (snode.hasAttributes()):
                        for j in range(snode.attributes.length):
                            attr = snode.attributes.item(j)
                            if (attr.nodeName == "name"):
                                pname = str(attr.nodeValue)
                            elif (attr.nodeName == "value"):
                                pval = str(attr.nodeValue)
                            else:
                                print("Warning: node "+node.nodeName+" has a subnode "+snode.nodeName+" with an invalid attribute: "+attr.nodeName)
                                self._log.writeLog(__name__, "node "+node.nodeName+" has a subnode "+snode.nodeName+" with an invalid attribute: "+attr.nodeName, type=fatboyLog.WARNING)
                    if (pname is None or pval is None):
                        print("Error: node "+snode.nodeName+" must specify name and value attributes!")
                        self._log.writeLog(__name__, "node "+snode.nodeName+" must specify name and value attributes!", type=fatboyLog.ERROR)
                        continue
                    #Assign property to dict.  Will be added to all FDUs in this object/calib below
                    properties[pname] = pval
                elif (isCalib and snode.nodeName == "object"):
                    #object should have <value> subtag(s)
                    if (snode.hasChildNodes()):
                        for tnode in snode.childNodes:
                            #tnode = tertiary node
                            #Ignore whitespace text nodes.  Just look at elements.
                            if (tnode.nodeType != Node.ELEMENT_NODE):
                                continue
                            if (tnode.nodeName == "value"):
                                if (tnode.hasChildNodes()):
                                    #value should contain a text node child only
                                    if (tnode.childNodes[0].nodeType == Node.TEXT_NODE):
                                        #append object to objTags list
                                        objTags.append(str(tnode.childNodes[0].nodeValue))
                            else:
                                print("Warning: node "+node.nodeName+" has child node: "+snode.nodeName+" with invalid granchild: "+tnode.nodeName)
                                self._log.writeLog(__name__, "node "+node.nodeName+" has child node: "+snode.nodeName+" with invalid granchild: "+tnode.nodeName, type=fatboyLog.WARNING)
                                continue
                elif (snode.nodeName == "timestamp"):
                    matchAllInDir = False #this node has subtags -- only match these
                    #index should have attributes start and stop
                    #index can also have value and except subtags
                    startTs = None
                    stopTs = None
                    excepts = []
                    vals = []
                    fmt = "%Y-%m-%dT%H:%M:%S.%f"
                    if (snode.hasAttributes()):
                        for j in range(snode.attributes.length):
                            attr = snode.attributes.item(j)
                            if (attr.nodeName == "start"):
                                startTs = attr.nodeValue
                            elif (attr.nodeName == "stop"):
                                stopTs = attr.nodeValue
                            elif (attr.nodeName == "format"):
                                fmt = attr.nodeValue
                            else:
                                print("Warning: node "+node.nodeName+" has a subnode "+snode.nodeName+" with an invalid attribute: "+attr.nodeName)
                                self._log.writeLog(__name__, "node "+node.nodeName+" has a subnode "+snode.nodeName+" with an invalid attribute: "+attr.nodeName, type=fatboyLog.WARNING)
                    if (startTs is None or stopTs is None):
                        print("Error: node "+snode.nodeName+" must specify start and stop attributes!")
                        self._log.writeLog(__name__, "node "+snode.nodeName+" must specify start and stop attributes!", type=fatboyLog.ERROR)
                        continue
                    try:
                        #now look for value and except subtags
                        if (snode.hasChildNodes()):
                            for tnode in snode.childNodes:
                                #tnode = tertiary node
                                #Ignore whitespace text nodes.  Just look at elements.
                                if (tnode.nodeType != Node.ELEMENT_NODE):
                                    continue
                                if (tnode.nodeName == "value"):
                                    if (tnode.hasChildNodes()):
                                        #value should contain a text node child only
                                        if (tnode.childNodes[0].nodeType == Node.TEXT_NODE):
                                            #append int value to vals list
                                            vals.append(str(tnode.childNodes[0].nodeValue))
                                elif (tnode.nodeName == "except"):
                                    if (tnode.hasChildNodes()):
                                        #except should contain a text node child only
                                        if (tnode.childNodes[0].nodeType == Node.TEXT_NODE):
                                            #append int value to excepts list
                                            excepts.append(str(tnode.childNodes[0].nodeValue))
                                else:
                                    print("Warning: node "+node.nodeName+" has child node: "+snode.nodeName+" with invalid granchild: "+tnode.nodeName)
                                    self._log.writeLog(__name__, "node "+node.nodeName+" has child node: "+snode.nodeName+" with invalid granchild: "+tnode.nodeName, type=fatboyLog.WARNING)
                                    continue
                        #glob on all files
                        matches = []
                        totalMatches = 0
                        if (pfix is not None):
                            #Check prefix first
                            if (self._dir is not None):
                                matches = glob.glob(self._dir+"/"+subdir+pfix+'*'+'.fits')
                                matches.extend(glob.glob(self._dir+"/"+subdir+pfix+'*'+'.fits.fz'))
                            else:
                                #if no dir specified, assume cwd
                                matches = glob.glob(subdir+pfix+'*'+'.fits')
                                matches.extend(glob.glob(subdir+pfix+'*'+'.fits.fz'))
                        elif (sfix is not None):
                            #Check suffix next, then pattern
                            if (self._dir is not None):
                                matches = glob.glob(self._dir+"/"+subdir+"*"+sfix+"*.fits")
                                matches.extend(glob.glob(self._dir+"/"+subdir+"*"+sfix+"*.fits.fz"))
                            else:
                                #if no dir specified, assume cwd
                                matches = glob.glob(subdir+"*"+sfix+"*.fits")
                                matches.extend(glob.glob(subdir+"*"+sfix+"*.fits.fz"))
                        matches.sort()
                        try:
                            startDt = datetime.strptime(startTs, fmt)
                            stopDt = datetime.strptime(stopTs, fmt)
                        except Exception as ex:
                            print(str(ex))
                            print("Warning: could not parse start and stop timestamps: "+str(startTs)+", "+str(stopTs))
                            self._log.writeLog(__name__, "could not parse start and stop timestamps: "+str(startTs)+", "+str(stopTs), type=fatboyLog.WARNING)
                            continue
                        for j in range(len(matches)):
                            isMatch = False
                            if (pfix is not None):
                                idx1 = matches[j].find(pfix)+len(pfix)
                                idx2 = matches[j].rfind('.fit')
                            elif (sfix is not None):
                                idx1 = matches[j].rfind('/')+1
                                idx2 = matches[j].rfind(sfix)
                            else:
                                break
                            ts = matches[j][idx1:idx2]
                            try:
                                dt = datetime.strptime(ts, fmt)
                            except Exception as ex:
                                print("Warning: could not parse timestamp "+str(ts))
                                self._log.writeLog(__name__, "could not parse timestamp "+str(ts), type=fatboyLog.WARNING)
                                continue
                            if (dt >= startDt and dt <= stopDt):
                                isMatch = True
                            elif (ts in vals):
                                isMatch = True
                            if (ts in excepts):
                                isMatch = False
                            if (isMatch):
                                totalMatches += 1
                                tsMatches += 1
                                fname = matches[j]
                                #create a fatboyImage and add to list
                                #_datatype is fatboyImage unless overridden with datatype attribute
                                currImage = _datatype(fname, log=self._log, tag=self._tag)
                                if (subtag is not None):
                                    #set subtag
                                    currImage.setTag(subtag, True)
                                if ((sfix is not None and fname.find(sfix) > 0)):
                                    currImage.setSuffix(True)
                                    #currImage.findIdentifier()
                                    fileprefix = sfix
                                else:
                                    fileprefix = pfix
                                #(fileprefix, sfileindex) = self.findIdentAndIndex(fname, suffix=currImage._suffix)
                                sfileindex = str(tsMatches) #Use tsMatches instead of totalMatches
                                while (len(sfileindex) < 6):
                                    sfileindex = '0'+sfileindex
                                if (name is None):
                                    #if name attribute not specified, use file prefix as ident
                                    ident = fileprefix.replace('/','_')
                                else:
                                    ident = name
                                currImage.setIdentifier("manual", ident, sfileindex)
                                #if (isCalib and thisType != "auto"):
                                if (thisType != "auto"):
                                    currImage.setType(thisType, False)
                                images.append(currImage)
                        if (totalMatches == 0):
                            #No matches found for entire index sequence.  Now print out error message
                            pfxStr = " matching "
                            if (pfix is not None):
                                pfxStr += "prefix "+str(pfix)
                            elif (sfix is not None):
                                pfxStr += "suffix "+str(sfix)
                            print("Warning: could not find any file index between "+startTs+" and "+stopTs+pfxStr)
                            self._log.writeLog(__name__, "could not find any file index between "+startTs+" and "+stopTs+pfxStr, type=fatboyLog.WARNING)
                    except Exception as ex:
                        print(ex)
                        print("Warning: Syntax Error processing node "+snode.nodeName)
                        self._log.writeLog(__name__, "Warning: Syntax Error processing node "+snode.nodeName, type=fatboyLog.WARNING)
                        continue
                else:
                    print("Warning: node "+node.nodeName+" has invalid child node: "+snode.nodeName)
                    self._log.writeLog(__name__, "node "+node.nodeName+" has invalid child node: "+snode.nodeName, type=fatboyLog.WARNING)
                    continue
        if (matchAllInDir):
            #No <value> or <index> child nodes = select all matching prefix*.fits or pattern.fits
            try:
                matches = None
                if (calibFilename is not None):
                    #Check calibFilename first
                    if (self._dir is not None):
                        if (os.access(self._dir+"/"+calibFilename, os.F_OK)):
                            matches = [self._dir+"/"+calibFilename]
                    else:
                        #if no dir specified, assume cwd
                        if (os.access(calibFilename, os.F_OK)):
                            matches = [calibFilename]
                elif (pfix is not None):
                    #Check prefix next
                    if (self._dir is not None):
                        matches = glob.glob(self._dir+"/"+subdir+pfix+'*.fits')
                        matches.extend(glob.glob(self._dir+"/"+subdir+pfix+'*.fits.fz'))
                    else:
                        #if no dir specified, assume cwd
                        matches = glob.glob(subdir+pfix+'*.fits')
                        matches.extend(glob.glob(subdir+pfix+'*.fits.fz'))
                elif (sfix is not None):
                    #Check suffix next, then pattern
                    if (self._dir is not None):
                        matches = glob.glob(self._dir+"/"+subdir+"*"+sfix+"*.fits")
                        matches.extend(glob.glob(self._dir+"/"+subdir+"*"+sfix+"*.fits.fz"))
                    else:
                        #if no dir specified, assume cwd
                        matches = glob.glob(subdir+"*"+sfix+"*.fits")
                        matches.extend(glob.glob(subdir+"*"+sfix+"*.fits.fz"))
                elif (pattern is not None):
                    #Don't add more wildcards to pattern
                    if (self._dir is not None):
                        matches = glob.glob(self._dir+"/"+subdir+pattern+'.fits')
                        matches.extend(glob.glob(self._dir+"/"+subdir+pattern+'.fits.fz'))
                    else:
                        matches = glob.glob(subdir+pattern+'.fits')
                        matches.extend(glob.glob(subdir+pattern+'.fits.fz'))
                if (matches is None):
                    #No matches found.
                    if (calibFilename is not None):
                        print("Warning: file "+calibFilename+" not found!")
                        self._log.writeLog(__name__, "file "+calibFilename+" not found!", type=fatboyLog.WARNING)
                    elif (pfix is not None):
                        print("Warning: found no images matching prefix "+pfix)
                        self._log.writeLog(__name__, "found no images matching prefix "+pfix, type=fatboyLog.WARNING)
                    elif (sfix is not None):
                        print("Warning: found no images matching suffix "+sfix)
                        self._log.writeLog(__name__, "found no images matching suffix "+sfix, type=fatboyLog.WARNING)
                    elif (pattern is not None):
                        print("Warning: found no images matching pattern "+pattern)
                        self._log.writeLog(__name__, "found no images matching pattern "+pattern, type=fatboyLog.WARNING)
                    return images
                if (len(matches) > 0):
                    #add ALL matches to filenames
                    matches.sort()
                    for j in range(len(matches)):
                        fname = matches[j]
                        #create a fatboyImage and add to list
                        #_datatype is fatboyImage unless overridden with datatype attribute
                        currImage = _datatype(fname, log=self._log, tag=self._tag)
                        if (subtag is not None):
                            #set subtag
                            currImage.setTag(subtag, True)
                        if (sfix is not None and fname.find(sfix) > 0):
                            currImage.setSuffix(True)
                            currImage.findIdentifier()
                        (fileprefix, sfileindex) = self.findIdentAndIndex(fname, suffix=currImage._suffix)
                        if (name is None):
                            #if name attribute not specified, use file prefix as ident
                            ident = fileprefix.replace('/','_')
                        else:
                            ident = name
                        currImage.setIdentifier("manual", ident, sfileindex)
                        #if (isCalib and thisType != "auto"):
                        if (thisType != "auto"):
                            currImage.setType(thisType, False)
                        images.append(currImage)
                        ident = None #Reset ident
            except Exception as ex:
                print(ex)
                print("Warning: Syntax Error processing node "+node.nodeName)
                self._log.writeLog(__name__, "Warning: Syntax Error processing node "+node.nodeName, type=fatboyLog.WARNING)
        #Loop over properties and assign each one to all FDUs in this object/calib
        for key in properties:
            value = properties[key]
            #Loop over images
            for image in images:
                image.setProperty(key, value)
        #Loop over objTags if a calib and assign to all FDUs in this object/calib
        if (isCalib):
            for tag in objTags:
                #Loop over images
                for image in images:
                    image.addObjectTag(tag)
        return images
    #end parseObject
