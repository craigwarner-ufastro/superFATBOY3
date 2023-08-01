## @package superFATBOY.datatypeExtensions
from superFATBOY.fatboyImage import *

class circeImage(fatboyImage):
    EXPMODE_FS = 0
    EXPMODE_URG = 1
    EXPMODE_URG_BYPASS = 2

    _name = "circeImage"
    _expmode = EXPMODE_FS
    ramp = 1 #By default ramp = 1 and section = 1 (the first ramp of every file)
    section = 1
    firstDataAccess = True

    def forgetData(self):
        outfile = self._fdb._tempdir+"/current_"+self.getFullId()
        if (not os.access(outfile, os.F_OK)):
            self.firstDataAccess = True #reset flag so that circeImage getData gets called to reread from disk
        #call superclass
        fatboyImage.forgetData(self)
    #end forgetData

    ## Get and return data. Only read from disk if necessary.
    ## OVERRIDE this method to return CDS difference of correct ramps on first access of data
    def getData(self, tag=None):
        if (self.firstDataAccess):
            self.firstDataAccess = False
            #Read from disk
            t = time.time()
            image = pyfits.open(self.filename)
            if (self._expmode == self.EXPMODE_FS):
                try:
                    #CIRCE data has nramps sets of 2x nreads frames
                    self._data = image[(self.ramp-1)*2*self.nreads+self.nreads+1].data.astype(int32) - image[(self.ramp-1)*2*self.nreads+1].data.astype(int32)
                    if (self.nreads > 1):
                        for read in range(2, self.nreads+1):
                            self._data += image[(self.ramp-1)*2*self.nreads+self.nreads+read].data.astype(int32) - image[(self.ramp-1)*2*self.nreads+read].data.astype(int32)
                    #self._data = image[self.ramp*2].data-image[self.ramp*2-1].data
                    self._data = self._data.astype(int32)
                except Exception:
                    self._data = None
                    print("circeImage::getData> Error: Could not find ramp "+str(self.ramp)+" in "+self.filename+"!  Discarding this frame!")
                    self._log.writeLog(__name__, "Could not find ramp "+str(self.ramp)+" in "+self.filename+"! Discarding this frame!", type=fatboyLog.ERROR)
                    self.disable()
                    return None
            elif (self._expmode == self.EXPMODE_URG):
                try:
                    #CIRCE URG data has nreads = 1, nramps sets of ngroups frames
                    #E.g., ngroups=4, nramps = 2 => [1,1], [2,1], [3,1], [4,1], RESET, [1,2], [2,2], [3,2], [4,2]
                    #Final output is nramps * (ngroups-1) frames, [2,1]-[1,1]; [3,1]-[2,1]; [4,1]-[3,1]; [2,2]-[1,2]; [3,2]-[2,2], [4,2]-[3,2]
                    self._data = image[(self.ramp-1)*self.getNGroups()+self.group+1].data.astype(int32) - image[(self.ramp-1)*self.getNGroups()+self.group].data.astype(int32)
                    self._data = self._data.astype(int32)
                except Exception:
                    self._data = None
                    print("circeImage::getData> Error: Could not find ramp "+str(self.ramp)+", group "+str(self.group)+" in "+self.filename+"!  Discarding this frame!")
                    self._log.writeLog(__name__, "Could not find ramp "+str(self.ramp)+", group "+str(self.group)+" in "+self.filename+"! Discarding this frame!", type=fatboyLog.ERROR)
                    self.disable()
                    return None
            elif (self._expmode == self.EXPMODE_URG_BYPASS):
                try:
                    #CIRCE URG BYPASS mode has nreads = 1, nramps sets of ngroups frames, but only returns read(final)-read(first) for each ramp
                    #E.g., ngroups=4, nramps = 2 => [1,1], [2,1], [3,1], [4,1], RESET, [1,2], [2,2], [3,2], [4,2]
                    #Final output is nramps frames, [4,1]-[1,1]; [4,2]-[1,2]
                    self._data = image[self.ramp*self.getNGroups()].data.astype(int32) - image[(self.ramp-1)*self.getNGroups()+1].data.astype(int32)
                    self._data = self._data.astype(int32)
                except Exception:
                    self._data = None
                    print("circeImage::getData> Error: Could not find ramp "+str(self.ramp)+" in "+self.filename+"!  Discarding this frame!")
                    self._log.writeLog(__name__, "Could not find ramp "+str(self.ramp)+" in "+self.filename+"! Discarding this frame!", type=fatboyLog.ERROR)
                    self.disable()
                    return None
            else:
                print("circeImage::getData> Error: Invalid expmode "+str(self._expmode)+"!  Discarding this frame!")
                self._log.writeLog(__name__, "Invalid expmode "+str(self._expmode)+"!  Discarding this frame!", type=fatboyLog.ERROR)
                self.disable()
                return None

            if (not self._data.dtype.isnative):
                #Byteswap
                self._data = self._data.byteswap()
                self._data = self._data.newbyteorder('<')
                self._data.dtype.newbyteorder('<')
            self._shape = self._data.shape
            image.close()
            if (self._fdb is not None):
                self._fdb.totalReadDataTime += (time.time()-t)
                self._fdb.checkMemoryManagement(self) #check memory status
            if (self.getObsType(True) == self.FDU_TYPE_BAD_PIXEL_MASK):
                #bad pixel masks should be type bool
                if (self._data.dtype != dtype("bool")):
                    self._data = self._data.astype("bool")
            return self._data
        else:
            #use superclass method
            return fatboyImage.getData(self, tag=tag)
    #end getData

    ## Get an individual read
    def getIndividualRead(self, n):
        #Read from disk
        t = time.time()
        image = pyfits.open(self.filename)
        if (n >= len(image)):
            print("circeImage::getIndividualRead> Error: Read "+str(n)+" does not exist!")
            self._log.writeLog(__name__, "Read "+str(n)+" does not exist!", type=fatboyLog.ERROR)
            return None
        data = image[n].data.astype(int32)
        image.close()
        if (self._fdb is not None):
            self._fdb.totalReadDataTime += (time.time()-t)
            self._fdb.checkMemoryManagement(self) #check memory status
        return data
    #end getIndividual Read

    ## Base class returns empty list.  Can be overridden to return a list of fatboyDataUnit (or subclass) representing multiple data extensions.
    ## Each should have a different fdu.section value.  For instance, newfirm has 4 detectors or CIRCE has multiple nramps.
    def getMultipleExtensions(self):
        extendedImages = []
        nramps = self.getNRamps()
        if (self._expmode == self.EXPMODE_FS):
            self.setRamp(1)
            for j in range(1, nramps):
                currImage = circeImage(self.filename, log=self._log, tag=self._tag)
                currImage.setExpMode(self._expmode)
                currImage.setRamp(j+1)
                currImage.setIdentifier("manual", self._id[:-2], self._index) #Strip S1 from end of _id
                if (self.getObsType() is not None):
                    currImage.setType(self.getObsType(), False) #Set obs type if identified in XML
                currImage._objectTags = self._objectTags #Copy over object tags for calibs from XML
                for key in self._properties:
                    #Set all properties from XML too
                    currImage.setProperty(key, self.getProperty(key))
                extendedImages.append(currImage)
        elif (self._expmode == self.EXPMODE_URG):
            ngroups = self.getNGroups()
            self.setRGS(1,1,1)
            section = 1
            for j in range(nramps):
                for g in range(ngroups-1):
                    if (j == 0 and g == 0):
                        #image already exists for first frame
                        continue
                    #increment section
                    section += 1
                    currImage = circeImage(self.filename, log=self._log, tag=self._tag)
                    currImage.setExpMode(self._expmode)
                    currImage.setRGS(j+1, g+1, section)
                    currImage.setIdentifier("manual", self._id[:-2], self._index) #Strip S1 from end of _id
                    if (self.getObsType() is not None):
                        currImage.setType(self.getObsType(), False) #Set obs type if identified in XML
                    currImage._objectTags = self._objectTags #Copy over object tags for calibs from XML
                    for key in self._properties:
                        #Set all properties from XML too
                        currImage.setProperty(key, self.getProperty(key))
                    extendedImages.append(currImage)
        elif (self._expmode == self.EXPMODE_URG_BYPASS):
            ngroups = self.getNGroups()
            self.setRGS(1,1,1)
            section = 1
            for j in range(1, nramps):
                #increment section
                section += 1
                currImage = circeImage(self.filename, log=self._log, tag=self._tag)
                currImage.setExpMode(self._expmode)
                currImage.setRGS(j+1, 1, section)
                currImage.setIdentifier("manual", self._id[:-2], self._index) #Strip S1 from end of _id
                if (self.getObsType() is not None):
                    currImage.setType(self.getObsType(), False) #Set obs type if identified in XML
                currImage._objectTags = self._objectTags #Copy over object tags for calibs from XML
                for key in self._properties:
                    #Set all properties from XML too
                    currImage.setProperty(key, self.getProperty(key))
                extendedImages.append(currImage)
        return extendedImages
    #end getMultipleExtensions

    def getNGroups(self):
        if (self.hasHeaderValue('NGROUPS')):
            return self.getHeaderValue('NGROUPS')
        return 0
    #end getNGroups

    def getNRamps(self):
        if (self.hasHeaderValue('NRAMPS')):
            return self.getHeaderValue('NRAMPS')
        return 0
    #end getNRamps

    ## This method can be overridden in subclasses to support images with multiple data extensions, represented as multiple fatboyDataUnits.
    def hasMultipleExtensions(self):
        t = time.time()
        self.readHeader()
        #Move EXPMODE check to here, need to know mode before determining extensions
        if (self.hasHeaderValue('EXPMODE')):
            if (self.getHeaderValue('EXPMODE') == 'URG'):
                self._expmode = self.EXPMODE_URG
        if (self._expmode == self.EXPMODE_URG and self.hasProperty("expmode")):
            if (self.getProperty("expmode").lower() == "bypass_intermediate_reads"):
                self._expmode = self.EXPMODE_URG_BYPASS
        nramps = self.getNRamps()
        if (nramps == 0):
            print("circeImage::hasMultipleExtensions> Warning: Unable to find keyword NRAMPS in "+self.filename+"!  Skipping this frame!")
            self._log.writeLog(__name__, "Unable to find keyword NRAMPS in "+self.filename+"!", type=fatboyLog.WARNING)
            self.disable()
            return False
        if (self._expmode == self.EXPMODE_FS):
            if (nramps > 1):
                return True
        elif (self._expmode == self.EXPMODE_URG):
            ngroups = self.getNGroups()
            if (ngroups == 0):
                print("circeImage::hasMultipleExtensions> Warning: Unable to find keyword NGROUPS in "+self.filename+"!  Skipping this frame!")
                self._log.writeLog(__name__, "Unable to find keyword NGROUPS in "+self.filename+"!", type=fatboyLog.WARNING)
                self.disable()
                return False
            if (nramps*(ngroups-1) > 1):
                return True
        elif (self._expmode == self.EXPMODE_URG_BYPASS):
            if (nramps > 1):
                return True
        return False
    #end hasMultipleExtensions

    ## Set expmode
    def setExpMode(self, expmode):
        self._expmode = expmode
    #end setExpMode

    ## Set the identifier for this data unit
    def setIdentifier(self, groupType, fileprefix, sindex=None, keyword=None):
        ##Call parent method
        fatboyImage.setIdentifier(self, groupType, fileprefix, sindex=sindex, keyword=keyword)
        ##CIRCE specific
        sramp = str(self.ramp)
        zeros = '0000'
        sramp = zeros[len(sramp):]+sramp
        if (self._expmode == self.EXPMODE_URG):
            #trailing index should be section number not ramp number for URG data
            sramp = str(self.section)
            sramp = zeros[len(sramp):]+sramp
        self._identFull = self._id+'.'+self._index+sramp+'.fits'
        self._identFull = self._identFull.replace('..','.') #for case of blank index in calibs
    #end setIdentifier

    ## Set the ramp and section of this circeImage
    def setRamp(self, ramp):
        self.ramp = ramp
        if (ramp == 1):
            self.section = 1
        else:
            self.section = 2
        #Add to header
        updateHeaderEntry(self._header, 'SECTION', self.section)
    #end setRamp

    ## Set the ramp and section of this circeURGImage
    def setRGS(self, ramp, group, section):
        self.ramp = ramp
        self.group = group
        self.section = section
        #Add to header
        updateHeaderEntry(self._header, 'SECTION', self.section)
    #end setRamp
