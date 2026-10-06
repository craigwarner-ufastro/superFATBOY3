## @package superFATBOY.datatypeExtensions
from superFATBOY.fatboyImage import *

class fourStarImage(fatboyImage):
    _name = "fourStarImage"
    section = 1 #default


    #No need to override forgetData or getData as osirisSpectrum and circeImage do
    #as long as no pre-processing needs to be done to data.
    #If we just assign a section and combine chips into full array at a later step this is the case.
    def findIdentifier(self, delim=None):
        dpos = self.filename.rfind('.')
        if (self.filename.endswith('.fz')):
            dpos = self.filename[:-3].rfind('.')
        cpos = dpos-1
        #fsr_0034_01_c4.fits
        if (self.filename[cpos-2:cpos] == '_c' and isDigit(self.filename[cpos])):
            #chip # - set section
            self.section = int(self.filename[cpos])
        #Now find index and id
        dpos = self.filename[:cpos].rfind('_')
        cpos = dpos-1
        #Find rightmost non-numerical character before .fits
        while((isDigit(self.filename[cpos]) or self.filename[cpos] == '_') and cpos > 0):
            cpos-=1
        self._id = self.filename[self.filename.rfind('/')+1:cpos+1]
        while (self._id.endswith('.') or self._id.endswith('-') or self._id.endswith('_')):
            self._id = self._id[:-1]
        #Replace _ between index and dither count with ''
        self._index = self.filename[cpos+1:dpos].replace('_','')
        self._id = str(self._id) #convert from unicode to str!!
        self._index = str(self._index) #convert fron unicode to str!!
        self._identFull = self._id+'.'+self._index+'.fits'
        return (self._id, self._index)

    ## Set the identifier for this data unit
    def setIdentifier(self, groupType, fileprefix, sindex=None, keyword=None):

        #Logic before or after parent method is called to set self.section
        #based on info from header or filename and if section is in filename,
        #make sure fileprefix and sindex (string-index e.g. '0001' is correct).

        #we want fileprefix == 'abc', sindex='0001' to have 4 fourStarImage objects,
        #each with self.section in 1,2,3,4.

        ##Call parent method
        fatboyImage.setIdentifier(self, groupType, fileprefix, sindex=sindex, keyword=keyword)
    #end setIdentifier
