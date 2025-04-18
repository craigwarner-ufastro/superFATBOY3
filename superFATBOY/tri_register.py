hasSep = True
try:
    import sep
except Exception:
    print("Warning: sep not installed")
    hasSep = False

import scipy, os, time
import itertools
from scipy.optimize import leastsq
from .fatboyLibs import *
from .fatboyLog import *
from .fatboyDataUnit import *

usePlot = True
try:
    import matplotlib.pyplot as plt
except Exception as ex:
    print("Warning: Could not import matplotlib!")
    usePlot = False

MODE_FITS = 0
MODE_RAW = 1
MODE_FDU = 2
MODE_FDU_DIFFERENCE = 3 #for twilight flats
MODE_FDU_TAG = 4 #tagged data from a specific step, e.g. preSkySubtraction

METHOD_DELAUNAY = 0
METHOD_ALL = 1

class Triangle:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys
        self.points = np.array([xs, ys]).T
        self.lengths = np.linalg.norm(self.points-np.roll(self.points,1,axis=0), axis=1)
        self.dx = xs-np.roll(xs,1)
        self.dy = ys-np.roll(ys,1)
        self.lengths.sort()
        self.dx.sort()
        self.dy.sort()
        self.ratios = self.lengths/np.roll(self.lengths,1)
        self.xs.sort()
        self.ys.sort()

    def compareTo(self, tri, atol=1.0, rtol=0.01):
        deltax = self.dx-tri.dx
        #print(f'{self.dx=} {tri.dx=} {deltax=}')
        if (np.abs(deltax).max() > atol):
            return False
        deltay = self.dy-tri.dy
        #print(f'{self.dy=} {tri.dy=} {deltay=}')
        if (np.abs(deltay).max() > atol):
            return False
        delta_lengths = np.abs(self.lengths-tri.lengths)
        #print(f'{self.lengths=} {tri.lengths=} {delta_lengths=}')
        if (delta_lengths.max() > atol):
            return False
        if ((delta_lengths/self.lengths).max() > rtol):
            return False
        delta_ratios = np.abs(self.ratios-tri.ratios)
        #print(f'{self.ratios=} {tri.ratios=} {delta_ratios=}')
        if ((delta_ratios/self.ratios).max() > rtol):
            return False
        shifts_x = self.xs-tri.xs
        shifts_y = self.ys-tri.ys
        #print(f'{shifts_x=} {shifts_y=}')
        #print (shifts_x.std(), shifts_y.std())
        if (shifts_x.std() > atol or shifts_y.std() > atol):
            return False
        return np.array([shifts_x.mean(), shifts_y.mean()])

def generate_triangles_vectorized_arrays(x_points, y_points, name=None, doplots=False, min_angle=30, max_angle=110):
    """Generates triangles with vectorization, taking x and y arrays as input."""

    if len(x_points) != len(y_points):
        raise ValueError("X and Y arrays must have the same length.")

    points_np = np.column_stack((x_points, y_points))
    num_points = len(points_np)

    # Delaunay triangles
    delaunay = scipy.spatial.Delaunay(points_np)
    delaunay_triangles = delaunay.simplices.tolist()

    # Generate all possible triangle combinations
    indices = np.array(list(itertools.combinations(range(num_points), 3)))

    p1 = points_np[indices[:, 0]]
    p2 = points_np[indices[:, 1]]
    p3 = points_np[indices[:, 2]]

    # Calculate side lengths
    a = np.linalg.norm(p2 - p3, axis=1)
    b = np.linalg.norm(p1 - p3, axis=1)
    c = np.linalg.norm(p1 - p2, axis=1)

    # Check for degenerate triangles
    valid_mask = (a > 0) & (b > 0) & (c > 0)

    # Calculate angles
    cos_angles1 = (b**2 + c**2 - a**2) / (2 * b * c)
    cos_angles2 = (a**2 + c**2 - b**2) / (2 * a * c)
    cos_angles3 = (a**2 + b**2 - c**2) / (2 * a * b)

    # Handle potential domain errors for arccos
    cos_angles1 = np.clip(cos_angles1, -1.0, 1.0)
    cos_angles2 = np.clip(cos_angles2, -1.0, 1.0)
    cos_angles3 = np.clip(cos_angles3, -1.0, 1.0)

    angles1 = np.degrees(np.arccos(cos_angles1))
    angles2 = np.degrees(np.arccos(cos_angles2))
    angles3 = np.degrees(np.arccos(cos_angles3))

    # Check for small angles
    valid_mask &= (angles1 >= min_angle) & (angles2 >= min_angle) & (angles3 >= min_angle)
    valid_mask &= (angles1 <= max_angle) & (angles2 <= max_angle) & (angles3 <= max_angle)

    # Filter valid triangles
    valid_triangles = indices[valid_mask].tolist()

    # Combine Delaunay and valid triangles
    all_triangles = delaunay_triangles + valid_triangles

    #Remove duplicates
    unique_triangles = []
    seen = set()
    for triangle in all_triangles:
        sorted_triangle = tuple(sorted(triangle))
        if sorted_triangle not in seen:
            unique_triangles.append(triangle)
            seen.add(sorted_triangle)

    if (usePlot and doplots and name is not None):
        plt.triplot(x_points, y_points, unique_triangles, color='g')
        plt.title(name)
        pltfile = name+".png"
        plt.savefig(pltfile, dpi=200)
        plt.close()

    tr = []
    for j in range(len(unique_triangles)):
        tr.append(Triangle(x_points[unique_triangles][j], y_points[unique_triangles][j]))
    #return (unique_triangles, angles1, angles2, angles3)
    return tr

def generate_triangles_delaunay(x_points, y_points, name=None, doplots=False):
    if len(x_points) != len(y_points):
        raise ValueError("X and Y arrays must have the same length.")

    points_np = np.column_stack((x_points, y_points))
    num_points = len(points_np)
    
    # Delaunay triangles
    delaunay = scipy.spatial.Delaunay(points_np)
    #delaunay_triangles = delaunay.simplices.tolist()
    dts = delaunay.simplices
    if (usePlot and doplots and name is not None):
        plt.triplot(x_points, y_points, dts, color='g')
        plt.title(name)        
        pltfile = name+".png"
        plt.savefig(pltfile, dpi=200)
        plt.close()
    tr = []
    for j in range(len(dts)):
        tr.append(Triangle(x_points[dts][j], y_points[dts][j]))
    return tr


def tri_register(frames, outfile=None, xcenter=-1, ycenter=-1, xboxsize=-1, yboxsize=-1, border=20, log=None, mef=0, gui=None, refframe=0, mode=None, dataTag=None, sepDetectThresh=3, method=METHOD_DELAUNAY, min_angle=30, max_angle=110, max_stars=None, doplots=False, plotdir=".", atol=2.0, rtol=0.025, sigma_clipping=False, sig_to_clip=3):
    t = time.time()
    _verbosity = fatboyLog.NORMAL
    #set log type
    logtype = LOGTYPE_NONE
    if (log is not None):
        if (isinstance(log, str)):
            #log given as a string
            log = open(log,'a')
            logtype = LOGTYPE_ASCII
        elif(isinstance(log, fatboyLog)):
            logtype = LOGTYPE_FATBOY
            _verbosity = log._verbosity

    nframes = len(frames)
    #Find type
    if (mode is None):
        mode = MODE_FITS
        if (isinstance(frames, str)):
            mode = MODE_FITS
        elif (isinstance(frames[0], str)):
            mode = MODE_FITS
        elif (isinstance(frames[0], ndarray)):
            mode = MODE_RAW
        elif (isinstance(frames[0], fatboyDataUnit)):
            mode = MODE_FDU

    if (mode == MODE_FITS):
        if (isinstance(frames, str)):
            if (os.access(frames, os.F_OK)):
                filelist = readFileIntoList(frames)
                nframes = len(filelist)
            else:
                print("triregister> Could not find file "+frames)
                write_fatboy_log(log, logtype, "Could not find file "+frames, __name__)
                return None
        else:
            filelist = frames
        #find refframe
        if (isinstance(refframe, str)):
            for j in range(len(filelist)):
                if (filelist[j].find(refframe) != -1):
                    refframe = j
                    print("triregister> Using "+filelist[j]+" as reference frame.")
                    write_fatboy_log(log, logtype, "Using "+filelist[j]+" as reference frame.", __name__)
                    break
            if (isinstance(refframe, str)):
                print("triregister> Could not find reference frame: "+refframe+"!  Using frame 0 = "+filelist[0])
                write_fatboy_log(log, logtype, "Could not find reference frame: "+refframe+"!  Using frame 0 = "+filelist[0], __name__)
                refframe = 0
    elif (mode == MODE_FDU or mode == MODE_FDU_DIFFERENCE or mode == MODE_FDU_TAG):
        #find refframe
        if (isinstance(refframe, str)):
            for j in range(len(frames)):
                if (frames[j].getFullId().find(refframe) != -1):
                    refframe = j
                    print("triregister> Using "+frames[j].getFullId()+" as reference frame.")
                    write_fatboy_log(log, logtype, "Using "+frames[j].getFullId()+" as reference frame.", __name__)
                    break
            if (isinstance(refframe, str)):
                print("triregister> Could not find reference frame: "+refframe+"!  Using frame 0 = "+frames[0].getFullId())
                write_fatboy_log(log, logtype, "Could not find reference frame: "+refframe+"!  Using frame 0 = "+frames[0].getFullId(), __name__)
                refframe = 0

    #Get reference frame
    if (mode == MODE_FITS):
        if (os.access(filelist[refframe], os.F_OK)):
            temp = pyfits.open(filelist[refframe])
            refData = temp[mef].data
            if (not refData.dtype.isnative):
                print("triregister> Byteswapping "+filelist[refframe])
                refData = float32(refData)
            refName = filelist[refframe]
            temp.close()
        else:
            print("triregister> Could not find file "+filelist[refframe])
            write_fatboy_log(log, logtype, "Could not find file "+filelist[refframe], __name__)
            return None
    elif (mode == MODE_RAW):
        refData = frames[refframe]
        refName = "index "+str(refframe)
    elif (mode == MODE_FDU):
        refData = frames[refframe].getData()
        refName = frames[refframe].getFullId()
    elif (mode == MODE_FDU_DIFFERENCE):
        refData = frames[refframe+1].getData()-frames[refframe].getData()
        refName = frames[refframe+1].getFullId()+"-"+frames[refframe].getFullId()
        #2-1, 3-2, etc.
        nframes -= 1
    elif (mode == MODE_FDU_TAG):
        refData = frames[refframe].getData(tag=dataTag)
        refName = frames[refframe].getFullId()+":"+dataTag
    else:
        print("triregister> Invalid input!  Exiting!")
        write_fatboy_log(log, logtype, "Invalid input!  Exiting!", __name__)
        return None

    shp = refData.shape

    if (xcenter == -1):
        xcenter = shp[1]//2
    if (ycenter == -1):
        ycenter = shp[0]//2
    if (xboxsize == -1):
        xboxsize = shp[1]
    if (yboxsize == -1):
        yboxsize = shp[0]

    print("Using ("+str(xboxsize)+", "+str(yboxsize)+") pixel wide box centered at ("+str(xcenter)+", "+str(ycenter)+") with "+str(border)+" pixel border.")
    write_fatboy_log(log, logtype, "Using ("+str(xboxsize)+", "+str(yboxsize)+") pixel wide box centered at ("+str(xcenter)+", "+str(ycenter)+") with "+str(border)+" pixel border.", __name__)

    x1 = xcenter-xboxsize//2
    x2 = xcenter+xboxsize//2
    y1 = ycenter-yboxsize//2
    y2 = ycenter+yboxsize//2
    if (x1 < 0):
        x1 = 0
    if (y1 < 0):
        y1 = 0
    if (x2 > shp[1]):
        x2 = shp[1]
    if (y2 > shp[0]):
        y2 = shp[0]

    xshifts = [0]
    yshifts = [0]
    ccmax = [0]

    refData = refData[y1:y2, x1:x2]
    bkg = sep.Background(refData)
    print("\tsep background = "+str(bkg.globalback)+" rms = "+str(bkg.globalrms))
    write_fatboy_log(log, logtype, "sep background = "+str(bkg.globalback)+" rms = "+str(bkg.globalrms), __name__)
    thresh = sepDetectThresh*bkg.globalrms #Default = 3
    #subtract background from data
    bkg.subfrom(refData)
    #extract objects
    refObjects = sep.extract(refData, thresh, minarea=9)
    keep = (refObjects['x'] > border)*(refObjects['x'] < refData.shape[1]-border)*(refObjects['y'] > border)*(refObjects['y'] < refData.shape[0]-border) 
    #throw away objects at edges
    refObjects = refObjects[keep]
    #sort by flux
    refObjects = refObjects[refObjects['flux'].argsort()[::-1]]
    print("\tsep extracted "+str(len(refObjects))+" objects using thresh = "+str(sepDetectThresh)+"*rms")
    write_fatboy_log(log, logtype, "sep extracted "+str(len(refObjects))+" objects using thresh = "+str(sepDetectThresh)+"*rms", __name__)

    ref_x = refObjects['x']
    ref_y = refObjects['y']
    if (max_stars is not None):
        ref_x = ref_x[:max_stars]
        ref_y = ref_y[:max_stars]

    if (method == METHOD_DELAUNAY):
        ref_triangles = generate_triangles_delaunay(ref_x, ref_y, name=plotdir+"/"+refName, doplots=doplots)
    else:
        ref_triangles = generate_triangles_vectorized_arrays(ref_x, ref_y, name=plotdir+"/"+refName, doplots=doplots, min_angle=min_angle, max_angle=max_angle)

    if (mode == MODE_FDU or mode == MODE_FDU_TAG):
        frames[refframe].setProperty("triangles", ref_triangles)

    all_triangles = []

    if (outfile is not None):
        f = open(outfile,'w')

    #if (_verbosity == fatboyLog.VERBOSE):
    if True:
        print("Initialize: ",time.time()-t)
    tt = time.time()

    for j in range(nframes):
        tt = time.time()
        if (j == refframe):
            continue
        #Get data
        if (mode == MODE_FITS):
            if (os.access(filelist[j], os.F_OK)):
                temp = pyfits.open(filelist[j])
                currData = temp[mef].data
                if (not currData.dtype.isnative):
                    print("triregister> Byteswapping "+filelist[j])
                    currData = float32(currData)
                currName = filelist[j]
                temp.close()
            else:
                print("triregister> Could not find file "+filelist[j])
                write_fatboy_log(log, logtype, "Could not find file "+filelist[j], __name__)
                return None
        elif (mode == MODE_RAW):
            currData = frames[j]
            currName = "index "+str(j)
        elif (mode == MODE_FDU):
            currData = frames[j].getData()
            currName = frames[j].getFullId()
        elif (mode == MODE_FDU_DIFFERENCE):
            currData = frames[j+1].getData()-frames[j].getData()
            currName = frame[j+1].getFullId()+"-"+frame[j].getFullId()
            #2-1, 3-2, etc.
        elif (mode == MODE_FDU_TAG):
            currData = frames[j].getData(tag=dataTag)
            currName = frames[j].getFullId()+":"+dataTag
        else:
            print("triregister> Invalid input!  Exiting!")
            write_fatboy_log(log, logtype, "Invalid input!  Exiting!", __name__)
            return None

        currData = currData[y1:y2, x1:x2]
        bkg = sep.Background(currData)
        print("\tsep background = "+str(bkg.globalback)+" rms = "+str(bkg.globalrms))
        write_fatboy_log(log, logtype, "sep background = "+str(bkg.globalback)+" rms = "+str(bkg.globalrms), __name__)
        thresh = sepDetectThresh*bkg.globalrms #Default = 3
        #subtract background from data
        bkg.subfrom(currData)
        #extract objects
        currObjects = sep.extract(currData, thresh, minarea=9)
        if True:
            print("Sep extract "+str(j)+":",time.time()-tt,"; Total: ",time.time()-t)
        tt = time.time()
        keep = (currObjects['x'] > border)*(currObjects['x'] < currData.shape[1]-border)*(currObjects['y'] > border)*(currObjects['y'] < currData.shape[0]-border)
        #throw away objects at edges
        currObjects = currObjects[keep]
        #sort by flux
        currObjects = currObjects[currObjects['flux'].argsort()[::-1]]
        print("\tsep extracted "+str(len(currObjects))+" objects using thresh = "+str(sepDetectThresh)+"*rms")
        write_fatboy_log(log, logtype, "sep extracted "+str(len(currObjects))+" objects using thresh = "+str(sepDetectThresh)+"*rms", __name__)

        curr_x = currObjects['x']
        curr_y = currObjects['y']
        if (max_stars is not None):
            curr_x = curr_x[:max_stars]
            curr_y = curr_y[:max_stars]

        if (method == METHOD_DELAUNAY):
            curr_triangles = generate_triangles_delaunay(curr_x, curr_y, name=plotdir+"/"+currName, doplots=doplots)
        else:
            curr_triangles = generate_triangles_vectorized_arrays(curr_x, curr_y, name=plotdir+"/"+currName, doplots=doplots, min_angle=min_angle, max_angle=max_angle)

        if (mode == MODE_FDU or mode == MODE_FDU_TAG):
            frames[j].setProperty("triangles", curr_triangles)

        all_triangles.append(curr_triangles)

        #if (_verbosity == fatboyLog.VERBOSE):
        if True:
            print("Find triangles "+str(j)+":",time.time()-tt,"; Total: ",time.time()-t)
        tt = time.time()

        curr_shifts = []
        for r in range(len(ref_triangles)):
           for i in range(len(curr_triangles)):
               z = ref_triangles[r].compareTo(curr_triangles[i],atol=atol,rtol=rtol)
               if z is not False:
                   curr_shifts.append(z)
                   break
        if len(curr_shifts) == 0:
            print("triregister> ERROR: Could not match any triangles from reference "+refName+" to "+currName)
            write_fatboy_log(log, logtype, "ERROR: Could not match any triangles from reference "+refName+" to "+currName, __name__)
            curr_shifts = [[0,0]]
        curr_shifts = np.array(curr_shifts).T
        xdiff = curr_shifts[0]
        ydiff = curr_shifts[1]
        if (sigma_clipping):
            nx = len(xdiff)
            ny = len(ydiff)
            xdiff = removeOutliersSigmaClip(xdiff, sig_to_clip, 5)
            ydiff = removeOutliersSigmaClip(ydiff, sig_to_clip, 5)
            print ("triregister> Sigma clipping: kept "+str(len(xdiff))+" of "+str(nx)+" X datapoints, "+str(len(ydiff))+" of "+str(ny)+" Y datapoints.")
            write_fatboy_log(log, logtype, "Sigma clipping: kept "+str(len(xdiff))+" of "+str(nx)+" X datapoints, "+str(len(ydiff))+" of "+str(ny)+" Y datapoints.", __name__)
        xshift = round(xdiff.mean(), 3)
        yshift = round(ydiff.mean(), 3)
        xsd = round(xdiff.std(), 3)
        ysd = round(ydiff.std(), 3)
        print("triregister> Used "+str(len(xdiff))+" matching triangles.  xshift = "+str(xshift)+" +/- "+str(xsd)+"; yshift = "+str(yshift)+" +/- "+str(ysd))
        write_fatboy_log(log, logtype, "Used "+str(len(xdiff))+" matching triangles.  xshift = "+str(xshift)+" +/- "+str(xsd)+"; yshift = "+str(yshift)+" +/- "+str(ysd), __name__)

        #if (_verbosity == fatboyLog.VERBOSE):
        if True:
            print("Match "+str(j)+":",time.time()-tt,"; Total: ",time.time()-t)
        tt = time.time()

        print("Shift from "+refName+" to "+currName+" is ("+str(xshift)+", "+str(yshift)+").")
        write_fatboy_log(log, logtype, "Shift from "+refName+" to "+currName+" is ("+str(xshift)+", "+str(yshift)+").", __name__)
        if (outfile is not None):
            f.write(str(xshift)+'\t'+str(yshift)+'\n')
        xshifts.append(xshift)
        yshifts.append(yshift)

        #GUI message:
        if (gui is not None):
            gui = (gui[0], gui[1]+1., gui[2], gui[3], gui[4])
            if (gui[0]): print("PROGRESS: "+str(int(gui[3]+gui[1]/gui[2]*gui[4])))

    if (outfile is not None):
        f.close()

    #if (_verbosity == fatboyLog.VERBOSE):
    if True:
        print("Cleanup "+str(j)+":",time.time()-tt,"; Total: ",time.time()-t)
    print("Tri-registered "+str(nframes)+" frames. Total time (s): "+str(time.time()-t))
    write_fatboy_log(log, logtype, "Tri-registered "+str(nframes)+" frames. Total time (s): "+str(time.time()-t), __name__)
    return [xshifts, yshifts]
