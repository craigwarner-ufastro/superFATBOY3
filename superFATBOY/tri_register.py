import numpy as np
hasSep = True
try:
    import sep
except Exception:
    print("Warning: sep not installed")
    hasSep = False

import scipy, os, time, math
from scipy.spatial import cKDTree
from scipy.stats import binom
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


#Match every reference triangle to the first current triangle that agrees (translation only).
#Returns an (n, 2) array of x, y shifts (reference - current), n = 0 if nothing matched.
#Legacy matching: every reference triangle is matched to the FIRST current triangle that agrees (translation only).
#Returns an (n, 2) array of x, y shifts (reference - current), n = 0 if nothing matched.
def matchTriangles(ref_triangles, curr_triangles, atol=2.0, rtol=0.025):
    curr_shifts = []
    for r in range(len(ref_triangles)):
        for i in range(len(curr_triangles)):
            z = ref_triangles[r].compareTo(curr_triangles[i], atol=atol, rtol=rtol)
            if z is not False:
                curr_shifts.append(z)
                break
    return np.array(curr_shifts).reshape(-1, 2)
#end matchTriangles

#All agreeing (reference, current) triangle pairs, not just the first per reference triangle.
def allTriangleMatches(ref_triangles, curr_triangles, atol=2.0, rtol=0.025):
    shifts = []
    for r in range(len(ref_triangles)):
        for i in range(len(curr_triangles)):
            z = ref_triangles[r].compareTo(curr_triangles[i], atol=atol, rtol=rtol)
            if z is not False:
                shifts.append(z)
    return np.array(shifts).reshape(-1, 2)
#end allTriangleMatches

#The original estimate: sigma-clipped mean of the first-match shifts.  Returns (xshift, yshift, ntriangles, scatter) or None.
def legacyShift(ref_triangles, curr_triangles, atol, rtol, sigma_clipping, sig_to_clip):
    curr_shifts = matchTriangles(ref_triangles, curr_triangles, atol=atol, rtol=rtol)
    if (len(curr_shifts) == 0):
        return None
    xdiff = curr_shifts[:,0]
    ydiff = curr_shifts[:,1]
    if (sigma_clipping):
        xdiff = removeOutliersSigmaClip(xdiff, sig_to_clip, 5)
        ydiff = removeOutliersSigmaClip(ydiff, sig_to_clip, 5)
    return (float(np.round(xdiff.mean(), 3)), float(np.round(ydiff.mean(), 3)), len(xdiff), float(max(xdiff.std(), ydiff.std())))
#end legacyShift

#Densest group of triangle shifts within tol of each other: (center, votes).
def clusterShifts(shifts, tol):
    if (len(shifts) == 0):
        return (None, 0)
    d = np.hypot(shifts[:,0][:,None]-shifts[:,0][None,:], shifts[:,1][:,None]-shifts[:,1][None,:])
    cnt = (d <= tol).sum(1)
    c = shifts[d[cnt.argmax()] <= tol].mean(0)
    for it in range(3):
        m = np.hypot(shifts[:,0]-c[0], shifts[:,1]-c[1]) <= tol
        if (m.sum() == 0):
            break
        c = shifts[m].mean(0)
    m = np.hypot(shifts[:,0]-c[0], shifts[:,1]-c[1]) <= tol
    return (c, int(m.sum()))
#end clusterShifts

#Successive densest clusters of triangle shifts: [(center, votes)] by decreasing votes.
def triangleClusters(shifts, tol, maxn=6):
    pts = shifts.copy()
    out = []
    while (len(pts) > 0 and len(out) < maxn):
        (c, v) = clusterShifts(pts, tol)
        if (c is None or v == 0):
            break
        out.append((c, v))
        pts = pts[np.hypot(pts[:,0]-c[0], pts[:,1]-c[1]) > tol]
    return out
#end triangleClusters

#Mutual nearest neighbour star matching for the translation s (current = reference - s), refined by iterating.
#Returns (refined shift, number of matched stars).
def matchStars(ref, cur, s, r):
    if (len(ref) == 0 or len(cur) == 0):
        return (s, 0)
    ct = cKDTree(cur)
    for it in range(3):
        pred = ref - s
        (d, i) = ct.query(pred)
        (d2, j) = cKDTree(pred).query(cur)
        ok = (d <= r) & (j[i] == np.arange(len(ref)))
        n = int(ok.sum())
        if (n == 0):
            return (s, 0)
        off = ref[ok] - cur[i[ok]]
        if (n < 3):
            s = np.array([np.median(off[:,0]), np.median(off[:,1])])
        else:
            s = off.mean(0)
    pred = ref - s
    (d, i) = ct.query(pred)
    (d2, j) = cKDTree(pred).query(cur)
    ok = (d <= r) & (j[i] == np.arange(len(ref)))
    return (s, int(ok.sum()))
#end matchStars

#Coarse to fine star matching (the plate scale and rotation of the field make the best translation depend on radius at large dithers).
#Returns (shift, matched stars, rms scatter of the matched offsets in pixels).
def refineShift(ref, cur, s0, radii=(5.0, 3.5, 2.5)):
    s = np.asarray(s0, float)
    for r in radii:
        (s, k) = matchStars(ref, cur, s, r)
        if (k == 0):
            return (s, 0, 0.)
    ct = cKDTree(cur)
    pred = ref - s
    (d, i) = ct.query(pred)
    (d2, j) = cKDTree(pred).query(cur)
    ok = (d <= radii[-1]) & (j[i] == np.arange(len(ref)))
    off = ref[ok] - cur[i[ok]]
    rms = 0.
    if (len(off) > 1):
        rms = float(np.sqrt(((off-off.mean(0))**2).sum(1).mean()))
    return (s, int(ok.sum()), rms)
#end refineShift

#How unlikely is it that k stars coincide by chance?  Returns -log10 of the probability, including a look-elsewhere factor for
#the number of distinguishable translations.  ref, cur are (n, 2) star positions; shapes are (ny, nx) of the frames.
def matchSignificance(ref, cur, s, k, shape_ref, shape_cur, r):
    (Hr, Wr) = shape_ref
    (Hc, Wc) = shape_cur
    x0 = max(0, s[0])
    x1 = min(Wr, s[0]+Wc)
    y0 = max(0, s[1])
    y1 = min(Hr, s[1]+Hc)
    if (x1 <= x0 or y1 <= y0):
        return 0.
    area = (x1-x0)*(y1-y0)
    nref = int(((ref[:,0] >= x0) & (ref[:,0] <= x1) & (ref[:,1] >= y0) & (ref[:,1] <= y1)).sum())
    c = cur + np.asarray(s)
    ncur = int(((c[:,0] >= x0) & (c[:,0] <= x1) & (c[:,1] >= y0) & (c[:,1] <= y1)).sum())
    if (nref == 0 or ncur == 0 or k <= 0):
        return 0.
    lam = ncur*math.pi*r*r/area
    p = 1-math.exp(-lam)
    P = float(binom.sf(k-1, nref, p))
    trials = 4.0*Wr*Hr/(math.pi*r*r)
    return float(-math.log10(max(min(P*trials, 1.0), 1e-300)))
#end matchSignificance

#Remove detections that recur at the same pixel in many other frames: detector or processing artifacts in a dithered sequence.  A real
#star only recurs in the frames of its own dither group.  If more than max_flag_frac of a frame's detections would be flagged the
#sequence is treated as undithered and that frame is left alone.  lists are (n, 2) arrays, one per frame.  Returns (lists, number removed).
def removeStationaryStars(lists, radius=0.7, min_frames=None, max_flag_frac=0.6, max_compare=60):
    n = len(lists)
    if (n < 4):
        return (lists, 0)
    rng = np.random.RandomState(12345)
    out = []
    removed = 0
    trees = [cKDTree(l) if len(l) > 0 else None for l in lists]
    for k in range(n):
        l = lists[k]
        if (len(l) == 0):
            out.append(l)
            continue
        others = [j for j in range(n) if j != k and trees[j] is not None]
        if (len(others) > max_compare):
            others = list(rng.choice(others, max_compare, replace=False))
        mf = min_frames
        if (mf is None):
            mf = max(4, int(math.ceil(0.15*len(others))))
        cnt = np.zeros(len(l), int)
        for j in others:
            (d, i) = trees[j].query(l)
            cnt += (d <= radius)
        flagged = cnt >= mf
        if (flagged.mean() > max_flag_frac):
            out.append(l)
        else:
            out.append(l[~flagged])
            removed += int(flagged.sum())
    return (out, removed)
#end removeStationaryStars

#Triangles from the max_stars brightest stars (lists are sorted by flux).  Returns [] if there are too few or degenerate points.
def starTriangles(x, y, method, max_stars, min_angle, max_angle, name=None, doplots=False):
    n = len(x)
    if (max_stars is not None):
        n = min(n, max_stars)
    if (n < 3):
        return []
    try:
        if (method == METHOD_DELAUNAY):
            return generate_triangles_delaunay(x[:n].copy(), y[:n].copy(), name=name, doplots=doplots)
        return generate_triangles_vectorized_arrays(x[:n].copy(), y[:n].copy(), name=name, doplots=doplots, min_angle=min_angle, max_angle=max_angle)
    except Exception as ex:
        return []
#end starTriangles

#Register one frame against another from their stars and triangles.  A shift is accepted only if (1) agreeing triangles vote for it
#and (2) at least min_stars stars coincide at that shift with a chance probability below 10**-min_sig (after a look-elsewhere
#correction).  Stars alone are never enough: tested on frames that cannot match (mirrored, transposed, other fields) every false
#acceptance came from star coincidences without triangle support.  Competing shifts are ranked by triangle votes; if the runner-up
#has more than half the votes the frame is declared ambiguous.  The original estimate is kept when it agrees with the verified
#shift (so good frames do not change), and replaced by the verified one when it does not.
#Returns a dict: status 'ok'/'fail', shift, source, nstars, sig, votes, rms, notes, reason, legacy.
def registerPair(ref, cur, ref_tris, cur_tris, shape_ref, shape_cur, atol=2.0, rtol=0.025, r=2.5, min_sig=6.0, min_stars=3, sigma_clipping=False, sig_to_clip=3, agree=1.5, merge=3.0, ratio=2.0, max_verify=3000):
    leg = legacyShift(ref_tris, cur_tris, atol, rtol, sigma_clipping, sig_to_clip)
    shifts = allTriangleMatches(ref_tris, cur_tris, atol=atol, rtol=rtol)
    if (len(shifts) == 0):
        return dict(status='fail', legacy=leg, reason='no matching triangles')
    rv = ref[:max_verify]
    cv = cur[:max_verify]
    hyps = []
    for (c, votes) in triangleClusters(shifts, atol):
        (s, k, rms) = refineShift(rv, cv, c)
        if (k < min_stars):
            continue
        sg = matchSignificance(rv, cv, s, k, shape_ref, shape_cur, r)
        if (sg < min_sig):
            continue
        for h in hyps:
            if (np.hypot(*(s-h['s'])) <= merge):
                h['votes'] += votes
                if (sg > h['sig']):
                    h.update(s=s, k=k, sig=sg, rms=rms)
                break
        else:
            hyps.append(dict(s=s, k=k, sig=sg, votes=votes, rms=rms))
    if (len(hyps) == 0):
        return dict(status='fail', legacy=leg, reason='triangle matches not confirmed by at least '+str(min_stars)+' coinciding stars (significance >= '+str(min_sig)+')')
    hyps.sort(key=lambda h: (-h['votes'], -h['sig']))
    best = hyps[0]
    notes = []
    if (len(hyps) > 1):
        o = hyps[1]
        if (o['votes']*ratio > best['votes']):
            return dict(status='fail', legacy=leg, reason='ambiguous: '+str(best['votes'])+' triangle votes at ('+str(round(best['s'][0],1))+', '+str(round(best['s'][1],1))+') vs '+str(o['votes'])+' at ('+str(round(o['s'][0],1))+', '+str(round(o['s'][1],1))+')')
        notes.append('a second, weaker shift ('+str(round(o['s'][0],1))+', '+str(round(o['s'][1],1))+') is also supported by '+str(o['k'])+' stars; the stronger one was used')
    if (best['rms'] > 2.0):
        notes.append('matched stars scatter by '+str(round(best['rms'],1))+' px (field distortion or rotation?)')
    if (best['sig'] < 10):
        notes.append('weak confirmation: '+str(best['k'])+' stars, significance '+str(round(best['sig'],1)))
    final = best['s']
    source = 'verified'
    if (leg is not None and np.hypot(leg[0]-best['s'][0], leg[1]-best['s'][1]) <= agree):
        final = np.array(leg[:2])
        source = 'original'
    return dict(status='ok', shift=final, source=source, nstars=best['k'], sig=best['sig'], votes=best['votes'], rms=best['rms'], notes=notes, legacy=leg)
#end registerPair

#Rescue frames that could not be registered against the reference by matching them against other frames that already have a shift,
#nearest in the sequence first, and composing the shifts: shift(ref -> j) = shift(ref -> c) + shift(c -> j).  Needed for large dithers
#over a sparse field where frames far from the reference share almost no stars with it.  Uses the same verified matching as above.
def chainRescue(refframe, xshifts, yshifts, pos_of, name_of, stars_of, tri_of, shape_of, log, logtype, pair_kw, max_candidates):
    pending = [j for j in pos_of if np.isnan(xshifts[pos_of[j]])]
    if (len(pending) == 0):
        return
    print("triregister> Chaining: trying to rescue "+str(len(pending))+" frame(s) via overlapping frames.")
    write_fatboy_log(log, logtype, "Chaining: trying to rescue "+str(len(pending))+" frame(s) via overlapping frames.", __name__)
    def haveShift(c):
        return c == refframe or not np.isnan(xshifts[pos_of[c]])
    def getShift(c):
        if (c == refframe):
            return (0., 0.)
        return (xshifts[pos_of[c]], yshifts[pos_of[c]])
    progress = True
    while (progress and len(pending) > 0):
        progress = False
        for j in sorted(pending):
            anchors = [c for c in stars_of if c != j and c != refframe and haveShift(c)]
            anchors.sort(key=lambda c: (abs(c-j), c))
            for c in anchors[:max_candidates]:
                res = registerPair(stars_of[c], stars_of[j], tri_of[c], tri_of[j], shape_of[c], shape_of[j], **pair_kw)
                if (res['status'] != 'ok'):
                    continue
                (cx, cy) = getShift(c)
                xshifts[pos_of[j]] = np.round(cx+res['shift'][0], 3)
                yshifts[pos_of[j]] = np.round(cy+res['shift'][1], 3)
                msg = "Frame "+name_of[j]+" registered via "+name_of[c]+" ("+str(res['votes'])+" triangle vote(s), "+str(res['nstars'])+" matched stars, significance "+str(round(res['sig'],1))+"); shift from reference = ("+str(xshifts[pos_of[j]])+", "+str(yshifts[pos_of[j]])+")."
                print("triregister> "+msg)
                write_fatboy_log(log, logtype, msg, __name__)
                for note in res['notes']:
                    print("triregister> WARNING: "+name_of[j]+" via "+name_of[c]+": "+note)
                    write_fatboy_log(log, logtype, name_of[j]+" via "+name_of[c]+": "+note, __name__, messageType=fatboyLog.WARNING)
                pending.remove(j)
                progress = True
                break
#end chainRescue

def tri_register(frames, outfile=None, xcenter=-1, ycenter=-1, xboxsize=-1, yboxsize=-1, border=20, log=None, mef=0, gui=None, refframe=0, mode=None, dataTag=None, sepDetectThresh=3, method=METHOD_DELAUNAY, min_angle=30, max_angle=110, max_stars=None, doplots=False, plotdir=".", atol=2.0, rtol=0.025, sigma_clipping=False, sig_to_clip=3, chain_overlapping_frames=False, chain_max_candidates=10, verify=True, min_stars=3, min_significance=6.0, match_radius=2.5, remove_stationary=True):
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
        elif (isinstance(frames[0], np.ndarray)):
            mode = MODE_RAW
        elif (isinstance(frames[0], fatboyDataUnit)):
            mode = MODE_FDU

    filelist = None
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
    if (mode == MODE_FDU_DIFFERENCE):
        #2-1, 3-2, etc.
        nframes -= 1

    #Get the data and name of frame j
    def getFrame(j):
        if (mode == MODE_FITS):
            if (os.access(filelist[j], os.F_OK)):
                temp = pyfits.open(filelist[j])
                data = temp[mef].data
                if (not data.dtype.isnative):
                    print("triregister> Byteswapping "+filelist[j])
                    data = data.astype(np.float32)
                name = filelist[j]
                temp.close()
                return (data, name)
            print("triregister> Could not find file "+filelist[j])
            write_fatboy_log(log, logtype, "Could not find file "+filelist[j], __name__)
            return (None, None)
        elif (mode == MODE_RAW):
            return (frames[j], "index "+str(j))
        elif (mode == MODE_FDU):
            return (frames[j].getData(), frames[j].getFullId())
        elif (mode == MODE_FDU_DIFFERENCE):
            return (frames[j+1].getData()-frames[j].getData(), frames[j+1].getFullId()+"-"+frames[j].getFullId())
        elif (mode == MODE_FDU_TAG):
            return (frames[j].getData(tag=dataTag), frames[j].getFullId()+":"+dataTag)
        print("triregister> Invalid input!  Exiting!")
        write_fatboy_log(log, logtype, "Invalid input!  Exiting!", __name__)
        return (None, None)

    #Pass 1: find the stars in every frame (brightest first, away from the edges)
    star_x = {}
    star_y = {}
    names = {}
    shapes = {}
    box = None
    for j in range(nframes):
        tt = time.time()
        (data, name) = getFrame(j)
        if (data is None):
            return None
        names[j] = name
        if (box is None):
            shp = data.shape
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
            x1 = max(xcenter-xboxsize//2, 0)
            x2 = min(xcenter+xboxsize//2, shp[1])
            y1 = max(ycenter-yboxsize//2, 0)
            y2 = min(ycenter+yboxsize//2, shp[0])
            box = (x1, x2, y1, y2)
        (x1, x2, y1, y2) = box
        #Work on a private, contiguous float copy: sep subtracts its background map in place, which used to modify the
        #caller's frames (and so the stacked science image), and it cannot take a non-contiguous crop.
        if (hasattr(data, 'get')):
            data = data.get()
        data = np.array(data[y1:y2, x1:x2], dtype=(data.dtype if data.dtype in (np.float32, np.float64) else np.float32), order='C')
        shapes[j] = data.shape
        bkg = sep.Background(data)
        print("\tsep background = "+str(bkg.globalback)+" rms = "+str(bkg.globalrms))
        write_fatboy_log(log, logtype, "sep background = "+str(bkg.globalback)+" rms = "+str(bkg.globalrms), __name__)
        thresh = sepDetectThresh*bkg.globalrms #Default = 3
        #subtract background from data
        bkg.subfrom(data)
        #extract objects
        objects = sep.extract(data, thresh, minarea=9)
        keep = (objects['x'] > border)*(objects['x'] < data.shape[1]-border)*(objects['y'] > border)*(objects['y'] < data.shape[0]-border)
        #throw away objects at edges
        objects = objects[keep]
        #sort by flux
        objects = objects[objects['flux'].argsort()[::-1]]
        print("\tsep extracted "+str(len(objects))+" objects using thresh = "+str(sepDetectThresh)+"*rms")
        write_fatboy_log(log, logtype, "sep extracted "+str(len(objects))+" objects using thresh = "+str(sepDetectThresh)+"*rms", __name__)
        star_x[j] = np.array(objects['x'], dtype=np.float64)
        star_y[j] = np.array(objects['y'], dtype=np.float64)
        if (_verbosity == fatboyLog.VERBOSE):
            print("Sep extract "+str(j)+":",time.time()-tt,"; Total: ",time.time()-t)

    #Remove artifacts that sit at the same pixel in many frames of a dithered sequence
    if (verify and remove_stationary):
        lists = [np.column_stack((star_x[j], star_y[j])) for j in range(nframes)]
        (lists, nremoved) = removeStationaryStars(lists)
        if (nremoved > 0):
            msg = "Removed "+str(nremoved)+" detections that recur at the same pixel in many frames (detector or sky-model artifacts) before matching."
            print("triregister> "+msg)
            write_fatboy_log(log, logtype, msg, __name__)
        for j in range(nframes):
            star_x[j] = lists[j][:,0].copy()
            star_y[j] = lists[j][:,1].copy()

    #Triangles for every frame
    tri_of = {}
    stars_of = {}
    for j in range(nframes):
        tt = time.time()
        stars_of[j] = np.column_stack((star_x[j], star_y[j]))
        tri_of[j] = starTriangles(star_x[j], star_y[j], method, max_stars, min_angle, max_angle, name=plotdir+"/"+names[j], doplots=doplots)
        if (mode == MODE_FDU or mode == MODE_FDU_TAG):
            frames[j].setProperty("triangles", tri_of[j])
    print("Initialize: ",time.time()-t)

    xshifts = [0]
    yshifts = [0]
    pos_of = {}
    pair_kw = dict(atol=atol, rtol=rtol, r=match_radius, min_sig=min_significance, min_stars=min_stars, sigma_clipping=sigma_clipping, sig_to_clip=sig_to_clip)
    refName = names[refframe]
    for j in range(nframes):
        if (j == refframe):
            continue
        tt = time.time()
        currName = names[j]
        if (verify):
            res = registerPair(stars_of[refframe], stars_of[j], tri_of[refframe], tri_of[j], shapes[refframe], shapes[j], **pair_kw)
        else:
            leg = legacyShift(tri_of[refframe], tri_of[j], atol, rtol, sigma_clipping, sig_to_clip)
            if (leg is None):
                res = dict(status='fail', legacy=None, reason='no matching triangles')
            else:
                res = dict(status='ok', shift=np.array(leg[:2]), source='original', nstars=0, sig=0., votes=leg[2], rms=0., notes=[], legacy=leg)
        pos_of[j] = len(xshifts)
        if (res['status'] != 'ok'):
            print("triregister> ERROR: Could not match "+currName+" to reference "+refName+": "+res['reason'])
            write_fatboy_log(log, logtype, "ERROR: Could not match "+currName+" to reference "+refName+": "+res['reason'], __name__, messageType=fatboyLog.ERROR)
            xshifts.append(np.nan)
            yshifts.append(np.nan)
        else:
            xshift = float(np.round(res['shift'][0], 3))
            yshift = float(np.round(res['shift'][1], 3))
            leg = res['legacy']
            sd = ""
            if (leg is not None):
                sd = " +/- "+str(np.round(leg[3], 3))
            msg = "Used "+str(res['votes'])+" matching triangles.  Shift = ("+str(xshift)+", "+str(yshift)+")"+sd
            if (verify):
                msg += "; confirmed by "+str(res['nstars'])+" coinciding stars (significance "+str(round(res['sig'],1))+", "+res['source']+" estimate)."
            print("triregister> "+msg)
            write_fatboy_log(log, logtype, msg, __name__)
            for note in res['notes']:
                print("triregister> WARNING: "+currName+": "+note)
                write_fatboy_log(log, logtype, currName+": "+note, __name__, messageType=fatboyLog.WARNING)
            print("Shift from "+refName+" to "+currName+" is ("+str(xshift)+", "+str(yshift)+").")
            write_fatboy_log(log, logtype, "Shift from "+refName+" to "+currName+" is ("+str(xshift)+", "+str(yshift)+").", __name__)
            xshifts.append(xshift)
            yshifts.append(yshift)
        if (_verbosity == fatboyLog.VERBOSE):
            print("Match "+str(j)+":",time.time()-tt,"; Total: ",time.time()-t)
        #GUI message:
        if (gui is not None):
            gui = (gui[0], gui[1]+1., gui[2], gui[3], gui[4])
            if (gui[0]): print("PROGRESS: "+str(int(gui[3]+gui[1]/gui[2]*gui[4])))

    if (chain_overlapping_frames and verify):
        chainRescue(refframe, xshifts, yshifts, pos_of, names, stars_of, tri_of, shapes, log, logtype, pair_kw, chain_max_candidates)

    #Frames that could not be registered at all: report them together (alignStack discards them)
    nfail = int(np.sum(np.isnan(xshifts)))
    if (nfail > 0):
        failed = [names[j] for j in sorted(pos_of, key=lambda k: pos_of[k]) if np.isnan(xshifts[pos_of[j]])]
        msg = str(nfail)+" of "+str(nframes)+" frames could NOT be registered and will be discarded: "+", ".join(failed)
        print("triregister> ERROR: "+msg)
        write_fatboy_log(log, logtype, msg, __name__, messageType=fatboyLog.ERROR)

    if (outfile is not None):
        f = open(outfile,'w')
        for k in range(1, len(xshifts)):
            f.write(str(xshifts[k])+'\t'+str(yshifts[k])+'\n')
        f.close()

    print("Tri-registered "+str(nframes)+" frames. Total time (s): "+str(time.time()-t))
    write_fatboy_log(log, logtype, "Tri-registered "+str(nframes)+" frames. Total time (s): "+str(time.time()-t), __name__)
    return [xshifts, yshifts]
