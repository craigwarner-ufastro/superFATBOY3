#Line-identification vote: a starting wavelength solution for a 1-d arc or sky cut that needs neither a good
#wavelength_scale_guess nor reliable line intensities (wavecal_initial_method = vote, or the vote fallback of
#wavelengthCalibrateProcess).
#
#1. Emission peaks of the cut (positions only).
#2. The cut is split into windows (several splittings, see PASSES).  In each window every (peak, line-list line) pair votes,
#   for each scale on a log grid, for the wavelength at the window centre (a Hough transform, as RASCAL does).  A cell is
#   scored by how improbable its number of distinct matched peaks is by chance, given how many list lines fall per pixel
#   there (binomial tail): a raw count would favor too large a scale, where a dense list puts a line near every peak.
#3. The best cells of neighboring windows are chained (dynamic programming) when they are consistent with one smooth
#   solution: lambda_B - lambda_A = (b_A+b_B)/2 (x_B-x_A), exact for a quadratic.  Chain scores are sums of log-probabilities.
#4. Each chain's matched pairs give a polynomial, refined by matching every peak to its nearest line.
#5. The solution's significance: the binomial chance of matching that many of the peaks within tol px, for the fraction of
#   the wavelength range within tol px of a list line, times the number of independent (scale, zero point) hypotheses
#   searched (look-elsewhere correction, as matchSignificance in tri_register.py).  S = -log10 of that.
#If the configured guess is a polynomial, its shape (not its scale) can be used to straighten the cut first: pixel x is
#replaced by P(x)/c1, so the solution is closer to linear in each window.
import math
import numpy as np
from scipy.special import betainc

#(tolerance in px, number of windows) of the Hough passes
PASSES = ((1.5, 1), (1.5, 2), (1.5, 3), (1.5, 4), (1.5, 6))
#tolerances (px) at which a solution's significance is computed; the best is used
SIG_TOLS = (0.3, 0.5, 1.0)

def polyval(c, x):
    return sum(c[i]*np.asarray(x, dtype=np.float64)**i for i in range(len(c)))

def polyderiv(c, x):
    x = np.asarray(x, dtype=np.float64)
    return sum(i*c[i]*x**(i-1) for i in range(1, len(c)))

#-log10 P(X >= k) for X ~ Binomial(n, p), vectorized
def binomialTail(k, n, p):
    p = np.clip(p, 1.e-9, 1-1.e-9)
    k = np.asarray(k, dtype=np.float64)
    v = np.where(k <= 0, 1.0, betainc(np.maximum(k, 1), np.maximum(n-k+1, 1.e-9), p))
    return -np.log10(np.maximum(v, 1.e-300))

#Emission peaks above nsig sigma (MAD) of the cut, at least 3 px apart, the nmax brightest, centroided with a 3-point
#parabola; sorted by position.  A cut that is mostly one fill value (MAD 0) uses the MAD of the other values.
def findPeaks(oned, nmax=60, nsig=3.0):
    y = np.asarray(oned, dtype=np.float64)
    nz = y[y != 0]
    if (len(nz) < 50):
        return np.array([])
    med = np.median(nz)
    scale = max(np.abs(nz).max(), 1.e-30)
    noise = 1.4826*np.median(np.abs(nz-med))
    if (noise <= 1.e-3*scale):
        dif = nz[np.abs(nz-med) > 1.e-3*scale]
        if (len(dif) > 10):
            noise = 1.4826*np.median(np.abs(dif-np.median(dif)))
        else:
            noise = nz.std()
    cand = np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]) & (y[1:-1] > med+nsig*noise))[0]+1
    cand = cand[np.argsort(y[cand])[::-1]]
    keep = []
    for c in cand:
        if (all(abs(c-k) >= 3 for k in keep)):
            keep.append(c)
        if (len(keep) >= nmax):
            break
    if (len(keep) == 0):
        return np.array([])
    keep = np.sort(np.array(keep, dtype=int))
    (a, b, c) = (y[keep-1], y[keep], y[keep+1])
    den = a-2*b+c
    shift = np.where(den != 0, 0.5*(a-c)/np.where(den != 0, den, 1), 0)
    return keep+np.clip(shift, -0.5, 0.5)

#Nearest value of sorted lines to each of lam
def nearestLine(lines, lam):
    j = np.clip(np.searchsorted(lines, lam), 1, len(lines)-1)
    a = lines[j-1]
    b = lines[j]
    return np.where(np.abs(a-lam) <= np.abs(b-lam), a, b)

#Fraction of [lo, hi] within width of a line
def lineCoverage(lines, lo, hi, width):
    near = lines[(lines > lo-width) & (lines < hi+width)]
    if (len(near) == 0 or hi <= lo):
        return 0.0
    covered = 0.0
    last = lo
    for x in near:
        a = max(x-width, last)
        b = min(x+width, hi)
        if (b > a):
            covered += b-a
            last = b
    return covered/(hi-lo)

#Peaks within tol px of a line under solution c (one peak per line).  Returns (mask, nearest lines).
def matchPeaks(c, xp, lines, tol):
    lam = polyval(c, xp)
    d = np.abs(polyderiv(c, xp))
    nl = nearestLine(lines, lam)
    ok = np.abs(nl-lam)/np.maximum(d, 1.e-12) < tol
    if (ok.sum() > 0):
        (u, ix) = np.unique(nl[ok], return_index=True)
        keep = np.zeros(len(xp), dtype=bool)
        keep[np.where(ok)[0][ix]] = True
        ok = keep
    return (ok, nl)

#Polynomial order for n points: linear below 7, quadratic below 12, then cubic (up to maxOrder)
def fitOrder(n, maxOrder):
    if (n >= 12):
        return min(3, maxOrder)
    if (n >= 7):
        return min(2, maxOrder)
    return 1

#Polynomial fit (power coefficients) with iterative 3-sigma (MAD) clipping
def robustPolyfit(x, w, order):
    use = np.ones(len(x), dtype=bool)
    for it in range(4):
        if (use.sum() < order+2):
            order = max(1, int(use.sum())-2)
        c = np.polynomial.polynomial.polyfit(x[use], w[use], order)
        r = w-polyval(c, x)
        s = 1.4826*np.median(np.abs(r[use]))
        nu = np.abs(r) < max(3*s, 1.e-9)
        if (nu.sum() < order+2 or (nu == use).all()):
            break
        use = nu
    return c

#Hough cells of one window: for each scale s of a log grid (fine enough that the window ends move tol/2 px per step), the
#pairs vote for the wavelength at the window centre wc (between lcmin and lcmax) in bins of tol px, on two grids offset by
#half a bin.  Each cell is scored by binomialTail of its number of distinct peaks.  Returns the ktop best distinct cells as
#(score, signed scale, wavelength at wc, wc).
def windowCells(xp, lines, smin, smax, lcmin, lcmax, wc, width, sign, tol, ktop):
    xi = xp[np.abs(xp-wc) <= width/2]
    n = len(xi)
    if (n < 3):
        return []
    dl = 0.5*tol/(width/2)
    scales = np.exp(np.arange(math.log(smin), math.log(smax)+dl, dl))
    #score by (count, chance probability rounded up to the next 1/NP)
    NP = 400
    table = binomialTail(np.arange(n+1)[:,None], n, (np.arange(NP)[None,:]+1.0)/NP)
    found = []
    for s in scales:
        b = sign*s
        lo = lcmin+b*(xi-wc)
        hi = lcmax+b*(xi-wc)
        (lo, hi) = (np.minimum(lo, hi), np.maximum(lo, hi))
        j0 = np.searchsorted(lines, lo)
        j1 = np.searchsorted(lines, hi)
        cnt = j1-j0
        tot = int(cnt.sum())
        if (tot == 0):
            continue
        I = np.repeat(np.arange(n), cnt)
        J = np.arange(tot)-np.repeat(np.cumsum(cnt)-cnt, cnt)+np.repeat(j0, cnt)
        lam = lines[J]-b*(xi[I]-wc)
        bw = tol*s
        ncell = int((lcmax-lcmin)/bw)+3
        for off in (0.0, 0.5):
            key = np.floor((lam-lcmin)/bw+off).astype(np.int64)
            #keys increase within each peak's run: drop repeats so a peak counts once per cell
            keep = np.ones(len(key), dtype=bool)
            keep[1:] = (key[1:] != key[:-1]) | (I[1:] != I[:-1])
            counts = np.bincount(key[keep], minlength=ncell)
            cells = np.where(counts >= 3)[0]
            if (len(cells) == 0):
                continue
            counts = counts[cells]
            lc = lcmin+(cells+0.5-off)*bw
            #chance a peak lands in a cell: list lines per px over the window's wavelength range times the bin width
            half = 0.5*width*s
            nl = np.searchsorted(lines, lc+half)-np.searchsorted(lines, lc-half)
            pbin = np.minimum((nl*tol/width*NP).astype(np.int64), NP-1)
            score = table[counts, pbin]
            if (len(score) > ktop):
                top = np.argpartition(score, -ktop)[-ktop:]
            else:
                top = np.arange(len(score))
            found.append((score[top], np.full(len(top), b), lc[top]))
    if (len(found) == 0):
        return []
    SC = np.concatenate([f[0] for f in found])
    B = np.concatenate([f[1] for f in found])
    LC = np.concatenate([f[2] for f in found])
    o = np.argsort(SC)[::-1]
    #near-duplicates (within 1% in scale and 1.5 px in wavelength): keep the best
    kb = np.round(np.log(np.abs(B[o]))/0.01).astype(np.int64)
    kl = np.round(LC[o]/(1.5*np.abs(B[o]))).astype(np.int64)
    (u, first) = np.unique(kb*(1 << 32)+kl, return_index=True)
    o = o[np.sort(first)][:ktop]
    return [(float(SC[t]), float(B[t]), float(LC[t]), wc) for t in o]

#Chain cells of consecutive windows (a window may be skipped) that fit one smooth solution: lambda_B - lambda_A =
#(b_A+b_B)/2 (x_B-x_A) within maxmis px per step and scales within a factor maxratio per step.  Dynamic programming on the
#summed scores.  Returns the nbest chains, best first, as (score, [cells]).
def chainCells(cellsByWin, wcs, maxmis=2.0, maxratio=1.5, nbest=40):
    nw = len(wcs)
    score = [np.array([c[0] for c in C], dtype=np.float64) for C in cellsByWin]
    back = [[None]*len(C) for C in cellsByWin]
    for i in range(nw):
        if (len(cellsByWin[i]) == 0):
            continue
        bB = np.array([c[1] for c in cellsByWin[i]])
        lB = np.array([c[2] for c in cellsByWin[i]])
        base = np.array([c[0] for c in cellsByWin[i]], dtype=np.float64)
        for gap in (1, 2):
            a = i-gap
            if (a < 0 or len(cellsByWin[a]) == 0):
                continue
            bA = np.array([c[1] for c in cellsByWin[a]])
            lA = np.array([c[2] for c in cellsByWin[a]])
            dx = wcs[i]-wcs[a]
            #predecessors with similar scale only (sorted by scale, a few blocks at a time to bound memory)
            for k0 in range(0, len(bB), 500):
                bb = bB[k0:k0+500]
                ll = lB[k0:k0+500]
                bmean = 0.5*(bA[:,None]+bb[None,:])
                mis = np.abs(ll[None,:]-(lA[:,None]+bmean*dx))/np.abs(bmean)
                rat = bb[None,:]/bA[:,None]
                ok = (mis < maxmis*gap) & (rat < maxratio**gap) & (rat > 1.0/maxratio**gap)
                cand = np.where(ok, score[a][:,None], -np.inf)
                jbest = np.argmax(cand, axis=0)
                v = cand[jbest, np.arange(len(bb))]
                for q in np.where(np.isfinite(v) & (v+base[k0:k0+500] > score[i][k0:k0+500]))[0]:
                    score[i][k0+q] = v[q]+base[k0+q]
                    back[i][k0+q] = (a, int(jbest[q]))
    ends = [(score[i][q], i, q) for i in range(nw) for q in range(len(cellsByWin[i]))]
    ends.sort(key=lambda e: -e[0])
    out = []
    used = set()
    for (sc, i, q) in ends:
        if ((i, q) in used):
            continue
        chain = []
        cur = (i, q)
        while (cur is not None):
            chain.append(cellsByWin[cur[0]][cur[1]])
            used.add(cur)
            cur = back[cur[0]][cur[1]]
        out.append((sc, chain[::-1]))
        if (len(out) >= nbest):
            break
    return out

#Polynomial (in pixels) from a chain: the pairs its cells matched (in straightened coordinates xw), then twice matching
#every peak within 2.5 px and twice within 1 px, refitting each time.  None if it falls apart or is not monotonic.
def refineChain(chain, xw, xp, lines, npix, maxOrder, width, tol):
    X = []
    L = []
    for (sc, b, lc, wc) in chain:
        sel = np.abs(xw-wc) <= width/2
        lam = lc+b*(xw[sel]-wc)
        nl = nearestLine(lines, lam)
        ok = np.abs(nl-lam) < tol*abs(b)
        X += list(xp[sel][ok])
        L += list(nl[ok])
    X = np.array(X)
    L = np.array(L)
    if (len(X) < 3):
        return None
    c = robustPolyfit(X, L, fitOrder(len(X), maxOrder))
    for it in range(4):
        t = 2.5
        if (it >= 2):
            t = 1.0
        (ok, nl) = matchPeaks(c, xp, lines, t)
        if (ok.sum() < 4):
            return None
        c = robustPolyfit(xp[ok], nl[ok], fitOrder(int(ok.sum()), maxOrder))
    d = polyderiv(c, np.linspace(0, npix-1, 50))
    if (np.any(d == 0) or np.any(np.sign(d) != np.sign(d[0]))):
        return None
    return c

#Matched peaks k of n within tol px, the chance p of a random position being within tol px of a list line, and
#-log10 of the binomial tail.  Returns (k, n, p, -log10 P).
def solutionChance(c, xp, lines, npix, tol):
    (ok, nl) = matchPeaks(c, xp, lines, tol)
    k = int(ok.sum())
    (lo, hi) = sorted(polyval(c, np.array([0, npix-1.0])))
    disp = abs(hi-lo)/(npix-1)
    p = lineCoverage(lines, lo, hi, tol*disp)
    return (k, len(xp), p, float(binomialTail(k, len(xp), max(p, 1.e-6))))

#Starting solutions for the 1-d cut oned from the line list (wavelengths), most significant first.  scaleGuess: rough
#dispersion (its sign is the direction); scaleRange: factors of it searched; window: (min, max) wavelength that the middle
#of the cut must fall in (None = the line list's range); shape: power coefficients c1, c2, ... of a polynomial guess
#whose shape straightens the cut (None = no straightening).  Each solution is a dict: coeffs (power series in pixels),
#significance S, nmatched, npeaks, chance (p), tol, ids = (peak pixels, line wavelengths) matched within 1 px.
def voteSolutions(oned, lines, scaleGuess, window=None, shape=None, scaleRange=(1/3.0, 3.0), maxOrder=3, passes=PASSES, ktop=2000, nchain=40, nmax=60, nsigs=(3.0, 2.0, 1.5), minPeaks=30):
    npix = len(oned)
    #3 sigma peaks, lower thresholds while there are fewer than minPeaks (faint lines are real lines too; the significance
    #allows for the extra noise peaks)
    for nsig in nsigs:
        xp = findPeaks(oned, nmax, nsig)
        if (len(xp) >= minPeaks):
            break
    lines = np.unique(np.asarray(lines, dtype=np.float64))
    if (len(xp) < 5 or len(lines) < 5 or scaleGuess == 0):
        return []
    sign = 1.0
    if (scaleGuess < 0):
        sign = -1.0
    smin = abs(scaleGuess)*scaleRange[0]
    smax = abs(scaleGuess)*scaleRange[1]
    if (window is None):
        window = (lines.min(), lines.max())
    (cmin, cmax) = (min(window), max(window))
    #straightened coordinates
    warp = None
    if (shape is not None and len(shape) > 1 and shape[0] != 0):
        cw = np.array([0.0]+list(shape), dtype=np.float64)/shape[0]
        if (np.all(polyderiv(cw, np.linspace(0, npix-1, 50)) > 0)):
            warp = cw
    if (warp is None):
        xw = xp.copy()
        (w0, w1, wmid) = (0.0, npix-1.0, npix/2.0)
    else:
        xw = polyval(warp, xp)
        (w0, w1, wmid) = polyval(warp, np.array([0.0, npix-1.0, npix/2.0]))
    xw = xw-w0
    wlen = w1-w0
    wmid -= w0
    sols = []
    for (tol, nwin) in passes:
        width = wlen/float(nwin)
        wcs = [width*(i+0.5) for i in range(nwin)]
        cellsByWin = []
        for wc in wcs:
            #the middle of the cut must fall in [cmin, cmax]
            d = abs(wc-wmid)*smax
            cellsByWin.append(windowCells(xw, lines, smin, smax, cmin-d, cmax+d, wc, width, sign, tol, ktop))
        for (sc, chain) in chainCells(cellsByWin, wcs, nbest=nchain):
            c = refineChain(chain, xw, xp, lines, npix, maxOrder, width, max(tol, 1.0))
            if (c is None):
                continue
            if (polyval(c, npix/2.0) < cmin or polyval(c, npix/2.0) > cmax):
                continue
            best = None
            for t in SIG_TOLS:
                r = solutionChance(c, xp, lines, npix, t)
                if (best is None or r[3] > best[3]):
                    best = r+(t,)
            sols.append((best, c))
    #independent hypotheses: scale cells (the cut's ends move 1 px) x zero-point cells (1 px), per pass and tolerance
    nscale = math.log(scaleRange[1]/scaleRange[0])/(2.0/npix)
    nzero = max((cmax-cmin)/abs(scaleGuess), 1.0)
    trials = len(SIG_TOLS)*len(passes)*nscale*nzero
    out = []
    for (best, c) in sols:
        S = best[3]-math.log10(trials)
        (ok, nl) = matchPeaks(c, xp, lines, 1.0)
        out.append({"coeffs": c, "significance": S, "nmatched": best[0], "npeaks": best[1], "chance": best[2], "tol": best[4], "ids": (xp[ok], nl[ok])})
    out.sort(key=lambda r: -r["significance"])
    #distinct solutions only (differing by more than 2 px somewhere)
    final = []
    xs = np.linspace(0, npix-1, 9)
    for r in out:
        if (all(np.abs(polyval(r["coeffs"], xs)-polyval(q["coeffs"], xs)).max() > 2*abs(scaleGuess) for q in final)):
            final.append(r)
    return final
