#!/usr/bin/env python3
## makeLineList.py - build a superFATBOY line list (wavelength, relative intensity, optional flag) for a set of
## spectra (e.g. "Ne I, Ar I, Xe I") and a wavelength range from the NIST Atomic Spectra Database.
##
## NIST relative intensities are not on a common scale between spectra (or always between sources), and are
## missing for some strong lines.  So: missing intensities can be estimated from g*A (scaled per spectrum to the lines
## that have both), each spectrum can be scaled (-s "Ar I=0.5"), and -m takes the intensities superFATBOY measured in
## your own calibrated data (wavelengthCalibrated/measured_lines_*.dat): per-spectrum scales are fit to them and,
## with --use-measured, the measured values replace NIST's.  Lines closer than --blend are flagged -1 (used for the
## template but not the fit) when the fainter one is at least --blend-ratio of the brighter.
##
## Examples:
##   makeLineList.py -e "Xe I,Xe II" -r 3400 5000 -o xenon_blue.dat
##   makeLineList.py -e "Ne I,Ar I" -r 13000 26000 --vacuum --min-intensity 50 -o NeAr_HK.dat
##   makeLineList.py -e "Xe I,Xe II" -r 3400 5000 -m wavelengthCalibrated/measured_lines_arc.dat --use-measured -o xe.dat
import argparse
import datetime
import hashlib
import math
import os
import re
import sys
import urllib.parse
import urllib.request

import numpy as np

NIST_URL = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl"

#Vacuum to air (IAU standard, Morton 2000 / Ciddor 1996), wavelengths in Angstrom; unchanged below 2000 A
def vacToAir(w):
    w = np.asarray(w, dtype=np.float64)
    s2 = (1.e4/w)**2
    n = 1+0.0000834254+0.02406147/(130-s2)+0.00015998/(38.9-s2)
    return np.where(w > 2000, w/n, w)

#Air to vacuum (inverse of vacToAir by iteration)
def airToVac(w):
    w = np.asarray(w, dtype=np.float64)
    v = w.copy()
    for i in range(4):
        v = v+(w-vacToAir(v))
    return v

#Query NIST ASD for one spectrum (e.g. "Ne I") between wlo and whi (vacuum Angstrom); cached in cachedir
def queryNist(spectrum, wlo, whi, cachedir):
    params = [("spectra", spectrum), ("limits_type", "0"), ("low_w", "%.3f" % wlo), ("upp_w", "%.3f" % whi), ("unit", "0"),
              ("de", "0"), ("format", "1"), ("line_out", "0"), ("en_unit", "0"), ("output", "0"), ("bibrefs", "1"),
              ("page_size", "15"), ("show_obs_wl", "1"), ("show_calc_wl", "1"), ("order_out", "0"), ("show_av", "3"),
              ("tsb_value", "0"), ("A_out", "0"), ("intens_out", "on"), ("allowed_out", "1"), ("forbid_out", "1"),
              ("conf_out", "on"), ("term_out", "on"), ("enrg_out", "on"), ("J_out", "on"), ("remove_js", "on"),
              ("unc_out", "1"), ("submit", "Retrieve Data")]
    url = NIST_URL+"?"+urllib.parse.urlencode(params)
    cfile = None
    if (cachedir is not None):
        os.makedirs(cachedir, exist_ok=True)
        cfile = os.path.join(cachedir, hashlib.md5(url.encode()).hexdigest()+".txt")
        if (os.access(cfile, os.F_OK)):
            return open(cfile, errors="replace").read()
    #NIST refuses Python's default user agent
    req = urllib.request.Request(url, headers={"User-Agent": "superFATBOY-makeLineList/1.0 (+https://github.com)"})
    text = urllib.request.urlopen(req, timeout=120).read().decode("latin-1")
    if ("Error Message" in text):
        msg = re.sub(r"<[^>]*>", "", text[text.find("Error Message"):])[:200]
        raise RuntimeError("NIST ASD query for "+spectrum+" failed: "+" ".join(msg.split()))
    if (cfile is not None):
        open(cfile, "w").write(text)
    return text

#J value from NIST ("2", "5/2"), or None
def parseJ(text):
    text = text.strip()
    try:
        if ("/" in text):
            (a, b) = text.split("/", 1)
            return float(a)/float(b)
        return float(text)
    except ValueError:
        return None

#Parse NIST ASCII (format=1) output into dicts: vacuum wavelength (observed, else Ritz), intensity (float or None),
#intensity code, Aki, upper-level g.  NIST's columns differ between spectra (uncertainty columns come and go), so they
#are located from the header row; the upper level's J is the 6th column after the Ei-Ek column.
def parseNist(text, spectrum):
    text = re.sub(r"<[^>]*>", "", text.replace("\r", ""))
    rows = text.split("\n")
    header = None
    for row in rows:
        if ("Observed" in row and "|" in row):
            header = [x.strip() for x in row.split("|")]
            break
    if (header is None):
        return []

    def col(name):
        for (k, h) in enumerate(header):
            if (h.startswith(name)):
                return k
        return None
    (iobs, iritz, irel, iaki, ien) = (col("Observed"), col("Ritz"), col("Rel"), col("Aki"), col("Ei"))
    lines = []
    for row in rows:
        f = [x.strip() for x in row.split("|")]
        if (len(f) < len(header)-2 or iobs is None or irel is None):
            continue
        try:
            obs = float(f[iobs]) if f[iobs] else None
        except ValueError:
            continue
        ritz = None
        if (iritz is not None and f[iritz]):
            try:
                ritz = float(re.sub(r"[^0-9.]", "", f[iritz]))
            except ValueError:
                pass
        w = obs if obs is not None else ritz
        if (w is None):
            continue
        m = re.match(r"\s*([0-9.]+)", f[irel])
        inten = float(m.group(1)) if m else None
        code = re.sub(r"[0-9.]", "", f[irel]).strip()
        aki = None
        if (iaki is not None and f[iaki]):
            try:
                aki = float(f[iaki])
            except ValueError:
                pass
        g = None
        if (ien is not None and ien+6 < len(f)):
            J = parseJ(f[ien+6])
            if (J is not None):
                g = 2*J+1
        lines.append({"wave": w, "intensity": inten, "code": code, "aki": aki, "g": g, "spectrum": spectrum, "source": "NIST"})
    return lines

#Read a superFATBOY line list or measured_lines file: (wavelength, intensity) pairs
def readList(fname):
    out = []
    for l in open(fname):
        if (l.strip() == "" or l.lstrip().startswith("#")):
            continue
        p = l.split()
        try:
            out.append((float(p[0]), float(p[1])))
        except (ValueError, IndexError):
            continue
    return out

def main():
    ap = argparse.ArgumentParser(description="Build a superFATBOY line list from the NIST Atomic Spectra Database.")
    ap.add_argument("-e", "--elements", required=True, help='spectra, comma-separated, e.g. "Ne I,Ar I,Xe I" (a bare element means its neutral spectrum)')
    ap.add_argument("-r", "--range", nargs=2, type=float, required=True, metavar=("MIN", "MAX"), help="wavelength range in Angstrom")
    ap.add_argument("-o", "--output", default=None, help="output file (default: stdout)")
    ap.add_argument("--vacuum", action="store_true", help="vacuum wavelengths (default: air above 2000 A)")
    ap.add_argument("--min-intensity", type=float, default=0, help="drop lines fainter than this (after scaling)")
    ap.add_argument("--max-lines", type=int, default=0, help="keep only the brightest N lines")
    ap.add_argument("--missing", default="estimate", help="lines with no NIST intensity: estimate (from g*A), skip, or a number")
    ap.add_argument("-s", "--scale", default="", help='intensity scale per spectrum, e.g. "Ar I=0.5,Ne I=2"')
    ap.add_argument("-m", "--measured", default=None, help="measured intensities (superFATBOY measured_lines_*.dat, or wavelength intensity columns) to fit per-spectrum scales to")
    ap.add_argument("--use-measured", action="store_true", help="replace NIST intensities with the measured ones where measured")
    ap.add_argument("--measured-medium", default="same", choices=["same", "air", "vacuum"], help="medium of the measured file's wavelengths (default: same as the output)")
    ap.add_argument("--match-tol", type=float, default=0.1, help="tolerance in Angstrom for matching measured lines (default 0.1)")
    ap.add_argument("--blend", type=float, default=0, help="flag lines closer than this many Angstrom as blends (-1), default 0 = off")
    ap.add_argument("--blend-ratio", type=float, default=0.1, help="only when the fainter line is at least this fraction of the brighter (default 0.1)")
    ap.add_argument("--nist-file", action="append", default=[], help="parse a saved NIST ASCII response instead of querying (SPECTRUM=FILE)")
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/superFATBOY/nist"), help="cache directory for NIST responses ('' for none)")
    args = ap.parse_args()

    (wmin, wmax) = sorted(args.range)
    spectra = []
    for s in args.elements.split(","):
        s = " ".join(s.split())
        if (s == ""):
            continue
        if (len(s.split()) == 1):
            s += " I"
        spectra.append(s)
    #query in vacuum, padded for the air-vacuum difference
    qlo = float(airToVac(wmin))-1
    qhi = float(airToVac(wmax))+1
    files = dict(x.split("=", 1) for x in args.nist_file)
    alllines = []
    notes = []
    for sp in spectra:
        if (sp in files):
            text = open(files[sp], errors="replace").read()
        else:
            text = queryNist(sp, qlo, qhi, args.cache if args.cache != "" else None)
        lines = parseNist(text, sp)
        print("makeLineList> "+sp+": "+str(len(lines))+" lines from NIST", file=sys.stderr)
        alllines.extend(lines)
    for l in alllines:
        l["out"] = l["wave"] if args.vacuum else float(vacToAir(l["wave"]))
    alllines = [l for l in alllines if wmin <= l["out"] <= wmax]

    #missing intensities: g*A scaled per spectrum (log-median ratio of intensity to g*A over lines with both)
    for sp in spectra:
        mine = [l for l in alllines if l["spectrum"] == sp]
        both = [l for l in mine if l["intensity"] and l["aki"] and l["g"]]
        k = None
        if (len(both) >= 3):
            k = math.exp(np.median([math.log(l["intensity"]/(l["aki"]*l["g"])) for l in both]))
        for l in mine:
            if (l["intensity"] is not None):
                continue
            if (args.missing == "estimate"):
                if (k is not None and l["aki"] and l["g"]):
                    l["intensity"] = k*l["aki"]*l["g"]
                    l["source"] = "NIST g*A estimate"
            elif (args.missing != "skip"):
                l["intensity"] = float(args.missing)
                l["source"] = "assigned"
    alllines = [l for l in alllines if l["intensity"] is not None and l["intensity"] > 0]

    #per-spectrum scales: given, or fit to measured intensities
    scales = dict((sp, 1.0) for sp in spectra)
    for item in args.scale.split(","):
        if ("=" in item):
            (sp, v) = item.split("=", 1)
            sp = " ".join(sp.split())
            if (len(sp.split()) == 1):
                sp += " I"
            scales[sp] = float(v)
    measured = []
    if (args.measured is not None):
        measured = readList(args.measured)
        if (args.measured_medium == "air" and args.vacuum):
            measured = [(float(airToVac(w)), i) for (w, i) in measured]
        elif (args.measured_medium == "vacuum" and not args.vacuum):
            measured = [(float(vacToAir(w)), i) for (w, i) in measured]
        mw = np.array([m[0] for m in measured])
        for sp in spectra:
            ratios = []
            for l in alllines:
                if (l["spectrum"] != sp or len(mw) == 0):
                    continue
                k = int(np.argmin(np.abs(mw-l["out"])))
                if (abs(mw[k]-l["out"]) < args.match_tol and measured[k][1] > 0):
                    l["measured"] = measured[k][1]
                    ratios.append(measured[k][1]/l["intensity"])
            if (len(ratios) >= 2):
                scales[sp] = float(np.median(ratios))
                notes.append(sp+": scaled by "+"%.4g" % scales[sp]+" (median measured/NIST over "+str(len(ratios))+" lines)")
            elif (len(ratios) > 0):
                notes.append(sp+": only "+str(len(ratios))+" measured line(s), scale not fit")
    for l in alllines:
        l["final"] = l["intensity"]*scales.get(l["spectrum"], 1.0)
        if (args.use_measured and "measured" in l):
            l["final"] = l["measured"]
            l["source"] = "measured"
    alllines = [l for l in alllines if l["final"] >= args.min_intensity]
    alllines.sort(key=lambda l: l["out"])
    #merge duplicate entries of the same line (observed in several sources)
    merged = []
    for l in alllines:
        if (len(merged) > 0 and merged[-1]["spectrum"] == l["spectrum"] and abs(merged[-1]["out"]-l["out"]) < 0.005):
            if (l["final"] > merged[-1]["final"]):
                merged[-1] = l
            continue
        merged.append(l)
    alllines = merged
    if (args.max_lines > 0 and len(alllines) > args.max_lines):
        keep = sorted(alllines, key=lambda l: -l["final"])[:args.max_lines]
        alllines = sorted(keep, key=lambda l: l["out"])
    #blends
    for l in alllines:
        l["flag"] = 0
    if (args.blend > 0):
        for i in range(len(alllines)-1):
            (a, b) = (alllines[i], alllines[i+1])
            if (b["out"]-a["out"] < args.blend and min(a["final"], b["final"]) >= args.blend_ratio*max(a["final"], b["final"])):
                a["flag"] = -1
                b["flag"] = -1
        for l in alllines:
            if ("bl" in l["code"]):
                l["flag"] = -1

    out = sys.stdout if args.output is None else open(args.output, "w")
    out.write("#superFATBOY line list: "+", ".join(spectra)+", "+("%.1f" % wmin)+"-"+("%.1f" % wmax)+" A, "+("vacuum" if args.vacuum else "air")+" wavelengths\n")
    out.write("#From the NIST Atomic Spectra Database (https://physics.nist.gov/asd), "+datetime.date.today().isoformat()+", by makeLineList.py.\n")
    out.write("#Columns: wavelength, relative intensity, flag (-1 = blend, not used in the fit), #spectrum, source, NIST intensity code.\n")
    out.write("#NIST intensities are not on a common scale between spectra; missing ones: "+args.missing+".\n")
    for n in notes:
        out.write("#"+n+"\n")
    for l in alllines:
        out.write("%.4f\t%.4g\t%d\t#%s %s%s\n" % (l["out"], l["final"], l["flag"], l["spectrum"], l["source"], (" "+l["code"]) if l["code"] else ""))
    if (args.output is not None):
        out.close()
        print("makeLineList> wrote "+str(len(alllines))+" lines to "+args.output, file=sys.stderr)

if __name__ == "__main__":
    main()
