#!/usr/bin/env python3
# Regenerate docs/options-reference.md from the live output of `superFatboy3.py -list`.
# Usage (from the repo root):  python3 docs/gen_options_reference.py
import os
import re
import subprocess
import sys

repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env = dict(os.environ)
env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
script = os.path.join(repo, "superFATBOY", "superFatboy3.py")
out = subprocess.run([sys.executable, script, "-list"], capture_output=True, text=True, env=env, cwd=repo).stdout


def clean_value(text):
    # Escape characters that would break a markdown table cell
    return text.replace("|", "\\|").replace("\n", " ")


version = "unknown"
params = []
processes = []
current = None
current_opt = None
section = None
for line in out.split("\n"):
    m = re.match(r"superFATBOY v(\S+)", line)
    if m:
        version = m.group(1)
    if line.startswith("FATBOY Parameters:"):
        section = "params"
        continue
    if line.startswith("FATBOY Processes:"):
        section = "procs"
        continue
    m = re.match(r"^\tProcess = (\S+)", line)
    if m and section == "procs":
        current = (m.group(1), [])
        processes.append(current)
        current_opt = None
        continue
    if section == "params":
        m = re.match(r"^\t(\S+) = (.*)$", line)
        if m:
            params.append([m.group(1), m.group(2), ""])
            current_opt = params[-1]
        elif line.startswith("\t\t*") and current_opt is not None:
            current_opt[2] += " " + line.strip().lstrip("* ")
        continue
    if section == "procs" and current is not None:
        m = re.match(r"^\t\t(\S+) = (.*)$", line)
        if m:
            current[1].append([m.group(1), m.group(2), ""])
            current_opt = current[1][-1]
        elif re.match(r"^\t\t\t", line) and current_opt is not None:
            current_opt[2] += " " + line.strip().lstrip("* ")

lines = []
lines.append("# Options reference")
lines.append("")
lines.append("*[Docs home](README.md)*")
lines.append("")
lines.append("**This file is generated.** It is a snapshot of `superFatboy3.py -list` for superFATBOY v%s." % version)
lines.append("Run `superFatboy3.py -list` yourself for the live list, or regenerate this file with")
lines.append("`python3 docs/gen_options_reference.py`. For prose descriptions of what each process does, see the")
lines.append("[process guide](processes/README.md).")
lines.append("")
lines.append("Every process also accepts `write_output`, `write_calib_output` and `create_calib_only`; processes that")
lines.append("carry a noisemap also accept `write_noisemaps`. They are listed here only where the process defines them.")
lines.append("")
lines.append("## Contents")
lines.append("")
lines.append("- [Global parameters](#global-parameters)")
for pname, _ in processes:
    lines.append("- [%s](#%s)" % (pname, pname.lower()))
lines.append("")
lines.append("## Global parameters")
lines.append("")
lines.append("Set in the `<parameters>` section of the XML file with `<param name=\"...\" value=\"...\"/>`.")
lines.append("")
lines.append("| Parameter | Default | Notes |")
lines.append("|---|---|---|")
for name, default, info in params:
    lines.append("| `%s` | `%s` | %s |" % (name, clean_value(default), clean_value(info.strip())))
lines.append("")
for pname, opts in processes:
    lines.append("## %s" % pname)
    lines.append("")
    if not opts:
        lines.append("No process-specific options.")
        lines.append("")
        continue
    lines.append("| Option | Default | Notes |")
    lines.append("|---|---|---|")
    for name, default, info in opts:
        lines.append("| `%s` | `%s` | %s |" % (name, clean_value(default), clean_value(info.strip())))
    lines.append("")

outfile = os.path.join(repo, "docs", "options-reference.md")
with open(outfile, "w") as f:
    f.write("\n".join(lines))
print("wrote %s (%d params, %d processes)" % (outfile, len(params), len(processes)))
