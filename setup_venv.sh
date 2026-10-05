#!/usr/bin/env bash
## setup_venv.sh - create a Python virtual environment with everything superFATBOY3 needs, on a desktop/laptop or an
## HPC cluster (no root needed).  Run it from the superFATBOY3 checkout:
##
##   ./setup_venv.sh                          # venv in ./venv, GPU support if an NVIDIA GPU is found
##   ./setup_venv.sh --venv ~/sfb-venv        # venv somewhere else
##   ./setup_venv.sh --cuda none              # CPU only
##   ./setup_venv.sh --cuda 12                # force the CUDA 12 CuPy (e.g. on an HPC login node with no GPU)
##
## then:  source <venv>/bin/activate  and run  superFatboy3.py my_data.xml
##
## What it does: creates the venv; installs numpy, scipy, astropy, matplotlib (and sep); installs the CuPy build that
## matches the NVIDIA driver; installs superFATBOY in editable mode (pip install -e .), so the superFatboy3.py on your
## PATH always runs this checkout - no reinstall after editing the code; builds the C extension (fatboyclib) and, if
## nvcc is available, the CUDA extensions (fatboycudalib, cp_select) against the venv's Python and numpy; then runs a
## short check.  The C/CUDA extensions are built in the source tree (superFATBOY/*.so), replacing any built before.
##
## HPC: load the modules first (Python 3.9+, a C++ compiler, and CUDA if you want the CUDA extensions), e.g.
##   module load python gcc cuda
## or pass them with --modules "python gcc cuda".  Build on a GPU node, or pass --cuda 11|12 on a login node.
##
## Options:
##   --venv DIR        venv location (default: ./venv)
##   --python PY       Python interpreter to create the venv with (default: python3)
##   --cuda auto|11|12|none   CuPy build: auto (from nvidia-smi, default), CUDA 11, CUDA 12, or none (CPU only)
##   --ccbin PATH      host C++ compiler for nvcc (e.g. /usr/bin/g++-9 when the default gcc is too new for nvcc)
##   --no-cuda-ext     don't build the optional CUDA extensions even if nvcc is found
##   --deepcr          also install deepCR (cosmic-ray removal with a neural net; pulls in PyTorch, several GB)
##   --minimal         skip optional packages (sep)
##   --no-editable     regular install into the venv instead of editable (code changes then need a reinstall)
##   --modules "..."   run "module load ..." first (HPC)
##   -h, --help        this help
set -e

VENV="./venv"
PY="python3"
CUDA="auto"
CCBIN=""
CUDA_EXT="yes"
DEEPCR="no"
MINIMAL="no"
EDITABLE="yes"
MODULES=""

usage() { sed -n '2,/^set -e/p' "$0" | sed -e 's/^## \{0,1\}//' -e '/^set -e/d'; }
while [ $# -gt 0 ]; do
    case "$1" in
        --venv) VENV="$2"; shift 2;;
        --python) PY="$2"; shift 2;;
        --cuda) CUDA="$2"; shift 2;;
        --ccbin) CCBIN="$2"; shift 2;;
        --no-cuda-ext) CUDA_EXT="no"; shift;;
        --deepcr) DEEPCR="yes"; shift;;
        --minimal) MINIMAL="yes"; shift;;
        --no-editable) EDITABLE="no"; shift;;
        --modules) MODULES="$2"; shift 2;;
        -h|--help) usage; exit 0;;
        *) echo "setup_venv> unknown option $1 (see --help)"; exit 1;;
    esac
done

say() { echo "setup_venv> $*"; }
die() { echo "setup_venv> ERROR: $*"; exit 1; }

SRC="$(cd "$(dirname "$0")" && pwd)"
[ -f "$SRC/setup.py" ] && [ -d "$SRC/superFATBOY" ] || die "run this script from the superFATBOY3 checkout"

if [ -n "$MODULES" ]; then
    command -v module > /dev/null 2>&1 || die "--modules given but there is no 'module' command"
    say "module load $MODULES"
    module load $MODULES
fi

#--- Python
command -v "$PY" > /dev/null 2>&1 || die "$PY not found (use --python, or load a Python module on HPC)"
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' || die "Python 3.9 or newer is needed ($("$PY" --version 2>&1))"
"$PY" -c 'import venv, ensurepip' 2> /dev/null || die "$PY has no venv/ensurepip module (Debian/Ubuntu: apt install python3-venv)"
say "Python: $("$PY" --version 2>&1) ($(command -v "$PY"))"

#--- GPU / CuPy flavor
if [ "$CUDA" = "auto" ]; then
    if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi > /dev/null 2>&1; then
        drv=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]*\).*/\1/p' | head -1)
        if [ -z "$drv" ]; then
            CUDA="none"
        elif [ "$drv" -ge 12 ]; then
            CUDA="12"
        else
            CUDA="11"
        fi
        say "NVIDIA driver supports CUDA ${drv:-?}: GPU support with CuPy for CUDA $CUDA"
    else
        CUDA="none"
        say "no NVIDIA GPU found: CPU only (use --cuda 11|12 to install GPU support anyway, e.g. on a login node)"
    fi
fi
case "$CUDA" in
    11) CUPY="cupy-cuda11x";;
    12) CUPY="cupy-cuda12x";;
    none) CUPY="";;
    *) die "--cuda must be auto, 11, 12 or none";;
esac

#--- venv
if [ -d "$VENV" ] && [ -x "$VENV/bin/python" ]; then
    say "using the existing venv $VENV"
else
    say "creating the venv $VENV"
    "$PY" -m venv "$VENV"
fi
VENV="$(cd "$VENV" && pwd)"
VPY="$VENV/bin/python"
"$VPY" -m pip install --quiet --upgrade pip setuptools wheel

#--- Python packages
say "installing numpy, scipy, astropy, matplotlib$( [ "$MINIMAL" = "no" ] && echo ", sep")"
if [ "$MINIMAL" = "yes" ]; then
    grep -v '^sep' "$SRC/requirements.txt" > "$VENV/requirements-minimal.txt"
    "$VPY" -m pip install --quiet -r "$VENV/requirements-minimal.txt"
else
    "$VPY" -m pip install --quiet -r "$SRC/requirements.txt" || {
        say "WARNING: installing with sep failed; retrying without it (sep is optional)"
        grep -v '^sep' "$SRC/requirements.txt" > "$VENV/requirements-minimal.txt"
        "$VPY" -m pip install --quiet -r "$VENV/requirements-minimal.txt"
    }
fi
if [ -n "$CUPY" ]; then
    say "installing $CUPY"
    "$VPY" -m pip install --quiet "$CUPY"
fi
if [ "$DEEPCR" = "yes" ]; then
    say "installing deepCR (and PyTorch)"
    "$VPY" -m pip install --quiet deepCR
fi

#--- superFATBOY and its C extension
cd "$SRC"
if [ "$EDITABLE" = "yes" ]; then
    say "installing superFATBOY (editable: the installed scripts run this checkout)"
    "$VPY" -m pip install --quiet --no-build-isolation --no-deps -e . > "$VENV/build.log" 2>&1 || { tail -20 "$VENV/build.log"; die "pip install -e . failed (full log: $VENV/build.log)"; }
    #make sure the C extension in the tree is built against this venv's numpy
    "$VPY" setup.py build_ext --inplace >> "$VENV/build.log" 2>&1 || { tail -20 "$VENV/build.log"; die "building fatboyclib failed (full log: $VENV/build.log)"; }
else
    say "installing superFATBOY into the venv"
    "$VPY" -m pip install --quiet --no-build-isolation --no-deps . > "$VENV/build.log" 2>&1 || { tail -20 "$VENV/build.log"; die "pip install . failed (full log: $VENV/build.log)"; }
fi

#--- optional CUDA extensions (faster medians); superFATBOY falls back to CuPy kernels without them
if [ -n "$CUPY" ] && [ "$CUDA_EXT" = "yes" ]; then
    if command -v nvcc > /dev/null 2>&1; then
        say "building the CUDA extensions with $(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1 | sed 's/^/nvcc /')"
        PYINC=$("$VPY" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
        NPINC=$("$VPY" -c 'import numpy; print(numpy.get_include())')
        CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}
        CUDA_LIB="$CUDA_HOME/lib64"
        [ -d "$CUDA_LIB" ] || CUDA_LIB="$CUDA_HOME/lib"
        NVFLAGS=""
        [ -n "$CCBIN" ] && NVFLAGS="-ccbin=$CCBIN"
        TMP=$(mktemp -d)
        built=0
        for m in fatboycudalib cp_select; do
            if nvcc $NVFLAGS -O2 -c "superFATBOY/${m}module.cu" --compiler-options '-fPIC' -I"$NPINC" -I"$PYINC" -o "$TMP/$m.o" > "$TMP/$m.log" 2>&1 \
               && ${CXX:-g++} -fPIC -shared -o "superFATBOY/$m.so" "$TMP/$m.o" -L"$CUDA_LIB" -lcudart >> "$TMP/$m.log" 2>&1; then
                built=$((built+1))
            else
                say "WARNING: could not build $m (optional); see the end of the log below"
                tail -5 "$TMP/$m.log"
                say "         if nvcc complains about the host compiler, rerun with --ccbin /path/to/an/older/g++"
            fi
        done
        rm -rf "$TMP"
        say "built $built of 2 CUDA extensions"
        if [ "$EDITABLE" = "no" ] && [ "$built" -gt 0 ]; then
            "$VPY" -m pip install --quiet --no-build-isolation --no-deps . >> "$VENV/build.log" 2>&1
        fi
    else
        say "nvcc not found: skipping the optional CUDA extensions (GPU mode still works through CuPy)"
    fi
fi

#--- check
say "checking the installation"
#run from the venv directory, not the checkout, so the check imports what was installed
cd "$(dirname "$(dirname "$VPY")")"
"$VPY" - <<'EOF'
import warnings
warnings.filterwarnings("ignore")
import numpy, scipy, astropy
import superFATBOY
from superFATBOY import fatboyclib
print("setup_venv>   superFATBOY %s, numpy %s, scipy %s, astropy %s" % (superFATBOY.__version__, numpy.__version__, scipy.__version__, astropy.__version__))
if tuple(int(v) for v in scipy.__version__.split(".")[:2]) >= (1, 15):
    print("setup_venv>   note: scipy >= 1.15 fits (leastsq) differ slightly from older scipy - compare runs made with the same scipy")
a = numpy.arange(11, dtype=numpy.float32)
assert fatboyclib.median(a) == 5.0, "fatboyclib.median gave the wrong answer"
print("setup_venv>   C extension fatboyclib: ok")
try:
    import cupy
    k = cupy.RawKernel(r'extern "C" __global__ void twice(float* x, int n) { int i = blockIdx.x*blockDim.x+threadIdx.x; if (i < n) x[i] *= 2; }', "twice")
    x = cupy.arange(4, dtype=cupy.float32)
    k((1,), (4,), (x, numpy.int32(4)))
    assert cupy.asnumpy(x).tolist() == [0, 2, 4, 6]
    print("setup_venv>   CuPy %s on %s: ok (GPU mode available)" % (cupy.__version__, cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()))
except ImportError:
    print("setup_venv>   CuPy not installed: CPU mode only (set gpumode = no in your XML files)")
except Exception as ex:
    print("setup_venv>   WARNING: CuPy is installed but could not run on a GPU here (%s): use gpumode = no, or run on a GPU node" % str(ex).splitlines()[0])
EOF
"$(dirname "$VPY")/superFatboy3.py" -list > /dev/null 2>&1 && say "  superFatboy3.py -list: ok" || say "WARNING: superFatboy3.py -list failed"

echo
say "done.  To use superFATBOY:"
say "    source $(cd "$VENV" && pwd)/bin/activate"
say "    superFatboy3.py my_data.xml"
