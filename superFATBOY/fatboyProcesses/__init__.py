print("__init__ fatboyProcesses")

import os
import glob
modules = glob.glob(os.path.dirname(__file__)+"/*.py")
__all__ = [ os.path.basename(f)[:-3] for f in modules]

__all__.remove('__init__')
__all__.remove('processDict')
