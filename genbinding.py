import glob, os
import subprocess, sys
from pathlib import Path
import argparse

#parser = argparse.ArgumentParser(description='Etaler Python binding generator')
#parser.add_argument('--home', dest='home', '')

etaler_home = "/usr/local/include"
rfldct = 'etaler'

# First we generate the redlection data
etaler_headers = [os.path.relpath(x, etaler_home) for x in Path(os.path.join(etaler_home, 'Etaler')).rglob('*.hpp')]

# Filter unwanted headers
etaler_headers = [x for x in etaler_headers if
        'OpenCL' not in x and         # FIXME: Backends causes loading problem
        'Interop' not in x and        # Disable interop
        'Util' not in x and           # uses std::chrono, but not supported by cppyy
        'CPU' not in x                # Same issue, causes loading problem
    ]

cmd = ' '.join(
    ['genreflex',                     # utility installed by pip when installing cppyy
     '--verbose',                     # Show information (somehow genreflex fail without this)
     '-s', 'selection.xml',           # election file
     '-o', '%s_rflx.cpp'%rfldct]+     # intermediate output file
     etaler_headers)                  # headers themselves
ret = os.system(cmd)
if ret != 0:
    print("genereflex failed. Exit code {}".format(ret))
    exit(ret)
else:
    print("genreflex done")

# Next we build the Python module itself
clingflags = subprocess.check_output(
    ['cling-config',               # utility installed by pip when installing cppyy
    '--cppflags'])

try:
    subprocess.check_output(
        ['c++']+                       # C++ compiler
         clingflags.split()+[          # extra flags provided by cling
         '-fPIC',                      # require position independent code
         '-shared',                    # generate shared library
         '-o', '%s_rflx.so'%rfldct,    # output file
         '-I'+etaler_home,             # include search path for Etaler headers
         '%s_rflx.cpp'%rfldct]+        # intermediate file to compile
         ['-lEtaler'])                 # link to Etaler
except subprocess.CalledProcessError as e:
    print('compilation failed (%d):' % e.returncode, e.output)
    exit(e.returncode)
else:
    print('compilation done')
