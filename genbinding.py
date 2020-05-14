import glob, os
import subprocess, sys
from pathlib import Path
import argparse

try:
    import cppyy
except:
    print('Failed to import cppyy. Please make sure cppyy is installed correctly')
    exit(1)

parser = argparse.ArgumentParser(description='Etaler Python binding generator')
parser.add_argument('--home', dest='home', default=None, help='Where Etaler\'s header folder is located')
parser.add_argument('--cxx', dest='cxx', default='c++',  help='the c++ compiler you use')
parser.add_argument('--opencl', dest='ocl', action='store_true', help='Wrapping the OpenCLBackned')
parser.add_argument('--out-dir', dest='out_dir', default='.', help='output directory')

args = parser.parse_args()

user_home = args.home
cxx = args.cxx
enable_ocl = args.ocl
rfldct = 'etaler'

etaler_homes = ['/usr/local/include', '/usr/include'] if user_home is None else [user_home]
etaler_headers = []
etaler_home = ''
# Attempt to find a valid home
for home in etaler_homes:
    etaler_headers = [os.path.relpath(x, home) for x in
            Path(os.path.join(home, 'Etaler')).rglob('*.hpp')]
    if len(etaler_headers) != 0:
        etaler_home = home
        break
if etaler_home == '': # not found
    raise IOError('Cannot find Etaler headers in ' + str(etaler_homes))

# Filter unwanted headers
etaler_headers = [x for x in etaler_headers if
        'Interop' not in x and        # Disable interop. TODO: add option to enable
        'ProgressDisplay' not in x    # uses std::chrono, but not supported by cppyy
    ]

if enable_ocl is False:
    etaler_headers = [x for x in etaler_headers if 'OpenCL' not in x]


print('---------generator information----------')
print('Etaler home   : %s'%etaler_home)
print('Interop       : Not supported now')
print('Backends      : Not supported now')
print('C++ compiler  : %s'%cxx)
print('OpenCL Backend: {}'.format(enable_ocl))
print('Generating via cppyy...')

# First we generate the redlection data
cmd = ' '.join(
    ['genreflex',                     # utility installed by pip when installing cppyy
     '--verbose',                     # Show information (somehow genreflex fail without this)
     '-s', 'selection.xml',           # selection file
     '-o', '%s_rflx.cpp'%rfldct]+     # intermediate output file
     etaler_headers)                  # headers themselves
ret = os.system(cmd)
if ret != 0:
    print("genereflex failed. Exit code {}".format(ret))
    exit(ret)
else:
    print("genreflex done")

# Next we build the Python module itself
print('Compiling binding...')
clingflags = subprocess.check_output(
    ['cling-config',               # utility installed by pip when installing cppyy
    '--cppflags'])

try:
    subprocess.check_output(
        [cxx]+                         # C++ compiler
         clingflags.split()+[          # extra flags provided by cling
         '-fPIC',                      # require position independent code
         '-shared',                    # generate shared library
         '-std=c++1z',                 # cppyy should have set this but to be safe.
         '-o', '%s_rflx.so'%rfldct,    # output file
         '-I'+etaler_home,             # include search path for Etaler headers
         '%s_rflx.cpp'%rfldct]+        # intermediate file to compile
         ['-lEtaler'])                 # link to Etaler
except subprocess.CalledProcessError as e:
    print('compilation failed (%d):' % e.returncode, e.output)
    exit(e.returncode)
else:
    print('compilation done')
