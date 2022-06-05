import numpy as np
import math
import linecache
from optparse import OptionParser
import sys

parser = OptionParser()
parser.add_option('-f','--file',dest='file_in',help='cube file for eigvec at amplitude level',default='')
parser.add_option('-t','--time',dest='time',help='last time to grab',default=1000)
parser.add_option('-d','--density',dest='dens',help='e- density',default=0.9)

(options, args) = parser.parse_args()
file_in = options.file_in
t = float(options.time)
dens = float(options.dens)

# starts here. Want to first read in the eigvector from cube file

ngrid = int(linecache.getline(file_in,4).split()[0])
natom = int(linecache.getline(file_in,3).split()[0])
header = natom+6
lpt_cube_in = int(ngrid*ngrid*np.ceil(ngrid/6)+header)

my_line1 = linecache.getline(file_in, 1).split()
my_line2 = linecache.getline(file_in, lpt_cube_in+1).split()
dt = float(my_line2[2])-float(my_line1[2])

if float(linecache.getline(file_in, int(t/dt)*lpt_cube_in+1).split()[2]) == t:
    t_frame = int(t/dt)
elif float(linecache.getline(file_in, int(t/dt-1)*lpt_cube_in+1).split()[2]) == t:
    t_frame = int(t/dt-1)
elif float(linecache.getline(file_in, int(t/dt+-1)*lpt_cube_in+1).split()[2]) == t:
    t_frame = int(t/dt+1)
else:
    print('Check time in .cube file')
    sys.exit()

eigvec = []
my_lines = range(t_frame*lpt_cube_in+header+1,(t_frame+1)*lpt_cube_in+1)
for line in my_lines:
    eigvec += linecache.getline(file_in,line).split()
eigvec = np.abs(np.array(eigvec, dtype='float'))
rho = eigvec**2

rho = -np.sort(-rho)/np.sum(rho)
running_sum = 0
n = 0
while running_sum < dens:
    running_sum += rho[n]
    n += 1

print('for e- dens of '+str(dens))
print('eigvec isoval = '+str(eigvec[n]))
print('rho isoval = '+str(rho[n]))
