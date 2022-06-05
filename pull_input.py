import numpy as np
import linecache
import shutil
import os
import sys
sys.path.append('/home/andy/py_scripts/')
import dimer_dissoc_analysis as lib
from optparse import OptionParser

parser = OptionParser()
parser.add_option('--i_dir', dest='i_dir', help='directory from which to pull na, ar config', default='./')
parser.add_option('--o_dir', dest='o_dir', help='output directory', default='./')
parser.add_option('--t0', dest='t0', help='start pulling config at t0')
parser.add_option('--dt', dest='dt', help='pull config every dt', default='1000')
parser.add_option('--d0', dest='d0', help='start writing config to dir d0')
parser.add_option('-N', '--N', dest='N', help='number of directories to create', default='1')

(options, args) = parser.parse_args()
i_dir = options.i_dir
o_dir = options.o_dir
t0 = float(options.t0)
dt = float(options.dt)
d0 = int(options.d0)
N = int(options.N)

def pullNa(file_in, t, unitl=21.9166090280612, n_solu=2):
    lpt = int(6*n_solu + 3)

    t0 = lib.convFloat(linecache.getline(file_in, 2).split()[4])
    t1 = lib.convFloat(linecache.getline(file_in, lpt+2).split()[4])
    dt = t1 - t0

    linenums = np.arange(lpt*int((t-t0)/dt) + 1, lpt*int((t-t0)/dt+1) + 1)

    # quick check to find time
    if lib.convFloat(linecache.getline(file_in, linenums[1]).split()[4]) != t:
        print('could not find t = '+str(t))
        return None

    na_lines = []
    for i in range(len(linenums)):
        na_lines += [linecache.getline(file_in, linenums[i])]

    return na_lines

def pullAr(file_in, t, unitl=21.9166090280612):
    '''
    Does not handle *.argon
    '''
    nmol = int(linecache.getline(file_in, 7).split()[1])
    lpt = int(nmol*3*3+2+8)

    t0 = lib.convFloat(linecache.getline(file_in, 7).split()[4])
    t1 = lib.convFloat(linecache.getline(file_in, lpt+7).split()[4])
    dt = t1-t0

    linenums = np.arange(lpt*int((t-t0)/dt) + 1, lpt*int((t-t0)/dt+1) + 1)

    # quick check to find time
    if lib.convFloat(linecache.getline(file_in, linenums[6]).split()[4]) != t:
        print('could not find t = '+str(t))
        return None

    ar_lines = []
    for i in range(len(linenums)):
        ar_lines += [linecache.getline(file_in, linenums[i])]

    return ar_lines

dirs = np.arange(d0, d0+N)

job_file = open('/media/andy/Samsung_T5/supermic/Na2+_Ar/first_es/fssh/61/1/hoffman_job.sh', 'r')
job_lines = job_file.readlines()

for i in range(len(dirs)):
    os.mkdir(str(dirs[i]))
    os.mkdir(str(dirs[i])+'/1/')

    na_lines = pullNa('out.na', t0+i*dt)
    with open(str(dirs[i])+'/1/'+'in.na', 'w') as na_out:
        na_out.writelines(my_line for my_line in na_lines)
    na_out.close()

    ar_lines = pullAr('out.water', t0+i*dt)
    with open(str(dirs[i])+'/1/'+'in.water', 'w') as ar_out:
        ar_out.writelines(my_line for my_line in ar_lines)
    ar_out.close()

    job_lines[18] = job_lines[18][:56] + str(dirs[i])+'/1/\n'
    with open(str(dirs[i])+'/1/'+'hoffman_job.sh', 'w') as job_out:
        job_out.writelines(my_line for my_line in job_lines)

    shutil.copy('phifile.dat', str(dirs[i])+'/1/'+'phifile.dat')

    del na_lines, ar_lines

print("To transfer newly created directories and files to hoffman, run")
print("find ./ -wholename './*/1/*.*' -printf %P\\0\\n | rsync -a --files-from=- ./ $hoffman2_scratch/supermic/Na2+_Ar/first_es/fssh/")
print("from terminal assuming that the output directory (o_dir from this script) is ./")



