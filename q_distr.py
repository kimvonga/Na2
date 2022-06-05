import numpy as np
import sys
sys.path.append('/home/andy/py_scripts/')
import dimer_dissoc_analysis as lib
from optparse import OptionParser

parser = OptionParser()
parser.add_option('--i_dir',dest='i_dir',help='input directory',default='./')
parser.add_option('--o_dir',dest='o_dir',help='output directory',default='./')
parser.add_option('--nstate',dest='nstate',help='electronic state of interest',default='2')

(options, args) = parser.parse_args()
i_dir = options.i_dir
o_dir = options.o_dir
nstate = int(options.nstate)

q_distr = lib.calcQSepTime(i_dir, nstate, r_cutoff=5, auleng=0.5291772108, dt=20)

np.savetxt(o_dir+'/q_distr.txt', q_distr, ['%10.2f', '%18.14f', '%18.14f'])
