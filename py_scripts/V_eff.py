import numpy as np
import sys
sys.path.append('/home/andy/py_scripts/')
import dimer_dissoc_analysis as lib
from optparse import OptionParser

parser = OptionParser()
parser.add_option('--i_dir',dest='i_dir',help='input directory',default='./')
parser.add_option('--o_dir',dest='o_dir',help='output directory',default='./')
parser.add_option('--t0', dest='t0', help='intial time', default='0')
parser.add_option('-t', '--t', dest='t', help='final time, inclusive', default='500')
parser.add_option('--dt', dest='dt', help='time interval in fs', default='4')

(options, args) = parser.parse_args()
i_dir = options.i_dir
o_dir = options.o_dir
t0 = float(options.t0)
t = float(options.t)
dt = float(options.dt)

t_frames = np.arange(t0, t+dt, dt)
V_eff = lib.calcEffVol(i_dir, t0, t, dt, n_solv=1600, auleng=0.5291772108, N=30, kappa=10, grid='manual')

np.savetxt(o_dir+'/V_eff.txt', V_eff, ['%10.2f', '%18.14f', '%18.14f'])
