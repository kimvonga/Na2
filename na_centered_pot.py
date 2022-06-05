import numpy as np
import sys
sys.path.append('/home/andy/py_scripts/')
import linecache
import dimer_dissoc_analysis as lib
from optparse import OptionParser

parser = OptionParser()
parser.add_option('--i_dir',dest='i_dir',help='input directory',default='./')
parser.add_option('--o_dir',dest='o_dir',help='output directory',default='./')
parser.add_option('--f_out',dest='f_out',help='file name to write out to',default='na_centered_pot.txt')
parser.add_option('--t0',dest='t0',help='initial time',default='0')
parser.add_option('-t',dest='t',help='final time',default='1000')
parser.add_option('--n_solu',dest='n_solu',help='number of Na atoms',default='2')

(options, args) = parser.parse_args()
i_dir = options.i_dir
o_dir = options.o_dir
f_out = options.f_out
t0 = float(options.t0)
t = float(options.t)
n_solu = int(options.n_solu)

t1 = lib.convFloat(linecache.getline(i_dir+'/out.na',2).split()[4])
t2 = lib.convFloat(linecache.getline(i_dir+'/out.na',n_solu*6+5).split()[4])
dt = t2 - t1

# for potential
t_frames = np.linspace(t0, t, num=int((t-t0)/dt+1))
ar_pot = np.zeros([len(t_frames),9])
for i in range(len(t_frames)):
    ar_pot[i,1:] = lib.calcArPot(i_dir, t_frames[i], n_solu=n_solu)
ar_pot[:,0] = t_frames

np.savetxt(o_dir+'/'+f_out, ar_pot, ['%8.2f','%12.8f','%12.8f','%12.8f','%12.8f','%12.8f','%12.8f','%12.8f','%12.8f'])

# for forces
#t_frames = np.linspace(t0,t,num=int((t-t0)/dt+1))
#f = np.zeros([len(t_frames), 3*n_solu+1])
#for i in range(len(t_frames)):
#    my_f = lib.calcArPot(i_dir, t_frames[i], n_solu=n_solu, PK=PK, Pol=Pol)
#    f1 = np.sum(my_f[0,:], axis=0)
#    f2 = np.sum(my_f[1,:], axis=0)
#
##    f[i,1:4] = f1 - (f1+f2)/2  # for forces rel to center of force
##    f[i,4:7] = f2 - (f1+f2)/2
#    f[i,1:4] = f1
#    f[i,4:7] = f2
#f[:,0] = t_frames
#
#np.savetxt(o_dir+'/ar_pseudo.txt', f, ['%10.2f', '%18.14f', '%18.14f', '%18.14f', '%18.14f', '%18.14f', '%18.14f'])

