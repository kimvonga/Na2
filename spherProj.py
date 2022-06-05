import numpy as np
from optparse import OptionParser

parser = OptionParser()
parser.add_option('-c','--center',dest='center',help='filename that stores position of center point',metavar='file')
parser.add_option('-s','--solvent',dest='solvent',help='filename that stores postions of surrounding solvent',metavar='file')
parser.add_option('-r','--radius',dest='radius',help='radius of spherical shell',metavar='float',default='3.5')
parser.add_option('--sigma',dest='sigma',help='LJ sigma of solvent',metavar='float',default='3.0')
parser.add_option('-N',dest='N',help='number of points on spheres surface',metavar='int',default='1000')

(options, args) = parser.parse_args()
center = np.loadtxt(options.center)
solvent = np.loadtxt(options.solvent)
radius = float(options.radius)
sigma = float(options.sigma)
num = int(options.N)

def dist(a,b):
    '''returns scalar distance between 2 sets of cartesian coordinates'''
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)

def fibonacciSphere(N, r=1):
    '''
    Returns N roughly equally spaced points on a sphere. 
    Points are spaced such that each point occupies roughly same area.
    Look up fibonacci spiral to get an idea of how it works.
    Returns as [N,3] in cartesian coordinates.
    '''
    points = np.zeros([N,3])

    indices = np.arange(0, N, dtype=float)+0.5

    phi = np.arccos(1-2*indices/N)
    theta = np.pi*(1+5**0.5)*indices

    points[:,0] = r*np.cos(theta)*np.sin(phi)
    points[:,1] = r*np.sin(theta)*np.sin(phi)
    points[:,2] = r*np.cos(phi)

    return points

def spherProjection(center, r_shell, eps, radius, N=1000):
    '''
    Projects solvent molecules onto shell around center point.
    Returns [N,4] with first 3 columns being cartesian coordinates
        of points on sphere. 4th column indicates whether the point
        is occupied by solvent or not (1=occupied, 0=not occupied).
    '''
    n_part = len(r_shell)
    shell_pts = np.append(fibonacciSphere(N, r=radius), [[0]]*N, axis=1)
    r_solvent = np.array(r_shell) - np.array(center)

#    commenting out linear projection of solvent onto sphere
#    radial_dist = np.zeros(n_part)
#    for i in range(n_part):
#        radial_dist[i] = norm(r_solvent[i])
#
#    r_eps = eps*radius/radial_dist

    for n in range(N):
        my_dist = np.zeros(n_part)
        for m in range(n_part):
            my_dist[m] = dist(shell_pts[n,:3], r_solvent[m])
        if min(my_dist) < eps:
            shell_pts[n,3] = 1

    return shell_pts

shell_pts = spherProjection(center, solvent, sigma/2, radius, N=num)
np.savetxt('spherical_projection.txt',shell_pts,fmt='% 10.6f% 10.6f% 10.6f% 4d')
