import numpy as np
import math, random
import linecache
import sys
sys.path.append('/u/home/k/kimvonga/python/')

def norm(a):
    '''returns the norm (length) of a vector'''
    return np.sqrt(a[0]**2 + a[1]**2 + a[2]**2)

def dist(a,b):
    '''returns scalar distance between 2 sets of cartesian coordinates'''
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)

def periodicBounds(A, box_dim):
    '''
    applies pbc to matrix A. Assumes A is a np.2darray
    simple function to only works if A[i,j] is not more than 1 box_dim out of box
    '''
    for i in range(0,len(A)):
        for j in range(0,3):
            if A[i,j] >= box_dim[j]:
                A[i,j] -= box_dim[j]
            elif A[i,j] < 0:
                A[i,j] += box_dim[j]
    
    return A

def minImage(r, box_dim):
    '''
    returns distance vector under minimum image. returns as [x,y,z] i.e. array size 3
    assumes distance vector is within the box
    '''
    for i in range(len(r)):
        if r[i] > box_dim/2:
            r[i] = r[i] - box_dim
        elif r[i] < -box_dim/2:
            r[i] = box_dim + r[i]

    return r

def minImageDist(a, b, box_dim):
    '''
    returns minimum image distance between two points (maps 2 points in R3 to scalar)
    '''
    min_dist = 0
    for j in range(0,3):
        min_dist += min(abs(a[j]-b[j]), box_dim[j] - abs(a[j]-b[j]))**2

    return np.sqrt(min_dist)

def sqrMinImageDist(a, b, box_dim):
    '''
    returns minimum image distance between two points (maps 2 points in R3 to scalar)
    '''
    min_dist = 0
    for j in range(0,3):
        min_dist += min(abs(a[j]-b[j]), box_dim[j] - abs(a[j]-b[j]))**2

    return min_dist

def vectorPBC(a, b, box_dim):
    '''
    returns vector between points a and b where a, b are under pbc. Assumes a, b are in Cartesian coordinates
        and vector ab is relative to point a
    '''
    vector_ab = np.zeros(3)
    for j in range(0,3):
        if abs(a[j]-b[j]) > box_dim[j]/2:
            if a[j] > b[j]:
                vector_ab[j] = b[j] + box_dim[j] - a[j]
            elif a[j] < b[j]:
                vector_ab[j] = b[j] - box_dim[j] - a[j]
        else:
            vector_ab[j] = b[j] - a[j]

    return vector_ab

def intersect(A,B):
    '''
    takes Nx3 matrices A, B and returns the rows shared by the two matrices
    intended to compare grid points. allows for case of N=1
    '''
    matches = []; intersect = np.array([])
    if len(A.shape) == 2 and len(B.shape) == 2:
        for j in range(0,len(B)):
            for i in range(0,len(A)):
                if B[j,0] == A[i,0] and B[j,1] == A[i,1] and B[j,2] == A[i,2]:
                    matches.append(i)
        matches = np.array(matches)

        indices = np.delete(np.arange(0,len(A)), matches, axis=0)
        intersect = np.delete(A,indices, axis=0)
    elif len(A.shape) == 1 and len(B.shape) == 2:
        for j in range(0,len(B)):
            if B[j,0] == A[0] and B[j,1] == A[1] and B[j,2] == A[2]:
                intersect = A
                break
    elif len(A.shape) == 2 and len(B.shape) == 1:
        for i in range(0,len(A)):
            if B[0] == A[i,0] and B[1] == A[i,1] and B[2] == A[i,2]:
                intersect = B
                break
    elif len(A.shape) == 1 and len(B.shape) == 1:
        if B[0] == A[0] and B[1] == A[1] and B[2] == A[2]:
            intersect = A

    return intersect

def cut(A,B):
    '''
    function to cut set B from A. B does not need to be contained in A.
    A, B need to be Nx3 matrices (was intended to compare grid point values)
    returns the A - intersect(A,B) as a Mx3 matrix
    '''
    matches = np.array([-1])
    for j in range(0,len(B)):
        for i in range(0,len(A)):
            if B[j,0] == A[i,0] and B[j,1] == A[i,1] and B[j,2] == A[i,2]:
                matches = np.append(matches, i)
    matches = np.delete(matches,0)

    cut = np.delete(A, matches, axis=0)

    return cut

def union(A,B):
    '''
    function such that union(A,B) = A + B - intersect(A,B)
    A, B need to be Nx3 matrices (was intended to compare grid points)
    '''
    union = np.concatenate(cut(A,B), B, axis=0)

    return union

def centerPoint(A):
    '''
    finds center of set A such that it is in set A.
    '''
    center = np.array([sum(A[::,0])/len(A), sum(A[::,1])/len(A), sum(A[::,2])/len(A)])
    dist = np.zeros(len(A))

    for i in range(0,len(A)):
        dist[i] = (center[0]-A[i,0])**2 + (center[1]-A[i,1])**2 + (center[2]-A[i,2])**2

    index = np.argwhere(dist == dist.min())[0]
    centerPoint = A[index][0]

    return centerPoint

def octants(A, center_pt=None):
    '''
    splits set A into octants relative to a given center_pt
        if no center_pt is given then it is taken as centerPoint(A).
    octant_1 contains intersection with xy, xz, yz planes.
    octant_2 contains intersection with xy, xz planes.
    octant_3 contains intersection with xy, yz planes.
    octant_4 contains intersection with xy plane.
    octant_5 contains intersection with xz, yz planes.
    octant_6 contains intersection with xz plane.
    octant_7 contains intersection with yz plane.
    octant_8 does not contain xy, xz, yz planes.
    returns a dictionary of lists

    while I like to keep things as np.ndarray, converting to python lists for speed
    '''
    listA = A.tolist()

    if center_pt is None:
        center_pt = centerPoint(A)
    center_pt = center_pt.tolist()

    octs = dict((str(i), []) for i in range(1,9))

    for i in range(0,len(listA)):
        if listA[i][0] >= center_pt[0] and listA[i][1] >= center_pt[1] and listA[i][2] >= center_pt[2]:
            octs['1'].append(listA[i])
        elif listA[i][0] < center_pt[0] and listA[i][1] >= center_pt[1] and listA[i][2] >= center_pt[2]:
            octs['2'].append(listA[i])
        elif listA[i][0] <= center_pt[0] and listA[i][1] < center_pt[1] and listA[i][2] >= center_pt[2]:
            octs['3'].append(listA[i])
        elif listA[i][0] > center_pt[0] and listA[i][1] < center_pt[1] and listA[i][2] >= center_pt[2]:
            octs['4'].append(listA[i])
        elif listA[i][0] >= center_pt[0] and listA[i][1] >= center_pt[1] and listA[i][2] < center_pt[2]:
            octs['5'].append(listA[i])
        elif listA[i][0] < center_pt[0] and listA[i][1] >= center_pt[1] and listA[i][2] < center_pt[2]:
            octs['6'].append(listA[i])
        elif listA[i][0] <= center_pt[0] and listA[i][1] < center_pt[1] and listA[i][2] < center_pt[2]:
            octs['7'].append(listA[i])
        else:
            octs['8'].append(listA[i])
    
    return octs

def relAngles(center, r_shell):
    '''
    Given a center position and positions of surrounding particles (in cartesian),
        returns angle between particles.
        e.g. for tetrahedron, expect 109 degrees between neighboring particles.
    Returns as NxN, upper diagonal.
    '''
    n_part = len(r_shell)
    rel_angles = np.zeros([n_part, n_part])

    # want to recenter particles relative to center pt and normalize
    r = r_shell - center
    r = np.array([r[i]/norm(r[i]) for i in range(n_part)])

    for m in range(n_part-1):
        for n in range(m+1, n_part):
            rel_angles[m,n] = 2*np.arcsin(0.5*dist(r[m], r[n]))

    return rel_angles

def relSpherCoords(center, r_shell):
    '''
    Given a center position and positions of surrounding particles (in cartesian),
        returns spherical coordinates of particles relative to center.
    Returns as
        [[radial dist, azimutal angle, polar angle],
         [radial dist, azimutal angle, polar angle], ...]
    '''
    n_part = len(r_shell)
    spher_coords = np.zeros([n_part,3])

    # want to recenter particles relative to center pt and normalize
    r = r_shell - center
    spher_coords[:,0] = np.array([norm(r[i]) for i in range(n_part)])  # rad distance
    r = np.array([r[i]/spher_coords[i,0] for i in range(n_part)])

    # want to then rotate particles such that first particle is along the z-axis
    if norm(np.cross(r[0], [0,0,1])) > 1e-6:
        rot_ax = np.cross(r[0], [0,0,1])/norm(np.cross(r[0], [0,0,1]))
        rot_ang = 2*np.arcsin(0.5*dist(r[0], [0,0,1]))
        cross_mat = np.array([[0,-rot_ax[2],rot_ax[1]],
            [rot_ax[2],0,-rot_ax[0]],[-rot_ax[1],rot_ax[0],0]])

        rot_m = np.cos(rot_ang)*np.identity(3) + \
                np.sin(rot_ang)*cross_mat + (1-np.cos(rot_ang))*np.outer(rot_ax,rot_ax)
        for i in range(n_part):
            r[i] = np.matmul(rot_m,r[i])

    for i in range(n_part):
        for j in range(3):
            if abs(r[i,j]) < 1e-6:
                r[i,j] = 0

    # want to do a second rotation s.t. second particle lies in xz-plane
    my_vect = np.array([r[1,0], r[1,1], 0])/norm([r[1,0], r[1,1], 0])
    if abs(my_vect[0]-1) > 1e-6:
        rot_ax = np.array([0,0,-1])
        if r[1,1] > 0:
            rot_ang = 2*np.arcsin(0.5*dist(my_vect, [1,0,0]))
        else:
            rot_ang = 2*np.pi - 2*np.arcsin(0.5*dist(my_vect, [1,0,0]))
        cross_mat = np.array([[0,-rot_ax[2],rot_ax[1]],
            [rot_ax[2],0,-rot_ax[0]],[-rot_ax[1],rot_ax[0],0]])

        rot_m = np.cos(rot_ang)*np.identity(3) + \
                np.sin(rot_ang)*cross_mat + (1-np.cos(rot_ang))*np.outer(rot_ax,rot_ax)
        for i in range(n_part):
            r[i] = np.matmul(rot_m,r[i])

    for i in range(n_part):
        for j in range(3):
            if abs(r[i,j]) < 1e-6:
                r[i,j] = 0

    # azimuthal angle. 8 conditions bc quadrants and screw numerical errors
    for i in range(n_part):
        if r[i,0] == 0 and r[i,1] > 0:
            spher_coords[i,1] = np.pi/2
        elif r[i,0] == 0 and r[i,1] < 0:
            spher_coords[i,1] = 3*np.pi/2
        elif r[i,1] == 0 and r[i,0] > 0:
            spher_coords[i,1] = 0
        elif r[i,1] == 0 and r[i,0] < 0:
            spher_coords[i,1] = np.pi
        elif r[i,0] > 0 and r[i,1] > 0:
            spher_coords[i,1] = np.arctan(r[i,1]/r[i,0])
        elif r[i,0] < 0 and r[i,1] > 0:
            spher_coords[i,1] = np.pi/2 + np.arctan(-r[i,0]/r[i,1])
        elif r[i,0] < 0 and r[i,1] < 0:
            spher_coords[i,1] = np.pi + np.arctan(r[i,0]/r[i,1])
        elif r[i,0] > 0 and r[i,1] < 0:
            spher_coords[i,1] = 3*np.pi/2 + np.arctan(-r[i,0]/r[i,1])
    
    # polar angle
    for i in range(n_part):
        spher_coords[i,2] = np.arccos(r[i,2]/1)

    return spher_coords.round(6)

def fibonacciSphere(N, r=1):
    '''
    Returns N roughly equally spaced points on a sphere. 
    Spaces points such that each point occupies roughly same area.
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

def spherProjection(center, r_shell, eps, radius, N=1000, project='None'):
    '''
    Projects solvent molecules onto shell around center point.
    Returns the area of the shell that is unoccupied vs occupied by solvent.
    '''
    n_part = len(r_shell)
    shell_pts = np.append(fibonacciSphere(N, r=radius), [[0]]*N, axis=1)
    r_solvent = r_shell - center

    if project=='None':
        for n in range(N):
            my_dist = np.zeros(n_part)
            for m in range(n_part):
                my_dist[m] = dist(shell_pts[n,:], r_solvent[m])/eps
            if min(my_dist) < 1:
                shell_pts[n,3] = 1

    elif project=='linear':
        radial_dist = np.zeros(n_part)
        for i in range(n_part):
            radial_dist[i] = norm(r_solvent[i])

        r_eps = eps*radius/radial_dist

        for n in range(N):
            my_dist = np.zeros(n_part)
            for m in range(n_part):
                my_dist[m] = dist(shell_pts[n,:3], radius*r_solvent[m]/radial_dist[m])/r_eps[m]
            if min(my_dist) < 1:
                shell_pts[n,3] = 1

    return shell_pts













