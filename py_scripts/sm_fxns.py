import numpy as np
from scipy import stats
import math
import linecache
import matplotlib.pyplot as plt
import set_fxns

def autocorrelation(ac_data):
    '''
    returns autocorrelation of whatever data used.
    Data should be in size NxM, with first column being time and second+ column being the data.
    Will work for things like bond velocity, etc.
    For statistical reasons, shouldnt trust second half of output.
    '''
    N = len(ac_data)
    ac = np.zeros([N,3])

    for i in range(N):
        for j in range(i,N):
            my_norm = np.dot(ac_data[i,1:], ac_data[i,1:])

            ac[int(j-i),1] += np.dot(ac_data[i,1:], ac_data[j,1:])/my_norm

    for i in range(N):
        ac[i,1] /= N-i
        ac[i,2] = ac[i,1]/np.sqrt(N-i)

    return ac

def radialDistrFunc(r_molcs, r0, r, dr, box_dim):
    '''
    Since only one type of molecule, only works for X-X correlation. 
    Also only for one time point so outer loop needs to time average
    '''
    bins = np.linspace(r0, r-dr, num=int((r-r0)/dr))
    hist = np.zeros([len(bins),2])
    hist[:,0] = bins

    r_molcs = set_fxns.periodicBounds(r_molcs, [box_dim]*3)

    for m in range(len(r_molcs)-1):
        for n in range(m+1, len(r_molcs)):
            r_mn = set_fxns.minImage(r_molcs[m]-r_molcs[n], box_dim)
            dist_mn = set_fxns.dist(r_mn, [0,0,0])

            if dist_mn < r0 or dist_mn > r:
                continue
            else:
                my_ind = int((dist_mn-r0)/dr)
                hist[my_ind,1] += 2

    rho = len(r_molcs)/(box_dim**3)
    for i in range(len(hist)):
        shell_vol = 4/3*np.pi*((bins[i]+dr)**3 - bins[i]**3)
        hist[i,1] = hist[i,1]/(shell_vol*len(r_molcs)*rho)

    return hist

def boltzPMF(rxn_coord, r0, r, dr):
    '''
    Calculates boltzmann weighted PMF for 1-D reaction coordinate, which is usually the bond distance.
    '''
    bins = np.linspace(r0, r-dr, num=np.round((r-r0)/dr))

    pmf = np.zeros([len(bins),3])
    pmf[:,0] = bins + dr/2

    for i in range(len(rxn_coord)):
        if rxn_coord[i,1] < r0 or rxn_coord[i,1] > r:
            continue

        my_ind = int((rxn_coord[i,1]-r0)/dr)
        pmf[my_ind,1] += 1

    max_count = np.max(pmf[:,1])
    pmf[:,2] = np.log(pmf[:,1]/max_count)/np.sqrt(pmf[:,1]) # Need to check if calculating error correctly
    pmf[:,1] = -np.log(pmf[:,1]/max_count)

    return pmf





