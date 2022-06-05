import numpy as np
from scipy import stats
import math
import linecache
import matplotlib.pyplot as plt
import set_fxns

def dist(a,b):
    '''returns scalar distance between 2 sets of cartesian coordinates'''
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)

def convFloat(mystring):
    '''Useful for converting time and Ar position, which have a stupid D instead of e for sci notation, to a float'''
    myval = list(mystring)
    myval[len(myval)-4] = 'e'
    return float("".join(myval))

def readAr(file_in, t, unitl=21.9166090280612):
    nmol = int(linecache.getline(file_in, 7).split()[1])
    suffix = "".join(list(file_in)[-5:])
    if suffix == 'water':
        lpt = int(nmol*3*3+2+8)     # lpt stands for lines for dt
    
        t0 = convFloat(linecache.getline(file_in, 7).split()[4])
        t1 = convFloat(linecache.getline(file_in, lpt+7).split()[4])
        dt = t1 - t0
        
        r = np.zeros([nmol,3])
        
        # quick check to find time
        if convFloat(linecache.getline(file_in, int((t-t0)/dt)*lpt+7).split()[4]) != t:
            print('could not find t = '+str(t))
            return None
        
        linenums = np.linspace(int((t-t0)/dt*lpt+9), int(((t-t0)/dt+1)*lpt), num=lpt-8, dtype=int)

        # part that does the reading and assigns values to matrix r
        r_counter = 0
        for i in range(len(linenums)):
            if i < math.ceil(nmol/3):
                my_line = linecache.getline(file_in, linenums[i]).split()
                for j in range(3):
                    if r_counter < nmol:
                        r[r_counter,0] = convFloat(my_line[j])
                        r_counter += 1
            elif i >= nmol and i < math.ceil(nmol*4/3):
                my_line = linecache.getline(file_in, linenums[i]).split()
                for j in range(3):
                    if r_counter < nmol:
                        r[r_counter,1] = convFloat(my_line[j])
                        r_counter += 1
            elif i >= 2*nmol and i < math.ceil(nmol*7/3):
                my_line = linecache.getline(file_in, linenums[i]).split()
                for j in range(3):
                    if r_counter < nmol:
                        r[r_counter,2] = convFloat(my_line[j])
                        r_counter += 1
            r_counter = 0 if r_counter == nmol else r_counter

        return r*unitl

    elif suffix == 'argon':
        lpt = int(nmol+8)

        t0 = convFloat(linecache.getline(file_in, 7).split()[4])
        t1 = convFloat(linecache.getline(file_in, lpt+7).split()[4])
        dt = t1-t0

        r = np.zeros([nmol,3])

        # quick check to find time
        if convFloat(linecache.getline(file_in, int((t-t0)/dt)*lpt+7).split()[4])!= t:
            print('could not find t = '+str(t))
            return None

        linenums = np.linspace(int((t-t0)/dt*lpt+9), int(((t-t0)/dt+1)*lpt), num=lpt-8, dtype=int)

        for i in range(len(linenums)):
            my_line = linecache.getline(file_in, linenums[i]).split()
            r[i] = my_line[:3]

        return r*unitl

def readNa(file_in, t, unitl=21.9166090280612, n_solu=2):
    lpt = int(6*n_solu + 3)
    
    t0 = convFloat(linecache.getline(file_in, 2).split()[4])
    t1 = convFloat(linecache.getline(file_in, lpt+2).split()[4])
    dt = t1 - t0
    
    r = np.zeros([n_solu,3])
    
    # quick check to find time
    if convFloat(linecache.getline(file_in, int((t-t0)/dt)*lpt+2).split()[4]) != t:
        print('could not find t = '+str(t))
        return None
    
    for i in range(0,n_solu):
        r_line = linecache.getline(file_in, int((t-t0)/dt)*lpt+6*i+5).split()
        r[i] = r_line[:3]
    
    return r*unitl

def readEDens(file_in, t, auleng=0.5291772108):
    '''
    Reads cube file to pull out electron density. Works for amplitude level as well since file format between
    density and amplitude file are the same. Reads in so that cartesian coordinates are relative to the origin in
    the cube file.
    Spits out as Nx4 with N=ngrid**3, first 3 columns are x,y,z coords and 4th col is density.
    '''
    ngrid = int(linecache.getline(file_in, 4).split()[0])
    n_atom = int(linecache.getline(file_in, 3).split()[0])
    origin = np.array(linecache.getline(file_in, 3).split()[1:4], dtype=float)*auleng
    dx = float(linecache.getline(file_in, 4).split()[1])*auleng
    dy = float(linecache.getline(file_in, 5).split()[2])*auleng
    dz = float(linecache.getline(file_in, 6).split()[3])*auleng
    lpt = n_atom + 6 + math.ceil(ngrid/6)*ngrid*ngrid

    t0 = float(linecache.getline(file_in, 1).split()[2])
    t1 = float(linecache.getline(file_in, lpt+1).split()[2])
    dt = t1 - t0

    # quick check to find time. If this doesnt work as intended, may be issue with data types (int vs float)
    if float(linecache.getline(file_in, int((t-t0)/dt)*lpt+1).split()[2]) != t:
        print('could not find t = '+str(t))
        return None

    e_dens = np.zeros([ngrid**3,4])
    line_nums = np.arange(int((t-t0)/dt)*lpt+n_atom+6+1, int((t-t0)/dt+1)*lpt+1) # +1 is for inclusize/exclusive
    dens_ct = 0
    x_ct = 0; y_ct = 0; z_ct = 0
    mod_val = math.ceil(ngrid/6)
    for i in range(len(line_nums)):
        my_line = np.array(linecache.getline(file_in, line_nums[i]).split())
        for j in range(len(my_line)):
            e_dens[dens_ct] = np.array([dx*x_ct, dy*y_ct, dz*z_ct, my_line[j]])
            dens_ct += 1
            z_ct += 1
        if i%mod_val == math.ceil(ngrid/6)-1:
            y_ct += 1
            z_ct = 0
            if y_ct == ngrid:
                y_ct = 0
                x_ct += 1
    e_dens[:,:3] += origin

    r_Na = np.zeros([2,3])
    r_Na[0] = np.array(linecache.getline(file_in, line_nums[0]-2).split()[2:], dtype=float)*auleng
    r_Na[1] = np.array(linecache.getline(file_in, line_nums[0]-1).split()[2:], dtype=float)*auleng

    return e_dens, r_Na

def readArVel(file_in, t, unitl=21.9166090280612):
    nmol = int(linecache.getline(file_in, 7).split()[1])
    del_t = convFloat(linecache.getline(file_in, 7).split()[3])

    suffix = "".join(list(file_in)[-5:])
    if suffix == 'water':
        lpt = int(nmol*3*3+2+8)     # lpt stands for lines for dt
        
        t0 = convFloat(linecache.getline(file_in, 7).split()[4])
        t1 = convFloat(linecache.getline(file_in, lpt+7).split()[4])
        dt = t1 - t0
        
        v = np.zeros([nmol,3])
        
        # quick check to find time
        if convFloat(linecache.getline(file_in, int((t-t0)/dt)*lpt+7).split()[4]) != t:
            print('could not find t = '+str(t))
            return None
        
        linenums = np.linspace(int((t-t0)/dt*lpt+9), int(((t-t0)/dt+1)*lpt), num=lpt-8, dtype=int)

        # part that does the reading and assigns values to matrix r
        v_counter = 0
        for i in range(len(linenums)):
            if i > 6*nmol+1 and i < math.ceil(nmol*19/3)+2:
                my_line = linecache.getline(file_in, linenums[i]).split()
                for j in range(3):
                    if v_counter < nmol:
                        v[v_counter,0] = convFloat(my_line[j])
                        v_counter += 1
            elif i > 7*nmol+1 and i < math.ceil(nmol*22/3)+2:
                my_line = linecache.getline(file_in, linenums[i]).split()
                for j in range(3):
                    if v_counter < nmol:
                        v[v_counter,1] = convFloat(my_line[j])
                        v_counter += 1
            elif i > 8*nmol+1 and i < math.ceil(nmol*25/3)+2:
                my_line = linecache.getline(file_in, linenums[i]).split()
                for j in range(3):
                    if v_counter < nmol:
                        v[v_counter,2] = convFloat(my_line[j])
                        v_counter += 1
            v_counter = 0 if v_counter == nmol else v_counter

        return v*unitl/del_t

    elif suffix == 'argon':
        lpt = int(nmol+8)

        t0 = convFloat(linecache.getline(file_in, 7).split()[4])
        t1 = convFloat(linecache.getline(file_in, lpt+7).split()[4])
        dt = t1 - t0
        
        v = np.zeros([nmol,3])
        
        # quick check to find time
        if convFloat(linecache.getline(file_in, int((t-t0)/dt)*lpt+7).split()[4]) != t:
            print('could not find t = '+str(t))
            return None

        linenums = np.linspace(int((t-t0)/dt*lpt+9), int(((t-t0)/dt+1)*lpt), num=lpt-8, dtype=int)

        for i in range(len(linenums)):
            my_line = linecache.getline(file_in, linenums[i]).split()
            v[i] = my_line[6:]

        return v*unitl/del_t

def readNaVel(file_in, t, unitl=21.9166090280612, n_solu=2):
    lpt = int(6*n_solu + 3)
    ref_file = list(file_in)
    del ref_file[-2:]
    ref_file = "".join(ref_file+list('water'))
    del_t = convFloat(linecache.getline(ref_file, 7).split()[3])
    
    t0 = convFloat(linecache.getline(file_in, 2).split()[4])
    t1 = convFloat(linecache.getline(file_in, lpt+2).split()[4])
    dt = t1 - t0
    
    v = np.zeros([n_solu,3])
    
    # quick check to find time
    if convFloat(linecache.getline(file_in, int((t-t0)/dt)*lpt+2).split()[4]) != t:
        print('could not find t = '+str(t))
        return None
    
    for i in range(0,n_solu):
        v_line = linecache.getline(file_in, int((t-t0)/dt)*lpt+6*i+9).split()
        v[i] = v_line[:3]
    
    return v*unitl/del_t

def readHF(my_dir, t, refst, n_solv=1600, n_solu=2):
    my_file = my_dir+'/hf_st'+str(refst)+'.out'
    lpt = n_solv+n_solu+3

    t0 = float(linecache.getline(my_file, 1).split()[2])
    t1 = float(linecache.getline(my_file,lpt+1).split()[2])
    dt = t1-t0

    # quick check to find time
    if float(linecache.getline(my_file, int(((t-t0)/dt)*lpt+1)).split()[2]) != t:
        print('could not find t = '+str(t))
        return None

    hf_na = np.zeros([n_solu,3]); hf_ar = np.zeros([n_solv,3])
    for i in range(n_solv):
        hf_ar[i] = linecache.getline(my_file, int(((t-t0)/dt)*lpt+4+i)).split()
    for i in range(n_solu):
        hf_na[i] = linecache.getline(my_file, int(((t-t0)/dt)*lpt+4+n_solv+i)).split()

    return hf_na, hf_ar

def readEcom(my_dir):
    # Only useful bc out.e_com spits out center of mass on 2 lines
    data = open(my_dir+'/out.e_com').readlines()
    e_com = np.zeros([int(len(data)/2),4])
    for i in range(len(e_com)):
        e_com[i] = np.array(data[2*i].split()+data[2*i+1].split()).astype(float)

    return e_com

def readDist(my_dir, n_solu=2, unitl=21.9166090280612):
    na_lines = open(my_dir+'/out.na').readlines()
    lpt_na = 6*n_solu+3

    na_pos = np.zeros([math.floor(len(na_lines)/lpt_na),7])
    for i in range(len(na_pos)):
        t = convFloat(na_lines[i*lpt_na+1].split()[4])
        na1 = na_lines[i*lpt_na+4].split()
        na2 = na_lines[i*lpt_na+10].split()
        na_pos[i] = [float(t),float(na1[0])*unitl,float(na1[1])*unitl,float(na1[2])*unitl,
                     float(na2[0])*unitl,float(na2[1])*unitl,float(na2[2])*unitl] 
    dt_na = na_pos[1,0]-na_pos[0,0]

    bond_dist = np.zeros([len(na_pos),2])
    bond_dist[:,0] = na_pos[:,0]
    for i in range(len(bond_dist)):
        bond_dist[i,1] = dist(na_pos[i,1:4],na_pos[i,4:])

    data = open(my_dir+'/out.e_com').readlines()
    full_ecom = np.zeros([int(len(data)/2),4])
    for i in range(len(full_ecom)):
        myvals = data[i*2].split()+data[i*2+1].split()
        full_ecom[i] = [float(myvals[0]),float(myvals[1]),float(myvals[2]),float(myvals[3])]
    dt_ecom = full_ecom[1,0]-full_ecom[0,0]
    #full_ecom[:,2:] *= unitl
    e_com = full_ecom[::int(dt_na/dt_ecom),:]

    na_e_dist = np.zeros([len(e_com),3])
    na_e_dist[:,0] = e_com[:,0]
    for i in range(len(na_e_dist)):
        na_e_dist[i,1] = dist(na_pos[i,1:4],e_com[i,1:])
        na_e_dist[i,2] = dist(na_pos[i,4:7],e_com[i,1:])

    return bond_dist, na_e_dist

def readEigval(my_dir, nstates=5):
    t1 = float(linecache.getline(my_dir+'/bondumb2.out',1).split()[0])
    t2 = float(linecache.getline(my_dir+'/bondumb2.out',2).split()[0])
    dt = t2-t1
    data = np.loadtxt(my_dir+'/out.eigstates')

    eigvals = np.zeros([int(len(data)/nstates),nstates+1])
    eigvals[:,0] = dt*np.arange(0,int(len(data)/nstates),1)
    for i in range(nstates):
        eigvals[:,i+1] = data[i::nstates,1]
    return eigvals

def readEnergy(my_dir):
    '''
    Goes into out.energies_mf and reads classical potential energy from all pairwise contributions
        and reads the quantum energy (eigenval)
    '''
    file_in = my_dir+'/out.energies_mf'
    
    data = np.loadtxt(file_in)

    return np.array([data[:,0],data[:,2]+data[:,4]]).T

def readEnergyDist(root, dirs, r0, r, dr):
    '''
    Goes into a root directory, reads energy and bins it according to bond distance.
    '''
    bins = np.linspace(r0, r-dr, num=int(round((r-r0)/dr)))+dr/2
    my_dict = {}
    for i in range(len(bins)):
        my_dict[str(bins[i])] = []

    for i in range(len(dirs)):
        my_dir = root+dirs[i]
        my_energy = readEnergy(my_dir)
        bond_dist = np.loadtxt(my_dir+'/bondumb2.out')

        for j in range(len(my_energy)):
            if bond_dist[j,1] < r0 or bond_dist[j,1] > r:
                continue
            else:
                my_ind = int((bond_dist[j,1]-r0)/dr)+0.5

            my_dict[str(round(r0+my_ind*dr,3))] += [my_energy[j,1]]

    energies = np.zeros([len(bins),3])
    energies[:,0] = bins
    for i in range(len(bins)):
        energies[i,1] = np.average(my_dict[str(bins[i])])
        energies[i,2] = stats.sem(my_dict[str(bins[i])])

    return energies

def readEigvec(file_in,t):
    '''
    May want to think about abandoning or rewriting this fxn
        because I may need to know where the grid is centered.
    Since cube file in au, need to conver bohr to Angstroms
        1 bohr = 0.529 A
    Should consider using readEDens() instead.
    '''
    au_len = 0.5291772108
    n_solv = 1600

    ngrid = int(linecache.getline(file_in,4).split()[0])
    natom = int(linecache.getline(file_in,3).split()[0])
    header = natom+6
    lpt_cube_in = int(ngrid*ngrid*np.ceil(ngrid/6)+header)
    
    t0 = float(linecache.getline(file_in, 1).split()[2])
    t1 = float(linecache.getline(file_in, lpt_cube_in+1).split()[2])
    dt = t1-t0

    if float(linecache.getline(file_in, int((t-t0)/dt)*lpt_cube_in+1).split()[2]) == t:
        t_frame = int((t-t0)/dt)
    elif float(linecache.getline(file_in, int((t-t0)/dt-1)*lpt_cube_in+1).split()[2]) == t:
        t_frame = int((t-t0)/dt-1)
    elif float(linecache.getline(file_in, int((t-t0)/dt+1)*lpt_cube_in+1).split()[2]) == t:
        t_frame = int((t-t0)/dt+1)
    else:
        print('Check time in .cube file')
        return None

    eigvec = []
    na_pos = np.zeros([natom-n_solv,3])
    my_lines = range(t_frame*lpt_cube_in+header+1,(t_frame+1)*lpt_cube_in+1)
    for line in my_lines:
        eigvec += linecache.getline(file_in,line).split()
    eigvec = np.array(eigvec, dtype='float').reshape([ngrid,ngrid,ngrid])

    for i in range(len(na_pos)):
        na_line = linecache.getline(file_in,my_lines[0]-(i+1)).split()
        na_pos[len(na_pos)-(i+1)] = np.array([float(na_line[2])*au_len,float(na_line[3])*au_len,float(na_line[4])*au_len])
    
    return eigvec, na_pos

def readOverlap(my_dir):
    data = open(my_dir+'/debug1.out','r').readlines()
    over = []
    time = []

    for i in range(len(data)):
        if i > 10:
            line = data[i].split()
            if line[0] == 'time':
                time = time + [float(line[2])]
            elif line[0] == 'overlap=':
                over = over + [float(line[1])]
        
    overlap = np.zeros([len(time)-1,2])
    for i in range(len(time)-1):
        overlap[i] = [time[i],over[i]]

    return overlap

def readMaxOverlap(my_dir):
    data = open(my_dir+'/overlap_test.out','r').readlines()
    over_st1 = []
    over_st2 = []
    time = []

    for i in range(len(data)):
        if i > 10:
            line = data[i].split()
            if line[0] == 'time':
                time = time + [float(line[2])]
            elif line[0] == 'overlap_st1=':
                over_st1 = over_st1 + [float(line[1])]
            elif line[0] == 'overlap_st2=':
                over_st2 = over_st2 + [float(line[1])]

    overlap = np.zeros([len(time)-1,3])
    for i in range(len(time)-1):
        overlap[i] = [time[i],over_st1[i],over_st2[i]]

    return overlap

def readHopping(my_dir):
    '''Returns time(s) when electron hops up from 1->2 and down 2->1'''
    data = np.loadtxt(my_dir+'/out.hop_prob', dtype=str)
    up = []
    down = []
    failed = []
                     
    for i in range(len(data)):
        if data[i,4] == 'T':
            if data[i,1] == data[i+1,1]:
                failed += [data[i,0]]
            else:
                if data[i,1] == '2':
                    down += [data[i,0]]
                else:
                    up += [data[i,0]]

    return np.array(up, dtype=float), np.array(down, dtype=float), np.array(failed, dtype=float)

def readCGrad(file_in, t, n_solv=1600, n_solu=2):
    '''Returns classical grad, which is essentially the velocity, for that particular time'''
    lpt = n_solv + n_solu + 1

    t0 = float(linecache.getline(file_in, 1).split()[2])
    t1 = float(linecache.getline(file_in, lpt+1).split()[2])
    dt = t1-t0

    cgrad = np.zeros([n_solv+n_solu,3])

    # quick check to find time
    if float(linecache.getline(file_in, int((t-t0)/dt*lpt+1)).split()[2]) != t:
        print('could not find t= '+str(t))
        return None

    for i in range(0,n_solu+n_solv):
        my_line = linecache.getline(file_in, int((t-t0)/dt*lpt+2+i)).split()
        cgrad[i] = np.array(my_line, dtype=float)

    return cgrad

def readQGrad(file_in, t, n_solv=1600, n_solu=2):
    '''Returns quantum grad, which is essentially the HF force, for that particular time'''
    lpt = n_solv + n_solu + 4

    t0 = float(linecache.getline(file_in, 1).split()[2])
    t1 = float(linecache.getline(file_in, lpt+1).split()[2])
    dt = t1-t0

    qgrad = np.zeros([n_solv+n_solu,3])

    # quick check to find time
    if float(linecache.getline(file_in, int((t-t0)/dt*lpt+1)).split()[2]) != t:
        print('could not find t= '+str(t))
        return None

    for i in range(0,n_solu+n_solv):
        my_line = linecache.getline(file_in, int((t-t0)/dt*lpt+5+i)).split()
        qgrad[i] = np.array(my_line, dtype=float)

    return qgrad

def calcNaArDist(r_Na, r_Ar, box_dim=43.833218056):
    '''Use with a single Na at a time'''
    Na_Ar_dist = np.zeros(len(r_Ar))
    for i in range(len(Na_Ar_dist)):
        Na_Ar_dist[i] = set_fxns.minImageDist(r_Na, r_Ar[i], [box_dim]*3)
    
    return Na_Ar_dist

def calcNaArDistTime(my_dir, n_solu=2, t0=0, t=5000, dt=20):
    n_solv = int(linecache.getline(my_dir+'/out.water', 7).split()[1])
    t_frames = np.linspace(t0,t,num=int((t-t0)/dt+1),dtype='int')
    na_ar_dist = np.zeros([n_solu, n_solv, len(t_frames)])

    for i in range(len(t_frames)):
        r_Ar = readAr(my_dir+'/out.water', t_frames[i])
        r_Na = readNa(my_dir+'/out.na', t_frames[i], n_solu=n_solu)
        for n in range(n_solu):
            na_ar_dist[n,:,i] = calcNaArDist(r_Na[n], r_Ar)
    
    return na_ar_dist, t_frames

def calcAvgEigvec(my_dir, refst=1, n_solv=1600, t0=0, t=1000, dt=20):
    '''
    Want this to go into a directory of my choice, read eigvector of my choice,
        then average them over a series of times
    Want to know Na position(s) relative ro grid
    Returns avg_eigvec as [x,y,z] cubic matrix.
        Also returns Na positions as [x,y,z] not on the grid
    Doesnt handle eigenstates with nodes bc at any time the sign can switch
    '''
    file_in = my_dir+'/eigvec'+str(refst)+'.cube'
    ngrid = int(linecache.getline(file_in, 4).split()[0])
    n_atom = int(linecache.getline(file_in, 3).split()[0])
    n_solu = n_atom-n_solv

    t_frames = np.linspace(t0, t, num=int((t-t0)/dt+1))

    avg_eigvec = np.zeros([ngrid,ngrid,ngrid])
    avg_Na_pos = np.zeros([n_solu, 3])
    for t in t_frames:
        my_eigvec, my_na_pos = readEigvec(file_in, t)
        avg_eigvec += np.abs(my_eigvec)
        avg_Na_pos += my_na_pos
    avg_eigvec /= len(t_frames)
    avg_Na_pos /= len(t_frames)

    return avg_eigvec, avg_Na_pos

def calcDensCorrWNa0(my_dir, refst, ref_eigvec, ref_na, n_solu=2, t0=0, t=1980, dt=20, spacng=25/32):
    '''
    For Na2+, the grid is centered on the dimer center of mass.
    For Na0, grid is centered on electron com.
    Returns correlation, which is 2D array, with 1st column being time and 2nd column being correlation
    '''
    ngrid = np.shape(ref_eigvec)[0]
    t_frames = np.linspace(t0,t,num=int((t-t0)/dt+1))
    corr = np.zeros([len(t_frames),n_solu+1])
    corr[:,0] = t_frames

    shift = np.zeros([n_solu,3], dtype='int')
    limits = np.zeros([3,2], dtype='int')
    for t in range(len(t_frames)):
        my_eigvec, my_na = readEigvec(my_dir+'/eigvec'+str(refst)+'.cube',t_frames[t])

        if np.shape(my_eigvec) != np.shape(ref_eigvec):
            print('dimensions of eigvec does not match dimensions of ref eigvec')
            return None

        for n in range(n_solu):
            for j in range(3):
                shift[n,j] = math.floor((ref_na[0,j]-my_na[n,j])/spacng)

        for n in range(n_solu):
            shifted_eigvec = np.zeros([ngrid,ngrid,ngrid])
            for d in range(3):
                if shift[n,d] < 0:
                    limits[d,0] = -shift[n,d]
                    limits[d,1] = ngrid
                elif shift[n,d] > 0:
                    limits[d,0] = 0
                    limits[d,1] = ngrid-shift[n,d]
                else:
                    limits[d,0] = 0
                    limits[d,1] = ngrid
            for i in range(limits[0,0], limits[0,1]):
                for j in range(limits[1,0], limits[1,1]):
                    for k in range(limits[2,0], limits[2,1]):
                        shifted_eigvec[i,j,k] = ref_eigvec[i+shift[n,0], j+shift[n,1], k+shift[n,2]]

            corr[t,n+1] = sum(my_eigvec.flatten()*shifted_eigvec.flatten())
 
    return corr

def calcDensCorrWNa2(my_dir, refst, ref_eigvec, ref_na, t0=0, t=1980, dt=20, spacng=25/32):
    '''
    Calculated eigvec corr with an averaged picture of Na2+
    Clearly in progress. Want to avoid interpolation, but not sure how to
        since bond axis can be pointed in any direction
    '''
    n_solu = 2
    ngrid = np.shape(ref_eigvec)[0]
    t_frames = np.linspace(t0,t,num=int((t-t0)/dt+1))
    corr = np.zeros([len(t_frames),n_solu+1])
    corr[:,0] = t_frames

    return None

def calcDivNa(my_dir, r_cutoff, n_solu=2, t0=0, t=500, dt=20, alpha=1, beta=6, form='poly'):
    '''
    want to calculate divergence around each sodium
    supply dir for out.water, out.na. Will calculate na_ar_dist internally
    recall na_ar_dist = np.array([n_solu, n_solv, len(t_frames)])
    6/6/2019 splitting to get positive and negative divergence
    '''
    if form == 'poly':
        def weight(r, dr, alpha=1, beta=6):
            return alpha * r**(-beta) * dr
    elif form == 'exp':
        def weight(r, dr, alpha=1, beta=6):
            return alpha * np.exp(-beta * r) * dr
    
    n_solv = int(linecache.getline(my_dir+'/out.water', 7).split()[1])
    t_frames = np.linspace(t0, t, num=int((t-t0)/dt+1), dtype='int')
    pos_div_na = np.zeros([n_solu, len(t_frames)])
    neg_div_na = np.zeros([n_solu, len(t_frames)])
    
    # read in positions
    ar_pos = np.zeros([n_solv, 3, len(t_frames)])
    na_pos = np.zeros([n_solu, 3, len(t_frames)])
    for k in range(len(t_frames)):
        ar_pos[:,:,k] = readAr(my_dir+'/out.water', t_frames[k])
        na_pos[:,:,k] = readNa(my_dir+'/out.na', t_frames[k], n_solu=n_solu)

    # need to read in velocities here
    ar_vel = np.zeros([n_solv, 3, len(t_frames)])
    na_vel = np.zeros([n_solu, 3, len(t_frames)])
    for k in range(len(t_frames)):
        ar_vel[:,:,k] = readArVel(my_dir+'/out.water', t_frames[k])
        na_vel[:,:,k] = readNaVel(my_dir+'/out.na', t_frames[k], n_solu=n_solu)
        
    # calculate dist
    na_ar_dist = np.zeros([n_solu, n_solv, len(t_frames)])
    for i in range(n_solu):
        for k in range(len(t_frames)):
            na_ar_dist[i,:,k] = calcNaArDist(na_pos[i,:,k], ar_pos[:,:,k])
    
    # calculate dr/dt by np.dot(v_Na - v_Ar, r_Na - r_Ar). Will do on the fly
    for i in range(n_solu):
        for j in range(n_solv):
            for k in range(len(t_frames)):
                if na_ar_dist[i,j,k] < r_cutoff:
                    rel_vel = np.dot(na_vel[i,:,k]-ar_vel[j,:,k], na_pos[i,:,k]-ar_pos[j,:,k])/na_ar_dist[i,j,k]
                    my_val = weight(na_ar_dist[i,j,k], rel_vel, alpha, beta)
                    if my_val > 0:
                        pos_div_na[i,k] += my_val
                    else:
                        neg_div_na[i,k] += my_val

    return pos_div_na, neg_div_na, t_frames

def calcAvgApproach(na_ar_dist, N, option='cation'):
    '''
    Calculates the average sodium-argon distance for the N closest Argons
    Since I'm likely to call this several time with different N, best to supply na_ar_dist
    recall np.shape(na_ar_dist) = [n_solu, n_solv, len(t_frames)]
    If option set to 'cation' then calculates avg approach around the cation(s)
        if set to 'dimer' then calculates avg approach around the dimer, where distance
        is the sum between Ar_i and Na1 and Ar_i and Na2. Which is dist for ellipse
    '''
    n_solu, n_solv, t_slices = np.shape(na_ar_dist)

    if option == 'cation':
        avg_approach = np.zeros([n_solu,t_slices])

        for i in range(n_solu):
            for k in range(t_slices):
                avg_approach[i,k] = np.sum(np.sort(na_ar_dist[i,:,k])[:N])/N

        return avg_approach

    elif option == 'dimer':
        if n_solu == 1:
            print('check input. n_solu=1')
            return None

        avg_approach = np.zeros(t_slices)
        dimer_dist = np.sum(na_ar_dist, axis=0)
        for k in range(t_slices):
            avg_approach[k] = np.sum(np.sort(dimer_dist[:,k])[:N])/N

        return avg_approach

def calcNumNeighbors(na_ar_dist, R):
    '''
    Calculates the number of Argons within distance R of each sodium
    Since I'm likely to call this several time with different R, best to supply na_ar_dist
    Using this instead of structure factor, S(r), and g(r) because of low statistics
    '''
    n_solu, n_solv, t_slices = np.shape(na_ar_dist)
    num_neighbors = np.zeros([n_solu, t_slices])

    for i in range(n_solu):
        for k in range(t_slices):
            num_neighbors[i,k] = n_solv - np.sum(np.ceil((na_ar_dist[i,:,k]-R)/100))

    return num_neighbors

def calcAvgAppDist(root, dirs, N=8, r0=3, r=9, dr=0.1, option='cation', method='time'):
    '''
    Returns Mx5 where M is the number of bins.
        1st column bin centers
        2nd and 3rd column avg approach and standard error of mean for Na1, respectively
        4th and 5th column avg approach and standard error of mean for Na2, respectively
    If option set to 'cation' then calculates avg approach around the cation(s)
        if set to 'dimer' then calculates avg approach around the dimer, where distance
        is the sum between Ar_i and Na1 and Ar_i and Na2. Which is dist for ellipse
    '''
    bins = np.linspace(r0, r, num=int((r-r0)/dr+1))
    if option=='cation' and method=='time':
        avg_app_hist = np.zeros([len(bins),5])

        my_dict1 = {}
        my_dict2 = {}
        for i in bins:
            my_dict1[str(i)] = []
            my_dict2[str(i)] = []

        for i in range(len(dirs)):
            na_e_dist = readDist(root+dirs[i])[1][::10]
            bond_dist = np.loadtxt(root+dirs[i]+'/bondumb2.out')[::50]
            na_ar_dist, t_frames = calcNaArDistTime(root+dirs[i], t0=bond_dist[0,0], t=bond_dist[-1,0], dt=bond_dist[1,0]-bond_dist[0,0])
            avg_approach = calcAvgApproach(na_ar_dist, N, option=option)

            for j in range(len(bond_dist)):
                if bond_dist[j,0] != t_frames[j]:
                    print('times do not match.')
                    print('dir = '+str(dirs[i]))
                    print('bond_dist['+str(j)+',0] = '+str(bond_dist[j,0])+'; t_frames['+str(j)+'] = '+str(t_frames[j]))
                    return None
                if bond_dist[j,1] > r:
                    continue
                elif bond_dist[j,1] < r0:
                    continue
                else:
                    my_index = int((bond_dist[j,1]-r0)/dr)

                if na_e_dist[0,0] < na_e_dist[0,1]:
                    my_dict1[str(round(r0+my_index*dr,2))] += [avg_approach[0,j]]
                    my_dict2[str(round(r0+my_index*dr,2))] += [avg_approach[1,j]]
                else:
                    my_dict1[str(round(r0+my_index*dr,2))] += [avg_approach[1,j]]
                    my_dict2[str(round(r0+my_index*dr,2))] += [avg_approach[0,j]]

        avg_app_hist[:,0] = bins+dr/2
        for i in range(len(bins)):
            avg_app_hist[i,1] = np.average(my_dict1[str(bins[i])])
            avg_app_hist[i,2] = stats.sem(my_dict1[str(bins[i])])
            avg_app_hist[i,3] = np.average(my_dict2[str(bins[i])])
            avg_app_hist[i,4] = stats.sem(my_dict2[str(bins[i])])

        return avg_app_hist

    if option=='cation' and method=='dist':
        avg_app_hist = np.zeros([len(bins),5])

        my_dict1 = {}
        my_dict2 = {}
        for i in bins:
            my_dict1[str(i)] = []
            my_dict2[str(i)] = []

        for i in range(len(bins)):
            bond_dist = np.loadtxt(root+dirs[i]+'/bondumb2.out')[::50]
            na_ar_dist, t_frames = calcNaArDistTime(root+dirs[i], t0=bond_dist[0,0], t=bond_dist[-1,0], dt=bond_dist[1,0]-bond_dist[0,0])
            avg_approach = calcAvgApproach(na_ar_dist, N, option=option)

            for j in range(len(bond_dist)):
                if bond_dist[j,0] != t_frames[j]:
                    print('times do not match')
                    print('dir = '+str(dirs[i]))
                    print('bond_dist['+str(j)+',0] = '+str(bond_dist[j,0])+'; t_frames['+str(j)+'] = '+str(t_frames[j]))
                    return None
                if bond_dist[j,1] > r:
                    continue
                elif bond_dist[j,1] < r0:
                    continue
                else:
                    my_index = int((bond_dist[j,1]-r0)/dr)

                if min(na_ar_dist[0,:,j]) < min(na_ar_dist[1,:,j]):
                    my_dict1[str(round(r0+my_index*dr,2))] += [avg_approach[0,j]]
                    my_dict2[str(round(r0+my_index*dr,2))] += [avg_approach[1,j]]
                else:
                    my_dict1[str(round(r0+my_index*dr,2))] += [avg_approach[1,j]]
                    my_dict2[str(round(r0+my_index*dr,2))] += [avg_approach[0,j]]

        avg_app_hist[:,0] = bins+dr/2
        for i in range(len(bins)):
            avg_app_hist[i,1] = np.average(my_dict1[str(bins[i])])
            avg_app_hist[i,2] = stats.sem(my_dict1[str(bins[i])])
            avg_app_hist[i,3] = np.average(my_dict2[str(bins[i])])
            avg_app_hist[i,4] = stats.sem(my_dict2[str(bins[i])])

        return avg_app_hist

    elif option=='dimer' and method=='time':
        avg_app_hist = np.zeros([len(bins),3])

        my_dict = {}
        for i in bins:
            my_dict[str(i)] = []

        for i in range(len(bins)):
            bond_dist = np.loadtxt(root+dirs[i]+'/bondumb2.out')[::15]
            na_ar_dist, t_frames = calcNaArDistTime(root+dirs[i], t0=bond_dist[0,0], t=bond_dist[-1,0], dt=bond_dist[1,0]-bond_dist[0,0])
            avg_approach = calcAvgApproach(na_ar_dist, N, option=option)

            for j in range(len(bond_dist)):
                if bond_dist[j,0] != t_frames[j]:
                    print('times do not match.')
                    print('dir = '+str(dirs[i]))
                    print('bond_dist['+str(j)+',0] = '+str(bond_dist[j,0])+'; t_frames['+str(j)+'] = '+str(t_frames[j]))
                    return None
                if bond_dist[j,1] > r:
                    my_index = int((r-r0)/dt)
                elif bond_dist[j,1] < r0:
                    continue
                else:
                    my_index = int((bond_dist[j,1]-r0)/dr)

                my_dict[str(round(r0+my_index*dr,2))] += [avg_approach[j]]

        avg_app_hist[:,0] = bins+dr/2
        for i in range(len(bins)):
            avg_app_hist[i,1] = np.average(my_dict[str(bins[i])])
            avg_app_hist[i,2] = stats.sem(my_dict[str(bins[i])])

        return avg_app_hist

def calcQSepDist(root, dirs, nstate, r0=3.0, r=9.0, dr=0.1, chkpt='False', resume='False'):
    '''
    Main challenge is to have this run on my computer. Since loading lots of data, need to
        organize very well. Possibly calculte in chunks and then combine and average at a
        later stage.
    '''
    if resume.lower() == 'false':
        bins = np.linspace(r0, r-dr, num=int((r-r0)/dr))
        my_dict = {}
        for i in bins:
            my_dict[str(np.round(i,2))] = []

        e_distri_hist = np.zeros([len(bins),4])
        e_distri_hist[:,0] = bins

        for i in range(len(dirs)):
            my_dir = root+dirs[i]

            bond_dist = np.loadtxt(my_dir+'/bondumb2.out')[:205:5]
            for j in range(len(bond_dist)-1):
                if bond_dist[j,1] > r or bond_dist[j,1] < r0:
                    continue
                else:
                    my_ind = int((bond_dist[j,1]-r0)/dr)

                e_distri = calcQSep(my_dir, bond_dist[j,0], nstate=nstate, r_cutoff=5, auleng=0.5291772108)
                my_dict[str(round(r0+my_ind*dr,2))] += [np.abs(e_distri[0] - e_distri[1])]

        if chkpt.lower() == 'false':
            for i in range(len(bins)):
                if len(my_dict[str(bins[i])]) != 0:
                    e_distri_hist[i,1] = np.average(my_dict[str(round(bins[i],2))])
                    e_distri_hist[i,2] = np.std(my_dict[str(round(bins[i],2))])
                    e_distri_hist[i,3] = stats.sem(my_dict[str(round(bins[i],2))])

            return e_distri_hist

        elif chkpt.lower() == 'true':
            return np.save('delta_q_chkpt.npy', my_dict)

    elif resume.lower() == 'true':
        bins = np.linspace(r0, r-dr, num=int((r-r0)/dr))
        my_dict = dict(enumerate(np.load('delta_q_chkpt.npy', allow_pickle=True).flatten(),1))[1]

        e_distri_hist = np.zeros([len(bins),4])
        e_distri_hist[:,0] = bins

        for i in range(len(dirs)):
            my_dir = root+dirs[i]

            bond_dist = np.loadtxt(my_dir+'/bondumb2.out')[500::50]
            for j in range(len(bond_dist)-1):
                if bond_dist[j,1] > r or bond_dist[j,1] < r0:
                    continue
                else:
                    my_ind = int((bond_dist[j,1]-r0)/dr)

                e_distri = calcQSep(my_dir, bond_dist[j,0], nstate=nstate, r_cutoff=5, auleng=0.5291772108)
                my_dict[str(round(r0+my_ind*dr,2))] += [np.abs(e_distri[0] - e_distri[1])]

        if chkpt.lower() == 'false':
            for i in range(len(bins)):
                if len(my_dict[str(round(bins[i],2))]) != 0:
                    e_distri_hist[i,1] = np.average(my_dict[str(round(bins[i],2))])
                    e_distri_hist[i,2] = np.std(my_dict[str(round(bins[i],2))])
                    e_distri_hist[i,3] = stats.sem(my_dict[str(round(bins[i],2))])

            return e_distri_hist

        elif chkpt.lower() == 'true':
            return np.save('delta_q_chkpt.npy', my_dict)

def calcQSepTime(my_dir, nstate, r_cutoff=5, auleng=0.5291772108, dt=20):
    '''
    Calculates the electronic density around each sodium and spits out as [N,3]
        where N is the number of time points. To get charge separation, take
        the difference between columns 2 and 3
    This is hard coded for 2 Na atoms
    2/3/2021 dt for .cube files set to 20 fs
    '''
    Na_file = open(my_dir+'/out.na').readlines()
    t0 = convFloat(Na_file[1].split()[4])
    del_t = convFloat(Na_file[16].split()[4]) - t0
    t = (len(Na_file)/15-2) * del_t + t0
    del Na_file

    t_frames = np.arange(t0, t+dt, dt)
    e_distri = np.zeros([len(t_frames),3])

    for i in range(len(t_frames)):
        e_distri[i,1:] = calcQSep(my_dir, t_frames[i], nstate=nstate, r_cutoff=r_cutoff, auleng=auleng)
    e_distri[:,0] = t_frames

    return e_distri

def calcNAC(my_dir, t0=0, t=500, dt=20, cfile='/out.cgrad', qfile='/out.qgrad21', n_solv=1600, n_solu=2):
    '''
    Goes into a directory and calculates the NAC contribution from each Argon.
    Returns as TxM+1 array with T being the number of time points, M the number of Argon plus number of Na
        and first column for time.
    Since I'm not sure how best to sort the Argons yet, will leave sorting to after the fact.
    '''
    t_frames = np.linspace(t0, t, num=int((t-t0)/dt+1))
    scalefac = float(linecache.getline(my_dir+qfile,2).split()[2])
    eigvals = readEigval(my_dir, dt=2, nstates=2)[int(t0/2):int(t/2)+1:int(dt/2)]

    NAC = np.zeros([len(t_frames), n_solv+n_solu+1])
    NAC[:,0] = t_frames

    for i in range(len(t_frames)):
        cgrad = readCGrad(my_dir+cfile, t_frames[i], n_solv=n_solv, n_solu=n_solu)
        qgrad = readQGrad(my_dir+qfile, t_frames[i], n_solv=n_solv, n_solu=n_solu)
        if t_frames[i] != eigvals[i,0]:
            print('Check that time from eigvals match t_frames')
            return None

        for j in range(n_solv+n_solu):
            NAC[i,j+1] = np.dot(cgrad[j],qgrad[j])
        NAC[i,1:] = NAC[i,1:]*scalefac/(eigvals[i,2]-eigvals[i,1])

    return NAC

def calcArArLJFor(my_dir, t):
    '''
    Calculates the LJ forces between Argons
    Returns as [n_solv, n_solv, 3]
    '''
    sig, eps = (3.405, 0.99607)

    r_Na = readNa(my_dir+'/out.na', t)
    r_Ar = readAr(my_dir+'/out.water', t)
    r_Na, r_Ar = rotDimerEnv(r_Na, r_Ar)

#    ar_ar_pot = np.zeros([len(r_Ar),len(r_Ar)])
    ar_ar_for = np.zeros([len(r_Ar), len(r_Ar), 3])
    ar_ar_dist = np.zeros([len(r_Ar), len(r_Ar), 3])
    ar_ar_scal = np.zeros([len(r_Ar), len(r_Ar)])

    for m in range(len(r_Ar)-1):
        for n in range(m+1,len(r_Ar)):
            ar_ar_dist[m,n,:] = set_fxns.minImage(r_Ar[m] - r_Ar[n], 43.833218056)

    for m in range(len(r_Ar)-1):
        for n in range(m+1,len(r_Ar)):
            r2 = (ar_ar_dist[m,n,0]**2 + ar_ar_dist[m,n,1]**2 + ar_ar_dist[m,n,2]**2)
            ar_ar_scal[m,n] = np.sqrt(r2)
            red_r6 = (sig**2/r2)**3
            r = np.sqrt(r2)
#            f = 4*eps*(red_r6**2-red_r6)    # this is for potential
            f = 48*eps/r*(red_r6**2-0.5*red_r6)    # this is the force

#            ar_ar_pot[m,n] = f
            ar_ar_for[m,n,:] = f*ar_ar_dist[m,n,:]/r

    return ar_ar_for

def calcNaArLJFor(my_dir, t, n_solu=2, core='cation', Pol='False'):
    '''
    Calculates the LJ forces between each Na and the Argons
        and taken from perspective of the Na
    Returns as [n_solu, n_solv, 3]
    For Na+ : sig_Na, eps_Na = (2.69, 0.5144) units of Ang and kj/mol
    For Na0 : sig_Na, eps_Na = (3.22, 1.5973)
    divide by 96.485 to convert kj/mol to eV
    '''
    sig_Ar, eps_Ar = (3.405, 0.99607)
    if core.lower() == 'cation':
        sig, eps = ((2.69+sig_Ar)/2, np.sqrt(0.5144*eps_Ar)/96.485)
    elif core.lower() == 'neutral':
        sig, eps = ((3.22+sig_Ar)/2, np.sqrt(1.5973*eps_Ar)/96.485)

    r_Na = readNa(my_dir+'/out.na', t, n_solu=n_solu)
    r_Ar = readAr(my_dir+'/out.water', t)
    r_Na, r_Ar = rotDimerEnv(r_Na, r_Ar)

    na_ar_for = np.zeros([len(r_Na), len(r_Ar), 3])
    na_ar_dist = np.zeros([len(r_Na), len(r_Ar), 3])

    for m in range(len(r_Na)):
        for n in range(len(r_Ar)):
            na_ar_dist[m,n,:] = set_fxns.minImage(r_Na[m] - r_Ar[n], 43.833218056)

    for m in range(len(r_Na)):
        for n in range(len(r_Ar)):
            r2 = (na_ar_dist[m,n,0]**2 + na_ar_dist[m,n,1]**2 + na_ar_dist[m,n,2]**2)
            red_r6 = (sig**2/r2)**3
            r = np.sqrt(r2)
#            f = 4*eps*(red_r6**2-red_r6)   # this is for potential
            f = 48*eps/r*(red_r6**2-0.5*red_r6) # this is for force
#            f = 48*eps/r*red_r6**2     # force focusing on repulsive
            if Pol.lower() == 'true':
#                f += -0.5 * 11.08*0.529177**3 * r2**-2  # for potential
                f += -2 * 11.08*0.529177**3 * r**-5 # for force

            na_ar_for[m,n,:] = f*na_ar_dist[m,n,:]/r

    return na_ar_for

def trackBelt(na_ar_dist, N):
    '''
    Follows idea of calcAvgApproach with otion='dimer'
    But returns the distance of belt Ar relative to each Na so that I can
        see if the motion of belt Ar (relative to each Na) tracks with e-com
    Returns as 3d array with shape=[t_slices, N, n_solu] with N number of Argons to track
    Note that changing order of dimensions between na_ar_dist and na_beltAr_dist
        because realized that its better to have the number of observations as the first
        dimension, not the last. Basically, I was an idiot and dont feel like going back
        to make all the necessary changes
    Turns out this data is hard to read because belt Ar can switch identity
    '''
    n_solu, n_solv, t_slices = np.shape(na_ar_dist)
    na_beltAr_dist = np.zeros([t_slices, N, n_solu])
    dimer_dist = np.sum(na_ar_dist, axis=0).T

    for i in range(t_slices):
        closest_Ar = list(np.argsort(dimer_dist[i])[:N])
        na_beltAr_dist[i] = na_ar_dist[:,closest_Ar,i].T

    return na_beltAr_dist

def calcNumBeltAr(my_dir, kappa):
    '''
    Calculates number of Ar in the belt of the dimer.
    Defined as being in the belt if dimer com-Ar distance is less than 
        h=1/2 * (elliptical dist**2 + bond distance**2)**(1/2) with a smoothing function
    Returns as Nx2, N being the number of time steps in the directory.
    In Progress.
    '''

    return None

def sumEigvec(file_in,t):
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
        return None

    eigvec = []
    my_lines = range(t_frame*lpt_cube_in+header+1,(t_frame+1)*lpt_cube_in+1)
    for line in my_lines:
        eigvec += linecache.getline(file_in,line).split()
    eigvec = np.array(eigvec, dtype='float')
    pos = [i for i in eigvec if i > 0]
    neg = [i for i in eigvec if i < 0]
    
    return np.sum(eigvec), np.sum(neg), np.sum(pos)

def closestAr(r_Na, r_Ar, n_cutoff=12, option='cation', index='False'):
    '''
    Calculates n_cutoff closest argons to each sodium and then appends the 2 lists
        while avoiding duplicates
    Returns as Nx3 matrix of cartesian coordinates where n_cutoff < N < 2*n_cutoff
    If option set to 'cation' then calculates avg approach around the cation(s)
        if set to 'dimer' then calculates avg approach around the dimer, where distance
        is the sum between Ar_i and Na1 and Ar_i and Na2. Which is dist for ellipse
    '''
    if option=='cation':
        ar_index = []
        for i in range(len(r_Na)):
            na_ar_dist = calcNaArDist(r_Na[i], r_Ar)
            ar_index += list(np.argsort(na_ar_dist)[:n_cutoff])
        ar_index = list(set(ar_index))

        closest_ar = r_Ar[ar_index,:]

        if index.lower() == 'false':
            return closest_ar
        elif index.lower() == 'true':
            return closest_ar, ar_index

    elif option=='dimer':
        ar_index = []
        dimer_dist = np.zeros(len(r_Ar))
        for i in range(len(r_Na)):
            dimer_dist += calcNaArDist(r_Na[i], r_Ar)

        ar_index += list(np.argsort(dimer_dist)[:n_cutoff])

        closest_ar = r_Ar[ar_index,:]

        if index.lower() == 'false':
            return closest_ar
        elif index.lower() == 'true':
            return closest_ar, ar_index

def axialAr(r_Na, r_Ar, r_cutoff=3.3, phi_cutoff=30, const=[1,1]):
    '''
    Finds closest axial Ar to each sodium. Returns Ar indices for each Na
        with that for Na first
    '''
    ar_na1_index = []
    ar_na2_index = []

    bond_axis = (r_Na[1]-r_Na[0])/set_fxns.norm(r_Na[1]-r_Na[0])
    
    na1_ar_dist = calcNaArDist(r_Na[0], r_Ar)
    for i in range(len(na1_ar_dist)):
        if na1_ar_dist[i] < r_cutoff:
            z = -np.dot(r_Ar[i], bond_axis)
            if np.arccos(z/na1_ar_dist[i]) < phi_cutoff/180*np.pi:
                ar_na1_index += [i]
        else:
            continue

    na2_ar_dist = calcNaArDist(r_Na[1], r_Ar)
    for i in range(len(na2_ar_dist)):
        if na2_ar_dist[i] < r_cutoff:
            z = np.dot(r_Ar[i], bond_axis)
            if np.arccos(z/na2_ar_dist[i]) < phi_cutoff/180*np.pi:
                ar_na2_index += [i]
        else:
            continue

    return np.array([ar_na1_index,ar_na2_index])

def spherProjDimer(r_Na, r_Ar, sigma, radius, n_cutoff=12, Ar_metric='cation', N=1000, project='None'):
    '''
    Given solute and solvent positions, creates shell around solute and projects
        solvent positions onto shell. Can either project by simple overlap or 
        something more complicated. Also needs
        sigma, which is LJ radius of solvent
        radius, radius of shell enscribing solute
    Returns Nx4 matrix, first 3 columns x,y,z coordinates of shell pts
        4th column identifies if points occupied(=1) or not(=0)
    '''
    bond_dist = dist(r_Na[0], r_Na[1])

    # need to create shell points while accounting for overlap of the 2 shells
    if bond_dist > 2*radius:
        shell_pts = np.append(set_fxns.fibonacciSphere(N, r=radius) + r_Na[0],
                    set_fxns.fibonacciSphere(N, r=radius) + r_Na[1], axis=0)
        shell_pts = np.append(shell_pts, [[0]]*(2*N), axis=1)
    else:
        shell_pts_na1 = set_fxns.fibonacciSphere(N, r=radius) + r_Na[0]
        shell_pts_na2 = set_fxns.fibonacciSphere(N, r=radius) + r_Na[1]
        del_list1 = []; del_list2 = []
        for i in range(N):
            if dist(shell_pts_na1[i], r_Na[1]) < radius:
                del_list1 += [i]
            if dist(shell_pts_na2[i], r_Na[0]) < radius:
                del_list2 += [i]
        shell_pts_na1 = np.delete(shell_pts_na1, del_list1, axis=0)
        shell_pts_na2 = np.delete(shell_pts_na2, del_list2, axis=0)

        shell_pts = np.append(shell_pts_na1, shell_pts_na2, axis=0)
        shell_pts = np.append(shell_pts, [[0]]*len(shell_pts), axis=1)

    # need to find n_cutoff closest argons to each sodium and then append the 2 lists
    # while avoiding duplicates
    closest_ar = closestAr(r_Na, r_Ar, n_cutoff=n_cutoff, option=Ar_metric)

    # now project argon onto shell
    if project == 'None':
        for n in range(len(shell_pts)):
            my_dist = np.zeros(len(closest_ar))
            for m in range(len(closest_ar)):
                my_dist[m] = dist(shell_pts[n], closest_ar[m])/sigma
            if min(my_dist) < 1:
                shell_pts[n,3] = 1

    return shell_pts, closest_ar

def calcOccupancy(root, dirs, eps, radius, r=9.0, r0=3.0, dr=0.1, n_cutoff=12, num=2000, project='None'):
    '''
    Calculates average occupancy of shell around dimer.
    '''
    bins = np.linspace(r0, r, num=int((r-r0)/dr+1))
    avg_occ = np.zeros([len(bins),3])
    my_dict = {}

    for i in bins:
        my_dict[str(i)] = []

    for i in range(len(dirs)):
        bond_dist = np.loadtxt(root+dirs[i]+'/bondumb2.out')[::50,:]
        for t in range(len(bond_dist)):
            r_Na = readNa(root+dirs[i]+'/out.na', bond_dist[t,0])
            r_Ar = readAr(root+dirs[i]+'/out.water', bond_dist[t,0])
            shell_pts = spherProjDimer(r_Na, r_Ar, eps, radius, N=num)[0]

            my_occ = np.sum(shell_pts[:,3])/len(shell_pts)

            if bond_dist[t,1] > r:
                my_index = int((r-r0)/dr)
            elif bond_dist[t,1] < r0:
                continue
            else:
                my_index = int((bond_dist[t,1]-r0)/dr)

            my_dict[str(round(r0+my_index*dr,2))] += [my_occ]

    avg_occ[:,0] = bins+dr/2
    for i in range(len(bins)):
        avg_occ[i,1] = np.average(my_dict[str(bins[i])])
        avg_occ[i,2] = stats.sem(my_dict[str(bins[i])])

    return avg_occ

def calcArPot(my_dir, t, n_solu=2):
    '''
    Calculates Ar-e and Ar-Na potential for all Ar
    Really, this is the Ar potential that the electron would feel if centered on the sodium
    In units of eV. Output is [N,8]
    Columns being full Ar-e pseudo on Na1 Na2, Ar-e static exchange, Ar-e polarization, Ar-Na Coulomb
    '''
    # parameters for static-exchange and polarization terms
    A = 0.28 * 27.211386
    B = 2.55 / 0.529177
    r_c = 3 * 0.529177

    del_e = 0.5 / 0.529177**2
    alpha_ar = 11.08 * 0.529177**3

    # for Coulomb
    k = 14.3996

    r_Na = readNa(my_dir+'/out.na', t, n_solu=n_solu)
    r_Ar = readAr(my_dir+'/out.water', t)
    r_Na, r_Ar = rotDimerEnv(r_Na, r_Ar)

    na_ar_dist = np.zeros([n_solu, len(r_Ar), 3])
    ar_exchange = np.zeros(n_solu)
    ar_pol = np.zeros(n_solu)
    ar_coulomb = np.zeros(n_solu)

    for m in range(n_solu):
        for n in range(len(r_Ar)):
            na_ar_dist[m,n,:] = set_fxns.minImage(r_Na[m] - r_Ar[n], 43.833218056)

    for m in range(n_solu):
        r2 = na_ar_dist[m,:,0]**2 + na_ar_dist[m,:,1]**2 + na_ar_dist[m,:,2]**2
        r = np.sqrt(r2)

        ar_exchange[m] = np.sum(A/(1+np.exp(B*(r-r_c))))
        ar_pol[m] = np.sum(-0.5*(1-np.exp(-1*del_e*r2))**2 * alpha_ar / r2**2)
        ar_coulomb[m] = np.sum(k/r)

    return np.array([ar_exchange+ar_pol, ar_exchange, ar_pol, ar_coulomb]).flatten()

def calcCoordNum(my_dir, r_cutoff, kappa, option='dimer', n_solu=2):
    '''
    calculates coordination number around each cation or around the dimer.
    Similar to structure factor.
    '''
    na_lines = open(my_dir+'/out.na').readlines()
    lpt_na = int(6*n_solu+3)
    t0 = convFloat(na_lines[1].split()[4])
    t1 = convFloat(na_lines[lpt_na+1].split()[4])
    dt_na = t1-t0

    t_frames = np.arange(t0, len(na_lines)/lpt_na*dt_na, dt_na)
    del na_lines, t0, t1

    if option == 'dimer':
        coord_num = np.zeros([len(t_frames),2])
        coord_num[:,0] = t_frames

        for i in range(len(t_frames)):
            r_Na = readNa(my_dir+'/out.na', t_frames[i])
            r_Ar = readAr(my_dir+'/out.water', t_frames[i])
            bond_dist = dist(r_Na[0], r_Na[1])

            dimer_dist = calcNaArDist(r_Na[0], r_Ar) + calcNaArDist(r_Na[1], r_Ar)

            coord_num[i,1] = np.sum(1/(np.exp((dimer_dist-(bond_dist+2*r_cutoff))/kappa)+1))

        return coord_num

    elif option == 'cation':
        coord_num = np.zeros([len(t_frames),3])
        coord_num[:,0] = t_frames

        for i in range(len(t_frames)):
            r_Na = readNa(my_dir+'/out.na', t_frames[i])
            r_Ar = readAr(my_dir+'/out.water', t_frames[i])

            coord_num[i,1] = np.sum(1/(np.exp((calcNaArDist(r_Na[0],r_Ar)-r_cutoff)/kappa)+1))
            coord_num[i,2] = np.sum(1/(np.exp((calcNaArDist(r_Na[1],r_Ar)-r_cutoff)/kappa)+1))

        return coord_num

def calcOvoidFluc(my_dir, r_cutoff, kappa, n_solu=2):
    '''
    returns scalar quantity wrt to time
    '''
    na_lines = open(my_dir+'/out.na').readlines()
    lpt_na = int(6*n_solu+3)
    t0 = convFloat(na_lines[1].split()[4])
    t1 = convFloat(na_lines[lpt_na+1].split()[4])
    dt_na = t1-t0

    t_frames = np.arange(t0, len(na_lines)/lpt_na*dt_na, dt_na)
    del na_lines, t0, t1

    ovoid_fluc = np.zeros([len(t_frames),4])
    ovoid_fluc[:,0] = t_frames

    for i in range(len(t_frames)):
        r_Na = readNa(my_dir+'/out.na', t_frames[i])
        r_Ar = readAr(my_dir+'/out.water', t_frames[i])
        bond_dist = dist(r_Na[0], r_Na[1])

        dimer_dist = calcNaArDist(r_Na[0], r_Ar) + calcNaArDist(r_Na[1], r_Ar)

        coord_num = int(np.round(np.sum(1/(np.exp((dimer_dist-(bond_dist+2*r_cutoff))/kappa)+1))))

        ovoid_fluc[i,1] = np.sum(np.sort(dimer_dist)[:coord_num] - (bond_dist+2*r_cutoff))/coord_num
        ovoid_fluc[i,2] = stats.sem(np.sort(dimer_dist)[:coord_num] - (bond_dist+2*r_cutoff))
        ovoid_fluc[i,3] = np.std(np.sort(dimer_dist)[:coord_num] - (bond_dist+2*r_cutoff))
#        ovoid_fluc[i,1] = np.sum(np.sort(dimer_dist)[:12] - (bond_dist+2*r_cutoff))/12

    return ovoid_fluc

def avgOccDist(root, dirs, eps, radius, r, dr, n_cutoff=12, Ar_metric='cation', num=2000, ordered='electron', dup=False, project='None'):
    '''
    Calculates avg occupance of shell around dimer for bond dist in [r, r+dr)
    Returns as numx4 just as spherProjDimer(), but 4th column is an average between [0,1]
    '''
    r_Na = np.array([[(r+dr/2)/2,0,0], [-(r+dr/2)/2,0,0]])
    shell_pts = spherProjDimer(r_Na, np.array([[0,0,2*r],[0,0,-2*r]]), 1e-6, radius, N=num)[0]

    norm_count = 0
    for i in range(len(dirs)):
        bond_dist = np.loadtxt(root+dirs[i]+'/bondumb2.out')[250::10,:]
        e_com = readEcom(root+dirs[i])[::5,:]
        for j in range(len(bond_dist)):
            if bond_dist[j,1] >= r and bond_dist[j,1] < r+dr:
                norm_count += 1

                if ordered.lower() == 'electron':
                    my_Na = readNa(root+dirs[i]+'/out.na', bond_dist[j,0])
                    my_Ar = readAr(root+dirs[i]+'/out.water', bond_dist[j,0])
                    my_Ar = closestAr(my_Na, my_Ar, n_cutoff=n_cutoff, option=Ar_metric)
                    if dist(my_Na[0], e_com[j,1:]) < dist(my_Na[1], e_com[j,1:]):
                        my_Na, my_Ar = rotDimerEnv(my_Na, my_Ar)
                    else:
                        my_Na, my_Ar = rotDimerEnv(my_Na, my_Ar)
                        my_Na *= [-1,1,1]; my_Ar *= [-1,1,1]

                elif ordered.lower() == 'ar':
                    my_Na = readNa(root+dirs[i]+'/out.na', bond_dist[j,0])
                    my_Ar = readAr(root+dirs[i]+'/out.water', bond_dist[j,0])
                    my_Ar = closestAr(my_Na, my_Ar, n_cutoff=n_cutoff, option=Ar_metric)
                    my_Na, my_Ar = rotDimerEnv(my_Na, my_Ar)

                    my_val1 = np.min(calcNaArDist(my_Na[0], my_Ar))
                    my_val2 = np.min(calcNaArDist(my_Na[1], my_Ar))
                    if my_val2 < my_val1:
                        my_Na *= [-1,1,1]; my_Ar *= [-1,1,1]
                
                if dup == True:
                    for k in range(6):
                        rot_ang = k*np.pi/3
                        rot_M = np.array([[1,0,0],[0,np.cos(rot_ang),-np.sin(rot_ang)],[0,np.sin(rot_ang),np.cos(rot_ang)]])
                        for l in range(len(my_Ar)):
                            my_Ar[l] = np.matmul(my_Ar[l], rot_M)

                        my_shell_pts = spherProjDimer(r_Na, my_Ar, eps, radius, N=num, project=project)[0]
                        shell_pts[:,3] += my_shell_pts[:,3]/6
                else:
                    my_shell_pts = spherProjDimer(r_Na, my_Ar, eps, radius, N=num, project=project)[0]
                    shell_pts[:,3] += my_shell_pts[:,3]

    print(norm_count)
    shell_pts[:,3] /= norm_count

    return shell_pts

def rotDimerEnv(r_Na, r_Ar, *argv):
    '''
    Rotates coordinates such that dimer bond axis is along x-axis with Na1 pointing toward +x-axis.
    '''
    # recenters sodiums and argon s.t. dimer center of mass at origin
    r_solu = r_Na - (r_Na[0]+r_Na[1])/2
    r_solv = r_Ar - (r_Na[0]+r_Na[1])/2

    # project solu and solvent onto unit sphere to make rotations easier
    radial_solu = np.array([[set_fxns.norm(r_solu[i])] for i in range(len(r_solu))])
    radial_solv = np.array([[set_fxns.norm(r_solv[i])] for i in range(len(r_solv))])

    r_solu /= radial_solu
    r_solv /= radial_solv

    # rotate atoms s.t. dimer is along x-axis with Na1 on +x-axis
    if set_fxns.norm(np.cross(r_solu[0], [1,0,0])) > 1e-6:
        rot_ax = np.cross(r_solu[0], [1,0,0])/set_fxns.norm(np.cross(r_solu[0], [1,0,0]))
        rot_ang = 2*np.arcsin(0.5*dist(r_solu[0], [1,0,0]))
        cross_mat = np.array([[0,-rot_ax[2],rot_ax[1]],
            [rot_ax[2],0,-rot_ax[0]],[-rot_ax[1],rot_ax[0],0]])

        rot_m = np.cos(rot_ang)*np.identity(3) + \
                np.sin(rot_ang)*cross_mat + (1-np.cos(rot_ang))*np.outer(rot_ax,rot_ax)
        for i in range(len(r_solu)):
            r_solu[i] = np.matmul(rot_m,r_solu[i])*radial_solu[i,0]
        for i in range(len(r_solv)):
            r_solv[i] = np.matmul(rot_m,r_solv[i])*radial_solv[i,0]

    if len(argv) == 0:
        return r_solu, r_solv

#    elif len(argv) == 1:
#        e_dens = argv[0]
#        rot_dens = np.zeros([len(e_dens),4])
#        for i in range(len(e_dens)):
#            rot_dens[i,:3] = np.matmul(rot_m, e_dens[i,:3])
#        rot_dens[:,3] = e_dens[:,3]
#
#        return r_solu, r_solv, rot_dens
    elif len(argv) == 1:
        verts = argv[0]
        rot_verts = np.zeros([len(verts),3])
        for i in range(len(verts)):
            rot_verts[i] = np.matmul(rot_m, verts[i])

        return r_solu, r_solv, rot_verts

    else:
        print('rotDimerEnv() does not support extra arguments. Supply the correct number of arguments')
        return None 

def stagAngle(r_Na, r_solv, Na1_THF, Na2_THF):
    '''
    I think Devon's Na2+ in THF for (5,5) configurations looks highly symmetric.
        Where the environment around Na_1 looks like a mirror of Na_2 with some rotation.
        Which is to say R(theta, r) M(sigma_1, sigma_2) (r_Na[0]T) = (r_Na[1]T)
            where M(sigma_1, sigma_2) is a mirror matrix about some plane definited by 
            vectors sigma_1, sigma_2, and R(theta, r) is a rotation matrix by angle theta
            about the bond axis, r.
    The challenge is to calculate that staggering angle, theta. 
    The goal is to say that prior to excitation, theta=0 and this angle the fes of the electron
        is actually bonding, but unstable. As time progresses, theta increases and the electron
        becomes nonbonding/antibonding. The bond dist then increases, which finally allows the 
        electron to rotate s.t. the node is perpendicular to the bond axis.
    Returns the staggering angle, theta, and the standard error, (Delta theta)^2
    For now, I expect len(r_solv) = 10.
    '''
    # in case r_solv includes all solvents
#    r_solv = closestAr(r_Na, r_solv, n_cutoff=5, option='cation')

    # rotate coordinates so that dimer is along x-axis with Na[0] on +x-axis
#    r_Na, r_solv = rotDimerEnv(r_Na, r_solv)

    # goes through r_solv and identifies which are closest to Na1 and Na[2], respectively
#    solv_Na2 = r_solv[np.argsort(r_solv, axis=0)[:5,0], :] # need to check that argsort will get the job done
#    solv_Na1 = r_solv[np.argsort(r_solv, axis=0)[5:,0], :] # and that indices work out
    solv_Na1 = r_solv[Na1_THF,:]
    solv_Na2 = r_solv[Na2_THF,:]

    # mirror matrix is set because I know the dimer is along the x-axis
    M = np.array([[-1,0,0], [0,1,0], [0,0,1]])

    # now the challenge is determining theta. The rotation axis is going to be the x-axis
    # I think what I'll do is calculate the rotation angle between each pair of non-axial solvent molecules
    # and from each pair, take the minimum. Should get 4 minima, and hopefully the spread among the minima is
    # negligible.
    # this is not going to elegant
    rot_ax = np.array([1,0,0])
    angles_Na1 = np.zeros(5)
    my_array = np.zeros(5)
    for i in range(5):
        my_dist = solv_Na1[i] - r_Na[0]
        z_proj = np.dot([0,0,1], my_dist)/set_fxns.norm(my_dist)
        y_proj = np.dot([0,1,0], my_dist)/set_fxns.norm(my_dist)
        my_array[i] = np.dot([1,0,0], my_dist)
        if y_proj > 0 and z_proj > 0:
            angles_Na1[i] = np.arctan(y_proj/z_proj)
        elif y_proj < 0 and z_proj > 0:
            angles_Na1[i] = np.arctan(y_proj/z_proj) + 2*np.pi
        else:
            angles_Na1[i] = np.arctan(y_proj/z_proj) + np.pi
    angles_Na1 = np.delete(angles_Na1, np.where(my_array == np.max(my_array))[0][0], axis=0)

    angles_Na2 = np.zeros(5)
    my_array = np.zeros(5)
    for i in range(5):
        my_dist = solv_Na2[i] - r_Na[1]
        z_proj = np.dot([0,0,1], my_dist)/set_fxns.norm(my_dist)
        y_proj = np.dot([0,1,0], my_dist)/set_fxns.norm(my_dist)
        my_array[i] = np.dot([-1,0,0], my_dist)
        if y_proj > 0 and z_proj > 0:
            angles_Na2[i] = np.arctan(y_proj/z_proj)
        elif y_proj < 0 and z_proj > 0:
            angles_Na2[i] = np.arctan(y_proj/z_proj) + 2*np.pi
        else:
            angles_Na2[i] = np.arctan(y_proj/z_proj) + np.pi
    angles_Na2 = np.delete(angles_Na2, np.where(my_array == np.max(my_array))[0][0], axis=0)

    diff = np.zeros([len(angles_Na1), len(angles_Na2)])
    for i in range(len(diff)):
        diff[i,:] = np.abs(angles_Na2 - angles_Na1[i])
    for i in range(len(angles_Na1)):  # min image
        for j in range(len(angles_Na2)):
            if diff[i,j] > np.pi:
                diff[i,j] = 2*np.pi - diff[i,j]
#    diff = np.abs(angles_Na2 - angles_Na1[0])
#    for i in range(len(diff)):  # minimum image 
#        if diff[i] > np.pi:
#            diff[i] = 2*np.pi - diff[i]
#    rot_angle = np.min(np.abs(angles_Na2 - angles_Na1[0]))

    return np.min(diff, axis=1)

def relSpherCoordDimer(r_Na, r_Ar):
    '''
    Takes positions of sodiums and argons
    Outputs positions as spherical coordinates with dimer along z-axis
    '''
    closest_Ar = closestAr(r_Na, r_Ar)

    center = (r_Na[0] + r_Na[1])/2
    r_shell = np.append(r_Na[0], closest_Ar, axis=0)

    spher_coords = set_fxns.relSpherCoords(center, r_shell)

    spher_Na = np.append(spher_coords[0], -spher_coords[0], axis=0)
    spher_Ar = np.delete(spher_coords, 0, axis=0)

    return spher_Na, spher_Ar

def calcQSep(my_dir, t, nstate=1, r_cutoff=5, auleng=0.5291772108):
    '''
    Calculates the electronic density around each sodium.
    Takes in file directory and time, spits out 2x1 array of scalars.
        The sum of scalars less than or equal to 1.
    '''
    e_distri = np.zeros(2)

    e_dens, r_Na = readEDens(my_dir+'/eigvec'+str(int(nstate))+'.cube', t, auleng=auleng)
    e_dens[:,3] = e_dens[:,3]**2

    for i in range(len(e_dens)):
        my_dist1 = dist(e_dens[i,:3], r_Na[0])
        my_dist2 = dist(e_dens[i,:3], r_Na[1])

        if np.min([my_dist1, my_dist2]) < r_cutoff:
            if my_dist1 < my_dist2:
                e_distri[0] += e_dens[i,3]
            else:
                e_distri[1] += e_dens[i,3]

    return e_distri

def calcTCFProj(my_dir, n_solu=2, n_solv=1600, unitl=21.9166090280612):
    '''
    Still in progress. Fixing mqc md code before going back to this
    '''
    hf_na, hf_ar = readHF(my_dir, n_solv=n_solv)
    t_frames = hf_na[:,0]

    v_na = np.zeros([len(t_frames), 7])
    v_ar = np.zeros([len(t_frames), n_solv, 3])
    for i in range(len(t_frames)):
        v_ar[i] = readArVel(my_dir+'/out.water', t_frames[i], unitl=unitl)
        v_na[i,1:] = readNaVel(my_dir+'/out.na', t_frames[i], unitl=unitl, n_solu=n_solu).flatten()
    v_na[:,0] = t_frames

    tcf = np.zeros(len(t_frames))
    for i in range(len(t_frames)):
        tcf[i] = -np.dot(hf_na[i,1:],v_na[i,1:]) - np.dot(hf_ar[i].flatten(),v_ar[i].flatten())

    return tcf

def calcEffVol(my_dir, t0, t, dt, n_solv=1600, auleng=0.5291772108, N=30, kappa=10, grid='manual'):
    '''
    Piggy backs off of waterballoon() to calculate the effective volume around each Na+
    kappa is a constant for the fermi filter
    Returns as [N,3].
    '''
    dV = 0.015625   # hard set from grid spacing being 0.25 Ang
    t_frames = np.linspace(t0, t, num=int((t-t0)/dt+1))
    delta_V = np.zeros([len(t_frames), 3])

    for i in range(len(t_frames)):
        pot_grid, r_Na = waterballoon(my_dir, t_frames[i], n_solv=n_solv, auleng=auleng, N=N, grid=grid)

        d = dist(r_Na[0], r_Na[1])
        isoval = -14.4*(1/3 + 1/(d+3))

        pot_grid[:,3] = 1/(1+np.exp(kappa*(pot_grid[:,3]-isoval)))
        pot_grid = pot_grid[(-pot_grid[:,3]).argsort()]
        m = 0
        while pot_grid[m,3] >= 1e-6:
            m += 1
        pot_grid = pot_grid[:m]

        n, m, split = 0, 0, 0
        for j in range(len(pot_grid)):
            if pot_grid[j,0] > 0.25:
                n += pot_grid[j,3]
            elif pot_grid[j,0] < -0.25:
                m += pot_grid[j,3]
            else:
                split += pot_grid[j,3]

#        delta_V[i,1] = (n-m)/(n+m+split)
        delta_V[i,1] = (n+split/2)*dV
        delta_V[i,2] = (m+split/2)*dV

    delta_V[:,0] = t_frames

    return delta_V

def arVAC(my_dir, t0, t):
    '''
    Calculates Ar velocity autocorrelation
    '''
    n_solv = int(linecache.getline(my_dir+'/out.water', 7).split()[1])
    lpt = int(n_solv*3*3+2+8)     # lpt stands for lines per dt

    t1 = convFloat(linecache.getline(my_dir+'/out.water', 7).split()[4])
    t2 = convFloat(linecache.getline(my_dir+'/out.water', lpt+7).split()[4])
    dt = t2-t1

    t_frames = np.linspace(t0, t, num=int((t-t0)/dt+1))
    v_Ar = np.zeros([len(t_frames),n_solv,3])
    for i in range(len(t_frames)):
        v_Ar[i] = readArVel(my_dir+'/out.water', t_frames[i])

    vac = np.zeros([len(t_frames),2])
    vac[:,0] = t_frames-t0
    for i in range(len(t_frames)):
        my_norm = np.dot(v_Ar[i], v_Ar[i].T).diagonal()
        for j in range(i,len(t_frames)):
            vac[j-i,1] += np.sum(np.dot(v_Ar[i], v_Ar[j].T).diagonal()/my_norm)/n_solv
    for i in range(len(t_frames)):
        vac[i,1] /= len(t_frames)-i

    return vac

def arMSD(my_dir, t0, t):
    '''
    Calculates Ar mean square displacement
    '''
    n_solv = int(linecache.getline(my_dir+'/out.water', 7).split()[1])
    lpt = int(n_solv*3*3+2+8)     # lpt stands for lines per dt

    t1 = convFloat(linecache.getline(my_dir+'/out.water', 7).split()[4])
    t2 = convFloat(linecache.getline(my_dir+'/out.water', lpt+7).split()[4])
    dt = t2-t1

    t_frames = np.linspace(t0, t, num=int((t-t0)/dt+1))
    r_Ar = np.zeros([len(t_frames),n_solv,3])
    for i in range(len(t_frames)):
        r_Ar[i] = readAr(my_dir+'/out.water', t_frames[i])

    msd = np.zeros([len(t_frames),2])
    msd[:,0] = t_frames-t0
    for i in range(len(t_frames)-1):
        for j in range(i+1,len(t_frames)):
            msd[j-i,1] += np.sum(np.sqrt(np.dot(r_Ar[i]-r_Ar[j], (r_Ar[i]-r_Ar[j]).T).diagonal()))/n_solv
    for i in range(len(t_frames)):
        msd[i,1] /= len(t_frames)-i

    return msd

def waterballoon(my_dir, t, n_solv=1600, auleng=0.5291772108, N=30, grid='manual'):
    '''
    Creates a potential mesh (in the format of a cube file) where the interactions
        considered are the Ar pseudopotential (including polarization) and a Coulombic Na+ potential
    '''
    # for Na+ potential, using coulomb which goes -14.3996/r but with cutoff to avoid asymptote
    def naPot(r):
        if r < 0.5:
            return -28.7992
        else:
            return -14.3996/r
    # parameters for Ar PK potential and polarization potential in eV and Ang
    def arPot(r):
        A = 0.28 * 27.211386
        B = 2.55 / 0.529177
        r_c = 3 * 0.529177

        del_e = 0.5 / 0.529177**2
        alpha_ar = 11.08 * 0.529177**3
        return A/(1+np.exp(B*(r-r_c))) - 0.5*(1-np.exp(-1*del_e*r**2))**2 * alpha_ar / r**4

    n_solu = int(linecache.getline(my_dir+'/eigvec2.cube',3).split()[0]) - n_solv
    r_Na = readNa(my_dir+'/out.na', t, n_solu=n_solu)
    r_Ar = closestAr(r_Na, readAr(my_dir+'/out.water', t), n_cutoff=N, option='cation')
    r_Na, r_Ar = rotDimerEnv(r_Na, r_Ar)

    if grid=='read':
        ngrid = int(linecache.getline(my_dir+'/eigvec2.cube', 4).split()[0])
        origin = np.array(linecache.getline(my_dir+'/eigvec2.cube', 3).split()[1:4], dtype=float)*auleng
        dx = float(linecache.getline(my_dir+'/eigvec2.cube', 4).split()[1])*auleng
        dy = float(linecache.getline(my_dir+'/eigvec2.cube', 5).split()[2])*auleng
        dz = float(linecache.getline(my_dir+'/eigvec2.cube', 6).split()[3])*auleng
    else:
        # hard coding spacing to be 0.25 Ang with 5 Ang overhang.
        dx, dy, dz = (0.25, 0.25, 0.25)
        if n_solu == 1:
            ngrid_x, ngrid_y, ngrid_z = (40, 40, 40)
            origin = r_Na[0] - np.array([5,5,5])
        elif n_solu == 2:
            ngrid_x = 40 + int(abs(r_Na[0,0] - r_Na[1,0])/dx)
            ngrid_y, ngrid_z = (40, 40)
            origin = (r_Na[0] + r_Na[1])/2 - np.array([ngrid_x/8,5,5])

    # generate grid in Ang
    balloon_pot = np.zeros([ngrid_x*ngrid_y*ngrid_z, 4])
    x_ct, y_ct, z_ct = (0, 0, 0)
    for i in range(len(balloon_pot)):
        balloon_pot[i] = np.array([dx*x_ct, dy*y_ct, dz*z_ct, 0])
        z_ct += 1
        if z_ct == ngrid_z:
            z_ct = 0
            y_ct += 1
            if y_ct == ngrid_y:
                x_ct += 1
                y_ct = 0
    balloon_pot[:,:3] += origin

    # go through each grid point and calculate sum of potential from Ar and Na
    for i in range(len(balloon_pot)):
        for m in range(n_solu):
            balloon_pot[i,3] += -14.3996/dist(balloon_pot[i,:3], r_Na[m])
        for n in range(len(r_Ar)):
            r = dist(balloon_pot[i,:3], r_Ar[n])
            if r > 6:
                continue
            else:
                balloon_pot[i,3] += arPot(r)

    return balloon_pot, r_Na

def SOAPSquareGrid(r_Na, r_Ar, sigma, kappa, x, y, z, dx):
    '''
    Creates a square grid of dimensions (x,y,z), in Ang, with spacing dx=dy=dz
    and projects positions of Ar to grid using gaussian smoothing of atomic positions
    '''
    grid = np.zeros([int(x/dx), int(y/dx), int(z/dx), 4])

    neighboring_Ar = closestAr(r_Na, r_Ar, n_cutoff=20, option='cation', index='False')


    return grid

def binScalarsBondDist(scalars, bond_dist, r0, r, dr):
    bins = np.round(np.arange(r0, r, dr), 2)
    my_dict = {}
    for i in bins:
        my_dict[str(i)] = []

    for i in range(len(scalars)):
        if bond_dist[i] > r or bond_dist[i] < r0:
            continue
        else:
            my_ind = int((bond_dist[i]-r0)/dr)

            my_dict[str(np.round(r0+my_ind*dr,2))] += [scalars[i]]

    binned_dat = np.zeros([len(bins),4])
    binned_dat[:,0] = bins+dr/2
    for i in range(len(bins)):
        binned_dat[i,1] = np.average(my_dict[str(bins[i])])
        binned_dat[i,2] = stats.sem(my_dict[str(bins[i])])
        binned_dat[i,3] = np.std(my_dict[str(bins[i])])

    return binned_dat

def binScalars2D(scalars, xs, ys, x0, x, dx, y0, y, dy):
    xbins = np.arange(x0, x, dx)
    ybins = np.arange(y0, y, dy)
    my_dict = {}
    for i in xbins:
        for j in ybins:
            my_dict[str((np.round(i,2),np.round(j,2)))] = []

    for n in range(len(scalars)):
        if xs[n] > x or xs[n] < x0 or ys[n] > y or ys[n] < y0:
            continue
        else:
            x_ind = int((xs[n]-x0)/dx)
            y_ind = int((ys[n]-y0)/dy)

            my_dict[str((np.round(x0+x_ind*dx,2),np.round(y0+y_ind*dy,2)))] += [[xs[n], ys[n],scalars[n]]]

    binned_dat = np.zeros([len(xbins),len(ybins),5])
    for i in range(len(xbins)):
        for j in range(len(ybins)):
            my_dict[str((np.round(xbins[i],2),np.round(ybins[j],2)))] = np.array(my_dict[str((np.round(xbins[i],2),np.round(ybins[j],2)))])
            binned_dat[i,j,0] = np.average(my_dict[str((np.round(xbins[i],2),np.round(ybins[j],2)))][:,0])
            binned_dat[i,j,1] = np.average(my_dict[str((np.round(xbins[i],2),np.round(ybins[j],2)))][:,1])
            binned_dat[i,j,2] = np.average(my_dict[str((np.round(xbins[i],2),np.round(ybins[j],2)))][:,2])
            binned_dat[i,j,3] = stats.sem(my_dict[str((np.round(xbins[i],2),np.round(ybins[j],2)))][:,2])
            binned_dat[i,j,4] = np.std(my_dict[str((np.round(xbins[i],2),np.round(ybins[j],2)))][:,2])

    return binned_dat
































