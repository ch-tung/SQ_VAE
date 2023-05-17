"""
May 2023
calculate Rg tensor from trajectory files
@author: CHTUNG
"""
import numpy as np
import numpy.matlib
from scipy.io import savemat

import time
tStart = time.time()

# import numba as nb

#%%
def is_header(x,n_particle):
    particlenumber = n_particle
    return np.remainder(x,particlenumber+9)>8

def loaddump(filename,n_particle,n_frame=1):
    # load file
    with open(filename,'r') as fp:
        lines = fp.readlines()
    lines_header = lines[0:9]
    l = np.genfromtxt(lines_header[5:8], delimiter=' ')
    
    # remove headers
    particlenumber = n_particle
    nframe = n_frame
    index_all = range((particlenumber+9)*nframe)
    
    index = list(filter(lambda x: is_header(x,n_particle), index_all))
    
    from operator import itemgetter
    lines_mod = list(itemgetter(*index)(lines))
    
    # convert to array
    data_np = np.genfromtxt(lines_mod, delimiter=' ')
    
    r = data_np[:,2:5]
    
    # reshape and permute
    r_all = np.reshape(r,(nframe,particlenumber,3))
    r_all = r_all.transpose((1,2,0))
    
    return r_all, l

# @nb.jit(forceobj=True)
def Rg_cov_matrix(rt, L, rc, sl):
    '''
    rt: Trajectory at a single timeframe
    L: Simulation cell PBC
    rc: Cutoff radius of coordination
    sl: Neighbor list
    '''
    Neig = sl
    N = Neig.size
    rjk = np.zeros((N, N, 3))
    rjk = rt[sl, :].reshape(N, 1, 3) - rt[sl, :].reshape(1, N, 3)
    
    for i_d in range(3):
        rjk[:, :, i_d] = rjk[:, :, i_d] - np.round(rjk[:, :, i_d] / L[i_d], 0) * L[i_d]
    
    djk2 = np.sum(np.power(rjk, 2), 2)
    
    weight = np.exp(-djk2 / (2 * rc**2))
    weight_sum = np.sum(weight, 1)
    
    rjk = np.transpose(rjk, [0, 2, 1])
    
    rg_all = np.zeros((3, 3, N))
    for i_1 in range(N):
        rg_all[:, :, i_1] = np.matmul(rjk[:, :, i_1].T, np.multiply(rjk[:, :, i_1], weight[i_1, :].reshape(N, 1))) / weight_sum[i_1]
    
    return rg_all
