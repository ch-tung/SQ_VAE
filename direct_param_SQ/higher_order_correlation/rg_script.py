"""
Mar 2021
calculate Rg tensor from trajectory files
@author: CHTUNG
"""
import numpy as np
import numpy.matlib
from scipy.io import savemat

import time
tStart = time.time()

###############################################
# define time interval
t = np.arange(1,10,1)
for it_p in range(7):
    t = np.append(t,np.arange(1,10,1)*(10**(it_p+1)))

t = t[np.where(t<=10000000)]
t = np.append(0,t)
t = t

###############################################
# define simulation
n_conf = 8
n_particle = 6912
n_frame = t.size
rs = 3
rc = 1

###############################################
#%%
# define functions
def is_header(x):
    particlenumber = n_particle
    return np.remainder(x,particlenumber+9)>8

def loaddump(filename):
    # load file
    with open(filename,'r') as fp:
        lines = fp.readlines()
    lines_header = lines[0:9]
    l = np.genfromtxt(lines_header[5:8], delimiter=' ')
    
    # remove headers
    particlenumber = n_particle
    nframe = n_frame
    index_all = range((particlenumber+9)*nframe)
    
    index = list(filter(is_header, index_all))
    
    from operator import itemgetter
    lines_mod = list(itemgetter(*index)(lines))
    
    # convert to array
    data_np = np.genfromtxt(lines_mod, delimiter=' ')
    
    r = data_np[:,2:5]
    
    # reshape and permute
    r_all = np.reshape(r,(nframe,particlenumber,3))
    r_all = r_all.transpose((1,2,0))
    
    return r_all, l

def Rg_cov_matrix(rt, L, rc, sl):
    Neig = sl
    N = Neig.size
    rjk = np.zeros((N,N,3))
    rjk = rt[sl,:].reshape(N,1,3) - rt[sl,:].reshape(1,N,3)
    
    for i_d in range(3):
        rjk[:,:,i_d] = rjk[:,:,i_d] - np.around(rjk[:,:,i_d]/L[i_d],0)*L[i_d]
    
    djk2 = np.sum(np.power(rjk,2),2)
    
    weight = np.exp(-djk2/2/rc**2)
    weight_sum = np.sum(weight,1)
    
    rjk = np.transpose(rjk,[0,2,1]);
    
    rg_all = np.zeros((3,3,N))
    for i_1 in range(N):
        rg_all[:,:,i_1] = np.matmul(rjk[:,:,i_1].T, np.multiply(rjk[:,:,i_1],weight[i_1,:].reshape(N,1)))/weight_sum[i_1]
    
    return rg_all

###############################################
#%%
#Temperature = np.arange(0,20,1)*0.1 + 0.1
#Temperature = np.arange(0,3,1)+3
Temperature = np.array([0.1, 0.4, 0.7, 1.0, 2.0, 5.0])

n_T = Temperature.shape[0]
n_t = t.shape[0]

rg_all_t = np.zeros((3,3,n_particle,n_t))
rg_all_t_c = np.zeros((3,3,n_particle,n_t,n_conf))

for iT in range(n_T):
    T_i = Temperature[iT]
    name_T = str('%.1f' % T_i)
    print(T_i)
    
    for ic in range(n_conf):
        name_P = str(ic+1)
        print(ic+1)
        
        filename = './' + name_T + '/eq.' + name_P + '.dump'
        r_all = loaddump(filename)[0]
        l = loaddump(filename)[1]
        L = l[:,1]-l[:,0]
                
        for it in range(n_t):
            rt = r_all[:,:,it]
            sl = np.arange(0,n_particle,1)
            
            rg_all = Rg_cov_matrix(rt, L, rc, sl)
            rg_all_t[:,:,:,it] = rg_all
            
        rg_all_t_c[:,:,:,:,ic] = rg_all_t
        
    filename_rg = 'rg_py_' + name_T + '.mat'
    
    mdic = {'rg_all_t_c':rg_all_t_c}
    savemat(filename_rg, mdic)

tEnd = time.time()
print('time used ' + tEnd-tStart + 'sec')