#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
# import matplotlib.pyplot as plt
# plt.rcParams["font.family"] = "Arial"
# from tqdm import tqdm

from rg import *


# In[2]:


# define time interval
t = np.array([0])

# define simulation inputs
n_conf = 21
n_particle = 16384
n_frame = t.size
rs = 6
rc = 2

# filename
filename_GT = '11__0.218_0.267_3.678/'
filename_NN = '11__0.225_0.3_3/'


# In[3]:


# load trajectory 
def eval_rg(filename_data):
    rg_all_c = np.zeros((3,3,n_particle,n_conf))
    for ic in range(n_conf):
        string_conf = '{:09d}'.format(ic*500)
        filename = './' + filename_data + 'dump.' + string_conf + '.txt'
        r_all, l = loaddump(filename,n_particle,n_frame)
        print('loaded '+filename)

        L = l[:,1]-l[:,0]
        rt = r_all[:,:,0]
        sl = np.arange(0,n_particle,1)

        rg_all = Rg_cov_matrix(rt, L, rc, sl)
        rg_all_c[:,:,:,ic] = rg_all
        
    return rg_all_c
    


# In[4]:


# GT
rg_all = eval_rg(filename_GT)

filename_rg = 'rg_py_GT.mat'

mdic = {'rg_all':rg_all}
savemat(filename_rg, mdic)


# In[5]:


# NN
rg_all = eval_rg(filename_NN)

filename_rg = 'rg_py_NN.mat'

mdic = {'rg_all':rg_all}
savemat(filename_rg, mdic)


# In[ ]:




