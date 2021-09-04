"""
Created on Sat Oct 24 21:31:30 2020

@author: CHTUNG
"""
import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import joblib
import random
import time
tStart = time.time()

SHOWFIG = True

FIG_COUNT = 0
OUTPUT_DIR = './'

# training set
# Choose which dataset to use
if 1:
    X_file = '../../../data/input_grid_all_GPR80.csv'
    Y_file = '../../../data/target_grid_all.csv'
else:
    X_file = '../../../data/input_random_all_GPR80.csv'
    Y_file = '../../../data/target_random_all.csv'
    
fX = open(X_file, 'r', encoding='utf-8-sig')
sq = np.genfromtxt(fX, delimiter=',')

fY = open(Y_file, 'r', encoding='utf-8-sig')
target = np.genfromtxt(fY, delimiter=',')

eta = target[:,0]
kappa = target[:,1]
Z = target[:,3]
A = target[:,2]
lnZ = np.log(Z)
lnA = np.log(A)

# test_set
# Choose which dataset to use
if 0:
    X_file = '../../../data/input_grid_all_GPR80.csv'
    Y_file = '../../../data/target_grid_all.csv'
else:
    X_file = '../../../data/input_random_all_GPR80.csv'
    Y_file = '../../../data/target_random_all.csv'
    
fX_test = open(X_file, 'r', encoding='utf-8-sig')
sq_test = np.genfromtxt(fX_test, delimiter=',')

fY_test = open(Y_file, 'r', encoding='utf-8-sig')
target_test = np.genfromtxt(fY_test, delimiter=',')

eta_test = target_test[:, 0]
kappa_test = target_test[:, 1]
Z_test = target_test[:, 3]
A_test = target_test[:, 2]
lnZ_test = np.log(Z_test)
lnA_test = np.log(A_test)

sq_all = np.concatenate((sq, sq_test), axis=0)
eta_all = np.concatenate((eta, eta_test), axis=0)
kappa_all = np.concatenate((kappa, kappa_test), axis=0)
Z_all = np.concatenate((Z, Z_test), axis=0)
A_all = np.concatenate((A, A_test), axis=0)
lnZ_all = np.concatenate((lnZ, lnZ_test), axis=0)
lnA_all = np.concatenate((lnA, lnA_test), axis=0)

# GPR #################################
# define kernel

X_all = sq_test

Y_all = lnA_test
#Y = kappa
#Y = lnZ

X=X_all
Y=Y_all

len_s = 0.373
#len_s = 0.05
#len_s = 0.1

sigma_y2 = 2e-2
#sigma_y2 = 2e-4
#sigma_y2 = 3e-2

kernel = RBF(len_s, (1e-2, 1e1)) + WhiteKernel(sigma_y2, (1e-3,1e-1))
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, n_restarts_optimizer=10)
gp.fit(X, Y)

print("GPML kernel: %s" % gp.kernel_)
print("Log-marginal-likelihood: %.3f"
      % gp.log_marginal_likelihood(gp.kernel_.theta))

tFit = time.time()

export_path_GPR = '../../saved_model/GPR/' 
model_name_GPR = 'sklearn/SQ_GPR_lnA'
export_name_GPR = export_path_GPR + model_name_GPR
joblib.dump(gp, export_name_GPR)

tEnd = time.time()
print('job end')

print("It cost %f sec" % (tEnd - tStart))
print("time GPfit = %f sec" % (tFit - tStart))