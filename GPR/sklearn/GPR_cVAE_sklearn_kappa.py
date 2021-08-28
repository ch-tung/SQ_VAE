#!/usr/bin/env python
# coding: utf-8

# # Variational Autoencoder for S(Q)

# Chi-Huan Tung
# National Tsing-Hua University
# Aug 2021
#
# This notebook is based on the example of Convolutional Variational Autoencoder (CVAE)
# on tensorflow.org/tutorials/generative/cvae

# Use convolution layer
# Sigmoid function output from network
# ln(S(Q))

# ## Setup

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import time
import scipy.interpolate as interp

# ## Load data

# ### Training set

# minimum sq
sq_min = np.exp(-5)

if 1:
    X_file = '../../../data/input_grid_all_GPR80.csv'
    Y_file = '../../../data/target_grid_all.csv'
else:
    X_file = '../../../data/input_random_all_GPR80.csv'
    Y_file = '../../../data/target_random_all.csv'
    
fX = open(X_file, 'r', encoding='utf-8-sig')
sq = np.genfromtxt(fX, delimiter=',').astype(np.float32)
sq[sq<=0] = sq_min

fY = open(Y_file, 'r', encoding='utf-8-sig')
target = np.genfromtxt(fY, delimiter=',').astype(np.float32)

if 0:
    sq = np.vstack((sq[0:7500,:],sq))
    target = np.vstack((target[0:7500,:],target))

sq.shape

eta = target[:,0]
kappa = target[:,1]
Z = target[:,3]
A = target[:,2]
lnZ = np.log(Z)
lnA = np.log(A)

sq_dim = sq.shape[1]
sample_train_dim = sq.shape[0]

q = (np.arange(sq_dim)+1)*0.2
q_rs = (np.arange(sq_dim)+1)*0.2
q_rs_dim = q_rs.shape[0]

# Rescale

r_eta = 1
sq_rs = np.zeros((sample_train_dim,q_rs_dim),dtype='float32')
for i in range(sample_train_dim):
    qr_eta = q*r_eta
    interpolating_function = interp.interp1d(qr_eta[3:],sq[i,3:],fill_value='extrapolate')
    sq_rs[i,:] = interpolating_function(q_rs).astype(np.float32)
sq_rs[sq_rs<=0] = sq_min

# ### Test set

if 0:
    X_file = '../../../data/input_grid_all_GPR80.csv'
    Y_file = '../../../data/target_grid_all.csv'
else:
    X_file = '../../../data/input_random_all_GPR80.csv'
    Y_file = '../../../data/target_random_all.csv'
    
fX_test = open(X_file, 'r', encoding='utf-8-sig')
sq_test = np.genfromtxt(fX_test, delimiter=',').astype(np.float32)
sq_test[sq_test<=0] = sq_min

fY_test = open(Y_file, 'r', encoding='utf-8-sig')
target_test = np.genfromtxt(fY_test, delimiter=',').astype(np.float32)

sq_test.shape

eta_test = target_test[:, 0]
kappa_test = target_test[:, 1]
Z_test = target_test[:, 3]
A_test = target_test[:, 2]
lnZ_test = np.log(Z_test)
lnA_test = np.log(A_test)

sample_test_dim = sq_test.shape[0]


# Rescale

r_eta_test = 1
sq_test_rs = np.zeros((sample_test_dim,q_rs_dim),dtype='float32')
for i in range(sample_test_dim):
    qr_eta = q*r_eta_test
    interpolating_function_test = interp.interp1d(qr_eta[3:],sq_test[i,3:],fill_value='extrapolate')
    sq_test_rs[i,:] = interpolating_function_test(q_rs)
sq_test_rs[sq_test_rs<=0] = sq_min

# ### Mask

mask_length = 0
sq_mask = sq_rs
sq_test_mask = sq_test_rs

for i in range(sample_train_dim):
    sq_mask[i,0:mask_length] = sq_rs[i,mask_length]
for i in range(sample_test_dim):
    sq_test_mask[i,0:mask_length] = sq_test_mask[i,mask_length]

# ### Preprocess/Postprocess

exp_scale = 3

def f_inp(sq):
    return np.log(sq)/exp_scale/2 + 0.5


def f_out(predictions):
    return np.exp((predictions*2-1)*exp_scale)

def to_tf(arg):
    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    return arg

# ## Network architecture

class VAE(tf.keras.Model):
    def __init__(self, latent_dim, sq_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        regularizer = None
        self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(sq_dim)),
            tf.keras.layers.Reshape((sq_dim,1)),
            tf.keras.layers.Conv1D(
                filters=32, kernel_size=3, strides=2, activation='relu',
                kernel_regularizer = regularizer,
                name='conv1d_en'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                latent_dim + latent_dim, 
                kernel_regularizer = regularizer,
                name='dense_en'),
        ]
        )
        
        self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(
                40*32, activation=tf.nn.relu, 
                kernel_regularizer = regularizer,
                name='dense_de'),
            tf.keras.layers.Reshape(target_shape=(40, 32)),
            tf.keras.layers.Conv1DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same', activation='relu',
                kernel_regularizer = regularizer,
                name='conv1dtrs_de'),
            tf.keras.layers.Conv1DTranspose(
                filters=1, kernel_size=3, strides=1, padding='same'),
            tf.keras.layers.Reshape((sq_dim,))
        ]
        )
        
    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(1000, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)
    
    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar
        
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean
    
    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

print('loading model')
latent_dim = 3
with tf.device('/cpu:0'):
	model = VAE(latent_dim, q_rs_dim)

# ## Load trained model

export_path = '../../saved_model/SQ_cVAE_MSE_ns/'
model_name = 'model_conv_stride2_batch32'
export_name = export_path + model_name

with tf.device('/cpu:0'):
    reload_sm = model.load_weights(export_name, by_name=False, skip_mismatch=False, options=None)
reload_sm.__dict__

print('model loaded')

model_r = reload_sm._root

x = to_tf(f_inp(sq_test_mask))
mean, logvar = model.encode(x)
z = model.reparameterize(mean, logvar)
x_logit = model.sample(z)
z = z.numpy()
z_mean = np.mean(z,axis=0)
zc = z-z_mean
F = zc.T

U, S, Vh = np.linalg.svd(F)

zs = np.matmul(zc,U)

for i in range(3):
    if np.abs(np.min(zs,axis = 0)[i]) > np.abs(np.max(zs,axis = 0)[i]):
        zs[:,i] = -zs[:,i]

np.std(zs,axis = 0)

d_zs = np.max(zs,axis = 0)-np.min(zs,axis = 0)
d_z = np.max(z,axis = 0)-np.min(z,axis = 0)

parameters = (eta_test,kappa_test,lnA_test)
parameters_GP = np.vstack(parameters).T
index_eta = np.arange(sq_test.shape[0])

print('GPR training set ready')

# ## GPR

# Use GPR to infer the potential parameters from to the latent vatiables in `Fit_cVAE` by the `GPflow` package.
print('import sklearn')
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import joblib

X = zs[:,:]
Y = parameters_GP[:,1] # kappa

len_s = 0.5964
sigma_y = 0.0011

kernel = RBF(len_s, (3e-1, 1e0)) + WhiteKernel(sigma_y, (5e-4,2e-3))
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, n_restarts_optimizer=0)

tStart = time.time()
gp.fit(X, Y)
print("GPML kernel: %s" % gp.kernel_)
print("Log-marginal-likelihood: %.3f"
    % gp.log_marginal_likelihood(gp.kernel_.theta))

model_name_GPR = 'sklearn/model_GPR_kappa'
export_name_GPR = export_path + model_name_GPR
joblib.dump(gp, export_name_GPR)

tEnd = time.time()
print('job end')

print("It cost %f sec" % (tEnd - tStart))