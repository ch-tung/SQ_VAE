import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp

import tensorflow as tf

# --------------------------------------
# Augmented decoder network
class Decoder_aug(tf.keras.Model):
    def __init__(self, latent_dim, sq_dim):
        super(Decoder_aug,self).__init__()
        self.latent_dim = latent_dim
        
        model_aug = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(3)),
            tf.keras.layers.Dense(6, 
                        kernel_regularizer = None,
                        name='dense_in'),
            tf.keras.layers.Dense(6, 
                        kernel_regularizer = None,
                        name='dense_in2'),
        ]
        )
        
        model_decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(
                40*32, activation=tf.nn.relu, 
                kernel_regularizer = None,
                name='dense_de'),
            tf.keras.layers.Reshape(target_shape=(40, 32)),
            tf.keras.layers.Conv1DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same', activation='relu',
                kernel_regularizer = None,
                name='conv1dtrs_de'),
            tf.keras.layers.Conv1DTranspose(
                filters=1, kernel_size=3, strides=1, padding='same'),
            tf.keras.layers.Reshape((sq_dim,))
        ]
        )
        
        
        self.aug_layers = model_aug
        self.decoder_layers = model_decoder
        
    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(64, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)
    
    def encode(self, x):
        mean, logvar = tf.split(self.aug_layers(x), num_or_size_splits=2, axis=1)
        return mean, logvar
        
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean
    
    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder_layers(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
    
    def sample_normal(self, x):
        mean, logvar = self.encode(x)
        eps = tf.random.normal(shape=(64, self.latent_dim))
#         z_samples = [e*tf.exp(logvar*.5) + mean for e in eps]
#         logits_samples = [self.decode(z, apply_sigmoid=True) for z in z_samples]
#         z_samples = eps*tf.exp(logvar*.5) + mean
#         logits_samples = self.decode(z_samples, apply_sigmoid=True)
        def zsample(e):
            return e*tf.exp(logvar*.5) + mean
        z_samples = tf.map_fn(zsample,eps)
        def logitsample(z):
            return self.decode(z, apply_sigmoid=True)
        logits_samples = tf.map_fn(logitsample,z_samples)
        
        logits_std = tf.math.reduce_std(logits_samples,axis=0)
        logits_mean = tf.math.reduce_mean(logits_samples,axis=0)
        
        return logits_mean, logits_std

# --------------------------------------
# load trained model
model = Decoder_aug(latent_dim=3, sq_dim=80)

export_path_aug = './saved_model/SQ_NLL_variation/SQ_NLL-hom_variation_aug_ft/'
model_name_aug = 'SQ_NLL-hom_variation_aug_ft'
export_name_aug = export_path_aug + model_name_aug

export_path_decoder = './saved_model/SQ_NLL_variation/SQ_NLL-hom_variation_decoder_ft/'
model_name_decoder = 'SQ_NLL-hom_variation_decoder_ft'
export_name_decoder = export_path_decoder + model_name_decoder

aug_layers_loaded = model.aug_layers.load_weights(export_name_aug, by_name=False, skip_mismatch=False, options=None)
model.aug_layers = aug_layers_loaded._root

decoder_layers_loaded = model.decoder_layers.load_weights(export_name_decoder, by_name=False, skip_mismatch=False, options=None)
model.decoder_layers = decoder_layers_loaded._root

# --------------------------------------
# Rescaling
# Transform the input to tensorflow tensor
def to_tf(arg):
    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    return arg

# rescale the training set SQ to range [0,1]
exp_scale = 6
def f_inp(sq):
    return tf.math.log(sq)/exp_scale/2 + 0.5

# transform the decoder output to SQ
def f_out(sq_pred):
    return np.exp((sq_pred*2-1)*exp_scale) # inverse of f_inp

@tf.function
def f_out_tf(predictions):
    return tf.math.exp((predictions*2-1)*exp_scale)

# --------------------------------------
# Scattering function
def SQ_NN(parameters):
    
    # mean and std of the training set labels
    mean = np.array([0.2325,0.2600,13.0000])
    std = np.sqrt(np.array([0.0169,0.0208,52.0000]))
    parameters_z = [(parameters[i]-mean[i])/std[i] for i in range(3)]
    
    x = tf.reshape(to_tf(parameters_z),(1,3))
    sample_mean, sample_std = model.sample_normal(x)
    
    return f_out_tf(sample_mean).numpy().reshape(80).astype('float64')

def SQ_NN_tf(parameters):
    
    # mean and std of the training set labels
    mean = np.array([0.2325,0.2600,13.0000])
    std = np.sqrt(np.array([0.0169,0.0208,52.0000]))
    parameters_z = [(parameters[i]-mean[i])/std[i] for i in range(3)]
    
    x = tf.reshape(to_tf(parameters_z),(1,3))
    sample_mean, sample_std = model.sample_normal(x)
    
    return f_out_tf(sample_mean)
# --------------------------------------
