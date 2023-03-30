import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import time
from tqdm import tqdm
from IPython import display
import os

import tensorflow as tf

from SQ_decoder import SQ_decoder, to_tf

q_rs = (np.arange(80)+1)*0.2

# rescale the training set SQ to range [0,1]
exp_scale = 6
def f_inp(sq):
    return tf.math.log(sq)/exp_scale/2 + 0.5

# transform the decoder output to SQ
def f_out_sample(x):
    return f_out_tf(tf.sigmoid(x))

# rescale the fitting parameters to range [0,1]
def fp_inp(parameters):
    return np.log(parameters)/exp_scale/2

def fp_out(parameters_pred):
    return np.exp((parameters_pred*2)*exp_scale)

def f_out(sq_pred):
    return np.exp((sq_pred*2-1)*exp_scale) # inverse of f_inp

def f_out_tf(sq_pred):
    return tf.math.exp((sq_pred*2-1)*exp_scale)

def compute_loss_l2(parameters, SQ, model):
    # SQ rescaled to range [0,1]

    err_l2 = tf.reduce_mean((f_inp(f_out_sample(model(parameters)))-SQ)**2)

    loss = err_l2

    return loss

class Train_NN:
    def __init__(self,n_epoch,batch_size,batch_size_validate,train_rate,f_loss,model,
                  data_train, data_test, test_SQ_sample, test_parameters_sample,
                  fig_path,model_path):
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.batch_size_validate = batch_size_validate
        self.optimizer = tf.keras.optimizers.Adam(train_rate)
        self.f_loss = f_loss
        self.model = model
        self.epoch_prev = 0
        self.epoch_counter = 0
        self.fig_path = fig_path
        if not os.path.isdir(self.fig_path):
            os.mkdir(self.fig_path)
        self.model_path = model_path
        if not os.path.isdir(self.model_path):
            os.mkdir(self.model_path)
            
        self.train_SQ_dataset, self.train_parameters_dataset = data_train
        self.test_SQ_dataset, self.test_parameters_dataset = data_test
        
        self.test_SQ_sample = test_SQ_sample
        self.test_parameters_sample = test_parameters_sample

    @tf.function
    def train_step(self, parameters, SQ, model, optimizer):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            loss = self.f_loss(parameters, SQ, model)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
    def generate_and_save_images(self, model, epoch, test_SQ_sample, test_parameters_sample):
        GT = f_out(test_SQ_sample)
        predictions = f_out_sample(model(test_parameters_sample))

        fig = plt.figure(figsize=(8, 8))

        for i in range(GT.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.plot(q_rs,GT[i,:],'k')
            plt.plot(q_rs,predictions[i,:],'.c')    
            plt.plot(q_rs,predictions[i,:],'-b')

            #plt.axis('off')
            plt.ylim(0, 3)

        # tight_layout minimizes the overlap between 2 sub-plots
        pngname = self.fig_path+'{:04d}.png'
        plt.savefig(pngname.format(epoch))
        plt.show()
    
    def fit(self):
        self.err_test_epoch = np.zeros(self.n_epoch)
        self.err_train_epoch = np.zeros(self.n_epoch)
        self.err_validate_epoch = np.zeros(self.n_epoch)
        for epoch in tqdm(range(1, self.n_epoch + 1)):
            # training set
            for train_SQ_batch in self.train_SQ_dataset.take(1):
                train_SQ = train_SQ_batch[0:self.batch_size, :]
            for train_parameters_batch in self.train_parameters_dataset.take(1):
                train_parameters = train_parameters_batch[0:self.batch_size, :]

            # test set    
            for test_SQ_batch in self.test_SQ_dataset.take(1):
                test_SQ = test_SQ_batch[0:self.batch_size, :]
            for test_parameters_batch in self.test_parameters_dataset.take(1):
                test_parameters = test_parameters_batch[0:self.batch_size, :]

            # validation set
            for validate_SQ_batch in self.test_SQ_dataset.take(1):
                validate_SQ = validate_SQ_batch[0:self.batch_size_validate, :]
            for validate_parameters_batch in self.test_parameters_dataset.take(1):
                validate_parameters = validate_parameters_batch[0:self.batch_size_validate, :]

            start_time = time.time()
            self.train_step(train_parameters, train_SQ, self.model, self.optimizer)    
            end_time = time.time()

            # loss
            err_test = compute_loss_l2(test_parameters, test_SQ, self.model)
            err_train = compute_loss_l2(train_parameters, train_SQ, self.model)
            err_validate = compute_loss_l2(validate_parameters, validate_SQ, self.model)
            self.err_test_epoch[epoch-1] = err_test
            self.err_train_epoch[epoch-1] = err_train
            self.err_validate_epoch[epoch-1] = err_validate

            display.clear_output(wait=0.5)
            self.generate_and_save_images(self.model, epoch+self.epoch_prev, 
                                          self.test_SQ_sample, self.test_parameters_sample)
        
        self.epoch_counter = epoch
        
    def save_model(self, model_path_sub = '', model_name='model_param_SQ'):
        if not os.path.isdir(self.model_path+model_path_sub):
            os.mkdir(self.model_path+model_path_sub)
        export_name = self.model_path + model_path_sub + model_name
        self.model.save_weights(export_name, overwrite=True, save_format=None, options=None)
        
class Train_NN_V:
    def __init__(self,n_epoch, batch_size, batch_size_validate, train_rate, f_loss, model,
                  data_train, data_test, test_SQ_sample, test_parameters_sample,
                  fig_path,model_path,save_fig=True):
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.batch_size_validate = batch_size_validate
        self.optimizer = tf.keras.optimizers.Adam(train_rate)
        self.f_loss = f_loss
        
        self.model = model
        
        self.epoch_prev = 0
        self.epoch_counter = 0
        self.fig_path = fig_path
        if not os.path.isdir(self.fig_path):
            os.mkdir(self.fig_path)
        self.model_path = model_path
        if not os.path.isdir(self.model_path):
            os.mkdir(self.model_path)
            
        self.train_SQ_dataset, self.train_parameters_dataset = data_train
        self.test_SQ_dataset, self.test_parameters_dataset = data_test
        
        self.test_SQ_sample = test_SQ_sample
        self.test_parameters_sample = test_parameters_sample
        self.save_fig = save_fig
    
    @tf.function
    def compute_loss_l2(self, parameters, SQ, model):
        # SQ rescaled to range [0,1]

        mean, logvar = model.encode(parameters)
        eps = model.reparameterize(mean, logvar)
        SQ_pred = model.sample(eps)

        err_l2 = tf.reduce_mean((SQ_pred-SQ)**2)
        loss = err_l2

        return loss

    @tf.function
    def train_step(self, parameters, SQ, model, optimizer):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            loss = self.f_loss(parameters, SQ, model)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    @tf.function
    def SQ_pred(self,x):
        mean, logvar = self.model.encode(x)
        eps = self.model.reparameterize(mean, logvar)
        return self.model.sample(eps)
        
    def generate_and_save_images(self, model, epoch, test_SQ_sample, test_parameters_sample):
        GT = f_out(test_SQ_sample)
        
        ## Decoded Means
        mean, logvar = model.encode(test_parameters_sample)
        predictions_0 = f_out(model.sample(mean))
        
        eps = model.reparameterize(mean, logvar)
        predictions = f_out(model.sample(eps))
        
        sample_mean, sample_std = model.sample_normal(test_parameters_sample)
        predictions_mean = f_out(sample_mean)
        predictions_mean_p = f_out(sample_mean+sample_std)
        predictions_mean_n = f_out(sample_mean-sample_std)

        fig = plt.figure(figsize=(8, 8))

        for i in range(GT.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.plot(q_rs,GT[i,:],'k')
#             plt.plot(q_rs,predictions[i,:],'.c')    
#             plt.plot(q_rs,predictions_0[i,:],'-b')
            plt.fill_between(q_rs,predictions_mean_p[i,:],predictions_mean_n[i,:],color='b',alpha=0.5)
            plt.plot(q_rs,predictions_mean[i,:],'-b')    

            #plt.axis('off')
            plt.ylim(0, 3)
        
        if self.save_fig:
            # tight_layout minimizes the overlap between 2 sub-plots
            pngname = self.fig_path+'{:04d}.png'
            plt.savefig(pngname.format(epoch))
        plt.show()
    
    def fit(self):
        self.err_test_epoch = np.zeros(self.n_epoch)
        self.err_train_epoch = np.zeros(self.n_epoch)
        self.err_validate_epoch = np.zeros(self.n_epoch)
        for epoch in tqdm(range(1, self.n_epoch + 1)):
            # training set
            for train_SQ_batch in self.train_SQ_dataset.take(1):
                train_SQ = train_SQ_batch[0:self.batch_size, :]
            for train_parameters_batch in self.train_parameters_dataset.take(1):
                train_parameters = train_parameters_batch[0:self.batch_size, :]

            # test set    
            for test_SQ_batch in self.test_SQ_dataset.take(1):
                test_SQ = test_SQ_batch[0:self.batch_size, :]
            for test_parameters_batch in self.test_parameters_dataset.take(1):
                test_parameters = test_parameters_batch[0:self.batch_size, :]

            # validation set
            for validate_SQ_batch in self.test_SQ_dataset.take(1):
                validate_SQ = validate_SQ_batch[0:self.batch_size_validate, :]
            for validate_parameters_batch in self.test_parameters_dataset.take(1):
                validate_parameters = validate_parameters_batch[0:self.batch_size_validate, :]

            start_time = time.time()
            self.train_step(train_parameters, train_SQ, self.model, self.optimizer)    
            end_time = time.time()

            # loss
            err_test = self.compute_loss_l2(test_parameters, test_SQ, self.model)
            err_train = self.compute_loss_l2(train_parameters, train_SQ, self.model)
            err_validate = self.compute_loss_l2(validate_parameters, validate_SQ, self.model)
            self.err_test_epoch[epoch-1] = err_test
            self.err_train_epoch[epoch-1] = err_train
            self.err_validate_epoch[epoch-1] = err_validate

            display.clear_output(wait=True)
            self.generate_and_save_images(self.model, epoch+self.epoch_prev, 
                                          self.test_SQ_sample, self.test_parameters_sample)
        
        self.epoch_counter = self.epoch_counter + epoch
        
    def save_model_aug(self, model_path_sub = '', model_name='model_aug'):
        if not os.path.isdir(self.model_path+model_path_sub):
            os.mkdir(self.model_path+model_path_sub)
        export_name = self.model_path + model_path_sub + model_name
        self.model.aug_layers.save_weights(export_name, overwrite=True, save_format=None, options=None)
        
    def save_model_decoder(self, model_path_sub = '', model_name='model_decoder'):
        if not os.path.isdir(self.model_path+model_path_sub):
            os.mkdir(self.model_path+model_path_sub)
        export_name = self.model_path + model_path_sub + model_name
        self.model.decoder_layers.save_weights(export_name, overwrite=True, save_format=None, options=None)