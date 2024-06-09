import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join
#import tensorflow_probability as tfp
tf.keras.backend.set_floatx('float32')

class Linear_transfer(tf.keras.layers.Layer):
    def __init__(self,rec_base,units,last=False):
        super(Linear_transfer, self).__init__()
        self.number_of_attractors = rec_base.shape[1]
        self.last = last
        self.units = units
                # Define the weights
        self.base_fix = self.add_weight(
            shape=rec_base.shape,
            initializer=tf.constant_initializer(rec_base),
            dtype=tf.float32,
            trainable=False,
            name='base_fix'
        )

        self.autov_fix = self.add_weight(
            shape=(self.number_of_attractors,),
            initializer=tf.zeros_initializer(),
            dtype=tf.float32,
            trainable=False,
            name='eigval_fix'
        )

        self.autov_tr = self.add_weight(
            shape=(int(units - self.number_of_attractors),),
            initializer=tf.random_normal_initializer(mean=-28., stddev=1.),
            dtype=tf.float32,
            trainable=True,
            name='eigenval_train'
        )
        self.base_train = self.add_weight(shape=(units, int(units - self.number_of_attractors)),initializer=tf.keras.initializers.Orthogonal(gain=1.15, seed=None),regularizer=tf.keras.regularizers.OrthogonalRegularizer(factor=0.01,mode='columns'),
                                          trainable=True,
                                          name='base_train')



    def call(self, inputs, **kwargs):
        eigenvalues_total = tf.concat([tf.transpose(self.autov_tr), tf.transpose(self.autov_fix)], axis=0)
        diagonal = tf.linalg.diag(tf.clip_by_value(eigenvalues_total, clip_value_min=-1000000., clip_value_max=20.))
        base = tf.concat([self.base_train, self.base_fix], axis=1)
        A = tf.matmul(base, tf.matmul(diagonal, tf.linalg.inv(base)))
        return tf.transpose(A), eigenvalues_total, base

    def return_base(self):
        return tf.concat([self.base_train, self.base_fix], axis=1)

    def return_diagonal(self):
        return tf.concat([self.autov_tr, self.autov_fix], axis=0)

class Cowan_network(tf.keras.Model):
    def __init__(self,
                 size,
                 tmax,
                 attractors,
                 my_attractors,
                 are_eigenvectors_trainable=True,
                 y_saddle_points=None,
                 x_saddle_points=None,
                 spectral=False,
                 a=0,
                 gamma=0.25,
                 wee=7.2,
                 wei=2.,
                 wie=0., 
                 wii=1.,
                 ae=1.5,
                 ai=0.4,
                 he=-1.2,
                 hi=0.1,
                 fe1=0.25,
                 fe2=0.65,
                 fi1=0.5,
                 fi2=0.5,
                 beta1=3.7,
                 beta2=1,
                 Dx=1,
                 Dy=1,
                 dt=0.1
                 ):
        super(Cowan_network, self).__init__()

        self.size = size
        self.iterations = tmax
        self.attractors = attractors
        self.my_attractors = tf.constant(my_attractors.T, dtype=tf.float32)
        self.are_eigenvectors_trainable = are_eigenvectors_trainable

        self.wee = wee 
        self.wei = wei
        self.wie = wie
        self.wii = wii
        self.Dy = 1./np.sqrt(self.size)
        self.Dx = 1./np.sqrt(self.size)
        self.he = he
        self.hi = hi
        self.fe1 = fe1
        self.fe2 = fe2
        self.fi1 = fi1
        self.fi2 = fi2
        self.ae = ae
        self.ai = ai 
        self.beta1 = beta1
        self.beta2 = beta2 
        self.gamma = self.add_weight(
            shape=(),
            initializer=tf.constant_initializer(gamma),
            dtype=tf.float32,
            trainable=True,
            name='gamma'
        )
        self.dt = dt
        self.spectral = spectral
        self.flag = False
        self.flag_inf = False
        self.M =1000000.
        self.my_y = None
        self.myA=None
        self.diag=None
        self.base=None
        self.binary=True

    def build(self, input_shape):
        self.adiacency = Linear_transfer(rec_base=self.attractors,units=self.size)

    def call(self, inputs, **kwargs):
        x = inputs
        x = tf.cast(x, dtype=tf.float32)
        y = x
        matrix_A, self.diag, self.base = self.adiacency(x)
        self.myA=matrix_A
        if self.flag:
            for k in range(self.iterations):
                I1 = self.wee * x - self.wei * y + self.he + self.Dx * tf.matmul(x, matrix_A)
                I2 = self.wie * x - self.wii * y + self.hi 
                self.fe = self.fe1 * tf.math.tanh(self.beta1 * I1) + self.fe2
                self.fi = self.fi1 * tf.math.tanh(self.beta2 * I2) + self.fi2
                x = x + self.dt * (-self.ae * x + (1 - x) * self.fe)
                y = y + (self.dt / self.gamma) * (-self.ai * y + (1 - y) * self.fi)
                x=tf.clip_by_value(x, clip_value_min=0., clip_value_max=1.)
                y=tf.clip_by_value(y, clip_value_min=0., clip_value_max=1.)
        if self.binary:
                
                y_final = x
                # Expand dimensions of y_final to enable broadcasting
                y_final_exp = tf.expand_dims(y_final, axis=1)
                # Create a tensor of ones with the same shape as my_attractors
                ones_tensor = tf.ones_like(self.my_attractors)
                # Perform the operation without a for loop
                diff = self.my_attractors - ones_tensor * y_final_exp
                x = tf.reduce_sum(diff * diff, axis=2)
                norm_factor = tf.sqrt(tf.reduce_sum(self.my_attractors**2, axis=1) * tf.reduce_sum((ones_tensor * y_final_exp)**2, axis=2))
                # Normalize x
                x = x / norm_factor

                # Final computation for x
                x = 1 / x
                x = x / tf.reduce_sum(x, axis=1, keepdims=True)
                x=x[:,0]
        return x
    
    def dynamics(self, inputs, **kwargs):
        x = inputs
        x = tf.cast(x, dtype=tf.float32)
        y = x
        matrix_A, self.diag, self.base = self.adiacency(x)
        self.myA=matrix_A
        if self.flag:
            for k in range(self.iterations):
                I1 = self.wee * x - self.wei * y + self.he + self.Dx * tf.matmul(x, matrix_A)
                I2 = self.wie * x - self.wii * y + self.hi 
                self.fe = self.fe1 * tf.math.tanh(self.beta1 * I1) + self.fe2
                self.fi = self.fi1 * tf.math.tanh(self.beta2 * I2) + self.fi2
                x = x + self.dt * (-self.ae * x + (1 - x) * self.fe)
                y = y + (self.dt / self.gamma) * (-self.ai * y + (1 - y) * self.fi)
            
                data=x[10,:].numpy()
                data.flatten()
                with open("outputx_forwardx.txt", "a") as f:
                    f.write(" ".join(map(str, data)) + "\n")
                data=y[10,:].numpy()
                data.flatten()
                with open("outputy_forwardy.txt", "a") as f:
                    f.write(" ".join(map(str, data)) + "\n")