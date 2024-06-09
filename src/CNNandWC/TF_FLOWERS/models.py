import tensorflow as tf
import keras
from keras.layers import Dense, Dropout, BatchNormalization, Flatten, Conv2D, MaxPooling2D, Input


import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join
#import tensorflow_probability as tfp
keras.backend.set_floatx('float32')



class NormalizeByMax(keras.layers.Layer):
    def __init__(self):
        super(NormalizeByMax, self).__init__()

    def call(self, inputs):
        max_values = tf.reduce_max(inputs, axis=1, keepdims=True)
        normalized_inputs = inputs / max_values
        return normalized_inputs

class Conv_NN(keras.layers.Layer):
    def __init__(self, input_shape, **kwargs):
        super(Conv_NN, self).__init__()
        input_tensor = Input(shape=(224, 224, 3))
        self.base_model = keras.applications.VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
        for layer in self.base_model.layers:
                layer.trainable = False
        self.flatten = Flatten()
        self.batch1 = BatchNormalization()
        self.dense3 = Dense(4096, activation='relu')
        self.batch2 = BatchNormalization()
        self.dense4 = Dense(2024, activation='relu')
        self.batch3 = BatchNormalization()
        self.dense5 = Dense(1048, activation='relu')
        self.batch4 = BatchNormalization()
        self.dense6 = Dense(784, activation='relu')
        
    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.flatten(x)
        x = self.batch1(x)
        x = self.dense3(x)
        x = self.batch2(x)
        x = self.dense4(x)
        x = self.batch3(x)
        x = self.dense5(x)
        x = self.batch4(x)
        x = self.dense6(x)
        return  x


class Linear_transfer(tf.keras.layers.Layer):
    def __init__(self,
                 rec_base,
                 units,
                 last=False
                 ):
        super(Linear_transfer, self).__init__()

        self.number_of_attractors = rec_base.shape[1]
        self.last = last
        self.units = units
        
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

        self.base_train = self.add_weight(shape=(units, int(units - self.number_of_attractors)),
                                          initializer=tf.keras.initializers.Orthogonal(gain=1.15, seed=None),
                                          regularizer=tf.keras.regularizers.OrthogonalRegularizer(factor=0.01,
                                                                                                  mode='columns'),
                                          trainable=True,
                                          name='base_train')
        
    def call(self, inputs, **kwargs):
        eigenvalues_total = tf.concat([tf.transpose(self.autov_tr), tf.transpose(self.autov_fix)], axis=0)
        diagonal = tf.linalg.diag(tf.clip_by_value(eigenvalues_total, clip_value_min=-1000000., clip_value_max=200.))
        base = tf.concat([self.base_train, self.base_fix], axis=1)
        A = tf.matmul(base, tf.matmul(diagonal, tf.linalg.inv(base)))

        return tf.transpose(A)

    def return_base(self):
        return tf.concat([self.base_train, self.base_fix], axis=1)

    def return_diagonal(self):
        return tf.concat([self.autov_tr, self.autov_fix], axis=0)

class Cowan_network(tf.keras.Model):
    def __init__(self,
                 size,
                 tmax,
                 attractors,
                 are_eigenvectors_trainable=True,
                 spectral=False,
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
        self.flag=False
        self.myA=None
        self.flag_train=False

    def build(self, input_shape):
        self.adiacency = Linear_transfer(rec_base=self.attractors,units=self.size)
        self.cnn = Conv_NN(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        x = tf.cast(x, dtype=tf.float32)
        x = self.cnn(x)
        matrix_A = self.adiacency(x)
        if self.flag_train:
            y = x
            for k in range(self.iterations):
                I1 = self.wee * x - self.wei * y + self.he + self.Dx * tf.matmul(x, matrix_A)
                I2 = self.wie * x - self.wii * y + self.hi 
                self.fe = self.fe1 * tf.math.tanh(self.beta1 * I1) + self.fe2
                self.fi = self.fi1 * tf.math.tanh(self.beta2 * I2) + self.fi2
                x = x + self.dt * (-self.ae * x + (1 - x) * self.fe)
                y = y + (self.dt / self.gamma) * (-self.ai * y + (1 - y) * self.fi)
        return x
    
    def dynamic(self, inputs, **kwargs):
        x = inputs
        x = tf.cast(x, dtype=tf.float32)
        x = self.cnn(x)
        y = x
        if self.flag:
            data=x[0,:].numpy()
            data.flatten()
            with open("outputx.txt", "a") as f:
                f.write(" ".join(map(str, data)) + "\n")
            data=y[0,:].numpy()
            data.flatten()
            with open("outputy.txt", "a") as f:
                f.write(" ".join(map(str, data)) + "\n")
        matrix_A = self.adiacency(x)
        for k in range(self.iterations):
            I1 = self.wee * x - self.wei * y + self.he + self.Dx * tf.matmul(x, matrix_A)
            I2 = self.wie * x - self.wii * y + self.hi 
            self.fe = self.fe1 * tf.math.tanh(self.beta1 * I1) + self.fe2
            self.fi = self.fi1 * tf.math.tanh(self.beta2 * I2) + self.fi2
            x = x + self.dt * (-self.ae * x + (1 - x) * self.fe)
            y = y + (self.dt / self.gamma) * (-self.ai * y + (1 - y) * self.fi)
            if self.flag:
                data=x[0,:].numpy()
                data.flatten()
                with open("outputx.txt", "a") as f:
                    f.write(" ".join(map(str, data)) + "\n")
                data=y[0,:].numpy()
                data.flatten()
                with open("outputy.txt", "a") as f:
                    f.write(" ".join(map(str, data)) + "\n")
        return x

    