import os
from os.path import join
import sys
#definisco le istruzioni per la GPU

dev = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = dev
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #
import tensorflow as tf
tf.keras.backend.set_floatx('float32')
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

if len(physical_devices) > 0:
    tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')  
    print("GPU available.")
else:
    print("No GPU available. Using CPU instead.")


#Fine istruzioni  per la GPU
import matplotlib
import matplotlib.pyplot as plt
from functions import DatasetMaker, LoadConfig
from models import Cowan_network
import numpy as np
import shutil

# Set up the config and log dir
config = LoadConfig('doppiabuca.yml')
log_dir = ".\\logs"
# clear log dir
shutil.rmtree(log_dir,
              ignore_errors=True)
# check if the directory is deleted correctly
if os.path.exists(log_dir):
    raise Exception('Directory not deleted')


noise=0.0
# Load the dataset
DM = DatasetMaker(config)

(flat_train, y_train), (flat_test, y_test)= DM.load_CIFAR_dataset(perturb=noise)#
y_test=y_test.reshape([y_test.shape[0],y_test.shape[2]])
y_train=y_train.reshape([y_train.shape[0],y_train.shape[2]])
my_test=y_test
my_train=y_train

(x_train, y_train_cifar), (x_test, y_test_cifar) = tf.keras.datasets.cifar10.load_data()
flat_train = x_train/255.0
flat_test = x_test/255.0

attractors = DM.attractor_matrix
my_attractors = attractors

y_train=y_train[:, :10] @ attractors[:,0:10].T
y_test=y_test[:, :10] @ attractors[:,0:10].T  
attractors = np.linalg.qr(attractors)[0]

t_max = 35 # non modificare questo numero tanto non serve a nulla. 

custom_model1 = Cowan_network(size=config['Network']['size'],
                attractors=attractors,
                tmax=t_max,
                **config['DynamicalParameters'])

custom_model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss='mse')
custom_model1(flat_train[0:2])
custom_model1.flag_train=False
custom_model1.summary()
A=custom_model1.myA.numpy()
L=custom_model1.diag.numpy()
Phi=custom_model1.base.numpy()
np.savetxt('matrixAinitial.txt', A)
np.savetxt('matrixLinitial.txt', L)
np.savetxt('matrixPhiinitial.txt', Phi)
custom_model1.fit(x=flat_train,
                y=y_train,
                shuffle=True,
                batch_size=10,
                epochs=70,
                verbose=1)


custom_model2 = Cowan_network(size=config['Network']['size'],
                attractors=attractors,
                tmax=t_max,
                **config['DynamicalParameters'])

custom_model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss='mse')
custom_model2(flat_train[0:2])
custom_model2.set_weights(custom_model1.get_weights())
custom_model2.flag_train=True
custom_model2.summary()
custom_model2.fit(x=flat_train,
                y=y_train,
                shuffle=True,
                batch_size=250,
                epochs=35,
                verbose=1)
size_pic=784

print('accuracy final')

custom_model2.iterations=800
y_pred=custom_model2(flat_test)
y_pred2=y_pred
y_final=y_pred.numpy()

for i in range(y_test.shape[0]):

    x=(my_attractors.T-(np.ones([10,size_pic])*y_final[i,:]))*(my_attractors.T-(np.ones([10,size_pic])*y_final[i,:])) 
    x=np.sum(x,axis=1)
    x=x/np.sqrt((np.sum(my_attractors.T**2,axis=1)*np.sum((np.ones([10,size_pic])*y_final[i,:])**2,axis=1)))
    x=(1/x)/np.sum((1/x))
    if  np.argmax(x)==np.argmax(my_test[i,:]):
        counter=counter+1

print(counter/y_test.shape[0])
print('gamma')
print(custom_model2.gamma.numpy())

