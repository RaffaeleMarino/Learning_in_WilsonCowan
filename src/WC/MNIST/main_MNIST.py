import os
from os.path import join
import sys
#GPU instructions 

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
    tf.config.experimental.set_memory_growth(physical_devices[0], True)    
    print("GPU available.")
else:
    print("No GPU available. Using CPU instead.")

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


DM = DatasetMaker(config)
(flat_train, y_train), (flat_test, y_test) = DM.load_MNIST_dataset(perturb=0.0)
attractors = DM.attractor_matrix
my_test=y_test
my_attractors = attractors
y_train=y_train[:, :10] @ attractors[:,0:10].T
y_test=y_test[:, :10] @ attractors[:,0:10].T  
attractors = np.linalg.qr(attractors)[0]
t_max = 25
# Model
model = Cowan_network(size=config['Network']['size'],
                attractors=attractors,
                tmax=t_max,
                **config['DynamicalParameters'])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
            loss='mse')
model(flat_train[0:2])
model.summary()
model.fit(x=flat_train,
        y=y_train,
        shuffle=True,
        batch_size=200,
        epochs=350,
        verbose=1)

#we increment the number of iteration only for testing
model.iterations=400
print('gamma value')
print(model.gamma.numpy())
y_pred=model(flat_test)
y_pred2=y_pred
counter=0
print('accuracy')
y_final=y_pred.numpy()

for i in range(y_test.shape[0]):

    x=(my_attractors.T-(np.ones([10,784])*y_final[i,:]))*(my_attractors.T-(np.ones([10,784])*y_final[i,:])) 
    x=np.sum(x,axis=1)
    x=x/np.sqrt((np.sum(my_attractors.T**2,axis=1)*np.sum((np.ones([10,784])*y_final[i,:])**2,axis=1)))
    x=(1/x)/np.sum((1/x))
    if  np.argmax(x)==np.argmax(my_test[i,:]):
        counter=counter+1

print(counter/y_test.shape[0])