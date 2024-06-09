import os
from os.path import join
import sys
#definisco le istruzioni per la GPU

dev = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = dev
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #
import tensorflow as tf
import keras
from keras.applications import VGG16
import matplotlib
import matplotlib.pyplot as plt
from functions import DatasetMaker, LoadConfig
from models import Cowan_network
import numpy as np
import shutil
keras.backend.set_floatx('float32')
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

if len(physical_devices) > 0:
    tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')  
    print("GPU available.")
else:
    print("No GPU available. Using CPU instead.")


#Fine istruzioni  per la GPU

import tensorflow_datasets as tfds


# Caricamento del dataset
(ds_train, ds_test), ds_info = tfds.load(
    'tf_flowers',  # Qui dovrebbe essere il nome del dataset ImageNet10 se disponibile
    split=['train[:90%]', 'train[90%:]'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

IMG_SIZE = 224  # Dimensione delle immagini (può variare in base al modello pre-addestrato utilizzato)
NUM_CLASSES = ds_info.features['label'].num_classes  # Numero di classi nel dataset

def preprocess_image(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0  # Normalizzazione
    return image, label

# Applicazione del preprocessing al dataset
ds_train = ds_train.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

# Ottimizzazione delle prestazioni del dataset
ds_train = ds_train.cache().shuffle(1000).batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)
ds_test = ds_test.batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)
# Ispezione del dataset
for images, labels in ds_train.take(1):
    print(f"Forma delle immagini nel batch: {images.shape}")
    print(f"Tipo di dati delle immagini: {images.dtype}")
    print(f"Forma delle etichette nel batch: {labels.shape}")
    print(f"Tipo di dati delle etichette: {labels.dtype}")
    
    # Visualizza la prima immagine del batch
    plt.imshow(images[0])
    plt.title(f'Label: {labels[0].numpy()}')
    plt.axis('off')
    plt.show()


# Estrazione e trasformazione dei dati da ds_train
def extract_and_transform_data(dataset):
    images_list = []
    labels_list = []
    for images, labels in dataset:
        images_list.extend(images.numpy())
        labels_list.extend(labels.numpy())

    # Convertire a tensori
    images_tensor = tf.convert_to_tensor(images_list)
    labels_tensor = tf.convert_to_tensor(labels_list)

    # Flatten delle immagini
    images_tensor_flattened = tf.reshape(images_tensor, (images_tensor.shape[0], -1))

    # Conversione delle etichette a vettori one-hot
    labels_tensor_one_hot = tf.one_hot(labels_tensor, depth=NUM_CLASSES)

    return images_tensor, labels_tensor_one_hot

# Estrazione dei dati da ds_train
x_train, y_train = extract_and_transform_data(ds_train)
x_test, y_test = extract_and_transform_data(ds_test)

print(f"Forma di x_train: {x_train.shape}")
print(f"Tipo di dati di x_train: {x_train.dtype}")
print(f"Forma di y_train: {y_train.shape}")
print(f"Tipo di dati di y_train: {y_train.dtype}")
print(f"Forma di x_test: {x_train.shape}")
print(f"Tipo di dati di x_test: {x_train.dtype}")
print(f"Forma di y_test: {y_train.shape}")
print(f"Tipo di dati di y_test: {y_train.dtype}")

# Estrai un batch di immagini dal dataset di addestramento
for images, labels in ds_train.take(1):
    # Converti il tensor a numpy array per visualizzare con matplotlib
    img_array = images.numpy()
    label_array = labels.numpy()

    # Visualizza la prima immagine del batch
    plt.imshow(img_array[0])
    plt.title(f'Label: {label_array[0]}')
    plt.axis('off')
    plt.show()

    # Salva l'immagine su disco
    plt.imsave('sample_image.png', img_array[0])
    
    plt.imshow(img_array[1])
    plt.title(f'Label: {label_array[1]}')
    plt.axis('off')
    plt.show()

    # Salva l'immagine su disco
    plt.imsave('sample_image1.png', img_array[1])
    
    plt.imshow(img_array[2])
    plt.title(f'Label: {label_array[2]}')
    plt.axis('off')
    plt.show()

    # Salva l'immagine su disco
    plt.imsave('sample_image2.png', img_array[2])


# Set up the config and log dir
config = LoadConfig('doppiabuca.yml')
log_dir = ".\\logs"
# clear log dir
shutil.rmtree(log_dir,
              ignore_errors=True)
# check if the directory is deleted correctly
if os.path.exists(log_dir):
    raise Exception('Directory not deleted')




#use_pretrained = True #no training
use_pretrained = True#si training
noise=0.0
# Load the dataset
DM = DatasetMaker(config)





my_test=y_test
my_train=y_train
flat_train = x_train
flat_test = x_test

attractors = DM.attractor_matrix
my_attractors = attractors
y_train=y_train.numpy()
y_test=y_test.numpy()
y_train=y_train[:, :5] @ attractors[:,0:5].T
y_test=y_test[:, :5] @ attractors[:,0:5].T  
attractors = np.linalg.qr(attractors)[0]
t_max = 35 

custom_model2 = Cowan_network(size=config['Network']['size'],
                attractors=attractors,
                tmax=t_max,
                **config['DynamicalParameters'])
custom_model2.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss='mse')



custom_model2(flat_train[0:2])
custom_model2.summary()
custom_model2.fit(x=flat_train,
                y=y_train,
                shuffle=True,
                batch_size=10,
                epochs=70,
                verbose=1)

custom_model2.flag_train=True

custom_model2.fit(x=flat_train,
                y=y_train,
                shuffle=True,
                batch_size=32,
                epochs=100,
                verbose=1)
size_pic=784

print('accuracy final')
custom_model2.iterations=300
counter=0

y_pred=custom_model2(flat_test[0:50, :])
y_pred2=y_pred
y_final=y_pred.numpy()

for i in range(50):

    x=(my_attractors.T-(np.ones([5,size_pic])*y_final[i,:]))*(my_attractors.T-(np.ones([5,size_pic])*y_final[i,:])) 
    x=np.sum(x,axis=1)
    x=x/np.sqrt((np.sum(my_attractors.T**2,axis=1)*np.sum((np.ones([5,size_pic])*y_final[i,:])**2,axis=1)))
    x=(1/x)/np.sum((1/x))
    if  np.argmax(x)==np.argmax(my_test[i,:]):
        counter=counter+1

y_pred=custom_model2(flat_test[50:100, :])
y_pred2=y_pred
y_final=y_pred.numpy()

for i in range(50):

    x=(my_attractors.T-(np.ones([5,size_pic])*y_final[i,:]))*(my_attractors.T-(np.ones([5,size_pic])*y_final[i,:])) 
    x=np.sum(x,axis=1)
    x=x/np.sqrt((np.sum(my_attractors.T**2,axis=1)*np.sum((np.ones([5,size_pic])*y_final[i,:])**2,axis=1)))
    x=(1/x)/np.sum((1/x))
    if  np.argmax(x)==np.argmax(my_test[i+50,:]):
        counter=counter+1

y_pred=custom_model2(flat_test[100:150, :])
y_pred2=y_pred
y_final=y_pred.numpy()

for i in range(50):

    x=(my_attractors.T-(np.ones([5,size_pic])*y_final[i,:]))*(my_attractors.T-(np.ones([5,size_pic])*y_final[i,:])) 
    x=np.sum(x,axis=1)
    x=x/np.sqrt((np.sum(my_attractors.T**2,axis=1)*np.sum((np.ones([5,size_pic])*y_final[i,:])**2,axis=1)))
    x=(1/x)/np.sum((1/x))
    if  np.argmax(x)==np.argmax(my_test[i+100,:]):
        counter=counter+1

y_pred=custom_model2(flat_test[150:200, :])
y_pred2=y_pred
y_final=y_pred.numpy()

for i in range(50):

    x=(my_attractors.T-(np.ones([5,size_pic])*y_final[i,:]))*(my_attractors.T-(np.ones([5,size_pic])*y_final[i,:])) 
    x=np.sum(x,axis=1)
    x=x/np.sqrt((np.sum(my_attractors.T**2,axis=1)*np.sum((np.ones([5,size_pic])*y_final[i,:])**2,axis=1)))
    x=(1/x)/np.sum((1/x))
    if  np.argmax(x)==np.argmax(my_test[i+150,:]):
        counter=counter+1
        
y_pred=custom_model2(flat_test[200:250, :])
y_pred2=y_pred
y_final=y_pred.numpy()

for i in range(50):

    x=(my_attractors.T-(np.ones([5,size_pic])*y_final[i,:]))*(my_attractors.T-(np.ones([5,size_pic])*y_final[i,:])) 
    x=np.sum(x,axis=1)
    x=x/np.sqrt((np.sum(my_attractors.T**2,axis=1)*np.sum((np.ones([5,size_pic])*y_final[i,:])**2,axis=1)))
    x=(1/x)/np.sum((1/x))
    if  np.argmax(x)==np.argmax(my_test[i+200,:]):
        counter=counter+1
        
y_pred=custom_model2(flat_test[250:300, :])
y_pred2=y_pred
y_final=y_pred.numpy()

for i in range(50):

    x=(my_attractors.T-(np.ones([5,size_pic])*y_final[i,:]))*(my_attractors.T-(np.ones([5,size_pic])*y_final[i,:])) 
    x=np.sum(x,axis=1)
    x=x/np.sqrt((np.sum(my_attractors.T**2,axis=1)*np.sum((np.ones([5,size_pic])*y_final[i,:])**2,axis=1)))
    x=(1/x)/np.sum((1/x))
    if  np.argmax(x)==np.argmax(my_test[i+250,:]):
        counter=counter+1
        
y_pred=custom_model2(flat_test[300:, :])
y_pred2=y_pred
y_final=y_pred.numpy()

for i in range(y_test[300:].shape[0]):

    x=(my_attractors.T-(np.ones([5,size_pic])*y_final[i,:]))*(my_attractors.T-(np.ones([5,size_pic])*y_final[i,:])) 
    x=np.sum(x,axis=1)
    x=x/np.sqrt((np.sum(my_attractors.T**2,axis=1)*np.sum((np.ones([5,size_pic])*y_final[i,:])**2,axis=1)))
    x=(1/x)/np.sum((1/x))
    if  np.argmax(x)==np.argmax(my_test[i+300,:]):
        counter=counter+1

print(counter/y_test.shape[0])
print('gamma')
print(custom_model2.gamma.numpy())
