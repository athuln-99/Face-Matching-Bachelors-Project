#!/usr/bin/env python


# Import standard dependencies
import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import csv
from itertools import zip_longest

# Import Dependencies for face detection and MTCNN Model
from numpy import asarray
from PIL import Image


# Import tensorflow dependencies - Functional API
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten, ZeroPadding2D, Convolution2D, Dropout, Activation
import tensorflow as tf

#Import Preprocessing packages
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from numpy import asarray

# VGG16 model
from tensorflow.keras.applications.vgg16 import VGG16

#Seed value for automatic shuffling from tensorflow datasets
SEED = 21

# Set GPU Growth
physical_devices = tf.config.list_physical_devices('GPU')
try:
    # Disable first GPU
    tf.config.set_visible_devices(physical_devices[1], 'GPU')
    logical_devices = tf.config.list_logical_devices('GPU')
    # Logical device was not created for first GPU
    assert len(logical_devices) == len(physical_devices) - 1
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

# Data scaling and preprocessing functions

def preprocess(file_path, required_size=(160,160)):
    raw = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(raw, channels=3, dct_method='INTEGER_ACCURATE')
    image = tf.image.resize(image,required_size, method='nearest')
    image = tf.cast(image, 'float32')
    return np.array(image)

def complete_preprocess(image_path, required_size=(224,224)):
    # load image and detect the face
    image = preprocess(image_path)

    #image = np.expand_dims(image, axis=0)

    #Preprocessing
    face_array = preprocess_input(image)


    # Scale image to be between 0 and 1
    face_array = (face_array - np.amin(face_array)) / (np.amax(face_array) - np.amin(face_array))
    # Scale image to be between -1 and 1
    face_array = 2*face_array - 1

    return tf.convert_to_tensor(face_array)


# ## Definition of VGG model in this section
def vvg_model():
    vgg_model = keras.Sequential()
    vgg_model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    vgg_model.add(Convolution2D(64, (3, 3), activation='relu'))
    vgg_model.add(ZeroPadding2D((1,1)))
    vgg_model.add(Convolution2D(64, (3, 3), activation='relu'))
    vgg_model.add(MaxPooling2D((2,2), strides=(2,2)))

    vgg_model.add(ZeroPadding2D((1,1)))
    vgg_model.add(Convolution2D(128, (3, 3), activation='relu'))
    vgg_model.add(ZeroPadding2D((1,1)))
    vgg_model.add(Convolution2D(128, (3, 3), activation='relu'))
    vgg_model.add(MaxPooling2D((2,2), strides=(2,2)))

    vgg_model.add(ZeroPadding2D((1,1)))
    vgg_model.add(Convolution2D(256, (3, 3), activation='relu'))
    vgg_model.add(ZeroPadding2D((1,1)))
    vgg_model.add(Convolution2D(256, (3, 3), activation='relu'))
    vgg_model.add(ZeroPadding2D((1,1)))
    vgg_model.add(Convolution2D(256, (3, 3), activation='relu'))
    vgg_model.add(MaxPooling2D((2,2), strides=(2,2)))

    vgg_model.add(ZeroPadding2D((1,1)))
    vgg_model.add(Convolution2D(512, (3, 3), activation='relu'))
    vgg_model.add(ZeroPadding2D((1,1)))
    vgg_model.add(Convolution2D(512, (3, 3), activation='relu'))
    vgg_model.add(ZeroPadding2D((1,1)))
    vgg_model.add(Convolution2D(512, (3, 3), activation='relu'))
    vgg_model.add(MaxPooling2D((2,2), strides=(2,2)))

    vgg_model.add(ZeroPadding2D((1,1)))
    vgg_model.add(Convolution2D(512, (3, 3), activation='relu'))
    vgg_model.add(ZeroPadding2D((1,1)))
    vgg_model.add(Convolution2D(512, (3, 3), activation='relu'))
    vgg_model.add(ZeroPadding2D((1,1)))
    vgg_model.add(Convolution2D(512, (3, 3), activation='relu'))
    vgg_model.add(MaxPooling2D((2,2), strides=(2,2)))

    vgg_model.add(Convolution2D(4096, (7, 7), activation='relu'))
    vgg_model.add(Dropout(0.5))
    vgg_model.add(Convolution2D(4096, (1, 1), activation='relu'))
    vgg_model.add(Dropout(0.5))
    vgg_model.add(Convolution2D(2622, (1, 1)))
    vgg_model.add(Flatten())
    vgg_model.add(Activation('softmax'))

    vgg_model.load_weights('vgg_face_weights.h5')

    vgg_face_embedding = Model(inputs=vgg_model.layers[0].input, outputs=vgg_model.layers[-2].output)

    return vgg_face_embedding


#IMPORTANT! "num" indicated which iteration of the cross validation training occurs
num = 1

#Prepare the training data into lists
file_name = 'cross_validation_data/siamese_training_data_cv_'+str(num)+'.csv'
df = pd.read_csv(file_name)

anchor_array = df.anchor.to_list()
full_array = df.other_image.to_list()
binary_array = df.binary.to_list()
anchor_label = df.anchor_label.to_list()
other_image_labels = df.other_image_label.to_list()


data = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(anchor_array)
                            ,tf.data.Dataset.from_tensor_slices(full_array),
                            tf.data.Dataset.from_tensor_slices(binary_array),
                            tf.data.Dataset.from_tensor_slices(anchor_label),
                            tf.data.Dataset.from_tensor_slices(other_image_labels)
                           ))



def preprocess_twin(anchor_img, other_img, label, a_lable, o_label):
    return(complete_preprocess(anchor_img), complete_preprocess(other_img), label)


# Build dataloader pipeline
data = data.map(lambda x, y, z, a, b: tf.py_function(preprocess_twin, inp = (x, y, z, a, b), Tout=(tf.float32, tf.float32, tf.int32)))
#data = data.cache()
data = data.shuffle(buffer_size=1024, seed = SEED)


# Training partition
train_data = data.take(round(len(data)*0.8))
train_data = train_data.batch(16)

#Prepare the testing data into lists
file_name = 'cross_validation_data/siamese_testing_data_cv_'+str(num)+'.csv'
dfa = pd.read_csv(file_name)

anchor_array = dfa.anchor.to_list()
full_array = dfa.other_image.to_list()
binary_array = dfa.binary.to_list()
anchor_label = dfa.anchor_label.to_list()
other_image_labels = dfa.other_image_label.to_list()

#Place lists into tf datasets for training process
data_next = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(anchor_array)
                            ,tf.data.Dataset.from_tensor_slices(full_array),
                            tf.data.Dataset.from_tensor_slices(binary_array),
                            tf.data.Dataset.from_tensor_slices(anchor_label),
                            tf.data.Dataset.from_tensor_slices(other_image_labels)
                           ))

# Build dataloader pipeline
data_next = data_next.map(lambda x, y, z, a, b: tf.py_function(preprocess_twin, inp = (x, y, z, a, b), Tout=(tf.float32, tf.float32, tf.int32)))
#data = data.cache()
data_next = data_next.shuffle(buffer_size=1024, seed = SEED)



test_data = data_next.take(round(len(data)*1))
test_data = test_data.batch(16)



# ## VGG Face model
vgg_model = vvg_model()


# Freeze four convolution blocks
for layer in vgg_model.layers:
    layer.trainable = False


def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))


# Siamese L1 Distance class (custom layer)
class L1Dist(Layer):

    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()

    # Similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

l1 = L1Dist()

# Siamese L2 Distance class (custom layer)
class L2Dist(Layer):

    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()

    # Similarity calculation
    def call(self, input_embedding, validation_embedding):
        sum_square = tf.math.reduce_sum(tf.math.square(input_embedding - validation_embedding), axis=1, keepdims=True)
        return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

#L2 Norm class (custom layer)
class L2Norm(Layer):

    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()

    # Similarity calculation
    def call(self, x):
        return x / tf.math.sqrt(tf.math.reduce_sum(tf.math.multiply(x, x), axis=1, keepdims=True))

def make_siamese_model(model):

    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(224,224,3))

    # Validation image in the network
    validation_image = Input(name='validation_img', shape=(224,224,3))

    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    norm1 = L2Norm()(model(input_image))
    norm2 = L2Norm()(model(validation_image))
    distances = siamese_layer(norm1, norm2)

    # Classification layer
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

siamese_model = make_siamese_model(vgg_model)


siamese_model.summary()

siamese_file_name = 'cross_validation_trained_models/siamesemodel_l1Dist_vgg_no_hard_mining_cv_'+str(num)+'.h5'

# Reload model
siamese_model.load_weights(siamese_file_name, by_name=True)



file_name = 'cross_validation_data/siamese_demonstration_not_straight_faces_data.csv'
dfa = pd.read_csv(file_name)

anchor_array = dfa.anchor.to_list()
full_array = dfa.other_image.to_list()
binary_array = dfa.binary.to_list()
anchor_label = dfa.anchor_label.to_list()
other_image_labels = dfa.other_image_label.to_list()


# In[ ]:


data_next = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(anchor_array)
                            ,tf.data.Dataset.from_tensor_slices(full_array),
                            tf.data.Dataset.from_tensor_slices(binary_array),
                            tf.data.Dataset.from_tensor_slices(anchor_label),
                            tf.data.Dataset.from_tensor_slices(other_image_labels)
                           ))

# Build dataloader pipeline
data_next = data_next.map(lambda x, y, z, a, b: tf.py_function(preprocess_twin, inp = (x, y, z, a, b), Tout=(tf.float32, tf.float32, tf.int32)))
#data = data.cache()
data_next = data_next.shuffle(buffer_size=1024, seed = SEED)



test_data = data_next.take(round(len(data_next)*1))
test_data = test_data.batch(16)





@tf.function
def facenet_test(batch):
    #Getting the embeddings for 3 images
    #Anchor is just an image of a person, it does not signify anything else
    #Positive is another picture of the same person in anchor
    #Negative is a random picture of someone that is NOT the person in anchor
    test = siamese_model([batch[0], batch[1]])

    return test

#Siamese training
positive_distances = []
negative_distances = []
for idx, batch in enumerate(test_data):
    test = facenet_test(batch)

    for idx in range(0, len(batch[0])):
        if(batch[2][idx] == 1):
            positive_distances.append(test[idx][0].numpy())
        else:
            negative_distances.append(test[idx][0].numpy())

d = [positive_distances, negative_distances]
export_data = zip_longest(*d, fillvalue = '')


file_name_save = 'cross_validation_results/demonstration2/demonstration_not_straight_faces_data_distances_vgg_no_mining_cv_'+str(num)+'.csv'
with open(file_name_save, 'w', encoding="ISO-8859-1", newline='') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(('positive_distances','negative_distances'))
    wr.writerows(export_data)
myfile.close()




file_name = 'cross_validation_data/siamese_demonstration_all_straight_faces_data.csv'
dfa = pd.read_csv(file_name)

anchor_array = dfa.anchor.to_list()
full_array = dfa.other_image.to_list()
binary_array = dfa.binary.to_list()
anchor_label = dfa.anchor_label.to_list()
other_image_labels = dfa.other_image_label.to_list()


# In[ ]:


data_next = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(anchor_array)
                            ,tf.data.Dataset.from_tensor_slices(full_array),
                            tf.data.Dataset.from_tensor_slices(binary_array),
                            tf.data.Dataset.from_tensor_slices(anchor_label),
                            tf.data.Dataset.from_tensor_slices(other_image_labels)
                           ))

# Build dataloader pipeline
data_next = data_next.map(lambda x, y, z, a, b: tf.py_function(preprocess_twin, inp = (x, y, z, a, b), Tout=(tf.float32, tf.float32, tf.int32)))
#data = data.cache()
data_next = data_next.shuffle(buffer_size=1024, seed = SEED)



test_data = data_next.take(round(len(data_next)*1))
test_data = test_data.batch(16)





@tf.function
def facenet_test(batch):
    #Getting the embeddings for 3 images
    #Anchor is just an image of a person, it does not signify anything else
    #Positive is another picture of the same person in anchor
    #Negative is a random picture of someone that is NOT the person in anchor
    test = siamese_model([batch[0], batch[1]])

    return test

#Siamese training
positive_distances = []
negative_distances = []
for idx, batch in enumerate(test_data):
    test = facenet_test(batch)

    for idx in range(0, len(batch[0])):
        if(batch[2][idx] == 1):
            positive_distances.append(test[idx][0].numpy())
        else:
            negative_distances.append(test[idx][0].numpy())

d = [positive_distances, negative_distances]
export_data = zip_longest(*d, fillvalue = '')


file_name_save = 'cross_validation_results/demonstration2/demonstration_all_straight_faces_data_distances_vgg_no_mining_cv_'+str(num)+'.csv'
with open(file_name_save, 'w', encoding="ISO-8859-1", newline='') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(('positive_distances','negative_distances'))
    wr.writerows(export_data)
myfile.close()
