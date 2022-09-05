#!/bin/env python

# Import standard dependencies
import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import csv
from itertools import zip_longest

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

#Import other dependencies
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.models import model_from_json

#Seed value for automatic shuffling from tensorflow datasets
SEED = 21


# Set GPU Growth
import sys
file_path = 'outputfile_vgg_1.txt'
sys.stdout = open(file_path, "w")

# Avoid OOM errors by setting GPU Memory Consumption Growth
"""gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)"""

physical_devices = tf.config.list_physical_devices('GPU')
try:
    # Disable first GPU
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    logical_devices = tf.config.list_logical_devices('GPU')
    # Logical device was not created for first GPU
    assert len(logical_devices) == len(physical_devices) - 1
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass





# Data scaling and preprocessing functions

def preprocess(file_path, required_size=(224,224)):
    raw = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(raw, channels=3, dct_method='INTEGER_ACCURATE')
    image = tf.image.resize(image,required_size, method='nearest')
    image = tf.cast(image, 'float32')
    return np.array(image)

def complete_preprocess(image_path, required_size=(224,224)):
    # load image and detect the face
    image = preprocess(image_path)

    #Preprocessing
    face_array = preprocess_input(image)


    # Scale image to be between 0 and 1
    face_array = (face_array - np.amin(face_array)) / (np.amax(face_array) - np.amin(face_array))
    # Scale image to be between -1 and 1
    face_array = 2*face_array - 1

    return tf.convert_to_tensor(face_array)


#Function used specifically to map tf datasets
def preprocess_twin(anchor_img, other_img, label, a_lable, o_label):
    return(complete_preprocess(anchor_img), complete_preprocess(other_img), label, a_lable, o_label)

#Function to l2 normalize numpy arrays
def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))


# ## Build Distance Layer

# Siamese L1 Distance class (custom layer)
class L1Dist(Layer):

    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()

    # Similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

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


# ## Make Siamese Model

def make_siamese_model(model, expected_size = (224,224,3)):

    # Anchor image input in the network
    input_image = Input(name='input_img', shape= expected_size)

    # Validation image in the network
    validation_image = Input(name='validation_img', shape= expected_size)

    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    norm1 = L2Norm()(model(input_image))
    norm2 = L2Norm()(model(validation_image))
    distances = siamese_layer(norm1, norm2)

    # Classification layer
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


# ## Section contains functions for training process

#Function for one training step
@tf.function
def train_step(batch, epoch, opt, binary_cross_loss, embedding_model, siamese_model):

    X = batch[:2]
    Y = batch[2]

    # Record all of our operations
    with tf.GradientTape() as tape:
        # Forward pass
        yhat = siamese_model(X, training=True)
        # Calculate loss
        loss = binary_cross_loss(Y, yhat)

    # Calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)

    # Calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))

    #Update loss metric
    train_loss_mean.update_state(loss)

    #Updating training metric
    train_acc_metric.update_state(Y, yhat)

#Function for one test step
@tf.function
def test_step(batch, epoch, binary_cross_loss, embedding_model, siamese_model):

    X = batch[:2]
    Y = batch[2]

    # Forward pass
    val_logits = siamese_model(X, training=False)

    # Calculate loss
    loss = binary_cross_loss(Y, val_logits)



    #Updating validation metric
    val_acc_metric.update_state(Y, val_logits)
    val_loss_mean.update_state(loss)



# Function to build Training Loop
def train(data, v_data, EPOCHS, opt, binary_cross_loss, embedding_model, siamese_model, number):
    epoch_l = []
    t_loss = []
    t_acc = []
    v_loss = []
    v_acc = []

    # Loop through epochs
    for epoch in range(1, EPOCHS+1):
        epoch_l.append(epoch)

        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))

        # Loop through each batch
        for idx, batch in enumerate(data):
            # Run train step here
            train_step(batch,epoch, opt, binary_cross_loss, embedding_model, siamese_model)
            progbar.update(idx+1)

        # Display metrics at the end of each epoch.
        train_loss = train_loss_mean.result()
        train_acc = train_acc_metric.result()
        t_loss.append(train_loss)
        t_acc.append(train_acc)
        print("Training Loss: %.4f Training accuracy: %.4f" % (float(train_loss),float(train_acc),))

        # Reset training metrics at the end of each epoch
        train_loss_mean.reset_states()
        train_acc_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        for batch in v_data:
            test_step(batch,epoch, binary_cross_loss, embedding_model, siamese_model)

        val_acc = val_acc_metric.result()
        val_loss = val_loss_mean.result()

        v_loss.append(val_loss)
        v_acc.append(val_acc)

        print("Validation Loss: %.4f Validation acc: %.4f" % (float(val_loss),float(val_acc),))

        val_loss_mean.reset_states()
        val_acc_metric.reset_states()

        # Save checkpoints
        if epoch % 10 == 0:
            name = 'training_checkpoints/training_'+str(number)+'/checkpoint_'+str(epoch)
            siamese_model.save_weights(name)
        
    save_name = 'cross_validation_trained_models/siamesemodel_l1Dist_vgg_no_hard_mining_cv_'+str(number)+'.h5'
    siamese_model.save_weights(save_name)

    return epoch_l, t_loss, t_acc, v_loss, v_acc


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

# ## Prepare the metrics
train_acc_metric = keras.metrics.BinaryAccuracy()
val_acc_metric = keras.metrics.BinaryAccuracy()

train_loss_mean = tf.keras.metrics.Mean()
val_loss_mean = tf.keras.metrics.Mean()

# ## Main function
if __name__ == "__main__":

    #IMPORTANT! "number" indicated which iteration of the cross validation training occurs
    number = 1

    #Prepare the training data into lists
    file_name = 'cross_validation_data/siamese_training_data_cv_'+str(number)+'.csv'
    df = pd.read_csv(file_name)

    anchor_array = df.anchor.to_list()
    full_array = df.other_image.to_list()
    binary_array = df.binary.to_list()
    anchor_label = df.anchor_label.to_list()
    other_image_labels = df.other_image_label.to_list()

    #Place lists into tf datasets for training process
    data = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(anchor_array)
                                ,tf.data.Dataset.from_tensor_slices(full_array),
                                tf.data.Dataset.from_tensor_slices(binary_array),
                                tf.data.Dataset.from_tensor_slices(anchor_label),
                                tf.data.Dataset.from_tensor_slices(other_image_labels)
                            ))


    # Build dataloader pipeline
    data = data.map(lambda x, y, z, a, b: tf.py_function(preprocess_twin, inp = (x, y, z, a, b), Tout=(tf.float32, tf.float32, tf.int32, tf.int32, tf.int32)))
    data = data.shuffle(buffer_size=1024)


    # Training partition
    train_data = data.take(round(len(data)*0.8))
    train_data = train_data.batch(32)


    # Validation partition
    test_data = data.skip(round(len(data)*0.8))
    test_data = test_data.take(round(len(data)*0.2))
    test_data = test_data.batch(32)



    # VGG model for the siamese network
    vgg_model = vvg_model()


    # Freeze four convolution blocks
    for layer in vgg_model.layers:
        layer.trainable = False

    # Making the siamese model
    siamese_model = make_siamese_model(vgg_model)

        
    # Loss function fro training
    binary_cross_loss = tf.losses.BinaryCrossentropy()
    #Set BinaryCrossentropy(from_logits=True) if the input of the loss function are not normalized

    #Setting the learning rate scheduler
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.01,
            decay_steps=10000,
            decay_rate=0.9)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    #Training the model
    EPOCHS = 100
    epoch_l, t_loss, t_acc, v_loss, v_acc = train(train_data,test_data, EPOCHS, opt, binary_cross_loss, vgg_model, siamese_model,number)


    # Saving loss and accuracy values from training
    t_loss1 = []
    t_acc1 = []
    v_loss1 = []
    v_acc1 = []

    for i in range(0,len(epoch_l)):
        t_loss1.append(t_loss[i].numpy())
        t_acc1.append(t_acc[i].numpy())
        v_loss1.append(v_loss[i].numpy())
        v_acc1.append(v_acc[i].numpy())

    #Saving values into a csv
    d = [epoch_l, t_loss1, v_loss1, t_acc1, v_acc1]
    export_data = zip_longest(*d, fillvalue = '')
    file_name_save = 'cross_validation_results/siamese_training_vgg_cv_no_hard_sampling_'+str(number)+'.csv'

    with open(file_name_save, 'w', encoding="ISO-8859-1", newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(("epoch", "t_loss", "v_loss", "t_acc", "v_acc"))
        wr.writerows(export_data)
    myfile.close()

    #Deleting memory
    del data
    del train_data
    del test_data
    del siamese_model
    del vgg_model

    del anchor_array
    del full_array
    del binary_array
    del anchor_label
    del other_image_labels
