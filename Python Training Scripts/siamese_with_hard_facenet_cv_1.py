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

#Inception resnet v1 - FaceNet file
from inception_resnet_v1 import *

#Seed value for automatic shuffling from tensorflow datasets
SEED = 21

# Set GPU Growth
# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


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


#Function used specifically to map tf datasets
def preprocess_twin(anchor_img, other_img, label, a_lable, o_label):
    return(complete_preprocess(anchor_img), complete_preprocess(other_img), label, a_lable, o_label)

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

def make_siamese_model(model):

    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(160,160,3))

    # Validation image in the network
    validation_image = Input(name='validation_img', shape=(160,160,3))

    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    norm1 = L2Norm()(model(input_image))
    norm2 = L2Norm()(model(validation_image))
    distances = siamese_layer(norm1, norm2)

    # Classification layer
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


# ## Section contains functions for online triplet mining
#Code from this repository: https://github.com/omoindrot/tensorflow-triplet-loss

# Calculate pairwise distances
def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.linalg.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.linalg.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.dtypes.cast(tf.equal(distances, 0.0), tf.float32)
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    # Combine the two masks
    mask = tf.logical_and(indices_not_equal, labels_equal)

    return mask


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask = tf.logical_not(labels_equal)

    return mask

def batch_hard_binary_sampling(images,labels, embeddings, squared=False):
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = tf.dtypes.cast(mask_anchor_positive, tf.float32)
    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

    # shape (batch_size, 1)
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)
    tf.summary.scalar("hardest_positive_dist", tf.reduce_mean(hardest_positive_dist))

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = tf.dtypes.cast(mask_anchor_negative, tf.float32)

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)
    tf.summary.scalar("hardest_negative_dist", tf.reduce_mean(hardest_negative_dist))


    #Number of triplets
    triplet_number = tf.dtypes.cast(tf.math.count_nonzero(hardest_positive_dist), tf.int32)

    #List of all the available positives (0's indivate that the picture has no positive and therefore no triple)
    available_positives_mask = tf.logical_not(tf.math.equal(hardest_positive_dist,0.0))
    available_positives = tf.dtypes.cast(available_positives_mask, tf.float32)

    # Find the position of all the max positive distances and min negative distances
    positive_max_position = tf.math.argmax(anchor_positive_dist, 1)
    negative_min_position = tf.math.argmin(anchor_negative_dist, 1)


    # Find all the positions where the positives are only 0, this is likely and indication there are no positives
    # It could also be there are
    hard_pos = hardest_positive_dist.numpy()
    zero_positions = np.where(hard_pos == 0)[0]

    # Making sure not to remove cases where images are close to identical (distance between positivies is 0)
    n_labels = list(labels.numpy())
    contains_positives = [n_labels[element] for element in zero_positions if n_labels.count(n_labels[element]) > 1]
    for c in contains_positives:
        zero_positions = np.delete(zero_positions, np.where(c == 0))

    #Remove the anchors that have no positives
    pos_max = np.delete(positive_max_position.numpy(), zero_positions)
    neg_min = np.delete(negative_min_position.numpy(), zero_positions)
    all_images = images.numpy()

    anchors = []
    other = []
    binary = []


    for i in range(0, len(pos_max)):
        anchors.append(all_images[i])
        other.append(all_images[pos_max[i]])
        binary.append(1.0)

        anchors.append(all_images[i])
        other.append(all_images[neg_min[i]])
        binary.append(0.0)

    return tf.convert_to_tensor(anchors), tf.convert_to_tensor(other), tf.dtypes.cast(tf.convert_to_tensor(binary), tf.float32)

# ## Section contains functions for training process

#Function for one training step
@tf.function
def train_step(batch, epoch, opt, binary_cross_loss, facenet_model, siamese_model):
    if epoch % 10 == 0:
        new = tf.concat([batch[0], batch[1]], 0)
        embedding = facenet_model(new)
        labels = tf.concat([batch[3], batch[4]], 0)
        anchor, other, binary = tf.py_function(batch_hard_binary_sampling, inp=[new, labels,embedding],
                                               Tout=[tf.float32, tf.float32, tf.float32])
        X = list([anchor,other])
        Y = binary
    else:
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
def test_step(batch, epoch, binary_cross_loss, facenet_model, siamese_model):

    if epoch % 10 == 0:
        new = tf.concat([batch[0], batch[1]], 0)
        embedding = facenet_model(new)
        labels = tf.concat([batch[3], batch[4]], 0)
        anchor, other, binary = tf.py_function(batch_hard_binary_sampling, inp=[new, labels,embedding],
                                               Tout=[tf.float32, tf.float32, tf.float32])
        X = list([anchor,other])
        Y = binary
    else:
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
def train(data, v_data, EPOCHS, opt, binary_cross_loss, facenet_model, siamese_model,number):
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
            train_step(batch,epoch, opt, binary_cross_loss, facenet_model, siamese_model)
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
            test_step(batch,epoch, binary_cross_loss, facenet_model, siamese_model)

        val_acc = val_acc_metric.result()
        val_loss = val_loss_mean.result()

        v_loss.append(val_loss)
        v_acc.append(val_acc)

        print("Validation Loss: %.4f Validation acc: %.4f" % (float(val_loss),float(val_acc),))

        val_loss_mean.reset_states()
        val_acc_metric.reset_states()
    
    save_name = 'cross_validation_trained_models/siamesemodel_l1Dist_facenet_with_hard_mining_cv_'+str(number)+'.h5'
    siamese_model.save_weights(save_name)

    return epoch_l, t_loss, t_acc, v_loss, v_acc


# Prepare the metrics
train_acc_metric = keras.metrics.BinaryAccuracy()
val_acc_metric = keras.metrics.BinaryAccuracy()

train_loss_mean = tf.keras.metrics.Mean()
val_loss_mean = tf.keras.metrics.Mean()



"""-----------------Start of the Cross Validation----------------------"""
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

    # FaceNet model for the siamese network
    facenet_model = InceptionResNetV1()
    facenet_model.load_weights('facenet_weights.h5')

    # Freeze four convolution blocks
    for layer in facenet_model.layers:
        layer.trainable = False

    # Making the siamese model
    siamese_model = make_siamese_model(facenet_model)

    # Loss function fro training
    binary_cross_loss = tf.losses.BinaryCrossentropy()
    #Set BinaryCrossentropy(from_logits=True) if the input of the loss function are not normalized

    # Setting the learning rate scheduler
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=10000,
        decay_rate=0.9)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # Training the model
    EPOCHS = 100
    epoch_l, t_loss, t_acc, v_loss, v_acc = train(train_data,test_data, EPOCHS, opt, binary_cross_loss, facenet_model, siamese_model, number)

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
    file_name_save = 'cross_validation_results/siamese_training_facenet_cv_with_hard_sampling_'+str(number)+'.csv'

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
    del facenet_model

    del anchor_array
    del full_array
    del binary_array
    del anchor_label
    del other_image_labels
