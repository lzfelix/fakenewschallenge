# coding:utf-8

import os
import logging
logging.basicConfig(format=logging.BASIC_FORMAT, level=logging.INFO)

import numpy as np
import tensorflow as tf
import cloudpickle
import random

# File paths
PATH_TRAIN_PICKLE = './dataset_train_encoded.pyk'
PATH_TEST_PICKLE  = './dataset_test_encoded.pyk'
PREDICTIONS_FILE  = 'output_predictions.csv'

# Model hyperparams. See Riedel1 et al. for details
INPUT_LAYER_SIZE = 5000
HIDDEN_LAYER_SIZE = 100
AMOUNT_CLASSES = 4

DROPOUT_KEEP_P = 0.6
L2_LAMBDA = 1e-5            # < paper says 1e-4, code uses 1e-5 (typo?)
LEARNING_RATE = 1e-2
GRAD_CLIP_RATIO = 5

TRAIN_BATCH_SIZE = 500
EPOCHS = 90

logging.info('Loading train set')
with open(PATH_TRAIN_PICKLE, 'rb') as file:
    stash = cloudpickle.load(file)
    X_train = np.asarray(stash['X'])
    y_train = np.asarray(stash['y'])

logging.info('Loading test set')
with open(PATH_TEST_PICKLE, 'rb') as file:
    X_test = cloudpickle.load(file)['X']
    X_test = np.asarray(X_test)

# Discovering the amount of samples (n) features (m)
n = len(X_train)
m = len(X_train[0])

def build_graph():
    # input paths
    x_pl = tf.placeholder(tf.float32, [None, m], 'x_layer')
    y_pl = tf.placeholder(tf.int64, [None], 'y_placeholder')
    dropout_proba_pl = tf.placeholder(tf.float32)

    # Magic trick to get the size of the batch without hard coding
    batch_size = tf.shape(x_pl)[0]

    # NN architecture
    hidden_layer = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.linear(x_pl, HIDDEN_LAYER_SIZE)), keep_prob=DROPOUT_KEEP_P)
    linear_layer = tf.nn.dropout(tf.contrib.layers.linear(hidden_layer, AMOUNT_CLASSES), keep_prob=DROPOUT_KEEP_P)
    logits = tf.reshape(linear_layer, [batch_size, AMOUNT_CLASSES])

    # adding regularization
    ws = tf.trainable_variables()
    l2_reg = tf.add_n([tf.nn.l2_loss(w) for w in ws if 'bias' not in w.name]) * L2_LAMBDA 

    # computing cross-entropy loss
    J = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_pl) + l2_reg)

    # adding prediction paths. Shortcut for prediction
    predict = tf.argmax(logits, axis=1)


    return (x_pl, y_pl, dropout_proba_pl, ws, J, predict)


# getting paths to the compute graph
x_pl, y_pl, dropout_proba_pl, W, loss_function, predict = build_graph()

optimizer = tf.train.AdamOptimizer(LEARNING_RATE)

# clipping the gradients, as described in the paper
grads = tf.gradients(loss_function, W)
clipped_grads, _ = tf.clip_by_global_norm(grads, clip_norm=GRAD_CLIP_RATIO)

# optimizing with clipped grads
optimization_step = optimizer.apply_gradients(zip(clipped_grads, W))

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    total_loss = 0
    for epoch in range(EPOCHS):
        logging.info('Training epoch {}/{}'.format(epoch+1, EPOCHS))

        # shuffling training samples for each epoch
        samples_indices = list(range(n))
        random.shuffle(samples_indices)
        epoch_loss = 0

        for i in range(n // TRAIN_BATCH_SIZE):
            start = i * TRAIN_BATCH_SIZE
            indices = samples_indices[start:start+TRAIN_BATCH_SIZE]

            X_batch = X_train[indices]
            y_batch = y_train[indices]

            _, Jt = session.run([optimization_step, loss_function], feed_dict={
                x_pl: X_batch,
                y_pl: y_batch,
                dropout_proba_pl: DROPOUT_KEEP_P,
            })

            epoch_loss += Jt

        logging.info('Done. Cost function: {:4.4}'.format(epoch_loss))

    # persisting the model
    tf.train.Saver().save(session, os.path.join(os.getcwd(), 'model/model'))


