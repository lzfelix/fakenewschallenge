# coding:utf-8
import csv
import os

import cloudpickle
import tensorflow as tf

MODEL_PATH = './model'
MODEL_META = os.path.join(MODEL_PATH, 'model.meta')

PATH_TRAIN_PICKLE = './dataset/dataset_train_encoded.pyk'
PATH_TEST_PICKLE  = './dataset/dataset_test_encoded.pyk'

PREDICTIONS_FILE  = './dataset/predicted_test.csv'
GOLD_LABEL_FILE   = './dataset/gold_labels.csv'


with open(PATH_TEST_PICKLE, 'rb') as file:
    X = cloudpickle.load(file)['X']

with open(PATH_TRAIN_PICKLE, 'rb') as file:
    label_encoder = cloudpickle.load(file)['label_encoder']


# loading the model graph and predicting
session = tf.Session()

loader = tf.train.import_meta_graph(MODEL_META)
loader.restore(session, tf.train.latest_checkpoint(MODEL_PATH))

graph = tf.get_default_graph()
x_in = graph.get_tensor_by_name('x_layer:0')
predict_op = graph.get_tensor_by_name('predict:0')
train_mode_in = graph.get_tensor_by_name('train_mode:0')

predictions = session.run(predict_op, feed_dict={
    x_in: X,
    train_mode_in: False
})

predictions = label_encoder.inverse_transform(predictions)


# storing predictions
gold_file  = open(GOLD_LABEL_FILE, 'r')
gold_reader = csv.DictReader(gold_file)

write_file = open(PREDICTIONS_FILE, 'w')

writer = csv.DictWriter(write_file, fieldnames=['Headline','Body ID','Stance'])
writer.writeheader()
for sample, prediction in zip(gold_reader, predictions.tolist()):
    writer.writerow({'Body ID': sample['Body ID'],'Headline': sample['Headline'],'Stance': prediction})

# finally, run "python scorer.py gold_labels.csv predicted_test.csv"

