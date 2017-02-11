'''
TODO:
-Learn how to save states (variables?) to keep track of best number of epochs
    -When optimum number of epochs is found, retrain on full data set
-Encapsulate training into function that can be passes learning rate, keep_rate etc.
-Impliment random batching
-Fix tensorboard, right now it measures accuracy on training set and not validation set (?)
-Fix tensorboard step count?
-Explore changing filter size
'''
# Imports and setting up some variables
import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split

TRAIN_PATH = './data/train.csv'
TEST_PATH = './data/test.csv'

LOGS_PATH = './tensorboard_files/'

NUM_CLASSES = 10
LEARNING_RATE = 0.001
KEEP_RATE = 0.8
KEEP_PROB = tf.placeholder(tf.float32)

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT

HM_EPOCH = 1
BATCH_SIZE = 100

START_TIME = datetime.datetime.now().strftime("(%Y-%m-%d_%H:%M:%S)")

x = tf.placeholder('float', [None, IMAGE_PIXELS], name='Feature_Input')
y = tf.placeholder('float', [None, NUM_CLASSES], name='Label_Input')


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Convert int to one hot
def dense_to_one_hot(labels_dense, NUM_CLASSES=10):
    labels_dense = np.array(labels_dense)
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * NUM_CLASSES
    labels_one_hot = np.zeros((num_labels, NUM_CLASSES))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def preprocess_data(df):
    df = df.astype(np.float32)
    df = df.values.tolist()
    df = np.array(df)
    df = (df - 255.0 / 2) / 255.0
    return df


# Model network
def neural_network(data):
    weights = {
        'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
        'W_conv2': tf.Variable(tf.random_normal([3, 3, 32, 64])),
        'W_fc': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
        'out': tf.Variable(tf.random_normal([1024, NUM_CLASSES]))
    }
    biases = {
        'b_conv1': tf.Variable(tf.random_normal([32])),
        'b_conv2': tf.Variable(tf.random_normal([64])),
        'b_fc': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([NUM_CLASSES]))
    }

    data = tf.reshape(data, shape=[-1, 28, 28, 1])

    conv1 = tf.nn.relu(conv2d(data, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, shape=[-1, 7 * 7 * 64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, KEEP_PROB)

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output


# Read in the training data
full_training = pd.read_csv(TRAIN_PATH)
testing = pd.read_csv(TEST_PATH)

# Preprocess data
labels = full_training['label'].values.tolist()
labels = dense_to_one_hot(labels, NUM_CLASSES)
full_training = full_training.drop('label', axis=1)
Xtrain, Xvalid, ytrain, yvalid = train_test_split(
    full_training, labels, test_size=0.1)

Xtrain = preprocess_data(Xtrain)
Xvalid = preprocess_data(Xvalid)
Xtest = preprocess_data(testing)

# Create graph and add cost and optimization
prediction = neural_network(x)
with tf.name_scope('cross_entropy'):
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(prediction, y))
with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

# Calculate accuracy on cross-validation
with tf.name_scope('Accuracy'):
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

tf.summary.scalar('Cost', cost)
tf.summary.scalar('Accuracy', accuracy)
summary_op = tf.summary.merge_all()

# Train model with select samples
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(
        LOGS_PATH + '{}-{}-{}-{}'.format(START_TIME, LEARNING_RATE, KEEP_RATE,
                                         HM_EPOCH),
        graph=tf.get_default_graph())
    for epoch in range(HM_EPOCH):
        epoch_loss = 0
        for i in range(int(Xtrain.shape[0] / BATCH_SIZE)):
            start = 0 + i * BATCH_SIZE
            end = start + BATCH_SIZE
            _, c, summary = sess.run(
                [optimizer, cost, summary_op],
                feed_dict={
                    x: Xtrain[start:end, :],
                    y: ytrain[start:end, :],
                    KEEP_PROB: KEEP_RATE
                })
            writer.add_summary(summary,
                               epoch * (Xtrain.shape[0] / BATCH_SIZE) + i)
            epoch_loss += c
        print('Epoch',
              int(epoch + 1), 'completed out of', HM_EPOCH, 'Loss:',
              epoch_loss)

        print('Accuracy:',
              accuracy.eval({
                  x: Xvalid,
                  y: yvalid,
                  KEEP_PROB: 1.0
              }))

    # Create submission predictions and write to .csv
    f_handle = open('submission' + '-{}-{}-{}-{}.csv'.format(
        START_TIME, LEARNING_RATE, KEEP_RATE, HM_EPOCH), 'ab')
    np.savetxt(f_handle, np.array([['ImageID', 'Label']]), delimiter=',', fmt='%s')
    final_output_predictions = tf.argmax(prediction, 1)
    for i in range(int(Xtest.shape[0] / BATCH_SIZE)):
        start = 0 + i * BATCH_SIZE
        end = start + BATCH_SIZE
        final_labels = final_output_predictions.eval({
            x: Xtest[start:end, :],
            KEEP_PROB: 1.0
        })
        indexes = np.array(range(start + 1, end + 1))
        submission_matrix = np.column_stack((indexes, final_labels))
        np.savetxt(f_handle, submission_matrix, delimiter=',')
    f_handle.close()
