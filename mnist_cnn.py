'''
TODO:
-Learn how to save states (variables?) to keep track of best number of epochs
    -When optimum number of epochs is found, retrain on full data set
-Encapsulate training into function that can be passes learning rate, keep_rate etc.
-Impliment random batching
-change submission file name to include relavent info (date/time/LR/KR/numEpochs etc.)
-Write submission file with headers
-Fix tensorboard, right now it measures accuracy on training set and not validation set (?)
-Change tensorboard log path to include relevant info (date/time/LR/KR/numEpochs etc.)
-Fix tensorboard step count?
-Explore changing filter size
'''
# Imports and setting up some variables
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

train_path = './data/train.csv'
test_path = './data/test.csv'

logs_path = './tensorboard_files/5'

num_classes = 10
LEARNING_RATE = 0.001
keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

image_width = 28
image_height = 28
image_pixels = image_width * image_height

hm_epoch = 20
batch_size = 100

x = tf.placeholder('float', [None, image_pixels], name='Feature_Input')
y = tf.placeholder('float', [None, num_classes], name='Label_Input')


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Convert int to one hot
def dense_to_one_hot(labels_dense, num_classes=10):
    labels_dense = np.array(labels_dense)
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
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
        'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        'W_fc': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
        'out': tf.Variable(tf.random_normal([1024, num_classes]))
    }
    biases = {
        'b_conv1': tf.Variable(tf.random_normal([32])),
        'b_conv2': tf.Variable(tf.random_normal([64])),
        'b_fc': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([num_classes]))
    }

    data = tf.reshape(data, shape=[-1, 28, 28, 1])

    conv1 = tf.nn.relu(conv2d(data, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, shape=[-1, 7 * 7 * 64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_prob)

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output


# Read in the training data
full_training = pd.read_csv(train_path)
testing = pd.read_csv(test_path)

# Preprocess data
labels = full_training['label'].values.tolist()
labels = dense_to_one_hot(labels, num_classes)
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
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    for epoch in range(hm_epoch):
        epoch_loss = 0
        for i in range(int(Xtrain.shape[0] / batch_size)):
            start = 0 + i * batch_size
            end = start + batch_size
            _, c, summary = sess.run(
                [optimizer, cost, summary_op],
                feed_dict={
                    x: Xtrain[start:end, :],
                    y: ytrain[start:end, :],
                    keep_prob: keep_rate
                })
            writer.add_summary(summary,
                               epoch * (Xtrain.shape[0] / batch_size) + i)
            epoch_loss += c
        print('Epoch',
              int(epoch + 1), 'completed out of', hm_epoch, 'Loss:',
              epoch_loss)

        print('Accuracy:',
              accuracy.eval({
                  x: Xvalid,
                  y: yvalid,
                  keep_prob: 1.0
              }))

    # Create submission predictions and write to .csv
    f_handle = open('submission.csv', 'ab')
    final_output_predictions = tf.argmax(prediction, 1)
    for i in range(int(Xtest.shape[0] / batch_size)):
        start = 0 + i * batch_size
        end = start + batch_size
        final_labels = final_output_predictions.eval({
            x: Xtest[start:end, :],
            keep_prob: 1.0
        })
        indexes = np.array(range(start + 1, end + 1))
        submission_matrix = np.column_stack((indexes, final_labels))
        np.savetxt(f_handle, submission_matrix, delimiter=',')
    f_handle.close()
