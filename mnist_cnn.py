'''
TODO:
-When optimum number of epochs is found, retrain on full data set
-Fix tensorboard, right now it measures accuracy on training set and not validation set (?)
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

LEARNING_RATE = 0.001
LEARNING_RATE_PH = tf.placeholder(tf.float32)
KEEP_RATE = 0.5
KEEP_RATE_PH = tf.placeholder(tf.float32)

NUM_CLASSES = 10
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT

HM_EPOCH = 30
BATCH_SIZE = 100

x = tf.placeholder('float', [None, IMAGE_PIXELS], name='Feature_Input')
y = tf.placeholder('float', [None, NUM_CLASSES], name='Label_Input')


def conv2d(x, W):
    """Convolution OP with 1,1,1,1 stride and SAME padding"""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    """Maxpooling with 1,2,2,1 size, 1,2,2,1 stide and SAME padding"""
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def dense_to_one_hot(labels_dense, NUM_CLASSES=10):
    """Converts dense labeling to one-hot"""
    labels_dense = np.array(labels_dense)
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * NUM_CLASSES
    labels_one_hot = np.zeros((num_labels, NUM_CLASSES))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def preprocess_feature_data(df):
    """Converts pixel data to float, then scales values to between 0-1"""
    df = df.astype(np.float32)
    df = df.values.tolist()
    df = np.array(df)
    df = (df - 255.0 / 2) / 255.0
    return df


def shuffle_training_data(training_features, training_labels):
    features_and_labels = np.concatenate(
        (training_features, training_labels), axis=1)
    np.random.shuffle(features_and_labels)
    shuffled_training_labels = features_and_labels[:, -10:]
    shuffled_training_features = features_and_labels[:, :-10]
    return shuffled_training_features, shuffled_training_labels


def neural_network(data):
    """Defines neural network model, not including cost/optimization/evaluation"""
    weights = {
        'W_conv1':
        tf.Variable(tf.random_normal([5, 5, 1, 32]), name='Conv1Weights'),
        'W_conv2':
        tf.Variable(tf.random_normal([3, 3, 32, 64]), name='Conv2Weights'),
        'W_fc':
        tf.Variable(tf.random_normal([7 * 7 * 64, 1024]), name='FCWeights'),
        'out':
        tf.Variable(tf.random_normal([1024, NUM_CLASSES]), name='OUTWeights')
    }
    biases = {
        'b_conv1': tf.Variable(tf.random_normal([32]), name='CONV1Biases'),
        'b_conv2': tf.Variable(tf.random_normal([64]), name='CONV2Biases'),
        'b_fc': tf.Variable(tf.random_normal([1024]), name='FCBiases'),
        'out': tf.Variable(tf.random_normal([NUM_CLASSES]), name='OUTBiases')
    }

    data = tf.reshape(data, shape=[-1, 28, 28, 1])
    with tf.name_scope("Conv1"):
        conv1 = tf.nn.relu(
            conv2d(data, weights['W_conv1']) + biases['b_conv1'])
        conv1 = maxpool2d(conv1)

    with tf.name_scope("Conv2"):
        conv2 = tf.nn.relu(
            conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
        conv2 = maxpool2d(conv2)

    with tf.name_scope("FC"):
        fc = tf.reshape(conv2, shape=[-1, 7 * 7 * 64])
        fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
        fc = tf.nn.dropout(fc, KEEP_RATE_PH)

    with tf.name_scope("Output"):
        output = tf.matmul(fc, weights['out']) + biases['out']

    return output


def train_network(Xtrain,
                  ytrain,
                  LEARNING_RATE=LEARNING_RATE,
                  KEEP_RATE=KEEP_RATE,
                  HM_EPOCH=HM_EPOCH,
                  BATCH_SIZE=BATCH_SIZE,
                  SHUFFLE_TRAINING_DATA=True):
    """Trains model defined in neural_network using x, y placeholders"""

    highest_accuracy = 0.0

    sess.run(tf.global_variables_initializer())
    START_TIME = datetime.datetime.now().strftime("(%Y-%m-%d_%H:%M:%S)")
    writer = tf.summary.FileWriter(
        LOGS_PATH + '{}-{}-{}-{}'.format(START_TIME, LEARNING_RATE, KEEP_RATE,
                                         HM_EPOCH),
        graph=tf.get_default_graph())
    for epoch in range(HM_EPOCH):
        epoch_loss = 0
        if SHUFFLE_TRAINING_DATA:
            Xtrain, ytrain = shuffle_training_data(Xtrain, ytrain)
        for i in range(int(Xtrain.shape[0] / BATCH_SIZE)):
            start = 0 + i * BATCH_SIZE
            end = start + BATCH_SIZE
            _, c, summary = sess.run(
                [optimizer, cost, summary_op],
                feed_dict={
                    x: Xtrain[start:end, :],
                    y: ytrain[start:end, :],
                    KEEP_RATE_PH: KEEP_RATE,
                    LEARNING_RATE_PH: LEARNING_RATE
                })
            writer.add_summary(summary,
                               (epoch * Xtrain.shape[0]) + BATCH_SIZE *
                               (i + 1))
            epoch_loss += c

        print('Epoch',
              int(epoch + 1), 'completed out of', HM_EPOCH, 'Loss:',
              epoch_loss)
        last_accuracy = float(accuracy.eval({x: Xvalid, y: yvalid, KEEP_RATE_PH: 1.0}))
        print('Accuracy:', last_accuracy)
        if last_accuracy > highest_accuracy:
            print('New highest accuracy')
            saver.save(sess, "./checkpoints/checkpoint.ckpt")
            highest_accuracy = last_accuracy
    return output


def evaluate_test_data(LEARNING_RATE=LEARNING_RATE,
                       KEEP_RATE=KEEP_RATE,
                       HM_EPOCH=HM_EPOCH,
                       BATCH_SIZE=BATCH_SIZE):
    # Create submission predictions and write to .csv
    START_TIME = datetime.datetime.now().strftime("(%Y-%m-%d_%H:%M:%S)")
    f_handle = open('submission' + '-{}-{}-{}-{}.csv'.format(
        START_TIME, LEARNING_RATE, KEEP_RATE, HM_EPOCH), 'ab')
    np.savetxt(
        f_handle, np.array([['ImageID', 'Label']]), delimiter=',', fmt='%s')
    final_output_predictions = tf.argmax(output, 1)
    for i in range(int(Xtest.shape[0] / BATCH_SIZE)):
        start = 0 + i * BATCH_SIZE
        end = start + BATCH_SIZE
        final_labels = final_output_predictions.eval({
            x: Xtest[start:end, :],
            KEEP_RATE_PH: 1.0
        })
        indexes = np.array(range(start + 1, end + 1))
        submission_matrix = np.column_stack((indexes, final_labels))
        np.savetxt(f_handle, submission_matrix, delimiter=',')
    f_handle.close()


# Read in the training data
full_training = pd.read_csv(TRAIN_PATH)
testing = pd.read_csv(TEST_PATH)

# Preprocess data
labels = full_training['label'].values.tolist()
labels = dense_to_one_hot(labels, NUM_CLASSES)

full_training = full_training.drop('label', axis=1)
Xtrain, Xvalid, ytrain, yvalid = train_test_split(
    full_training, labels, test_size=0.1)

Xtrain = preprocess_feature_data(Xtrain)
Xvalid = preprocess_feature_data(Xvalid)
Xtest = preprocess_feature_data(testing)

# Train model with select samples
with tf.Session() as sess:
    # Create graph and add cost and optimization
    output = neural_network(x)
    with tf.name_scope("Cost/Optimizer"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE_PH).minimize(cost)

    # Calculate accuracy on cross-validation
    with tf.name_scope("Accuracy_Calc"):
        correct = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    # Add ops for saving variables
    saver = tf.train.Saver()

    # Create summary scalars
    tf.summary.scalar('Cost', cost)
    tf.summary.scalar('Accuracy', accuracy)
    summary_op = tf.summary.merge_all()

    train_network(Xtrain, ytrain, KEEP_RATE=0.8)
    saver.restore(sess, "./checkpoints/checkpoint.ckpt")
    evaluate_test_data()
