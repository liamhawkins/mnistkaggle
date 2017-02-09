# Imports and setting up some variables
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

train_path = './data/train.csv'
test_path = './data/test.csv'

logs_path = './tensorboard_files/1'

num_classes = 10

image_width = 28
image_height = 28
image_pixels = image_width * image_height

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

batch_size = 100
hm_epoch = 3
LEARNING_RATE = 0.001

x = tf.placeholder('float', [None, image_pixels], name='Feature_Input')
y = tf.placeholder('float', [None, num_classes], name='Label_Input')


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
    hidden_layer_1 = {
        'weights':
        tf.Variable(tf.random_normal([784, n_nodes_hl1]), name='HL1_Weights'),
        'biases':
        tf.Variable(tf.random_normal([n_nodes_hl1]), name='HL1_Biases')
    }

    hidden_layer_2 = {
        'weights': tf.Variable(
            tf.random_normal([n_nodes_hl1, n_nodes_hl2]), name='HL2_Weights'),
        'biases':
        tf.Variable(tf.random_normal([n_nodes_hl2]), name='HL2_Biases')
    }

    hidden_layer_3 = {
        'weights': tf.Variable(
            tf.random_normal([n_nodes_hl2, n_nodes_hl3]), name='HL3_Weights'),
        'biases':
        tf.Variable(tf.random_normal([n_nodes_hl3]), name='HL3_Biases')
    }

    output_layer = {
        'weights': tf.Variable(
            tf.random_normal([n_nodes_hl3, num_classes]), name='OL_Weights'),
        'biases':
        tf.Variable(tf.random_normal([num_classes]), name='OL_Biases')
    }

    with tf.name_scope('Hidden_Layer_1'):
        l1 = tf.add(
            tf.matmul(data, hidden_layer_1['weights']),
            hidden_layer_1['biases'])
        l1 = tf.nn.relu(l1)

    with tf.name_scope('Hidden_Layer_2'):
        l2 = tf.add(
            tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
        l2 = tf.nn.relu(l2)

    with tf.name_scope('Hidden_Layer_3'):
        l3 = tf.add(
            tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['biases'])
        l3 = tf.nn.relu(l3)

    with tf.name_scope('Output_Layer'):
        output = tf.add(
            tf.matmul(l3, output_layer['weights']), output_layer['biases'])

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
                feed_dict={x: Xtrain[start:end, :],
                           y: ytrain[start:end, :]})
            writer.add_summary(summary,
                               epoch * (Xtrain.shape[0] / batch_size) + i)
            epoch_loss += c
        print('Epoch',
              int(epoch + 1), 'completed out of', hm_epoch, 'Loss:',
              epoch_loss)

    print('Accuracy:', accuracy.eval({x: Xvalid, y: yvalid}))

    # Create submission predictions and write to .csv
    final_output_predictions = tf.argmax(prediction, 1)
    final_labels = final_output_predictions.eval({x: Xtest})
    indexes = np.array(range(1, 28001))
    submission_matrix = np.column_stack((indexes, final_labels))
    np.savetxt('submission.csv', submission_matrix, delimiter=',')
