from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tensorflow.examples.tutorials.mnist import input_data
LOG_DIR = os.path.join(os.path.dirname(__file__),"/log")
if os.path.exists(LOG_DIR) is False:
    os.mkdir(LOG_DIR)

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
#Learning late for gradient descend
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')

#get mnist data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def inference(X,keep_prob):
    with tf.name_scope('fc1') as scope:
        W_fc1 = tf.Variable(tf.random_normal([784, 625], mean=0.0, stddev=0.05))
        b_fc1 = tf.Variable(tf.zeros([625]))
        h_fc1 = tf.nn.relu(tf.matmul(X, W_fc1) + b_fc1)
        # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2') as scope:
        W_fc2 = tf.Variable(tf.random_normal([625, 10], mean=0.0, stddev=0.05))
        b_fc2 = tf.Variable(tf.zeros([10]))
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
        # h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    with tf.name_scope('softmax') as scope:
        y = tf.nn.softmax(h_fc2)

    return y


def loss(label,y_inf):
    # Cost Function basic term
    with tf.name_scope('loss'):
        cross_entropy = -tf.reduce_sum(label * tf.log(y_inf))
    tf.summary.scalar("cross_entropy", cross_entropy)
    return cross_entropy


def training(loss, learning_rate):

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step


def accuracy(y_inf, labels):
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_inf, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar("accuracy", accuracy)
    return accuracy


"""Model cofiguation"""

with tf.Graph().as_default():
    # Variables
    x = tf.placeholder("float", [None, 784])
    y_ = tf.placeholder("float", [None, 10])
    keep_prob = tf.placeholder("float")

    #inf: prediction of labels
    inf = inference(x,0.5)

    #compute lossfunction.  y_:labels, inf:predictions
    loss_value = loss(y_,inf)

    train_op = training(loss_value, FLAGS.learning_rate)
    acc = accuracy(inf, y_)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    tf.summary.FileWriter(LOG_DIR, sess.graph)
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter("./log/", sess.graph_def)

    """
    training
    """

    for i in range(20001):
        batch_xs,batch_ys = mnist.train.next_batch(100)
        sess.run(train_op, feed_dict={
            x: batch_xs,
            y_: batch_ys,
            keep_prob: 0.5
        })
        if i%2000 == 0:
            train_accuracy = sess.run(acc, feed_dict={
                x: batch_xs,
                y_: batch_ys,
                keep_prob: 1.0
            })
            print("train_accuracy: ",train_accuracy)

    test_accuracy = sess.run(acc, feed_dict={
        x: mnist.test.images,
        y_: mnist.test.labels,
        keep_prob: 1.0
    })
    print("test_accuracy: ",test_accuracy)

