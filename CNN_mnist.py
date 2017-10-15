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
    """
    :param X: training image
    :param keep_prob: drop out probability
    :return: predicted labels
    we need to handle data as image style(matrix) in convolution layers and as concatenated vector
    in fully connected layers.
    """
    x_image = tf.reshape(X, [-1, 28, 28, 1])
    #convolution layer 1
    with tf.name_scope('conv1') as scope:
        W_conv1 = tf.Variable(tf.random_normal([5,5,1,8], mean=0.0, stddev=0.05))
        b_conv1 = tf.Variable(tf.zeros([8]))
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image,  W_conv1, strides=[1, 1, 1, 1],padding='SAME')+ b_conv1)
    tf.summary.image('conv1', tf.reshape(h_conv1, [-1, 28, 28, 1]), 8)

    # pooling layer 1
    with tf.name_scope('pool1') as scope:
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # convolution layer 2
    with tf.name_scope('conv2') as scope:
        W_conv2 = tf.Variable(tf.random_normal([5,5,8,16], mean=0.0, stddev=0.05))
        b_conv2 = tf.Variable(tf.zeros([16]))
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1,W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
    tf.summary.image('conv2', tf.reshape(h_conv2, [-1, 28, 28, 1]), 10)

    # pooling layer 2
    with tf.name_scope('pool2') as scope:
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    print h_conv1.shape
    print h_pool1.shape
    print h_conv2.shape
    print h_pool2.shape

    #fully connected layer 1
    with tf.name_scope('fc1') as scope:
        W_fc1 = tf.Variable(tf.random_normal([7*7*16, 625], mean=0.0, stddev=0.05))
        b_fc1 = tf.Variable(tf.zeros([625]))
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 16]) #concatenate the image(h_pool2)
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    tf.summary.image('fc1',tf.reshape(W_fc1, [-1, 28, 28, 1]), 10)
    #fully connected layer 2
    with tf.name_scope('fc2') as scope:
        W_fc2 = tf.Variable(tf.random_normal([625, 10], mean=0.0, stddev=0.05))
        b_fc2 = tf.Variable(tf.zeros([10]))
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    tf.summary.image('fc2', tf.reshape(W_fc2, [-1, 25, 25, 1]), 10)

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
    summary_writer = tf.summary.FileWriter("./log/", sess.graph_def)
    summary_op = tf.summary.merge_all()


    """
    training
    """

    patience = 0.0
    previous_train_accuracy = 0
    for i in range(20001):

        batch_xs,batch_ys = mnist.train.next_batch(100)
        _, summaries_str = sess.run([train_op, summary_op], feed_dict={
            x: batch_xs,
            y_: batch_ys,
            keep_prob: 0.5
        })
        summary_writer.add_summary(summaries_str, global_step=i)


        if i%2000 == 0:
            train_accuracy = sess.run(acc, feed_dict={
                x: batch_xs,
                y_: batch_ys,
                keep_prob: 1.0
            })
            print("train_accuracy: ",train_accuracy)
            if train_accuracy <= previous_train_accuracy:
                print train_accuracy
                print previous_train_accuracy
                patience += 1
                previous_train_accuracy = train_accuracy

            if patience >= 4:
                print "early stopping"
                break

    test_accuracy = sess.run(acc, feed_dict={
        x: mnist.test.images,
        y_: mnist.test.labels,
        keep_prob: 1.0
    })
    print("test_accuracy: ",test_accuracy)

