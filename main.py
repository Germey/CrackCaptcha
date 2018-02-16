import argparse
import tensorflow as tf
from config import *
from os.path import join, exists
from os import makedirs
import pickle
import math

from sklearn.model_selection import train_test_split

FLAGS = None


def load_data():
    """
    load data from pickle
    :return:
    """
    with open(join(FLAGS.source_data), 'rb') as f:
        data_x = pickle.load(f)
        data_y = pickle.load(f)
        return data_x, data_y


def get_data(data_x, data_y):
    """
    split data from loaded data
    :param data_x:
    :param data_y:
    :return: Arrays
    """
    print('Data X Length', len(data_x), 'Data Y Length', len(data_y))
    print('Data X Example', data_x[0])
    print('Data Y Example', data_y[0])
    
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.4, random_state=40)
    dev_x, test_x, dev_y, test_y, = train_test_split(test_x, test_y, test_size=0.5, random_state=40)
    
    print('Train X Shape', train_x.shape, 'Train Y Shape', train_y.shape)
    print('Dev X Shape', dev_x.shape, 'Dev Y Shape', dev_y.shape)
    print('Test Y Shape', test_x.shape, 'Test Y Shape', test_y.shape)
    return train_x, train_y, dev_x, dev_y, test_x, test_y


def weight(shape, stddev=0.1, mean=0):
    initial = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
    return tf.Variable(initial)


def bias(shape, value=0.1):
    initial = tf.constant(value=value, shape=shape)
    return tf.Variable(initial)


def conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME'):
    return tf.nn.conv2d(input, filter, strides=strides, padding=padding)


def max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'):
    return tf.nn.max_pool(input, ksize=ksize, strides=strides, padding=padding)


def main():
    data_x, data_y = load_data()
    train_x, train_y, dev_x, dev_y, test_x, test_y = get_data(data_x, data_y)
    
    train_steps = math.ceil(train_x.shape[0] / FLAGS.train_batch_size)
    dev_steps = math.ceil(dev_x.shape[0] / FLAGS.dev_batch_size)
    test_steps = math.ceil(test_x.shape[0] / FLAGS.test_batch_size)
    
    global_step = tf.Variable(-1, trainable=False, name='global_step')
    
    # train and dev dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_dataset = train_dataset.batch(FLAGS.train_batch_size)
    
    dev_dataset = tf.data.Dataset.from_tensor_slices((dev_x, dev_y))
    dev_dataset = dev_dataset.batch(FLAGS.dev_batch_size)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    test_dataset = test_dataset.batch(FLAGS.test_batch_size)
    
    print(train_dataset.output_types, test_dataset.output_shapes)
    
    # a reinitializable iterator
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    
    train_initializer = iterator.make_initializer(train_dataset)
    dev_initializer = iterator.make_initializer(dev_dataset)
    test_initializer = iterator.make_initializer(test_dataset)
    
    # input Layer
    with tf.variable_scope('inputs'):
        # x.shape = [-1, 60, 160, 3]
        x, y_label = iterator.get_next()
    print(x.shape)
    
    keep_prob = tf.placeholder(tf.float32, [])
    print('keepprob', keep_prob)
    # layer1
    w_conv1 = weight([3, 3, 3, 32])
    b_conv1 = bias([32])
    # h_conv1.shape: [-1, 60, 160, 32]
    h_conv1 = tf.nn.relu(conv2d(x, w_conv1) + b_conv1)
    print('H Conv1', h_conv1)
    # h_pool1.shape: [-1, 30, 80, 32]
    h_pool1 = max_pool(h_conv1)
    print('H Pool1', h_pool1)
    h_drop1 = tf.nn.dropout(h_pool1, keep_prob)
    print('H Drop1', h_drop1)
    
    # layer2
    w_conv2 = weight([3, 3, 32, 64])
    b_conv2 = bias([64])
    # h_conv2.shape: [-1, 30, 80, 64]
    h_conv2 = tf.nn.relu(conv2d(h_drop1, w_conv2) + b_conv2)
    # h_pool2.shape: [-1, 15, 40, 64]
    h_pool2 = max_pool(h_conv2)
    h_drop2 = tf.nn.dropout(h_pool2, keep_prob)
    print('H Drop2', h_drop2)
    # layer3
    w_conv3 = weight([3, 3, 64, 64])
    b_conv3 = bias([64])
    # h_conv3.shape: [-1, 15, 40, 64]
    h_conv3 = tf.nn.relu(conv2d(h_drop2, w_conv3) + b_conv3)
    # h_pool3.shape: [-1, 8, 20, 64]
    h_pool3 = max_pool(h_conv3)
    h_drop3 = tf.nn.dropout(h_pool3, keep_prob)
    print('H Drop3', h_drop3)
    
    h_reshape = tf.reshape(h_drop3, [-1, 8 * 20 * 64])
    # fully connected layer1
    w_f1 = weight([8 * 20 * 64, 1024])
    b_f1 = bias([1024])
    # h_f1.shape: [batch_size, 1024]
    h_f1 = tf.nn.relu(tf.matmul(h_reshape, w_f1) + b_f1)
    h_d1 = tf.nn.dropout(h_f1, keep_prob)
    
    print('H D1', h_d1)
    
    # fully connected layer2
    w_f2 = weight([1024, CAPTCHA_LENGTH * VOCAB_LENGTH])
    b_f2 = bias([CAPTCHA_LENGTH * VOCAB_LENGTH])
    # h_f2.shape: [batch_size, CAPTCHA_LENGTH * VOCAB_LENGTH]
    h_f2 = tf.nn.relu(tf.matmul(h_d1, w_f2) + b_f2)
    # h_d2 = tf.nn.dropout(h_f2, keep_prob)
    
    h_f2_reshape = tf.reshape(h_f2, [-1, VOCAB_LENGTH])
    y_label_reshape = tf.reshape(y_label, [-1, VOCAB_LENGTH])
    
    # loss
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h_f2_reshape, labels=y_label_reshape))
    
    max_index_predict = tf.argmax(h_f2_reshape, axis=-1)
    print('Max Index Predict', max_index_predict)
    max_index_label = tf.argmax(y_label_reshape, axis=-1)
    print('Max Index Label', max_index_label)
    
    correct_predict = tf.equal(max_index_predict, max_index_label)
    print('Correct predict', correct_predict)
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
    
    # Train
    train = tf.train.RMSPropOptimizer(FLAGS.learning_rate).minimize(cross_entropy, global_step=global_step)
    # Saver
    saver = tf.train.Saver()
    
    # Iterator
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # Global step
    gstep = 0
    
    if FLAGS.train:
        
        if tf.gfile.Exists(FLAGS.summaries_dir):
            tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
        
        for epoch in range(FLAGS.epoch_num):
            tf.train.global_step(sess, global_step_tensor=global_step)
            # Train
            sess.run(train_initializer)
            for step in range(int(train_steps)):
                loss, acc, gstep, _ = sess.run([cross_entropy, accuracy, global_step, train],
                                               feed_dict={keep_prob: FLAGS.keep_prob})
                # Print log
                if step % FLAGS.steps_per_print == 0:
                    print('Global Step', gstep, 'Step', step, 'Train Loss', loss, 'Accuracy', acc)
            
            if epoch % FLAGS.epochs_per_dev == 0:
                # Dev
                sess.run(dev_initializer)
                for step in range(int(dev_steps)):
                    if step % FLAGS.steps_per_print == 0:
                        print('Dev Accuracy', sess.run(accuracy, feed_dict={keep_prob: 1}), 'Step', step)
            
            # Save model
            if epoch % FLAGS.epochs_per_save == 0:
                saver.save(sess, FLAGS.checkpoint_dir, global_step=gstep)
    
    else:
        # Load model
        ckpt = tf.train.get_checkpoint_state('ckpt')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Restore from', ckpt.model_checkpoint_path)
        sess.run(test_initializer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Captcha')
    parser.add_argument('--train_batch_size', help='train batch size', default=100)
    parser.add_argument('--dev_batch_size', help='dev batch size', default=50)
    parser.add_argument('--test_batch_size', help='test batch size', default=500)
    parser.add_argument('--source_data', help='source size', default='./data/data.pkl')
    parser.add_argument('--num_layer', help='num of layer', default=2, type=int)
    parser.add_argument('--num_units', help='num of units', default=64, type=int)
    parser.add_argument('--time_step', help='time steps', default=32, type=int)
    parser.add_argument('--embedding_size', help='time steps', default=64, type=int)
    parser.add_argument('--category_num', help='category num', default=5, type=int)
    parser.add_argument('--learning_rate', help='learning rate', default=0.001, type=float)
    parser.add_argument('--epoch_num', help='num of epoch', default=10000, type=int)
    parser.add_argument('--epochs_per_test', help='epochs per test', default=100, type=int)
    parser.add_argument('--epochs_per_dev', help='epochs per dev', default=2, type=int)
    parser.add_argument('--epochs_per_save', help='epochs per save', default=2, type=int)
    parser.add_argument('--steps_per_print', help='steps per print', default=2, type=int)
    parser.add_argument('--steps_per_summary', help='steps per summary', default=100, type=int)
    parser.add_argument('--keep_prob', help='train keep prob dropout', default=0.5, type=float)
    parser.add_argument('--checkpoint_dir', help='checkpoint dir', default='ckpt/model.ckpt', type=str)
    parser.add_argument('--summaries_dir', help='summaries dir', default='summaries/', type=str)
    parser.add_argument('--train', help='train', default=True, type=bool)
    
    FLAGS, args = parser.parse_known_args()
    
    main()
