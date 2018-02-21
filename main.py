import argparse
import tensorflow as tf
from config import *
from os.path import join, exists
from os import makedirs
import pickle
import math
from sklearn.model_selection import train_test_split

FLAGS = None


def standardize(x):
    return (x - x.mean()) / x.std()


def load_data():
    """
    load data from pickle
    :return:
    """
    with open(join(FLAGS.source_data), 'rb') as f:
        data_x = pickle.load(f)
        data_y = pickle.load(f)
        return standardize(data_x), data_y


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


def main():
    data_x, data_y = load_data()
    train_x, train_y, dev_x, dev_y, test_x, test_y = get_data(data_x, data_y)
    train_steps = math.ceil(train_x.shape[0] / FLAGS.train_batch_size)
    dev_steps = math.ceil(dev_x.shape[0] / FLAGS.dev_batch_size)
    test_steps = math.ceil(test_x.shape[0] / FLAGS.test_batch_size)
    
    global_step = tf.Variable(-1, trainable=False, name='global_step')
    
    # train and dev dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(10000)
    train_dataset = train_dataset.batch(FLAGS.train_batch_size)
    
    dev_dataset = tf.data.Dataset.from_tensor_slices((dev_x, dev_y))
    dev_dataset = dev_dataset.batch(FLAGS.dev_batch_size)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    test_dataset = test_dataset.batch(FLAGS.test_batch_size)
    
    # a reinitializable iterator
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    
    train_initializer = iterator.make_initializer(train_dataset)
    dev_initializer = iterator.make_initializer(dev_dataset)
    test_initializer = iterator.make_initializer(test_dataset)
    
    # input Layer
    with tf.variable_scope('inputs'):
        # x.shape = [-1, 60, 160, 3]
        x, y_label = iterator.get_next()
    
    keep_prob = tf.placeholder(tf.float32, [])
    
    y = tf.cast(x, tf.float32)
    
    # 3 CNN layers
    for _ in range(3):
        y = tf.layers.conv2d(y, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)
        y = tf.layers.max_pooling2d(y, pool_size=2, strides=2, padding='same')
        # y = tf.layers.dropout(y, rate=keep_prob)
    
    # 2 dense layers
    y = tf.layers.flatten(y)
    y = tf.layers.dense(y, 1024, activation=tf.nn.relu)
    y = tf.layers.dropout(y, rate=keep_prob)
    y = tf.layers.dense(y, VOCAB_LENGTH)
    
    y_reshape = tf.reshape(y, [-1, VOCAB_LENGTH])
    y_label_reshape = tf.reshape(y_label, [-1, VOCAB_LENGTH])
    
    # loss
    cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y_reshape, labels=y_label_reshape))
    
    # accuracy
    max_index_predict = tf.argmax(y_reshape, axis=-1)
    max_index_label = tf.argmax(y_label_reshape, axis=-1)
    correct_predict = tf.equal(max_index_predict, max_index_label)
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
    
    # train
    train_op = tf.train.RMSPropOptimizer(FLAGS.learning_rate).minimize(cross_entropy, global_step=global_step)
    
    # saver
    saver = tf.train.Saver()
    
    # iterator
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # global step
    gstep = 0
    
    # checkpoint dir
    if not exists(FLAGS.checkpoint_dir):
        makedirs(FLAGS.checkpoint_dir)
    
    if FLAGS.train:
        for epoch in range(FLAGS.epoch_num):
            tf.train.global_step(sess, global_step_tensor=global_step)
            # train
            sess.run(train_initializer)
            for step in range(int(train_steps)):
                loss, acc, gstep, _ = sess.run([cross_entropy, accuracy, global_step, train_op],
                                               feed_dict={keep_prob: FLAGS.keep_prob})
                # print log
                if step % FLAGS.steps_per_print == 0:
                    print('Global Step', gstep, 'Step', step, 'Train Loss', loss, 'Accuracy', acc)
            
            if epoch % FLAGS.epochs_per_dev == 0:
                # dev
                sess.run(dev_initializer)
                for step in range(int(dev_steps)):
                    if step % FLAGS.steps_per_print == 0:
                        print('Dev Accuracy', sess.run(accuracy, feed_dict={keep_prob: 1}), 'Step', step)
            
            # save model
            if epoch % FLAGS.epochs_per_save == 0:
                saver.save(sess, FLAGS.checkpoint_dir, global_step=gstep)
    
    else:
        # load model
        ckpt = tf.train.get_checkpoint_state('ckpt')
        if ckpt:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Restore from', ckpt.model_checkpoint_path)
            sess.run(test_initializer)
            for step in range(int(test_steps)):
                if step % FLAGS.steps_per_print == 0:
                    print('Test Accuracy', sess.run(accuracy, feed_dict={keep_prob: 1}), 'Step', step)
        else:
            print('No Model Found')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Captcha')
    parser.add_argument('--train_batch_size', help='train batch size', default=128)
    parser.add_argument('--dev_batch_size', help='dev batch size', default=256)
    parser.add_argument('--test_batch_size', help='test batch size', default=256)
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
    parser.add_argument('--epochs_per_save', help='epochs per save', default=10, type=int)
    parser.add_argument('--steps_per_print', help='steps per print', default=2, type=int)
    parser.add_argument('--steps_per_summary', help='steps per summary', default=100, type=int)
    parser.add_argument('--keep_prob', help='train keep prob dropout', default=0.5, type=float)
    parser.add_argument('--checkpoint_dir', help='checkpoint dir', default='ckpt/model.ckpt', type=str)
    parser.add_argument('--summaries_dir', help='summaries dir', default='summaries/', type=str)
    parser.add_argument('--train', help='train', default=1, type=int)
    
    FLAGS, args = parser.parse_known_args()
    main()
