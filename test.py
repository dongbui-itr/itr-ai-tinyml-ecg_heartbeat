import tensorflow as tf
import numpy as np


# print(tf.config.list_physical_devices('GPU'))
with tf.device('/gpu:0'):
    a = tf.constant(np.arange(1, 9001), shape=[10, 900], name='a')
    b = tf.constant(np.arange(1, 9001), shape=[900, 10], name='b')
    c = tf.matmul(a, b)



