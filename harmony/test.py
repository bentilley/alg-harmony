
import tensorflow as tf
import numpy as np

with tf.Session() as sess:
    #a = tf.Variable(5.0, name='a')
    #b = tf.Variable(6.0, name='b')
    #c = tf.maximum(a, b, name="c")

    coefficients = tf.Variable(np.matrix([[1, 0.5, 0, 0],
                                          [0.5, 1, 0.5, 0],
                                          [0, 0.5, 1, 0.5],
                                          [0, 0, 0.5, 1]]), name="coefficients", dtype=tf.float32)

    attractees = tf.Variable(np.array([0.5, 1, 1, 2]),
                             name="attractees",
                             dtype=tf.float32)

    sess.run(tf.global_variables_initializer())

    normalised = tf.norm(coefficients, axis=1)

    print('normalised ', normalised.eval())
    print('tf.exp(0) ', tf.exp(0.).eval())
