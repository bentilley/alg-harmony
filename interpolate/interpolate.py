
import os
import tensorflow as tf
import numpy

class Interpolate:

    def __init__(self):

        self.session = tf.Session()

        self.build()

        init = tf.variables_initializer(tf.global_variables(), name='init_all_vars_op')
        self.session.run(init)

        self.export_test()

    def build(self):
        pass

    def export(self):
        pass
