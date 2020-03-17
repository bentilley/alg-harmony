
import os
import sys
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat
from google.protobuf import text_format

script_dir = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.join(script_dir, 'models')
checkpoints_dir = os.path.join(script_dir, 'checkpoints')

class TensorflowModel:

    def __init__(self, scope_name):

        self.session = tf.Session()

        with tf.name_scope(scope_name):
            self.build()

            tf.variables_initializer(tf.global_variables(), name='init_all_vars_op')
            #self.session.run(init)

            self.export()

    def setup_layers(self):
        pass

    def setup_weights(self):
        pass

    def export(self, name="graph.pb", path="/opt/harmony-engine"):

        tf.train.write_graph(self.session.graph_def, path, name)

        '''
        with tf.Session() as sess:

            model_filename = os.path.join(path, name)
            with gfile.FastGFile(model_filename, 'rb') as f:
                #data = compat.as_bytes(f.read())
                #sm = saved_model_pb2.SavedModel()
                #sm.ParseFromString(data)
                # print(sm)
                #if 1 != len(sm.meta_graphs):
                #    print('More than one graph found. Not sure which to write')
                #    sys.exit(1)

                graph_def = tf.GraphDef()
                text_format.Merge(f, graph_def)
                #graph_def = tf.GraphDef()
                #graph_def.ParseFromString(sm.meta_graphs[0])
                #g_in = tf.import_graph_def(sm.meta_graphs[0].graph_def)
        '''
        LOGDIR = '/opt/harmony-engine/logs'
        train_writer = tf.summary.FileWriter(LOGDIR)
        train_writer.add_graph(self.session.graph)
        train_writer.flush()
        train_writer.close()

        '''
            model_filename = os.path.join(path, name)
            print("model_filename ", model_filename )
            with gfile.FastGFile(model_filename, 'rb') as f:
                print("f ", f )
                graph_def = tf.GraphDef()
                print("f.read() ", f.read())
                #graph_def.ParseFromString(f.read())
                #g_in = tf.import_graph_def(graph_def)

            LOGDIR = '/opt/harmony-engine/logs'
            train_writer = tf.summary.FileWriter(LOGDIR)
            train_writer.add_graph(self.session.graph)
            train_writer.flush()
            train_writer.close()
        '''

    def build(self):
        pass

    def sigmoid(self, values, slope, threshold):
        e = tf.exp(-1 * slope * (values - threshold))
        return 1 / (1 + e)

