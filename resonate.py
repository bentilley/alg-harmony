
import tensorflow as tf
from model import TensorflowModel

class Resonate(TensorflowModel):

    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        TensorflowModel.__init__(self, "resonate")

    def build(self):

        self.inputs = tf.Variable(tf.zeros(self.num_nodes), name="resonate_inputs", dtype=tf.float32)

        outputs = self.inputs + 1
        self.outputs = tf.identity(outputs, name='resonate_outputs')

    def export(self):
        TensorflowModel.export(self, "resonate.pb")

resonate = Resonate(120)
