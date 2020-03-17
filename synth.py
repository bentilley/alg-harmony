

import tensorflow as tf
from model import TensorflowModel

class Synth(TensorflowModel):

    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        TensorflowModel.__init__(self, "synth")

    def build(self):

        num_samples = 1
        #num_samples = tf.Variable(512, name="num_samples", dtype=tf.int32)
        self.inputs = tf.Variable(tf.zeros(self.num_nodes), name="inputs", dtype=tf.float32)
        #.outputs = tf.Variable(tf.zeros(self.num_nodes), name="outputs", dtype=tf.float32)

        #i = tf.constant(55)
        #ijk_0 = (0, (1, 2))

        #ijk_0 = tf.Print(ijk_0, [ijk_0[0]], "Testing the printing")

        #body = lambda i, jk: (i + 1, jk)

        output_ta = tf.TensorArray(size=num_samples, dtype=tf.float32)
        input_ta = tf.TensorArray(size=num_samples, dtype=tf.float32)
        input_ta = input_ta.unstack(self.inputs)

        loop_args = (tf.Variable(0, dtype=tf.int32),
                     tf.Variable(1, dtype=tf.float32),
                     output_ta)

        def body(sample_num, state, out):
            #xt = input_ta.read(sample_num)
            #new_output, new_state = cell(xt, state)
            out = out.write(sample_num, tf.sin(tf.cast(sample_num, tf.float32) / 128))
            return (sample_num + 1, state, out)

        condition = lambda sample_num, _, __: sample_num < num_samples

        (sample_num_final, state_final, output_ta_final) \
            = tf.while_loop(condition,
                            body, loop_args)

        output_final = output_ta_final.stack()
        #lambda i: i + 1, [p])
        #p1 = tf.Print(output_final, [sample_num_final], "sample_num_final")
        #p2 = tf.Print(p1, [output_final], "sample_num_final")

        #outputs = p + 1
        #outputs = self.inputs + 1
        self.outputs = tf.identity(output_final, name='outputs')

    def export(self):
        TensorflowModel.export(self, "synth.pb")

synth = Synth(120)
