import sys
import math

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops.init_ops import Initializer
from tensorflow.python.util.tf_export import tf_export

#For debug, print whole numpy arrays/matrices
np.set_printoptions(threshold=sys.maxsize)

class LinearGradient():

    def __init__(self,
                 dtype=dtypes.float32):

        self.dtype = dtypes.as_dtype(dtype)

    def __call__(self,
                 shape,
                 dtype=None,
                 partition_info=None):
        self.shape = shape
        if dtype is None:
            dtype = self.dtype
        self.matrix = np.zeros(shape)

        for index in range(0, shape[0]):
            self.matrix[index] = (index * (-1 / shape[0])) + 1

        return tf.convert_to_tensor(self.matrix, np.float32)


def safe_inv_square(attractor, attractee):
    diff = attractor - attractee
    return 1. / pow(diff, 2) if diff != 0 else 0

class Gravitate():

    def __init__(self,
                 dtype=dtypes.float32,
                 get_attraction=lambda attractor, attractee: safe_inv_square(attractor, attractee)):

        self.get_attraction = get_attraction
        self.dtype = dtypes.as_dtype(dtype)

    def __call__(self,
                 shape,
                 dtype=None,
                 partition_info=None):
        self.shape = shape
        if dtype is None:
            dtype = self.dtype

        self.matrix = np.zeros(shape)

        # TODO Is shape iterating over correct row/column shape
        for pitch in range(0, shape[0]):
            self.get_attractions(pitch)

        # Normalize rows
        get_normalization_coefficients = np.vectorize(lambda x: (1/x) if x != 0 else 0)
        normalization_coefficients = get_normalization_coefficients(self.matrix.sum(axis=1))
        self.matrix = self.matrix * normalization_coefficients[:, np.newaxis]

        return tf.convert_to_tensor(self.matrix, np.float32)

    def get_attractions(self, attractor):
        for attractee in range(0, self.shape[0]):
            self.matrix[attractee, attractor] = self.get_attraction(attractor, attractee)

    def get_config(self):
        pass


sign = lambda x: (1, -1)[x < 0]

class Partial():

    def __init__(self, fundamental, harmonic_number):
        self.fundamental = fundamental
        self.pitch = fundamental + (12 * sign(harmonic_number) * math.log(abs(harmonic_number)) / math.log(2))
        self.amplitude = (1 / abs(harmonic_number))# if (harmonic_number < 1) else harmonic_number

@tf_export("keras.initializers.Resonate", "initializers.resonate",
           "resonate_initializer")
class Resonate(Initializer):
    """Initializer that generates tensors initialized to 0."""

    def __init__(self, dtype=dtypes.float32):
        self.dtype = dtypes.as_dtype(dtype)

    def __call__(self, shape, dtype=None, partition_info=None):
        self.shape = shape
        if dtype is None:
            dtype = self.dtype
        self.matrix = np.zeros(shape)

        for pitch in range(0, shape[0]):
            self.apply_partials(pitch)

        return tf.convert_to_tensor(self.matrix, np.float32)

    def apply_partials(self, fundamental):

        for harmonic_number in range(2, 24):
            partial = Partial(fundamental, harmonic_number)
            if not self.apply_partial(partial):
                break

        for harmonic_number in range(2, 24):
            partial = Partial(fundamental, -1 * harmonic_number)
            if not self.apply_partial(partial):
                break

    def apply_partial(self, partial):

        quantised_pitch = round(partial.pitch)
        if quantised_pitch < 0 or quantised_pitch >= self.shape[0]:
            return False
        pitch_difference = math.fabs(quantised_pitch - partial.pitch)
        amplitude = partial.amplitude * ((0.5 - pitch_difference) / 0.5)

        self.matrix[partial.fundamental, quantised_pitch] = amplitude
        return True
