from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops, init_ops, math_ops, nn_ops

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


class SRUCell(RNNCell):
    def __init__(self, num_units, activation=None, reuse=None, name=None):
        super(SRUCell, self).__init__(_reuse=reuse, name=name)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self.input_spec = base_layer.InputSpec(ndim=2)

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError(
                "Expected inputs.shape[-1] to be known, saw shape: %s" % inputs_shape)

        input_depth = inputs_shape[1].value

        self._kernel = self.add_variable(
            _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth, 4 * self._num_units])

        self._bias = self.add_variable(
            _BIAS_VARIABLE_NAME,
            shape=[2 * self._num_units],
            initializer=init_ops.constant_initializer(0.0, dtype=self.dtype))

        self._built = True

    def call(self, inputs, state):
        u = math_ops.matmul(inputs, self._kernel)
        x_bar, f_intermediate, r_intermediate, x_tx = array_ops.split(
            value=u, num_or_size_splits=4, axis=1)

        f_r = math_ops.sigmoid(
            nn_ops.bias_add(
                array_ops.concat([f_intermediate, r_intermediate], 1), self._bias))
        f, r = array_ops.split(value=f_r, num_or_size_splits=2, axis=1)

        c = f * state + (1.0 - f) * x_bar
        h = r * self._activation(c) + (1.0 - r) * x_tx

        return h, c
