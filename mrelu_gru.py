from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops import array_ops, init_ops, math_ops, nn_ops

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


class RevisedGRUCell(GRUCell):
    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value
        self._gate_kernel = self.add_variable(
            "gates/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + self._num_units, 2 * self._num_units],
            initializer=self._kernel_initializer)
        self._gate_bias = self.add_variable(
            "gates/%s" % _BIAS_VARIABLE_NAME,
            shape=[2 * self._num_units],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)))

        self.built = True

    def call(self, inputs, state):
        gate_inputs = math_ops.matmul(array_ops.concat([inputs, state], 1), self._gate_kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

        z, c = array_ops.split(value=gate_inputs, num_or_size_splits=2, axis=1)
        z = math_ops.sigmoid(z)
        c = nn_ops.relu(c)

        new_h = z * state + (1 - z) * c
        return new_h, new_h
