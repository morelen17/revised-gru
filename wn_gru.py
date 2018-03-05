from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops, math_ops, nn_impl, nn_ops
from tensorflow.python.ops import variable_scope as vs


class WeightNormGRUCell(GRUCell):
    @staticmethod
    def _normalize(weight, name):
        output_size = weight.get_shape().as_list()[1]
        g = vs.get_variable(name, [output_size], dtype=weight.dtype)
        return nn_impl.l2_normalize(weight, dim=0) * g

    def call(self, inputs, state):
        with ops.control_dependencies(None):
            self._gate_kernel = self._normalize(self._gate_kernel, name='weight_norm_gate')
            self._candidate_kernel = self._normalize(self._candidate_kernel, name='weight_norm_candidate')

        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state], 1), self._gate_kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

        value = math_ops.sigmoid(gate_inputs)
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state

        candidate = math_ops.matmul(
            array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        candidate = nn_ops.bias_add(candidate, self._candidate_bias)

        c = self._activation(candidate)
        new_h = u * state + (1 - u) * c
        return new_h, new_h
