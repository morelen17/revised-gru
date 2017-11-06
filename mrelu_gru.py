from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.ops import array_ops, init_ops, math_ops, nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.nn import relu
from tensorflow.python.util import nest

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


class _Linear(object):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of weight variable.
      dtype: data type for variables.
      build_bias: boolean, whether to build a bias variable.
      bias_initializer: starting value to initialize the bias
        (default is all zeros).
      kernel_initializer: starting value to initialize the weight.

    Raises:
      ValueError: if inputs_shape is wrong.
    """

    def __init__(self,
                 args,
                 output_size,
                 build_bias,
                 bias_initializer=None,
                 kernel_initializer=None):
        self._build_bias = build_bias

        if args is None or (nest.is_sequence(args) and not args):
            raise ValueError("`args` must be specified")
        if not nest.is_sequence(args):
            args = [args]
            self._is_sequence = False
        else:
            self._is_sequence = True

        # Calculate the total size of arguments on dimension 1.
        total_arg_size = 0
        shapes = [a.get_shape() for a in args]
        for shape in shapes:
            if shape.ndims != 2:
                raise ValueError("linear is expecting 2D arguments: %s" % shapes)
            if shape[1].value is None:
                raise ValueError("linear expects shape[1] to be provided for shape %s, "
                                 "but saw %s" % (shape, shape[1]))
            else:
                total_arg_size += shape[1].value

        dtype = [a.dtype for a in args][0]

        scope = vs.get_variable_scope()
        with vs.variable_scope(scope) as outer_scope:
            self._weights = vs.get_variable(
                _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
                dtype=dtype,
                initializer=kernel_initializer)
            if build_bias:
                with vs.variable_scope(outer_scope) as inner_scope:
                    inner_scope.set_partitioner(None)
                    if bias_initializer is None:
                        bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
                    self._biases = vs.get_variable(
                        _BIAS_VARIABLE_NAME, [output_size],
                        dtype=dtype,
                        initializer=bias_initializer)

    def __call__(self, args):
        if not self._is_sequence:
            args = [args]

        if len(args) == 1:
            res = math_ops.matmul(args[0], self._weights)
        else:
            res = math_ops.matmul(array_ops.concat(args, 1), self._weights)
        if self._build_bias:
            res = nn_ops.bias_add(res, self._biases)
        return res


class MReluGRUCell(RNNCell):
    """
        Revised Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1710.00641).
    """

    def __init__(self,
                 num_units,
                 reuse=None,
                 gate_kernel_initializer=None,
                 gate_bias_initializer=None,
                 candidate_kernel_initializer=None,
                 candidate_bias_initializer=None):
        super(MReluGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._gate_kernel_initializer = gate_kernel_initializer
        self._gate_bias_initializer = gate_bias_initializer
        self._candidate_kernel_initializer = candidate_kernel_initializer or xavier_initializer
        self._candidate_bias_initializer = candidate_bias_initializer
        self._gate_linear = None
        self._candidate_linear = None

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        """Gated recurrent unit (GRU) with nunits cells."""
        if self._gate_linear is None:
            bias_ones = self._gate_bias_initializer
            if self._gate_bias_initializer is None:
                bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
            with vs.variable_scope("update_gate"):
                self._gate_linear = _Linear(
                    [inputs, state],
                    self._num_units,
                    True,
                    bias_initializer=bias_ones,
                    kernel_initializer=self._gate_kernel_initializer)

        z = math_ops.sigmoid(self._gate_linear([inputs, state]))

        if self._candidate_linear is None:
            with vs.variable_scope("candidate"):
                self._candidate_linear = _Linear(
                    [inputs, state],
                    self._num_units,
                    True,
                    bias_initializer=self._candidate_bias_initializer,
                    kernel_initializer=self._candidate_kernel_initializer)
        c = relu(self._candidate_linear([inputs, state]))
        new_h = z * state + (1 - z) * c
        return new_h, new_h
