from tensorflow.contrib.rnn import LSTMStateTuple, RNNCell
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops, clip_ops, init_ops, math_ops, nn_impl, nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest


class WeightNormLSTMCell(RNNCell):
    def __init__(self, num_units, norm=True, use_peepholes=False,
                 cell_clip=None, initializer=None, num_proj=None,
                 proj_clip=None, forget_bias=1, activation=None,
                 reuse=None):
        super(WeightNormLSTMCell, self).__init__(_reuse=reuse)

        self._scope = 'wn_lstm_cell'
        self._num_units = num_units
        self._norm = norm
        self._initializer = initializer
        self._use_peepholes = use_peepholes
        self._cell_clip = cell_clip
        self._num_proj = num_proj
        self._proj_clip = proj_clip
        self._activation = activation or math_ops.tanh
        self._forget_bias = forget_bias

        self._weights_variable_name = "kernel"
        self._bias_variable_name = "bias"

        if num_proj:
            self._state_size = LSTMStateTuple(num_units, num_proj)
            self._output_size = num_proj
        else:
            self._state_size = LSTMStateTuple(num_units, num_units)
            self._output_size = num_units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def _normalize(self, weight, name):
        output_size = weight.get_shape().as_list()[1]
        g = vs.get_variable(name, [output_size], dtype=weight.dtype)
        return nn_impl.l2_normalize(weight, dim=0) * g

    def _linear(self, args,
                output_size,
                norm,
                bias,
                bias_initializer=None,
                kernel_initializer=None):
        """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
        Args:
          args: a 2D Tensor or a list of 2D, batch x n, Tensors.
          output_size: int, second dimension of W[i].
          bias: boolean, whether to add a bias term or not.
          bias_initializer: starting value to initialize the bias
            (default is all zeros).
          kernel_initializer: starting value to initialize the weight.
        Returns:
          A 2D Tensor with shape [batch x output_size] equal to
          sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
        Raises:
          ValueError: if some of the arguments has unspecified or wrong shape.
        """
        if args is None or (nest.is_sequence(args) and not args):
            raise ValueError("`args` must be specified")
        if not nest.is_sequence(args):
            args = [args]

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
            weights = vs.get_variable(
                self._weights_variable_name, [total_arg_size, output_size],
                dtype=dtype,
                initializer=kernel_initializer)
            if norm:
                wn = []
                st = 0
                with ops.control_dependencies(None):
                    for i in range(len(args)):
                        en = st + shapes[i][1].value
                        wn.append(self._normalize(weights[st:en, :],
                                                  name='norm_{}'.format(i)))
                        st = en

                    weights = array_ops.concat(wn, axis=0)

            if len(args) == 1:
                res = math_ops.matmul(args[0], weights)
            else:
                res = math_ops.matmul(array_ops.concat(args, 1), weights)
            if not bias:
                return res

            with vs.variable_scope(outer_scope) as inner_scope:
                inner_scope.set_partitioner(None)
                if bias_initializer is None:
                    bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)

                biases = vs.get_variable(
                    self._bias_variable_name, [output_size],
                    dtype=dtype,
                    initializer=bias_initializer)

            return nn_ops.bias_add(res, biases)

    def call(self, inputs, state):
        dtype = inputs.dtype
        num_units = self._num_units
        sigmoid = math_ops.sigmoid
        c, h = state

        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

        with vs.variable_scope(self._scope, initializer=self._initializer):

            concat = self._linear([inputs, h], 4 * num_units,
                                  norm=self._norm, bias=True)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)

            if self._use_peepholes:
                w_f_diag = vs.get_variable("w_f_diag", shape=[num_units], dtype=dtype)
                w_i_diag = vs.get_variable("w_i_diag", shape=[num_units], dtype=dtype)
                w_o_diag = vs.get_variable("w_o_diag", shape=[num_units], dtype=dtype)

                new_c = (c * sigmoid(f + self._forget_bias + w_f_diag * c)
                         + sigmoid(i + w_i_diag * c) * self._activation(j))
            else:
                new_c = (c * sigmoid(f + self._forget_bias)
                         + sigmoid(i) * self._activation(j))

            if self._cell_clip is not None:
                new_c = clip_ops.clip_by_value(new_c, -self._cell_clip, self._cell_clip)
            if self._use_peepholes:
                new_h = sigmoid(o + w_o_diag * new_c) * self._activation(new_c)
            else:
                new_h = sigmoid(o) * self._activation(new_c)

            if self._num_proj is not None:
                with vs.variable_scope("projection"):
                    new_h = self._linear(new_h,
                                         self._num_proj,
                                         norm=self._norm,
                                         bias=False)

                if self._proj_clip is not None:
                    new_h = clip_ops.clip_by_value(new_h,
                                                   -self._proj_clip,
                                                   self._proj_clip)

            new_state = LSTMStateTuple(new_c, new_h)
            return new_h, new_state
