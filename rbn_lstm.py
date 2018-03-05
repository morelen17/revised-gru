from tensorflow.contrib.rnn import LSTMStateTuple, RNNCell
from tensorflow.python.layers import base as base_layer
from tensorflow.python.layers.normalization import BatchNormalization
from tensorflow.python.ops import array_ops, clip_ops, init_ops, math_ops, nn, nn_ops, partitioned_variables
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


class BNLSTMCell(RNNCell):
    def __init__(self,
                 num_units,
                 use_peepholes=False,
                 cell_clip=None,
                 initializer=None,
                 num_proj=None,
                 proj_clip=None,
                 num_unit_shards=None,
                 num_proj_shards=None,
                 forget_bias=1.0,
                 state_is_tuple=True,
                 activation=None,
                 reuse=None,
                 normalize_in_to_hidden=False,
                 normalize_in_together=True,
                 normalize_cell=False,
                 normalize_config=None,
                 name=None):
        super(BNLSTMCell, self).__init__(_reuse=reuse, name=name)
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is slower and will soon be "
                         "deprecated.  Use state_is_tuple=True.", self)
        if num_unit_shards is not None or num_proj_shards is not None:
            logging.warn(
                "%s: The num_unit_shards and proj_unit_shards parameters are "
                "deprecated and will be removed in Jan 2017.  "
                "Use a variable scope with a partitioner instead.", self)

        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units
        self._use_peepholes = use_peepholes
        self._cell_clip = cell_clip
        self._initializer = initializer
        self._num_proj = num_proj
        self._proj_clip = proj_clip
        self._num_unit_shards = num_unit_shards
        self._num_proj_shards = num_proj_shards
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or math_ops.tanh
        self._normalize_in_to_hidden = normalize_in_to_hidden
        self._normalize_in_together = normalize_in_to_hidden and normalize_in_together
        self._normalize_cell = normalize_cell
        self._normalize_config = normalize_config

        if num_proj:
            self._state_size = (
                LSTMStateTuple(num_units, num_proj)
                if state_is_tuple else num_units + num_proj)
            self._output_size = num_proj
        else:
            self._state_size = (
                LSTMStateTuple(num_units, num_units)
                if state_is_tuple else 2 * num_units)
            self._output_size = num_units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value
        h_depth = self._num_units if self._num_proj is None else self._num_proj
        maybe_partitioner = (
            partitioned_variables.fixed_size_partitioner(self._num_unit_shards)
            if self._num_unit_shards is not None
            else None)

        if self._normalize_in_to_hidden or self._normalize_cell:
            if self._normalize_config is None:
                self._normalize_config = {'center': False,
                                          'scale': True,
                                          'gamma_initializer': init_ops.constant_initializer(0.1, dtype=self.dtype)}
            else:
                self._normalize_config['center'] = False

        if not self._normalize_in_to_hidden or self._normalize_in_together:
            self._kernel = self.add_variable(
                _WEIGHTS_VARIABLE_NAME,
                shape=[input_depth + h_depth, 4 * self._num_units],
                initializer=self._initializer,
                partitioner=maybe_partitioner)
            if self._normalize_in_to_hidden:
                self._bn = BatchNormalization(**self._normalize_config)
        else:
            self._kernel_m = self.add_variable(
                "i_scope/%s" % _WEIGHTS_VARIABLE_NAME,
                shape=[input_depth, 4 * self._num_units],
                initializer=self._initializer,
                partitioner=maybe_partitioner)
            with vs.variable_scope(None, "i_scope"):
                self._bn_i = BatchNormalization(**self._normalize_config)

            self._kernel_m = self.add_variable(
                "m_scope/%s" % _WEIGHTS_VARIABLE_NAME,
                shape=[h_depth, 4 * self._num_units],
                initializer=self._initializer,
                partitioner=maybe_partitioner)
            with vs.variable_scope(None, "m_scope"):
                self._bn_m = BatchNormalization(**self._normalize_config)

        self._bias = self.add_variable(
            _BIAS_VARIABLE_NAME,
            shape=[4 * self._num_units],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))

        if self._normalize_cell:
            self._normalize_config_cell = self._normalize_config
            self._normalize_config_cell['center'] = True
            self._bn_c = BatchNormalization(**self._normalize_config_cell)

        if self._use_peepholes:
            self._w_f_diag = self.add_variable("w_f_diag", shape=[self._num_units],
                                               initializer=self._initializer)
            self._w_i_diag = self.add_variable("w_i_diag", shape=[self._num_units],
                                               initializer=self._initializer)
            self._w_o_diag = self.add_variable("w_o_diag", shape=[self._num_units],
                                               initializer=self._initializer)

        if self._num_proj is not None:
            maybe_proj_partitioner = (
                partitioned_variables.fixed_size_partitioner(self._num_proj_shards)
                if self._num_proj_shards is not None
                else None)
            self._proj_kernel = self.add_variable(
                "projection/%s" % _WEIGHTS_VARIABLE_NAME,
                shape=[self._num_units, self._num_proj],
                initializer=self._initializer,
                partitioner=maybe_proj_partitioner)

        self.built = True

    def call(self, inputs, state, training=False):
        num_proj = self._num_units if self._num_proj is None else self._num_proj
        sigmoid = math_ops.sigmoid

        if self._state_is_tuple:
            (c_prev, m_prev) = state
        else:
            c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
            m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])

        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        if not self._normalize_in_to_hidden or self._normalize_in_together:
            lstm_matrix = math_ops.matmul(
                array_ops.concat([inputs, m_prev], 1), self._kernel)
            if self._normalize_in_to_hidden:
                lstm_matrix = self._bn(lstm_matrix, training=training)
        else:
            op_i = math_ops.matmul(inputs, self._kernel_i)
            op_m = math_ops.matmul(m_prev, self._kernel_m)
            lstm_matrix = self._bn_i(op_i, training=training)
            lstm_matrix += self._bn_m(op_m, training=training)

        lstm_matrix = nn_ops.bias_add(lstm_matrix, self._bias)

        i, j, f, o = array_ops.split(
            value=lstm_matrix, num_or_size_splits=4, axis=1)
        if self._use_peepholes:
            c = (sigmoid(f + self._forget_bias + self._w_f_diag * c_prev) * c_prev +
                 sigmoid(i + self._w_i_diag * c_prev) * self._activation(j))
        else:
            c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) *
                 self._activation(j))

        if self._cell_clip is not None:
            c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)

        if not self._normalize_cell:
            c_new = c
        else:
            c_new = self._bn_c(c, training=training)

        if self._use_peepholes:
            m = sigmoid(o + self._w_o_diag * c_new) * self._activation(c_new)
        else:
            m = sigmoid(o) * self._activation(c_new)

        if self._num_proj is not None:
            m = math_ops.matmul(m, self._proj_kernel)

            if self._proj_clip is not None:
                m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)

        new_state = (LSTMStateTuple(c, m) if self._state_is_tuple else
                     array_ops.concat([c, m], 1))
        return m, new_state
