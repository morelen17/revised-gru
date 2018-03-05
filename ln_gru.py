from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.ops import array_ops, init_ops, math_ops, nn_ops


class LayerNormBasicGRUCell(RNNCell):
    def __init__(self, num_units,
                 activation=math_ops.tanh,
                 layer_norm=True,
                 norm_gain=1.0,
                 norm_shift=0.0,
                 reuse=None):
        super(LayerNormBasicGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation
        self._layer_norm = layer_norm
        self._g = norm_gain
        self._b = norm_shift
        self._reuse = reuse

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def _linear(self, args, kernel, bias):
        out = math_ops.matmul(args, kernel)
        if not self._layer_norm:
            out = nn_ops.bias_add(out, bias)
        return out

    def _norm(self, inp, scope):
        return layers.layer_norm(inp, reuse=True, scope=scope)

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value

        if self._layer_norm:
            scopes = ["update_gate",
                      "reset_gate",
                      "candidate_linear_x",
                      "candidate_linear_h"]
            for scope in scopes:
                self.add_variable(scope + "/gamma",
                                  shape=[self._num_units],
                                  initializer=init_ops.constant_initializer(self._g))
                self.add_variable(scope + "/beta",
                                  shape=[self._num_units],
                                  initializer=init_ops.constant_initializer(self._b))

        self.update_gate_kernel = self.add_variable(
            "update_gate/kernel",
            shape=[input_depth + self._num_units, self._num_units])
        self.reset_gate_kernel = self.add_variable(
            "reset_gate/kernel",
            shape=[input_depth + self._num_units, self._num_units])
        self.candidate_linear_x_kernel = self.add_variable(
            "candidate_linear_x/kernel",
            shape=[input_depth, self._num_units])
        self.candidate_linear_h_kernel = self.add_variable(
            "candidate_linear_h/kernel",
            shape=[self._num_units, self._num_units])

        self.update_gate_bias = self.add_variable(
            "update_gate/bias",
            shape=[self._num_units]) if not self._layer_norm else None
        self.reset_gate_bias = self.add_variable(
            "reset_gate/bias",
            shape=[self._num_units]) if not self._layer_norm else None
        self.candidate_linear_x_bias = self.add_variable(
            "candidate_linear_x/bias",
            shape=[self._num_units]) if not self._layer_norm else None
        self.candidate_linear_h_bias = self.add_variable(
            "candidate_linear_h/bias",
            shape=[self._num_units]) if not self._layer_norm else None

        self.built = True

    def call(self, inputs, state):
        args = array_ops.concat([inputs, state], 1)

        z = self._linear(args,
                         kernel=self.update_gate_kernel,
                         bias=self.update_gate_bias)
        r = self._linear(args,
                         kernel=self.reset_gate_kernel,
                         bias=self.reset_gate_bias)

        if self._layer_norm:
            z = self._norm(z, "update_gate")
            r = self._norm(r, "reset_gate")

        z = math_ops.sigmoid(z)
        r = math_ops.sigmoid(r)

        _x = self._linear(inputs,
                          kernel=self.candidate_linear_x_kernel,
                          bias=self.candidate_linear_x_bias)
        _h = self._linear(state,
                          kernel=self.candidate_linear_h_kernel,
                          bias=self.candidate_linear_h_bias)

        if self._layer_norm:
            _x = self._norm(_x, scope="candidate_linear_x")
            _h = self._norm(_h, scope="candidate_linear_h")

        candidate = self._activation(_x + r * _h)

        new_h = (1 - z) * state + z * candidate

        return new_h, new_h
