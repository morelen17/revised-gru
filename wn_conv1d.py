from tensorflow.python.framework import ops
from tensorflow.python.layers.convolutional import Conv1D
from tensorflow.python.ops import array_ops, init_ops, math_ops, nn, nn_impl
from tensorflow.python.ops import variable_scope as vs
import tensorflow as tf


class WeightNormConv1D(Conv1D):
    def __init__(self, filters, kernel_size, **kwargs):
        super().__init__(filters, kernel_size, **kwargs)
        self.wn_init = tf.Variable(True, name='is_init_wn', trainable=False)  # for data-dependent initialization

    def call(self, inputs):
        output_size = self.kernel.get_shape().as_list()[1]
        g = vs.get_variable('weight_norm',
                            [output_size],
                            initializer=init_ops.constant_initializer(1.0),
                            dtype=self.kernel.dtype)  # trainable

        self.kernel = nn_impl.l2_normalize(self.kernel, dim=0) * g

        outputs = self._convolution_op(inputs, self.kernel)

        if self.use_bias:
            if self.data_format == 'channels_first':
                bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                outputs += bias
            else:
                outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

        #  data-dependent initialization
        if self.wn_init:
            mean, variance = nn_impl.moments(outputs, axes=[0, 1, 2])
            scale_init = 1. / math_ops.sqrt(variance + 1e-10)
            with ops.control_dependencies([g.assign(g * scale_init), self.bias.assign_add(-mean * scale_init)]):
                outputs = array_ops.identity(outputs)
            tf.assign(self.wn_init, False)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs
