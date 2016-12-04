
import collections
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs


_HmLstmStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h", "z"))


class HmLstmStateTuple(_HmLstmStateTuple):
  """Tuple used by HmLstm Cells for `state_size`, `zero_state`, and output state.

  Stores three elements: `(c, h, z)`, in that order.
  """
  __slots__ = ()

  @property
  def dtype(self):
    (c, h, z) = self
    if not c.dtype == h.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(c.dtype), str(h.dtype)))
    return c.dtype

  
class HmLstmCell(tf.nn.rnn_cell.RNNCell):
  def __init__(self, num_units):
    # self._num_units determines the size of c and h.
    self._num_units = num_units

  @property
  def state_size(self):
    return HmLstmStateTuple(self._num_units, self._num_units, 1)

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    # In the last layer L, the top-down connection is ignored
    # In the first layer, we use h^0_t = x^t.    
    # vars from different layers.
    # At the lowest level, h_bottom is an input vector.
    h_top_prev, h_bottom, z_bottom = inputs
    # vars from the previous time step on the same lyayer
    c_prev, h_prev, z_prev = state

    gate_slice_sizes = [self._num_units, self._num_units, self._num_units, self._num_units, 1]
    num_rows = sum(gate_slice_sizes)

    # scope: optional name for the variable scope, defaults to "HmLstmCell"
    with vs.variable_scope(scope or type(self).__name__):  # "HmLstmCell"
      # Matrix U_l^l
      U_curr = vs.get_variable("U_curr", [num_rows, len(h_prev)], dtype=tf.float64)
      # Matrix U_{l+1}^l
      # TODO This imples that the U matrix there has the same dimensionality as the
      # one used in equation 5. but that would only be true if you forced the h vectors
      # on the above layer to be equal in size to the ones below them. Is that a real restriction?
      # Or am I misunderstanding?
      U_top = vs.get_variable("U_top", [num_rows, len(h_bottom)], dtype=tf.float64)
      # Matrix W_{l-1}^l
      W_bottom = vs.get_variable("W_bottom", [num_rows, len(h_bottom)], dtype=tf.float64)
      # b_l
      bias = vs.get_variable("bias", [num_rows], dtype=tf.float64) 

      s_curr = tf.matmul(U_curr, h_prev)
      s_top = z_prev * tf.matmul(U_top, h_top)
      s_bottom = z_bottom * tf.matmul(W_bottom, h_bottom)
      gate_logits = s_curr + s_top + s_bottom + bias
      
      f_logits, i_logits, o_logits, g_logits, z_t_logit = tf.split_v(0, gate_slice_sizes, gate_logits)

      # TODO check that z_t_logit is a scalar, or squeeze it so it becomes one
      assert z_t_logit.get_shape() == [], 'z_t_logit should be scalar: {}'.format(z_t_logit)
      
      f = tf.sigmoid(f_logits)
      i = tf.sigmoid(i_logits)
      o = tf.sigmoid(o_logits)
      g = tf.tanh(g_logits)

      # TODO commenting out all this until we figure out how to do the fancy gradient stuff.
      # Just doing the 'update' move all the time, which is basically normal LSTM.
      c_new = f * c_prev + i * g
      h_new = o * tf.tanh(c_new)
      z_new = tf.sigmoid(z_t_logit) # Unused at the moment. change to hard sigmoid.
      
      # slope = 1 # TODO slope annealing trick
      # z_tilda = tf.maximum(0, tf.minimum(1, (slope * z_t_logit) / 2))
      # # TODO you have to do something special with the gradient here.
      # z_new = 1 if z_tilda > 0.5 else 0
      
      # if z_prev == 0 and z_below == 1:  # UPDATE
      #   c_new = f * c_prev + i * g
      #   h_new = o * tf.tanh(c_new)
      # elif z_prev == 0 and z_below == 0:  # COPY
      #   c_new = c_prev
      #   h_new = h_prev
      # elif z_prev == 1:  # FLUSH
      #   c_new = i * g
      #   h_new = o * tf.tanh(c_new)
      # else:
      #   raise Exception('Invalid z values. z_prev: {0:.4f}, z_below: {1:.4f}'.format(
      #     z_prev, z_below))

    state_new = LSTMStateTuple(c_new, h_new, z_new)
    return h_new, state_new


# from the original rnn_cell.py in tensorflow
# TODO not sure if I can use it as effectively here...
def _linear(args, output_size, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".

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
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  with vs.variable_scope(scope or "Linear"):
    matrix = vs.get_variable(
        "Matrix", [total_arg_size, output_size], dtype=dtype)
    if len(args) == 1:
      res = math_ops.matmul(args[0], matrix)
    else:
      res = math_ops.matmul(array_ops.concat(1, args), matrix)
    if not bias:
      return res
    bias_term = vs.get_variable(
        "Bias", [output_size],
        dtype=dtype,
        initializer=init_ops.constant_initializer(
            bias_start, dtype=dtype))
  return res + bias_term
