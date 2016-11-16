
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


  
# Using a tuple state for simplicity
# TODO add forget_bias?
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

  # From the paper:
  # In the last layer L, the top-down connection is ignored
  # In the first layer, we use h^0_t = x^t.
  def __call__(self, inputs, state, scope=None):
    h_prev, h_top, h_bottom, z_b, z_t = inputs
    c, h, z = state

    # each of the 4 gates is num_unites long, plus one for the z.
    num_concated_rows = (4 * self._num_units) + 1
    matrix_dims = [num_concated_rows, num_concated_rows]

    # Unlike regular LSTM, it doesn't appear to concatenate the inputs and h.
   
    # inputs: (h_prev, h_top, h_bottom, z_b, z_t)
    # state: HmLstmTuple (c, h, z)
    # scope: optional name for the variable scope, defaults to "HmLstmCell"
    with vs.variable_scope(scope or type(self).__name__):  # "HmLstmCell"
      U_curr = vs.get_variable("U_curr", matrix_dims, dtype=tf.float64)
      U_top = vs.get_variable("U_top", matrix_dims, dtype=tf.float64)
      W_bottom = vs.get_variable("W_bottom", matrix_dims, dtype=tf.float64)
      bias = vs.get_variable("bias", [num_concated_rows], dtype=tf.float64) 

      # TODO lots of these computations can be skipped dependings on z_b and z_t.
      
      s_curr = tf.matmul(U_curr, h_prev)
      s_top = tf.mul(z_t, tf.matmul(U_top, h_top)))
      s_bottom = tf.mul(z_b, tf.matmul(W_bottom, h_bottom))

      f_logits, i_logits, o_logits, g_logits, z_tilda = tf.split(1, 5, s_curr + s_top + s_bottom + bias)

      f = tf.sigmoid(f_logits)
      i = tf.sigmoid(i_logits)
      o = tf.sigmoid(o_logits)
      g = tf.tanh(g_logits)
      # TODO This is wrong. The real z is a hard sigmoid that's translated into a step function during the foward pass.
      new_z = tf.sigmoid(z_tilda)

      new_c = [] #TODO
      new_h = [] #TODO

      new_state = LSTMStateTuple(new_c, new_h, new_z)
      return new_h, new_state



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
