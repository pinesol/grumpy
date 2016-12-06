
# Note: Requires Tensorflow 0.12

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

  # TODO make the type changeable instead of defaulting to float32?
  
  # All RNNs return (output, new_state). For this type of LSTM, its 'output' is still its h vector,
  # and it's cell state is a c,h,z tuple.
  def __call__(self, inputs, state, scope=None):
    # In the last layer L, the top-down connection is ignored
    # In the first layer, we use h^0_t = x^t.    
    # vars from different layers.
    # At the lowest level, h_bottom is an input vector.
    h_bottom, z_bottom, h_top_prev = inputs
    # vars from the previous time step on the same lyayer
    c_prev, h_prev, z_prev = state

    gate_slice_sizes = [self._num_units, self._num_units, self._num_units, self._num_units, 1]
    num_rows = sum(gate_slice_sizes)

    # scope: optional name for the variable scope, defaults to "HmLstmCell"
    with vs.variable_scope(scope or type(self).__name__):  # "HmLstmCell"
      # Matrix U_l^l
      U_curr = vs.get_variable("U_curr", [h_prev.get_shape()[1], num_rows], dtype=tf.float32)
      # Matrix U_{l+1}^l
      # TODO This imples that the U matrix there has the same dimensionality as the
      # one used in equation 5. but that would only be true if you forced the h vectors
      # on the above layer to be equal in size to the ones below them. Is that a real restriction?
      # Or am I misunderstanding?
      U_top = vs.get_variable("U_top", [h_bottom.get_shape()[1], num_rows], dtype=tf.float32)
      # Matrix W_{l-1}^l
      W_bottom = vs.get_variable("W_bottom", [h_bottom.get_shape()[1], num_rows],
                                 dtype=tf.float32)
      # b_l
      bias = vs.get_variable("bias", [num_rows], dtype=tf.float32)

      s_curr = tf.matmul(h_prev, U_curr)
      s_top = z_prev * tf.matmul(h_top_prev, U_top)
#      print('h_bottom dimensions: {}'.format(h_bottom.get_shape())) # TODO      
#      print('W_bottom dimensions: {}'.format(W_bottom.get_shape())) # TODO
#      print('z_bottom dimensions: {}'.format(z_bottom.get_shape())) # TODO
# TODO this is printing out as having ?, ? shape....
#      print('z_prev dimensions: {}'.format(z_prev.get_shape())) # TODO
      s_bottom = z_bottom * tf.matmul(h_bottom, W_bottom)
      gate_logits = s_curr + s_top + s_bottom + bias

#TODO      print('gate_slice_sizes: {}'.format(gate_slice_sizes)) # TODO
      f_logits, i_logits, o_logits, g_logits, z_t_logit = tf.split_v(
        gate_logits, gate_slice_sizes, split_dim=1)      
     
      # TODO check that z_t_logit is a scalar, or squeeze it so it becomes one
#      assert z_t_logit.get_shape() == [], 'z_t_logit should be scalar: {}'.format(z_t_logit.get_shape())
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

    state_new = HmLstmStateTuple(c_new, h_new, z_new)
    return h_new, state_new



# The output for this is a list of h_vectors, one for each cell.
class MultiHmRNNCell(tf.nn.rnn_cell.RNNCell):
  def __init__(self, cells, output_embedding_size):
    """Create a RNN cell composed sequentially of a number of HmRNNCells.

    Args:
      cells: list of HmRNNCells that will be composed in this order.
    """
    if not cells:
      raise ValueError("Must specify at least one cell for MultiHmRNNCell.")
    self._cells = cells
    self._output_embedding_size = output_embedding_size

  @property
  def state_size(self):
    return tuple(cell.state_size for cell in self._cells)

  @property
  def output_size(self):
    return self._output_embedding_size

  # inputs should be a batch of word vectors
  # state should be a list of HM cell state tuples of the same length as self._cells
  def __call__(self, inputs, state, scope=None):
    """Run this multi-layer cell on inputs, starting from state."""
    assert len(state) == len(self._cells)
    with vs.variable_scope(scope or type(self).__name__):  # "MultiHmRNNCell"
      if len(self._cells) > 1:
        h_prev_top = state[1].h
      else:
        h_prev_top = np.zeros(state[0].h.get_shape())
      # h_bottom, z_bottom, h_prev_top
      current_input = inputs, tf.ones(inputs.get_shape()[0]), h_prev_top
      new_h_list = []
      new_states = []
      # Go through each cell in the different layers, going bottom to top
      for i, cell in enumerate(self._cells):
        with vs.variable_scope("Cell%d" % i):
          new_h, new_state = cell(current_input, state[i]) # state[i] = c_prev, h_prev, z_prev
          assert new_h == new_state.h # TODO remove after you're done coding
          # if this is the last element, the h_prev_top vector should be zeros
          if i < len(self._cells) - 2:
            h_prev_top = state[i+2].h
          else:
            h_prev_top = np.zeros(state[i+1].h.get_shape())
          current_input = new_state.h, new_state.z, h_prev_top  # h_bottom, z_bottom, h_prev_top
          new_h_list.append(new_h)
          new_states.append(new_state)
      # Output layer
      concat_new_h = tf.concat(0, new_h_list)
      output_logits = []
      for i in range(new_h_list):
        # w^l
        gating_unit_weight = vs.get_variable("w{}".format(i), concat_new_h.get_shape(), dtype=tf.float32)
        # g_t^l
        gating_unit = tf.sigmoid(tf.reduce_sum(gating_unit_weight * concat_new_h))
        # W_l^e
        output_embedding_matrix = vs.get_variable("W{}".format(i),
                                                  [self._output_embedding_size, new_h_list[i]], dtype=tf.float32)
        output_logits = gating_unit * tf.matmul(output_embedding_matrix, new_h_list[i]) # TODO new_h_list[i] has rank 1, must have rank2?
      output_h = tf.relu(tf.sum_n(output_logits))
      
    return output_h, tuple(new_states)




