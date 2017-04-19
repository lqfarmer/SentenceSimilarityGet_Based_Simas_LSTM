import tensorflow as tf
import numpy as np

class SiameseLSTM(object):
    """
    A LSTM based deep Siamese network for text similarity.
    Uses an character embedding layer, followed by a biLSTM and Energy Loss layer.
    """
    
    def BiRNN(self, x, dropout, scope, embedding_size, sequence_length):
        n_input=embedding_size
        n_steps=sequence_length
        n_hidden=200
        n_layers=2
        # Prepare data shape to match `bidirectional_rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input) (?, seq_len, embedding_size)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
        # Permuting batch_size and n_steps
        x = tf.transpose(x, [1, 0, 2])
        # Reshape to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, n_input])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(axis=0, num_or_size_splits=n_steps, value=x)
        print(x)
        # Define lstm cells with tensorflow
        # Forward direction cell
        with tf.name_scope("fw"+scope),tf.variable_scope("fw"+scope):
            print(tf.get_variable_scope().name)
#             fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
            fw_cell = tf.contrib.rnn.GRUCell(n_hidden,activation=tf.tanh)
            lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell,output_keep_prob=dropout)
            lstm_fw_cell_m=tf.contrib.rnn.MultiRNNCell([lstm_fw_cell]*n_layers, state_is_tuple=True)
        # Backward direction cell
        with tf.name_scope("bw"+scope),tf.variable_scope("bw"+scope):
            print(tf.get_variable_scope().name)
#             bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
            bw_cell = tf.contrib.rnn.GRUCell(n_hidden, activation=tf.tanh)
            lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell,output_keep_prob=dropout)
            lstm_bw_cell_m = tf.contrib.rnn.MultiRNNCell([lstm_bw_cell]*n_layers, state_is_tuple=True)
        # Get lstm cell output
        #try:
        with tf.name_scope("bw"+scope),tf.variable_scope("bw"+scope):
            outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x, dtype=tf.float32)
            #         except Exception: # Old TensorFlow version only returns outputs not states
            #             outputs = tf.nn.bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x,
            #                                             dtype=tf.float32)
                # create a dense layer and get final output
        output_temp = tf.layers.batch_normalization(outputs[-1])
        with tf.name_scope("dw"+scope),tf.variable_scope("denselayer_w"+scope),tf.variable_scope("denselayer_b"+scope),tf.variable_scope("denselayer_output"+scope):
            output = tf.nn.xw_plus_b(output_temp,
                             tf.get_variable("denselayer_w", [2 * n_hidden, n_hidden]),
                             tf.get_variable("denselayer_b", [n_hidden]))
            output = tf.nn.relu(output, "densepayer_output")
        with tf.name_scope("dw2"+scope),tf.variable_scope("denselayer_w2"+scope),tf.variable_scope("denselayer_b2"+scope),tf.variable_scope("denselayer_output2"+scope):
            output2 = tf.nn.xw_plus_b(output,
                             tf.get_variable("denselayer_w2", [n_hidden, 60]),
                             tf.get_variable("denselayer_b2", [60]))
            output2 = tf.nn.relu(output2, "densepayer_output2")
        output2 = tf.layers.batch_normalization(output2)
        return output2
    
    def contrastive_loss(self, y,d,batch_size):
        tmp= y *tf.square(d)
        #tmp= tf.mul(y,tf.square(d))
        tmp2 = (1-y) *tf.square(tf.maximum((1 - d),0))
        return tf.reduce_sum(tmp +tmp2)/batch_size/2
    
    def __init__(
      self, sequence_length, vocab_size, embedding_size, hidden_units, l2_reg_lambda, batch_size):

      # Placeholders for input, output and dropout
      self.input_x1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x1")
      self.input_x2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x2")
      self.input_y = tf.placeholder(tf.float32, [None], name="input_y")
      self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
      num_class = 1

      # Keeping track of l2 regularization loss (optional)
      l2_loss = tf.constant(0.0, name="l2_loss")
          
      # Embedding layer
      with tf.name_scope("embedding"):
          self.W = tf.Variable(
              tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
              trainable=True,name="W")
          self.embedded_chars1 = tf.nn.embedding_lookup(self.W, self.input_x1)
          #self.embedded_chars_expanded1 = tf.expand_dims(self.embedded_chars1, -1)
          self.embedded_chars2 = tf.nn.embedding_lookup(self.W, self.input_x2)
          #self.embedded_chars_expanded2 = tf.expand_dims(self.embedded_chars2, -1)

      # Create a convolution + maxpool layer for each filter size
      with tf.name_scope("output"):
        self.out1=self.BiRNN(self.embedded_chars1, self.dropout_keep_prob, "side1", embedding_size, sequence_length)
        self.out2=self.BiRNN(self.embedded_chars2, self.dropout_keep_prob, "side2", embedding_size, sequence_length)
        self.out = tf.concat([self.out1, self.out2], 1, "finalOutput")
        logits = tf.nn.xw_plus_b(self.out,
                             tf.get_variable("softmax_w", [120, num_class]),
                             tf.get_variable("softmax_b", [num_class]))
        self.results = tf.argmax(logits, 1)
        
      with tf.name_scope("loss"):
          #calculate contrastive_loss, and the formula mentioned in paper: Learning Text Similarity with Siamese Recurrent Networks,but a similar difference: 1 / 4 to all
#         self.loss = self.contrastive_loss(self.input_y,self.results, batch_size) 
        batch_size = tf.size(self.input_y)
        labels = tf.expand_dims(self.input_y, 1)
        indices = tf.expand_dims(tf.range(0, batch_size), 1)
        concated = tf.concat(1, [indices, labels])
        onehot_labels = tf.sparse_to_dense(concated, tf.pack([batch_size, num_class]), 1.0, 0.0)
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits,onehot_labels,name='xentropy')
        
      with tf.name_scope("accuracy"):
        correct_predictions = tf.equal(self.results, self.input_y)
        self.accuracy=tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")