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
        n_hidden=400
        n_layers=3
        # Prepare data shape to match `bidirectional_rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input) (?, seq_len, embedding_size)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
        # Permuting batch_size and n_steps
        x = tf.transpose(x, [1, 0, 2])
        # Reshape to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, n_input])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        #x = tf.split(0, n_steps, x)
        x = tf.split(x, n_steps, 0)
        print(x)
        # Define lstm cells with tensorflow
        # Forward direction cell
        with tf.name_scope("fw"+scope),tf.variable_scope("fw"+scope):
            print(tf.get_variable_scope().name)
            fw_cell = tf.contrib.rnn.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
#            fw_cell = tf.contrib.rnn.GRUCell(n_hidden,activation=tf.tanh)
            lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell,output_keep_prob=dropout)
            lstm_fw_cell_m=tf.contrib.rnn.MultiRNNCell([lstm_fw_cell]*n_layers, state_is_tuple=True)
        # Backward direction cell
        with tf.name_scope("bw"+scope),tf.variable_scope("bw"+scope):
            print(tf.get_variable_scope().name)
            bw_cell = tf.contrib.rnn.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
#            bw_cell = tf.contrib.rnn.GRUCell(n_hidden, activation=tf.tanh)
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
        with tf.name_scope("dw1"+scope),tf.variable_scope("denselayer_w1"+scope),tf.variable_scope("denselayer_b1"+scope),tf.variable_scope("denselayer_output1"+scope):
            output1 = tf.nn.xw_plus_b(output_temp,
                             tf.get_variable("denselayer_w1", [2 * n_hidden, n_hidden]),
                             tf.get_variable("denselayer_b1", [n_hidden]))
            output1 = tf.nn.elu(output1, "densepayer_output1")
            output1 = tf.layers.batch_normalization(output1)
        with tf.name_scope("dw2"+scope),tf.variable_scope("denselayer_w2"+scope),tf.variable_scope("denselayer_b2"+scope),tf.variable_scope("denselayer_output2"+scope):
            output2 = tf.nn.xw_plus_b(output1,
                             tf.get_variable("denselayer_w2", [n_hidden, 60]),
                             tf.get_variable("denselayer_b2", [60]))
            output2 = tf.nn.elu(output2, "densepayer_output2")
#             output2 = tf.layers.batch_normalization(output2)
#         with tf.name_scope("dw3"+scope),tf.variable_scope("denselayer_w3"+scope),tf.variable_scope("denselayer_b3"+scope),tf.variable_scope("denselayer_output3"+scope):
#             output3 = tf.nn.xw_plus_b(output2,
#                              tf.get_variable("denselayer_w3", [200, 60]),
#                              tf.get_variable("denselayer_b3", [60]))
#             output3 = tf.nn.relu(output3, "densepayer_output3")
        return output2
    
    def contrastive_loss(self, y,d,batch_size):
        tmp= y *tf.square(d)
        #tmp= tf.mul(y,tf.square(d))
        tmp2 = (1-y) *tf.square(tf.maximum((1 - d),0))
        return tf.reduce_sum(tmp +tmp2)/batch_size/2
    
    def __init__(
      self, sequence_length, embedding_size, hidden_units, l2_reg_lambda, batch_size, word_embeddings = None):
      # Placeholders for input, output and dropout
      self.input_x1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x1")
      self.input_x2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x2")
      self.input_y = tf.placeholder(tf.float32, [None], name="input_y")
      self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
#       self.input_x1 = tf.placeholder_with_default(x1_batch, [None, sequence_length], name="input_x1")
#       self.input_x2 = tf.placeholder_with_default(x2_batch, [None, sequence_length], name="input_x1")
#       self.input_y = tf.placeholder_with_default(y_batch, [None], name="input_y")
#       self.input_x1 = tf.cast(self.input_x1, tf.int32)
#       self.input_x2 = tf.cast(self.input_x2, tf.int32)
#       self.input_y = tf.cast(self.input_y, tf.float32)
#       self.dropout_keep_prob = dropout_keep_prob
#       self.dropout_keep_prob = tf.cast(self.dropout_keep_prob, tf.float32)

      # Keeping track of l2 regularization loss (optional)
      l2_loss = tf.constant(0.0, name="l2_loss")
          
      # Embedding layer
      with tf.name_scope("embedding"):
          if(word_embeddings == None):
            self.embedding = tf.Variable(tf.random_uniform([4200000, 100], -1.0, 1.0),trainable=True,name="W")
          else:
            rows = word_embeddings.shape[0]
            cols = word_embeddings.shape[1]
            self.embedding = tf.get_variable("embedding", [rows, cols], trainable=False)
          self.embedded_chars1 = tf.nn.embedding_lookup(self.embedding, self.input_x1)
          self.embedded_chars2 = tf.nn.embedding_lookup(self.embedding, self.input_x2)

      # Create a convolution + maxpool layer for each filter size
      with tf.name_scope("output"):
        # get hidden layer output:h1 and h2
        self.out1=self.BiRNN(self.embedded_chars1, self.dropout_keep_prob, "side1", embedding_size, sequence_length)
        self.out2=self.BiRNN(self.embedded_chars2, self.dropout_keep_prob, "side2", embedding_size, sequence_length)
        #caculate cosine similarity between h1 and h2
        self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.out1,self.out2)),1,keep_dims=True))
        self.distance = tf.div(self.distance, tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.out1),1,keep_dims=True)),tf.sqrt(tf.reduce_sum(tf.square(self.out2),1,keep_dims=True))))
        self.distance = tf.reshape(self.distance, [-1], name="distance")
      with tf.name_scope("loss"):
          #calculate contrastive_loss, and the formula mentioned in paper: Learning Text Similarity with Siamese Recurrent Networks,but a similar difference: 1 / 4 to all
          self.loss = self.contrastive_loss(self.input_y,self.distance, batch_size) 
      with tf.name_scope("accuracy"):
          correct_predictions = tf.equal(self.distance, self.input_y)
          self.accuracy=tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
#       with tf.name_scope("grads"):
#           self.grads = tf.gradients(self.loss, tf.trainable_variables())
