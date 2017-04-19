import tensorflow as tf
import numpy as np

class SiameseLSTM(object):
    """
    A LSTM based deep Siamese network for text similarity.
    Uses an character embedding layer, followed by a biLSTM and Energy Loss layer.
    """
    
    def BiRNN_FeatureExtract(self, x, dropout, scope, embedding_size, sequence_length):
        n_input=embedding_size
        n_steps=sequence_length
        n_hidden=240
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
        with tf.name_scope("fwF"+scope),tf.variable_scope("fwF"+scope):
            print(tf.get_variable_scope().name)
            fw_cell = tf.contrib.rnn.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
#            fw_cell = tf.contrib.rnn.GRUCell(n_hidden,activation=tf.tanh)
            lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell,output_keep_prob=dropout)
            lstm_fw_cell_m=tf.contrib.rnn.MultiRNNCell([lstm_fw_cell]*n_layers, state_is_tuple=True)
        # Backward direction cell
        with tf.name_scope("bwF"+scope),tf.variable_scope("bwF"+scope):
            print(tf.get_variable_scope().name)
            bw_cell = tf.contrib.rnn.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
#            bw_cell = tf.contrib.rnn.GRUCell(n_hidden, activation=tf.tanh)
            lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell,output_keep_prob=dropout)
            lstm_bw_cell_m = tf.contrib.rnn.MultiRNNCell([lstm_bw_cell]*n_layers, state_is_tuple=True)
        # Get lstm cell output
        with tf.name_scope("bwF"+scope),tf.variable_scope("bwF"+scope):
            outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x, dtype=tf.float32)

        # create a dense layer and get final output
        output_temp = tf.layers.batch_normalization(tf.reduce_sum(outputs,0))
        outputL2=tf.nn.l2_normalize(output_temp,1)
        with tf.name_scope("dwF"+scope),tf.variable_scope("denselayer_wF"+scope),tf.variable_scope("denselayer_bF"+scope),tf.variable_scope("denselayer_outputF"+scope):
            output = tf.nn.xw_plus_b(output_temp,
                             tf.get_variable("denselayer_wF", [2 * n_hidden, n_hidden]),
                             tf.get_variable("denselayer_bF", [n_hidden]))
            output = tf.nn.elu(output, "densepayer_outputF")
        output = tf.layers.batch_normalization(output)
        with tf.name_scope("dw2F"+scope),tf.variable_scope("denselayer_w2F"+scope),tf.variable_scope("denselayer_b2F"+scope),tf.variable_scope("denselayer_output2F"+scope):
            output2 = tf.nn.xw_plus_b(output,
                             tf.get_variable("denselayer_w2F", [n_hidden, sequence_length]),
                             tf.get_variable("denselayer_b2F", [sequence_length]))
            output2 = tf.nn.elu(output2, "densepayer_output2F")
        output2 = tf.layers.batch_normalization(output2)
        return output2
    
    def BiRNN_AttentionMechanism(self, x, dropout, scope, embedding_size, sequence_length):
        n_input=embedding_size
        n_steps=sequence_length
        n_hidden=100
        n_layers=1
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
        with tf.name_scope("fwA"+scope),tf.variable_scope("fwA"+scope):
            print(tf.get_variable_scope().name)
            fw_cell = tf.contrib.rnn.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
#            fw_cell = tf.contrib.rnn.GRUCell(n_hidden,activation=tf.tanh)
            lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell,output_keep_prob=dropout)
            lstm_fw_cell_m=tf.contrib.rnn.MultiRNNCell([lstm_fw_cell]*n_layers, state_is_tuple=True)
        # Backward direction cell
        with tf.name_scope("bwA"+scope),tf.variable_scope("bwA"+scope):
            print(tf.get_variable_scope().name)
            bw_cell = tf.contrib.rnn.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
#            bw_cell = tf.contrib.rnn.GRUCell(n_hidden, activation=tf.tanh)
            lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell,output_keep_prob=dropout)
            lstm_bw_cell_m = tf.contrib.rnn.MultiRNNCell([lstm_bw_cell]*n_layers, state_is_tuple=True)
        # Get lstm cell output
        #try:
        with tf.name_scope("bwA"+scope),tf.variable_scope("bwA"+scope):
            outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x, dtype=tf.float32)
            #         except Exception: # Old TensorFlow version only returns outputs not states
            #             outputs = tf.nn.bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x,
            #                                             dtype=tf.float32)
        # create a dense layer and get final output,noml and regulize to range [-1,1]
        output_temp = tf.layers.batch_normalization(tf.reduce_sum(outputs,0))
#         outputL2=tf.nn.l2_normalize(output_temp,1)
        with tf.name_scope("dwA"+scope),tf.variable_scope("denselayer_wA"+scope),tf.variable_scope("denselayer_bA"+scope),tf.variable_scope("denselayer_outputA"+scope):
            output = tf.nn.xw_plus_b(output_temp,
                             tf.get_variable("denselayer_wA", [2 * n_hidden, sequence_length]),
                             tf.get_variable("denselayer_bA", [sequence_length]))
            output = tf.nn.elu(output, "densepayer_outputA")
        output = tf.layers.batch_normalization(output)
#         with tf.name_scope("dw2"+scope),tf.variable_scope("denselayer_w2"+scope),tf.variable_scope("denselayer_b2"+scope),tf.variable_scope("denselayer_output2"+scope):
#             output2 = tf.nn.xw_plus_b(output,
#                              tf.get_variable("denselayer_w2", [n_hidden, 50]),
#                              tf.get_variable("denselayer_b2", [50]))
#             output2 = tf.nn.elu(output2, "densepayer_output2")
#         output2 = tf.layers.batch_normalization(output2)
        return output
    
    def contrastive_loss(self, y,d,batch_size):
        tmp= y *tf.square(d)
        #tmp= tf.mul(y,tf.square(d))
        tmp2 = (1-y) *tf.square(tf.maximum((1 - d),0))
        return tf.reduce_sum(tmp +tmp2)/batch_size/2
    
    def cosine_dis(self,out1,out2):
        distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(out1,out2)),1,keep_dims=True))
        distance = tf.div(distance, tf.add(tf.sqrt(tf.reduce_sum(tf.square(out1),1,keep_dims=True)),tf.sqrt(tf.reduce_sum(tf.square(out2),1,keep_dims=True))))
#         distance = tf.reshape(distance, [-1], name="distance")
        return distance
    
    def __init__(
      self, sequence_length, embedding_size, hidden_units, l2_reg_lambda, batch_size, word_embeddings = None):

      # Placeholders for input, output and dropout
      self.input_x1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x1")
      self.input_x2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x2")
      self.input_y = tf.placeholder(tf.int32, [None], name="input_y")
      self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
      num_class = 2

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
        self.out1_F=self.BiRNN_FeatureExtract(self.embedded_chars1, self.dropout_keep_prob, "side1", embedding_size, sequence_length)
        self.out1_A=self.BiRNN_AttentionMechanism(self.embedded_chars1, self.dropout_keep_prob, "side1", embedding_size, sequence_length)
        self.out2_F=self.BiRNN_FeatureExtract(self.embedded_chars2, self.dropout_keep_prob, "side2", embedding_size, sequence_length)
        self.out2_A=self.BiRNN_AttentionMechanism(self.embedded_chars2, self.dropout_keep_prob, "side2", embedding_size, sequence_length)
        self.cos_F = self.cosine_dis(self.out1_F,self.out2_F)
        self.cos_A = self.cosine_dis(self.out1_A,self.out2_A)
        
        self.out = tf.concat(axis=1, values=[self.out1_F,self.out1_A,self.out2_F, self.out2_A,self.cos_F,self.cos_A], name="finalOutput")
        logit = tf.nn.xw_plus_b(self.out,
                             tf.get_variable("dense_w1", [sequence_length * 4 + 2, sequence_length]),
                             tf.get_variable("dense_b1", [sequence_length]))
        logit = tf.nn.elu(logit, "dense_elu")
        logit = tf.layers.batch_normalization(logit)
        logits = tf.nn.xw_plus_b(logit,
                             tf.get_variable("dense_w2", [sequence_length, num_class]),
                             tf.get_variable("dense_b2", [num_class]))
        self.results = tf.argmax(logits, 1)
        
      with tf.name_scope("loss"):
          #calculate contrastive_loss, and the formula mentioned in paper: Learning Text Similarity with Siamese Recurrent Networks,but a similar difference: 1 / 4 to all
#         self.loss = self.contrastive_loss(self.input_y,self.results, batch_size) 
        labels = tf.expand_dims(self.input_y, 1)
        indices = tf.expand_dims(tf.range(0, batch_size), 1)
        concated = tf.concat(axis=1,values=[indices, labels],name="one_hot_lable")
        onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, num_class]), 1.0, 0.0)
        loss_t = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=onehot_labels,name='xentropy')
        self.loss = loss = tf.reduce_mean(loss_t, name='xentropy_mean')
        
#         self._lr = tf.Variable(0.0, trainable=False)
#         tvars = tf.trainable_variables()
#         grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
#                                       config.max_grad_norm)
#         optimizer = tf.train.GradientDescentOptimizer(self.lr)
#         self._train_op = optimizer.apply_gradients(zip(grads, tvars))
        
      with tf.name_scope("accuracy"):
        correct_predictions = tf.equal(tf.cast(self.results, tf.int32), self.input_y)
        self.accuracy=tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")