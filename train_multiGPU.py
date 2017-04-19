import tensorflow as tf
import numpy as np
import re
import os
import time
import datetime
import gc
# from input_helpers import InputHelper
import data_util_siamese as datatool
from siamese_network_multigpu import SiameseLSTM
from tensorflow.contrib import learn
import gzip
from random import random
import codecs
from tensorflow.contrib.metrics.python.metrics.classification import accuracy

# Parameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_string("data_path", "/search/odin/data/liuqi/lstm/deep-siamese-text-similarity/data/", "training file (default: None)")
tf.flags.DEFINE_integer("hidden_units", 100, "Number of hidden units in softmax regression layer (default:50)")
tf.flags.DEFINE_integer("max_document_length", 40, "Number of hidden units in softmax regression layer (default:50)")
tf.flags.DEFINE_integer("num_gpus", 4, "Number of hidden units in softmax regression layer (default:50)")

tf.flags.DEFINE_integer("batch_size", 200, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_iteration", 50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 5000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 5000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

# critital class define
def average_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    grads = []
    for g, _ in grad_and_vars:
      expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads
              
def multi_gpu_model(num_gpus=4, word_embeddings = None):
    grads = []
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-3)
    with tf.variable_scope(tf.get_variable_scope()) as initScope:
        for i in range(num_gpus):
            with tf.device("/gpu:%d"%i):
              with tf.name_scope("tower_%d"%i):
                siameseModel = SiameseLSTM(
                                sequence_length=FLAGS.max_document_length,
                                embedding_size=FLAGS.embedding_dim,
                                hidden_units=FLAGS.hidden_units,
                                l2_reg_lambda=FLAGS.l2_reg_lambda,
                                batch_size=FLAGS.batch_size,
                                word_embeddings=word_embeddings)
                tf.get_variable_scope().reuse_variables()
                tf.add_to_collection("train_model", siameseModel)
                grad_and_var = optimizer.compute_gradients(siameseModel.loss)
                grads.append(grad_and_var)
                tf.add_to_collection("loss",siameseModel.loss)
                tf.add_to_collection("accuracy",siameseModel.accuracy)
                tf.add_to_collection("distance",siameseModel.distance)                  
    with tf.device("cpu:0"):
        averaged_gradients = average_gradients(grads)
        train_op = optimizer.apply_gradients(averaged_gradients, global_step=global_step) 
    return train_op,global_step
  	
def generate_feed_dic(sess, batch_generator,feed_dict,train_op):
    
    SMS = tf.get_collection("train_model")
    for siameseModel in SMS:
        x1_batch, x2_batch, y_batch = batch_generator.next()
#         print x1_batch,y_batch
        if random()>0.5:
            feed_dict[siameseModel.input_x1] = x1_batch
            feed_dict[siameseModel.input_x2] = x2_batch
            feed_dict[siameseModel.input_y] = y_batch
            feed_dict[siameseModel.dropout_keep_prob] = FLAGS.dropout_keep_prob
        else:
            feed_dict[siameseModel.input_x1] = x2_batch
            feed_dict[siameseModel.input_x2] = x1_batch
            feed_dict[siameseModel.input_y] = y_batch
            feed_dict[siameseModel.dropout_keep_prob] = FLAGS.dropout_keep_prob
    return feed_dict            
def run_epoch(sess,train_x1_idsList,train_x2_idsList,train_y,scope,global_step,train_op=None,is_training=False):
    if is_training:
        epoches = len(train_x1_idsList) // FLAGS.batch_size
        batch_generator = datatool.data_iterator(train_x1_idsList, train_x2_idsList, train_y,FLAGS.batch_size,FLAGS.max_document_length)
#         siameseModels = tf.get_collection("train_model")
        while epoches > 0:
            feed_dict = {}
            epoches -= 1
            feed_dict = generate_feed_dic(sess,batch_generator,feed_dict,train_op)
            i = FLAGS.num_iteration
            while i > 0:
                i = i - 1
                losses = tf.get_collection("loss")
                accuracy = tf.get_collection("accuracy")
                distance = tf.get_collection("distance")
                
                total_accuracy = tf.add_n(losses, name='total_accu')
                total_distance = tf.add_n(losses, name='total_distance')
                total_loss = tf.add_n(losses, name='total_loss')
                
                avg_losses = total_loss / 4
                avg_accu = total_accuracy / 4
                avg_dist = total_distance / 4
                time_str = datetime.datetime.now().isoformat()
                _,step,avg_losses,avg_accu,avg_dist = sess.run([train_op,global_step,total_loss,avg_accu,avg_dist],feed_dict)
                print("TRAIN {}: step {}, avg_loss {:g}, avg_dist {:g}, avg_acc {:g}".format(time_str, step, avg_losses, avg_dist, avg_accu))
              
        
def main(argv=None):
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")
  
    word2id, word_embeddings = datatool.load_word2vec("/search/odin/doc2vec_jar/lstm/vector.skip.win2.100.float.for_python")
    print("load train data")
    (train_x1_idsList, train_x2_idsList,train_y),(valid_x1_idsList, valid_x2_lList,valid_y) = datatool.get_data_for_siamese(word2id, FLAGS.data_path)
    
    print("starting graph def")
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    with tf.Graph().as_default():#,tf.device('/cpu:0')
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement,
            gpu_options=gpu_options)
        sess = tf.Session(config=session_conf)
        
        print("started session")
        print ("build multiple model")
        with tf.name_scope("train") as train_scope:
            print("define multiple gpu model and init the training operation")
            train_op,global_step = multi_gpu_model(FLAGS.num_gpus,word_embeddings)
            print ("init all variable")
            sess.run(tf.global_variables_initializer())
            print ("run epoche stage")
            run_epoch(sess,train_x1_idsList,train_x2_idsList,train_y,train_scope,global_step,train_op,True)
        
        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        timestamp = str(int(time.time()))
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
 
if __name__ == '__main__':
    tf.app.run()
