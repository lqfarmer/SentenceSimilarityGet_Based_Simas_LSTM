#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import re
import os
import time
import datetime
import gc
# from input_helpers import InputHelper
import data_util_siamese as datatool
from siamese_network_2 import SiameseLSTM
from tensorflow.contrib import learn
import gzip
from random import random
import codecs
# Parameters
# ==================================================

tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_string("data_path", "/search/odin/data/liuqi/lstm/deep-siamese-text-similarity/data/", "training file (default: None)")
tf.flags.DEFINE_integer("hidden_units", 100, "Number of hidden units in softmax regression layer (default:50)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 500, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 5, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 5000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 5000, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# if FLAGS.training_files==None:
#     print "Input Files List is empty. use --training_files argument."
#     exit()
 
max_document_length=40
# inpH = InputHelper()
word2id, word_embeddings = datatool.load_word2vec("/search/odin/doc2vec_jar/lstm/vector.skip.win2.100.float.for_python")
(train_x1_idsList, train_x2_idsList,train_y),(valid_x1_idsList, valid_x2_lList,valid_y) = datatool.get_data_for_siamese(word2id, FLAGS.data_path)

# train_set, dev_set, vocab_processor,sum_no_of_batches = inpH.getDataSets(FLAGS.training_files,max_document_length, 10, FLAGS.batch_size)

# Training
# ==================================================

print("starting graph def")
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
with tf.Graph().as_default():
#     for d in ['/gpu:4', '/gpu:5','/gpu:6', '/gpu:7']:
        with tf.device('/gpu:6'):
            session_conf = tf.ConfigProto(
              allow_soft_placement=FLAGS.allow_soft_placement,
              log_device_placement=FLAGS.log_device_placement,
              gpu_options=gpu_options)
            sess = tf.Session(config=session_conf)
            print("started session")
            with sess.as_default():
                siameseModel = SiameseLSTM(
                    sequence_length=max_document_length,
                    embedding_size=FLAGS.embedding_dim,
                    hidden_units=FLAGS.hidden_units,
                    l2_reg_lambda=FLAGS.l2_reg_lambda,
                    batch_size=FLAGS.batch_size,
                    word_embeddings=word_embeddings)
         
                # Define Training procedure
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(1e-3)
#                 optimizer = tf.train.AdadeltaOptimizer(1e-3)
                print("initialized siameseModel object")
             
            grads_and_vars=optimizer.compute_gradients(siameseModel.loss)
            tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            print("defined training_ops")
            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)
            print("defined gradient summaries")
            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))
         
            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
         
            # Write vocabulary
        #     vocab_processor.save(os.path.join(checkpoint_dir, "vocab"))
         
            # Initialize all variables
            sess.run(tf.global_variables_initializer())
             
            print("init all variables")
            graph_def = tf.get_default_graph().as_graph_def()
            graphpb_txt = str(graph_def)
            with open(os.path.join(checkpoint_dir, "graphpb.txt"), 'w') as f:
                f.write(graphpb_txt)
         
         
            def train_step(x1_batch, x2_batch, y_batch):
                """
                A single training step
                """
                if random()>0.5:
                    feed_dict = {
                                     siameseModel.input_x1: x1_batch,
                                     siameseModel.input_x2: x2_batch,
                                     siameseModel.input_y: y_batch,
                                     siameseModel.dropout_keep_prob: FLAGS.dropout_keep_prob,
                    }
                else:
                    feed_dict = {
                                     siameseModel.input_x1: x2_batch,
                                     siameseModel.input_x2: x1_batch,
                                     siameseModel.input_y: y_batch,
                                     siameseModel.dropout_keep_prob: FLAGS.dropout_keep_prob,
                    }
                _, step, _, accuracy, results = sess.run([tr_op_set, global_step, siameseModel.loss, siameseModel.accuracy, siameseModel.results],  feed_dict)
                time_str = datetime.datetime.now().isoformat()
                
#                 d = np.copy(dist)
#                 d[d>=0.5]=999.0
#                 d[d<0.5]=1
#                 d[d>1.0]=0
#                 accuracy = np.mean(y_batch==d)
                
#                 print("TRAIN {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
#                 print y_batch, dist, d
                return step,accuracy,results
         
            def dev_step(x1_batch, x2_batch, y_batch):
                """
                A single training step
                """ 
                if random()>0.5:
                    feed_dict = {
                                     siameseModel.input_x1: x1_batch,
                                     siameseModel.input_x2: x2_batch,
                                     siameseModel.input_y: y_batch,
                                     siameseModel.dropout_keep_prob: FLAGS.dropout_keep_prob,
                    }
                else:
                    feed_dict = {
                                     siameseModel.input_x1: x2_batch,
                                     siameseModel.input_x2: x1_batch,
                                     siameseModel.input_y: y_batch,
                                     siameseModel.dropout_keep_prob: FLAGS.dropout_keep_prob,
                    }
                step, loss, accuracy, results = sess.run([global_step, siameseModel.loss, siameseModel.accuracy, siameseModel.results],  feed_dict)
                time_str = datetime.datetime.now().isoformat()
#                 d = np.copy(dist)
#                 d[d>=0.5]=999.0
#                 d[d<0.5]=1
#                 d[d>1.0]=0
#                 accuracy = np.mean(y_batch==d)
                print("DEV {}: step {}, acc {:g}".format(time_str, step, accuracy))
        #         print y_batch, dist, d
                return accuracy
         
            # Generate batches
        #     batches=inpH.batch_iter(
        #                 list(zip(train_set[0], train_set[1], train_set[2])), FLAGS.batch_size, FLAGS.num_epochs)
         
            ptr=0
            max_validation_acc=0.0
            total_accuracy = 0.0
            fout = codecs.open("//search//odin//data//liuqi//lstm//deep-siamese-text-similarity//data//eval_result","w","utf-8")
            for step,(x1_batch,x2_batch,y_batch) in enumerate(datatool.data_iterator(train_x1_idsList, train_x2_idsList, train_y,FLAGS.batch_size,max_document_length)):
                if len(y_batch) < 1:
                    break;
                i = FLAGS.num_epochs
                while i > 0:
                    i = i - 1
                    step,accuracy,_ = train_step(x1_batch, x2_batch, y_batch)
                    total_accuracy = total_accuracy + accuracy
                    print("TRAING: step {}, ave acc {:g}".format(step,total_accuracy / step))
                    current_step = tf.train.global_step(sess, global_step)
                    sum_acc=0.0
                    if current_step % FLAGS.evaluate_every == 0:
                        print("\nEvaluation:")
                        valTetFreq = 0
                        for step2,(x1__val_batch,x2_val_batch,y_val_batch) in enumerate(datatool.data_iterator(valid_x1_idsList, valid_x2_lList, valid_y,FLAGS.batch_size,max_document_length)):
                            acc = dev_step(x1__val_batch, x2_val_batch,y_val_batch)
                            sum_acc = sum_acc + acc
                            valTetFreq = valTetFreq + 1
                        print("Average valid test accuracy : {:g}".format(sum_acc / valTetFreq))
                        sum_acc = sum_acc / valTetFreq
                    if current_step % FLAGS.checkpoint_every == 0:
                        print("\nSave Model")
                        if sum_acc >= max_validation_acc:
                            max_validation_acc = sum_acc
                            fout.write(str(sum_acc))
                            fout.write("\n")
                            saver.save(sess, checkpoint_prefix, global_step=current_step)
                            tf.train.write_graph(sess.graph.as_graph_def(), checkpoint_prefix, "graph"+str(step)+".pb", as_text=False)
                            print("Saved model {} with sum_accuracy={} checkpoint to {}\n".format(step, max_validation_acc, checkpoint_prefix))
