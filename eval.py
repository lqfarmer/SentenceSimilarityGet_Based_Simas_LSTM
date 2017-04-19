#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.contrib import learn
# from input_helpers import InputHelper
import data_util_siamese as datatool
import codecs
# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 100000, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "/search/odin/data/liuqi/lstm/deep-siamese-text-similarity/runs/1490872607/checkpoints/", "Checkpoint directory from training run")
# tf.flags.DEFINE_string("eval_filepath", "match_valid.tsv", "Evaluate on this data (Default: None)")
# tf.flags.DEFINE_string("vocab_filepath", "runs/1479874609/checkpoints/vocab", "Load training time vocabulary (Default: None)")
tf.flags.DEFINE_string("data_path", "/search/odin/data/liuqi/lstm/deep-siamese-text-similarity/data/test_data/", "training file (default: None)")
tf.flags.DEFINE_string("model", "/search/odin/data/liuqi/lstm/deep-siamese-text-similarity/runs/1490872607/checkpoints/model-350000", "Load trained model checkpoint (Default: None)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

if FLAGS.model==None :#or FLAGS.vocab_filepath==None FLAGS.eval_filepath==None or 
    print("Model filepaths are empty.")
    exit()

# load data and map id-transform based on training time vocabulary
# inpH = datatool()
# x1_test,x2_test,y_test = inpH.getTestDataSet(FLAGS.eval_filepath, FLAGS.vocab_filepath, 30)
# _,_,y = datatool._read_words(FLAGS.data_path+"valid.txt")
word2id, word_embeddings = datatool.load_word2vec("/search/odin/doc2vec_jar/lstm/vector.skip.win2.100.float.for_python")

(train_x1_idsList, train_x2_idsList,train_y),(valid_x1_idsList, valid_x2_lList,valid_y) = datatool.get_data_for_siamese(word2id, FLAGS.data_path)

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = FLAGS.model
print checkpoint_file
graph = tf.Graph()
start = time.clock()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
with graph.as_default():
    with tf.device('/gpu:5'):
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement,
          gpu_options=gpu_options)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, checkpoint_file)
    
            # Get the placeholders from the graph by name
            input_x1 = graph.get_operation_by_name("input_x1").outputs[0]
            input_x2 = graph.get_operation_by_name("input_x2").outputs[0]
            input_y = graph.get_operation_by_name("input_y").outputs[0]
    
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/distance").outputs[0]
    
            accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]
            #emb = graph.get_operation_by_name("embedding/W").outputs[0]
            #embedded_chars = tf.nn.embedding_lookup(emb,input_x)
            # Generate batches for one epoch
    #         batches = inpH.batch_iter(list(zip(x1_test,x2_test,y_test)), 2*FLAGS.batch_size, 1, shuffle=False)
            # Collect the predictions here
            all_predictions = []
            all_d=[]
            total_accu = 0.0
            max_document_length = 30
            fout = codecs.open("//search//odin//data//liuqi//lstm//deep-siamese-text-similarity//data//test_data//result","w","utf-8")
            for step,(x1_batch,x2_batch,y_batch) in enumerate(datatool.data_iterator(valid_x1_idsList, valid_x2_lList, valid_y,FLAGS.batch_size,max_document_length)):
                batch_predictions, batch_acc = sess.run([predictions,accuracy], {input_x1: x1_batch, input_x2: x2_batch, input_y:y_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])
    #             print(batch_predictions)
                d = np.copy(batch_predictions)
                d[d>=0.5]=999.0
                d[d<0.5]=1
                d[d>1.0]=0
                batch_acc = np.mean(y_batch==d)
                for i in range(len(d)):
                    fout.write(str(d[i]))
                    fout.write("\n")
                all_d = np.concatenate([all_d, d])
                total_accu = batch_acc + total_accu
#                 print("DEV acc {}".format(total_accu / step))
    #         for ex in all_predictions:
    #             print ex 
            correct_predictions = float(np.mean(all_d == valid_y))
            print("Accuracy: {:g}".format(correct_predictions))
            end = time.clock()
            print "read: %f s" % (end - start)
