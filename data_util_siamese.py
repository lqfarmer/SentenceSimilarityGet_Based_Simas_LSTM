from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import time
import struct

import tensorflow.python.platform
from random import random
import numpy as np
from six.moves import xrange
import tensorflow as tf

from tensorflow.python.platform import gfile

TRAIN_FILE='train.txt'
VALID_FILE='valid.txt'

def _read_words(filename):
    with gfile.GFile(filename, "r") as f:
        lines = f.read().split('\n')
        x1 = []
        x2 = []
        y = []
        for line in lines:
            if len(line) == 0:
                break
            l = line.strip().split("\t")
            if len(l) < 3:
                continue
            if random() > 0.5: #random shuffle location of x1 and x2
                x1.append(l[0])
                x2.append(l[1])
            else:
                x1.append(l[1])
                x2.append(l[0])
            if l[2]=='1':
                y.append(1)
            else:
                y.append(0)
#             y.append(l[2])
        x1_S_list = []
        x2_S_list = []
        for s in x1:
            words = s.split(' ')
            x1_S_list.append(words)
        for s in x2:
            words = s.split(' ')
            x2_S_list.append(words)
        del x1
        del x2
        return np.asarray(x1_S_list),np.asarray(x2_S_list),y

def _list_to_word_ids(sList, word_to_id):
  unkown_word_id = len(word_to_id)
  word_ids = []
  for word in sList:
    if word in word_to_id: 
      word_ids.append(word_to_id[word])
    else:
      word_ids.append(unkown_word_id)          
  return word_ids


def file_to_ids(data_full_path, word_to_id):
  x1_S_list,x2_S_list,label = _read_words(data_full_path)  
  x1_Id_List = []
  x2_Id_List = []
 
  for s in x1_S_list:
      ids = _list_to_word_ids(s, word_to_id)
      x1_Id_List.append(ids)
  for s in x2_S_list:
      ids = _list_to_word_ids(s, word_to_id)
      x2_Id_List.append(ids)
  return (x1_Id_List, x2_Id_List,label)

def load_word2vec(path):
  fileData = open(path, 'rb')
  vocab_size, = struct.unpack("i", fileData.read(4))
  #vocab_size = int(vocab_size / 3) #[gaoteng]: Max graph size can't exceed 2GB!!!
  #vocab_size = int(50) #[gaoteng]: Max graph size can't exceed 2GB!!!
  dim, = struct.unpack("i", fileData.read(4))
  word2id = {}
  word_embeddings = np.zeros((vocab_size, dim), dtype = np.float32)
  print ("Loading Word Enbeding, embeding size : ")
  print(word_embeddings.shape)

  for i in range(vocab_size - 1):
    word_len, = struct.unpack("i", fileData.read(4))
    word_str, = struct.unpack(str(word_len) + "s", fileData.read(word_len))
    word2id[word_str] = i
    for j in range(dim):
        elem_value, = struct.unpack("f", fileData.read(4))
        word_embeddings[i, j] = elem_value    
        #print("(%f, %f)" % (word_embeddings[i, j], elem_value))
  for i in range(dim):
    word_embeddings[vocab_size - 1, i] = 0.0
  return word2id, word_embeddings
  
def get_data_for_siamese(word_to_id,data_path=None):
    train_path = os.path.join(data_path, TRAIN_FILE)
    valid_path = os.path.join(data_path, VALID_FILE)
    vocabu_len = len(word_to_id) + 1
    
    train_x1_idsList, train_x2_idsList,train_y = file_to_ids(train_path, word_to_id)
    valid_x1_idsList, valid_x2_idsList,valid_y = file_to_ids(valid_path, word_to_id)
    print ("Total num of cases in train Set : %d"%(len(train_x1_idsList)))
    print ("Total num of cases in validate Set : %d"%(len(valid_x1_idsList)))
    return (train_x1_idsList, train_x2_idsList,train_y),(valid_x1_idsList, valid_x2_idsList,valid_y)
    
def data_iterator(x1_idList, x2_idList, y_List,batch_size,maxSenLen):
    sentenceCount = len(x1_idList)
    epoch_size = sentenceCount // batch_size
    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or maxSenLen")
    validSentenceCount = epoch_size * batch_size
    x1_temp = np.zeros([validSentenceCount, maxSenLen], dtype=np.int32)
    x2_temp = np.zeros([validSentenceCount, maxSenLen], dtype=np.int32)
    for i in range(validSentenceCount):
        arr_x1 = np.asarray(x1_idList[i])
        arr_x2 = np.asarray(x2_idList[i])
        arr_x1.resize((maxSenLen))
        arr_x2.resize((maxSenLen))
        x1_temp[i] = arr_x1
        x2_temp[i] = arr_x2
    for i in range(epoch_size):
        x1_batch = x1_temp[i * batch_size : (i + 1) * batch_size, :]
        x2_batch = x2_temp[i * batch_size : (i + 1) * batch_size, :]
        y_batch = y_List[i * batch_size : (i + 1) * batch_size]
        yield (x1_batch,x2_batch,y_batch)
  