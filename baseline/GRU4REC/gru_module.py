
# coding: utf-8

# In[ ]:

import os
import random
import numpy as np
import tensorflow as tf


# In[ ]:

random.seed(1)
np.random.seed(1)


# In[ ]:

class gru4rec():
    def __init__(self, num_items, embedding_size=100, hidden_size=100, num_layers=1, gpu='0'):                   
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(1)
            with tf.name_scope('inputs'):
                self.seq_I, self.seq_L, self.length, self.learning_rate, self.keep_prob = self.get_inputs()
            with tf.name_scope('gru'):
                self.logits = self.gru(self.num_items, self.seq_I, self.length, self.embedding_size, self.hidden_size, self.num_layers, self.keep_prob)
            with tf.name_scope('loss'):
                self.loss = self.cal_loss(self.seq_L, self.logits)
            with tf.name_scope('optimizer'):
                self.train_op = self.optimizer(self.loss, self.learning_rate)
        
    def get_inputs(self):
        seq_I = tf.placeholder(dtype=tf.int32, shape=[None,None], name='seq_I')
        seq_L = tf.placeholder(dtype=tf.int32, shape=[None,], name='seq_L')
        length = tf.placeholder(dtype=tf.int32, shape=[None,], name='length')
        learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
        keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
        return seq_I, seq_L, length, learning_rate, keep_prob
    
    def get_gru_cell(self, hidden_size, keep_prob):
        gru_cell = tf.contrib.rnn.GRUCell(hidden_size, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        gru_cell = tf.contrib.rnn.DropoutWrapper(gru_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob, state_keep_prob=keep_prob)  
        return gru_cell
    
    def gru(self, num_items, seq_I, length, embedding_size, hidden_size, num_layers, keep_prob):  
        embedding_matrix = tf.get_variable(dtype=tf.float32, name='embedding_matrix', shape=[num_items,embedding_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False))  
        print(embedding_matrix)
        embbed_seq_I = tf.nn.embedding_lookup(embedding_matrix, seq_I)#embbed_seq_I=[batch_size,timestamp,embedding_size]
        gru_cell = tf.contrib.rnn.MultiRNNCell([self.get_gru_cell(hidden_size, keep_prob) for _ in range(num_layers)])
        output,state = tf.nn.dynamic_rnn(gru_cell, embbed_seq_I, sequence_length=length, dtype=tf.float32)#output=[batch_size,timestamp_T,hidden_size], state=([batch_size,hidden_size]*num_layers,)   
        print(output)
        print(state)
        state = tf.nn.dropout(state[-1], keep_prob)
        logits = tf.layers.dense(state, num_items, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False), name='output_layer', reuse=None)#logits=[batch_size,num_items]
        print(logits)
        return logits
    
    def cal_loss(self, seq_L, logits):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=seq_L, logits=logits)
        loss = tf.reduce_mean(loss, name='loss')
        return loss
    
    def optimizer(self, loss, learning_rate):
        optimizer = tf.train.AdamOptimizer(learning_rate)                             
        gradients = optimizer.compute_gradients(loss)
        capped_gradients = [(tf.clip_by_value(grad,-5.,5.),var) for grad,var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
        return train_op