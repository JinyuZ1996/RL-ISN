
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

class hrnn():
    def __init__(self, num_users, num_items, embedding_size=100, hidden_size=100, num_layers=1, gpu='0'):             
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.num_items = num_items
        self.num_users = num_users
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(1)
            with tf.name_scope('inputs'):
                self.ind_U, self.seq_I, self.seq_L, self.length, self.learning_rate, self.keep_prob = self.get_inputs()
            with tf.name_scope('history'):
                self.history_matrix, self.state_matrix, self.his_U = self.get_history(self.ind_U, self.num_users, self.hidden_size)
            with tf.name_scope('gru'):
                self.embedding_matrix, self.logits, self.state = self.gru(self.his_U, self.num_items, self.seq_I, self.length, self.embedding_size, self.hidden_size, self.num_layers, self.keep_prob)
            with tf.name_scope('loss'):
                self.loss = self.cal_loss(self.seq_L, self.logits)
            with tf.name_scope('optimizer'):
                self.train_op = self.optimizer(self.loss, self.learning_rate)
            with tf.name_scope('update'):
                self.update_h, self.update_s, self.update_h_test, self.update_s_test = self.update(self.train_op, self.ind_U, self.history_matrix, self.state_matrix, self.his_U, self.state)
        
    def get_inputs(self):
        ind_U = tf.placeholder(dtype=tf.int32, shape=[None,], name='ind_U')
        seq_I = tf.placeholder(dtype=tf.int32, shape=[None,None], name='seq_I')
        seq_L = tf.placeholder(dtype=tf.int32, shape=[None,], name='seq_L')
        length = tf.placeholder(dtype=tf.int32, shape=[None,], name='length')
        learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
        keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
        return ind_U, seq_I, seq_L, length, learning_rate, keep_prob
    
    def get_history(self, ind_U, num_users, hidden_size):
        history_matrix = tf.get_variable(dtype=tf.float32, name='history_matrix', shape=[num_users, hidden_size], initializer=tf.zeros_initializer(), trainable=False)
        print(history_matrix)
        state_matrix = tf.get_variable(dtype=tf.float32, name='state_matrix', shape=[num_users, hidden_size], initializer=tf.zeros_initializer(), trainable=False)
        print(state_matrix)
        his_U = tf.nn.embedding_lookup(history_matrix, ind_U)#his_U=[batch_size,hidden_size]
        print(his_U)
        pre_S = tf.nn.embedding_lookup(state_matrix, ind_U)#pre_S=[batch_size,hidden_size]
        con = tf.concat([pre_S, his_U], axis=-1)#con=[batch_size,2*hidden_size]
        print(con)
        zr = tf.layers.dense(con, 2*hidden_size, activation=tf.nn.sigmoid, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False), name='gate')#weight=[2*hidden_size,2*hidden_size]；zr=[batch_size,2*hidden_size]
        print(zr)
        z, r = tf.split(zr, num_or_size_splits=2, axis=1)#z=[batch_size,hidden_size]，r=[batch_size,hidden_size]
        print(z)
        print(r)
        ur = his_U*r#ur=[batch_size,hidden_size]
        print(ur)
        con = tf.concat([pre_S, ur], axis=-1)#con=[batch_size,2*hidden_size]
        print(con)
        h = tf.layers.dense(con, hidden_size, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False), name='candidate')#weight=[2*hidden_size,hidden_size]；_h=[batch_size,hidden_size]
        print(h)
        new_his_U = (1-z)*h+z*his_U#new_his_U=[batch_size,hidden_size]
        print(new_his_U)
        return history_matrix, state_matrix, new_his_U
    
    def get_gru_cell(self, hidden_size, keep_prob):
        gru_cell = tf.contrib.rnn.GRUCell(hidden_size, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        gru_cell = tf.contrib.rnn.DropoutWrapper(gru_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob, state_keep_prob=keep_prob)  
        return gru_cell
    
    def gru(self, his_U, num_items, seq_I, length, embedding_size, hidden_size, num_layers, keep_prob):
        embedding_matrix = tf.get_variable(dtype=tf.float32, name='embedding_matrix', shape=[num_items, embedding_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False)) 
        print(embedding_matrix)
        embbed_seq_I = tf.nn.embedding_lookup(embedding_matrix, seq_I)#embbed_seq_I=[batch_size,timestamp,embedding_size]
        gru_cell = tf.contrib.rnn.MultiRNNCell([self.get_gru_cell(hidden_size, keep_prob) for _ in range(num_layers)])
        output,state = tf.nn.dynamic_rnn(gru_cell, embbed_seq_I, sequence_length=length, initial_state=(his_U,)*num_layers, dtype=tf.float32)#output=[batch_size,timestamp_T,hidden_size], state=([batch_size,hidden_size]*num_layers,)      
        print(output)
        print(state)
        state_drop = tf.nn.dropout(state[-1], keep_prob)
        logits = tf.layers.dense(state_drop, num_items, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False), name='output_layer', reuse=None)#logits=[batch_size,num_items]
        print(logits)
        return embedding_matrix, logits, state[-1]
    
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
    
    def update(self, train_op, ind_U, history_matrix, state_matrix, his_U, state):
        indices = tf.expand_dims(ind_U, -1)#indices=[batch_size,1]
        print(indices)
        with tf.control_dependencies([train_op]):
            update_h = tf.scatter_nd_update(history_matrix, indices, his_U)
            update_s = tf.scatter_nd_update(state_matrix, indices, state)
        update_h_test = tf.scatter_nd_update(history_matrix, indices, his_U)
        update_s_test = tf.scatter_nd_update(state_matrix, indices, state)
        return update_h, update_s, update_h_test, update_s_test