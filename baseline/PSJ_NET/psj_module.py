#!/usr/bin/env python
# coding: utf-8
import os
import random
import numpy as np
import tensorflow as tf
from PSJ_net.PSJ_settings import *

args = setting()
random.seed(1)
np.random.seed(1)



def generate_single_mask(seq_len):
    mask = np.ones((len(seq_len), max(seq_len)))
    for i in range(len(seq_len)):
        mask[i,seq_len[i]:] = 0
    return mask

def generate_mutual_mask(len_A, len_B):
    mask_A2B = np.ones((len(len_A), max(len_A), max(len_B)))
    for i in range(len(len_A)):
        mask_A2B[i,len_A[i]:,:] = 0
        mask_A2B[i,:,len_B[i]:] = 0
    mask_B2A = np.transpose(mask_A2B, (0,2,1))                   
    return mask_A2B, mask_B2A


class FilterCell(tf.contrib.rnn.RNNCell):
    def __init__(self, num_units, num_members, member_embedding, activation=None, reuse=None, kernel_initializer=None, bias_initializer=None):
        super(FilterCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._num_members = num_members
        self._member_embedding = member_embedding
        self._activation = activation or tf.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer 
        
    @property
    def state_size(self):
        return tf.TensorShape([self._num_members, self._num_units])#We need use tf.TensorShape to define the shape of state and output

    @property
    def output_size(self):
        return tf.TensorShape([self._num_members, self._num_units])
    
    def call(self, inputs, state):#inputs=[batch_size,hidden_size+hidden_size] (it is one timestamp), state=[batch_size,self._num_members,self._num_units] (the initial state is zero defined by RNNCell, we don't need to override it)
        inputs_A, inputs_T = tf.split(inputs, num_or_size_splits=2, axis=1)#inputs_A=[batch_size,hidden_size]，inputs_T=[batch_size,hidden_size]; note that it is corresponding to the concat in the filter method 
        if self._kernel_initializer is None:
            self._kernel_initializer = tf.contrib.layers.xavier_initializer(uniform=False)
        if self._bias_initializer is None:
            self._bias_initializer = tf.constant_initializer(1.0)   
        with tf.variable_scope('member_gate'):#sigmoid([i_A|i_T|s_(t-1)]*[W_fA;W_fT;U_f]+emb*V_f+b_f)
            self.W_f = tf.get_variable(dtype=tf.float32, name='W_f', shape=[inputs.get_shape()[-1].value+state.get_shape()[-1].value, self._num_units], initializer=self._kernel_initializer)#for get_variable, we can only use get_shape instead of shape （it is not deterministic when we define the model）; note that the values we get by get_shape are all deterministic (we can checked by print) when we define the model (if the value is depend on the batch_size, we can't use get_shape either, because it is not deterministic)           
            self.V_f = tf.get_variable(dtype=tf.float32, name='V_f', shape=[self._member_embedding.get_shape()[-1].value, self._num_units], initializer=self._kernel_initializer)#we can't self.V_f = op(self.V_f), but can _V_f = op(self.V_f)
            self.b_f = tf.get_variable(dtype=tf.float32, name='b_f', shape=[self._num_units,], initializer=self._bias_initializer)
            u = tf.matmul(self._member_embedding, self.V_f)#u=[self._num_members,self._num_units]
            i = tf.expand_dims(inputs, 1)#i=[batch_size,1,hidden_size+hidden_size]
            i = tf.tile(i, [1,self._num_members,1])#i=[batch_size,self._num_members,hidden_size+hidden_size]
            f = tf.concat([i, state], axis=-1)#f=[batch_size,self._num_members,hidden_size+hidden_size+self._num_units]
            w = tf.expand_dims(self.W_f, 0)#w=[1,hidden_size+hidden_size+self._num_units,self._num_units]
            w = tf.tile(w, [tf.shape(f)[0],1,1])#w=[batch_size,hidden_size+hidden_size+self._num_units,self._num_units]
            f = tf.matmul(f, w)#f=[batch_size,self._num_members,self._num_units]
            f = f+u+self.b_f#f=[batch_size,self._num_members,self._num_units]
            f = tf.sigmoid(f)
            #f = tf.exp(f)
        with tf.variable_scope('none_gate'):#sigmoid([i_A|i_T]*[W_nA;W_nT]+s_(t-1)*V_f+b_n)
            self.W_n = tf.get_variable(dtype=tf.float32, name='W_n', shape=[inputs.get_shape()[-1].value, self._num_units], initializer=self._kernel_initializer)     
            self.V_n_1 = tf.get_variable(dtype=tf.float32, name='V_n_1', shape=[self._num_units, self._num_units], initializer=self._kernel_initializer)
            self.V_n_2 = tf.get_variable(dtype=tf.float32, name='V_n_2', shape=[self._num_members,self._num_units], initializer=self._kernel_initializer)
            self.b_n = tf.get_variable(dtype=tf.float32, name='b_n', shape=[self._num_units,], initializer=self._bias_initializer)
            v = tf.einsum('ijk,kl->ijl', state, self.V_n_1)#v=[batch_size,self._num_members,self._num_units]
            v = tf.einsum('ijk,jk->ik', v, self.V_n_2)#v=[batch_size,self._num_units]
            g = tf.matmul(inputs, self.W_n)#g=[batch_size,self._num_units]
            g = g+v
            g = g+self.b_n#g=[batch_size,self._num_units]
            g = tf.sigmoid(g)
            #g = tf.exp(g)
        sum_f = tf.reduce_sum(f, axis=1)#sum_f=[batch_size,self._num_units]
        sum_f = sum_f+g#sum_f=[batch_size,self._num_units]
        sum_f = sum_f+1e-10#Avoid sum is zero
        f = f/tf.expand_dims(sum_f, axis=1)#Because f is the output of sigmoid, we use sum for normalization
        g = g/sum_f
        g = tf.expand_dims(g,1)#g=[batch_size,1,self._num_units]
        g = tf.tile(g, [1,self._num_members,1])#g=[batch_size,self._num_members,self._num_units]
        with tf.variable_scope('candidate'):#tanh([i_A|s_(t-1)]*[W_s;U_s]+emb*V_s+b_s)
            self.W_s = tf.get_variable(dtype=tf.float32, name='W_s', shape=[inputs_A.get_shape()[-1].value+state.get_shape()[-1].value, self._num_units], initializer=self._kernel_initializer) 
            self.V_s = tf.get_variable(dtype=tf.float32, name='V_s', shape=[self._member_embedding.get_shape()[-1].value, self._num_units], initializer=self._kernel_initializer)
            self.b_s = tf.get_variable(dtype=tf.float32, name='b_s', shape=[self._num_units,], initializer=self._bias_initializer)
            _u = tf.matmul(self._member_embedding, self.V_s)#_u=[self._num_members,self._num_units]
            _i = tf.expand_dims(inputs_A, 1)#_i=[batch_size,1,hidden_size]
            _i = tf.tile(_i, [1,self._num_members,1])#_i=[batch_size,self._num_members,hidden_size]
            _s = tf.concat([_i, state], axis=-1)#_s=[batch_size,self._num_members,hidden_size+self._num_units]
            _w = tf.expand_dims(self.W_s, 0)#_w=[1,hidden_size+self._num_units,self._num_units]
            _w = tf.tile(_w, [tf.shape(_s)[0],1,1])#_w=[batch_size,hidden_size+self._num_units,self._num_units]
            _s = tf.matmul(_s, _w)#_s=[batch_size,self._num_members,self._num_units]
            _s = _s+u+self.b_s#_s=[batch_size,self._num_members,self._num_units]
            _s = self._activation(_s)
        new_s = f*_s+g*state#new_s=[batch_size,self._num_members,self._num_units]
        return new_s, new_s


# In[ ]:


class PSJ_Net:
    def __init__(self, items_max_num_A, items_max_num_B, num_members=args.default_account_num,
                 embedding_size=args.recommender_embedding_size, hidden_size=args.recommender_weight_size,
                 num_layers=args.num_layers, gpu=args.gpu_num):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.num_items_A = items_max_num_A
        self.num_items_B = items_max_num_B
        self.num_members = num_members
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(1)
            with tf.name_scope('inputs'):
                self.seq_A, self.seq_B, self.len_A, self.len_B, self.pos_A, self.pos_B, self.target_A, self.target_B, self.mask_A, self.mask_B, self.mask_A2B, self.mask_B2A, self.learning_rate, self.keep_prob = self.get_inputs()                                         
            with tf.name_scope('encoder_A'):
                encoder_output_A,encoder_state_A = self.encoder_A(self.num_items_A, self.seq_A, self.len_A, self.embedding_size, self.hidden_size, self.num_layers, self.keep_prob)                           
            with tf.name_scope('encoder_B'):
                encoder_output_B,encoder_state_B = self.encoder_B(self.num_items_B, self.seq_B, self.len_B, self.embedding_size, self.hidden_size, self.num_layers, self.keep_prob)
            with tf.name_scope('segregation_B'):
                filter_output_B,filter_state_B = self.filter_B(encoder_output_A, encoder_output_B, self.len_B, self.pos_B, self.num_members, self.embedding_size, self.hidden_size, self.num_layers, self.keep_prob)
                transfer_output_B,transfer_state_B = self.transfer_B(filter_output_B, self.num_members, self.len_B, self.hidden_size, self.num_layers, self.keep_prob)           
            with tf.name_scope('integration_B'):
                context_B = self.attention_A2B(encoder_output_A, transfer_output_B, self.mask_A2B, self.mask_A, self.num_members)
            with tf.name_scope('prediction_A'):
                self.pred_A = self.prediction_A(self.num_items_A, encoder_state_A, context_B, self.keep_prob, self.hidden_size)
            with tf.name_scope('segregation_A'):
                filter_output_A,filter_state_A = self.filter_A(encoder_output_B, encoder_output_A, self.len_A, self.pos_A, self.num_members, self.embedding_size, self.hidden_size, self.num_layers, self.keep_prob)
                transfer_output_A,transfer_state_A = self.transfer_A(filter_output_A, self.num_members, self.len_A, self.hidden_size, self.num_layers, self.keep_prob)            
            with tf.name_scope('integration_A'):
                context_A = self.attention_B2A(encoder_output_B, transfer_output_A, self.mask_B2A, self.mask_B, self.num_members)
            with tf.name_scope('prediction_B'):
                self.pred_B = self.prediction_B(self.num_items_B, encoder_state_B, context_A, self.keep_prob, self.hidden_size)            
            with tf.name_scope('loss'):
                self.loss = self.cal_loss(self.target_A, self.pred_A, self.target_B, self.pred_B)
            with tf.name_scope('optimizer'):
                self.train_op = self.optimizer(self.loss, self.learning_rate)
        
    def get_inputs(self):
        seq_A = tf.placeholder(dtype=tf.int32, shape=[None,None], name='seq_A')
        seq_B = tf.placeholder(dtype=tf.int32, shape=[None,None], name='seq_B')
        len_A = tf.placeholder(dtype=tf.float32, shape=[None,], name='len_A')
        len_B = tf.placeholder(dtype=tf.float32, shape=[None,], name='len_B')
        pos_A = tf.placeholder(dtype=tf.int32, shape=[None,None,2], name='pos_A')
        pos_B = tf.placeholder(dtype=tf.int32, shape=[None,None,2], name='pos_B')
        target_A = tf.placeholder(dtype=tf.int32, shape=[None,], name='target_A') 
        target_B = tf.placeholder(dtype=tf.int32, shape=[None,], name='target_B') 
        mask_A = tf.placeholder(dtype=tf.float32, shape=[None,None], name='mask_A')
        mask_B = tf.placeholder(dtype=tf.float32, shape=[None,None], name='mask_B')
        mask_A2B = tf.placeholder(dtype=tf.float32, shape=[None,None,None], name='mask_A2B')
        mask_B2A = tf.placeholder(dtype=tf.float32, shape=[None,None,None], name='mask_B2A')
        learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
        keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
        return seq_A, seq_B, len_A, len_B, pos_A, pos_B, target_A, target_B, mask_A, mask_B, mask_A2B, mask_B2A, learning_rate, keep_prob
    
    def get_gru_cell(self, hidden_size, keep_prob):
        gru_cell = tf.contrib.rnn.GRUCell(hidden_size, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        gru_cell = tf.contrib.rnn.DropoutWrapper(gru_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob, state_keep_prob=keep_prob)  
        return gru_cell
    
    def get_filter_cell(self, hidden_size, num_members, member_embedding, keep_prob):
        filter_cell = FilterCell(hidden_size, num_members, member_embedding)
        filter_cell = tf.contrib.rnn.DropoutWrapper(filter_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob, state_keep_prob=keep_prob)  
        return filter_cell   
    
    def dot_matrix(self, encoder_output, transfer_output):
        encoder_output_ext = tf.expand_dims(encoder_output, axis=2)#encoder_output_ext=[batch_size,timestamp_e,1,hidden_size]
        encoder_output_ext = tf.tile(encoder_output_ext, [1,1,tf.shape(transfer_output)[1],1])#encoder_output_ext=[batch_size,timestamp_e,timestamp_t,hidden_size]
        transfer_output_ext = tf.expand_dims(transfer_output, axis=1)#transfer_output_ext=[batch_size,1,timestamp_t,hidden_size]
        transfer_output_ext = tf.tile(transfer_output_ext, [1,tf.shape(encoder_output)[1],1,1])#transfer_output_ext=[batch_size,timestamp_e,timestamp_t,hidden_size]
        dot = tf.concat([encoder_output_ext,transfer_output_ext,encoder_output_ext*transfer_output_ext], axis=-1)#dot=[batch_size,timestamp_e,timestamp_t,hidden_size*3]
        dot = tf.layers.dense(dot, 1, activation=None, use_bias=False)#dot=[batch_size,timestamp_e,timestamp_t,1]
        dot = tf.squeeze(dot)#dot=[batch_size,timestamp_e,timestamp_t]
        return dot
    
    def attention_1(self, encoder_output, transfer_output, mask_e2t):#encoder_output=[batch_size,timestamp_e,hidden_size], transfer_output=[batch_size,timestamp_t,hidden_size], mask_e2t=[batch_size,timestamp_e,timestamp_t]         
        dot = self.dot_matrix(encoder_output, transfer_output)#dot=[batch_size,timestamp_e,timestamp_t]
        dot = dot - tf.reduce_max(dot, [-2,-1], keep_dims=True)#Avoidg overflow; note that the normalization here takes into account the two dimensions, so we subtract the maximum of two dimensions (here has a bug, the max may be from padding)
        dot = tf.exp(dot)*mask_e2t#We first use exp to make dot>0, then we use mask_e2t to mask padding part, so we make sure padding part doesn't affect max and sum 
        max_t = tf.reduce_max(dot, axis=1)#max_t=[batch_size,timestamp_t]
        sum_t = tf.reduce_sum(max_t, axis=-1, keep_dims=True)#sum_t=[batch_size,1]
        attn_t = max_t/sum_t#attn_t=[batch_size,timestamp_t]
        attn_t = tf.expand_dims(attn_t, axis=1)#attn_t=[batch_size,1,timestamp_t]
        ctx_t = tf.matmul(attn_t,transfer_output)#ctx_t=[batch_size,1,hidden_size]
        return ctx_t
    
    def attention_2(self, encoder_output, transfer_context, mask_e):#encoder_output=[batch_size,timestamp_e,hidden_size], transfer_context=[batch_size,num_members,hidden_size], mask_e=[batch_size,timestamp_e]         
        dot = self.dot_matrix(encoder_output, transfer_context)#dot=[batch_size,timestamp_e,num_members]
        if self.num_members == 1:#In this case, the dot returned by dot_matrix will be [batch_size,timestamp_e] (squeeze)
            dot = tf.expand_dims(dot,-1)
        dot = dot - tf.reduce_max(dot, [-2,-1], keep_dims=True)#Avoid overflow; note that the normalization here takes into account the two dimensions, so we subtract the maximum of two dimensions (here has a bug, the max may be from padding)
        dot = tf.exp(dot)*tf.expand_dims(mask_e,-1)#mask_e=[batch_size,timestamp_e,1]
        print(dot)#Note that shape is unknown
        max_t = tf.reduce_max(dot, axis=1)#max_t=[batch_size,num_members]
        sum_t = tf.reduce_sum(max_t, axis=-1, keep_dims=True)#sum_t=[batch_size,1]
        attn_t = max_t/sum_t#attn_t=[batch_size,num_members]
        attn_t = tf.expand_dims(attn_t, axis=1)#attn_t=[batch_size,1,num_members]
        ctx_t = tf.matmul(attn_t,transfer_context)#ctx_t=[batch_size,1,hidden_size]
        print(ctx_t)#Note that shape is known
        ctx_t = tf.squeeze(ctx_t)#ctx_t=[batch_size,hidden_size]
        print(ctx_t)#Note that shape becomes unknown; that's why we can't use dense layer in prediction
        return ctx_t
        
    def encoder_A(self, num_items_A, seq_A, len_A, embedding_size, hidden_size, num_layers, keep_prob):
        with tf.variable_scope('encoder_A'):#It looks like that name_scope can't help us differentiate different dynamic_rnn (more specifically the parameters in dynamic_rnn, like embedding_matrix; if we don't use variable_scope and use the same name for embedding_matrix, it will report an error), so we have to use variable_scope.      
            embedding_matrix_A = tf.get_variable(dtype=tf.float32, name='embedding_matrix_A', shape=[num_items_A,embedding_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False))#0 is a item id and is also the pad id (no problem in here)   
            print(embedding_matrix_A)
            embbed_seq_A = tf.nn.embedding_lookup(embedding_matrix_A, seq_A)#embbed_seq_A=[batch_size,timestamp_A,embedding_size]
            encoder_cell_A = tf.contrib.rnn.MultiRNNCell([self.get_gru_cell(hidden_size, keep_prob) for _ in range(num_layers)])
            encoder_output_A,encoder_state_A = tf.nn.dynamic_rnn(encoder_cell_A, embbed_seq_A, sequence_length=len_A, dtype=tf.float32)#encoder_output_A=[batch_size,timestamp_A,hidden_size], encoder_state_A=([batch_size,hidden_size]*num_layers)       
            print(encoder_output_A)
            print(encoder_state_A)
        return encoder_output_A,encoder_state_A
    
    def encoder_B(self, num_items_B, seq_B, len_B, embedding_size, hidden_size, num_layers, keep_prob):
        with tf.variable_scope('encoder_B'):
            embedding_matrix_B = tf.get_variable(dtype=tf.float32, name='embedding_matrix_B', shape=[num_items_B,embedding_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False))   
            print(embedding_matrix_B)
            embbed_seq_B = tf.nn.embedding_lookup(embedding_matrix_B, seq_B)#embbed_seq_B=[batch_size,timestamp_B,embedding_size]
            encoder_cell_B = tf.contrib.rnn.MultiRNNCell([self.get_gru_cell(hidden_size, keep_prob) for _ in range(num_layers)])
            encoder_output_B,encoder_state_B = tf.nn.dynamic_rnn(encoder_cell_B, embbed_seq_B, sequence_length=len_B, dtype=tf.float32)#encoder_output_B=[batch_size,timestamp_B,hidden_size], encoder_state_B=([batch_size,hidden_size]*num_layers)    
            print(encoder_output_B)
            print(encoder_state_B)
        return encoder_output_B,encoder_state_B
    
    def filter_B(self, encoder_output_A, encoder_output_B, len_B, pos_B, num_members, embedding_size, hidden_size, num_layers, keep_prob):
        with tf.variable_scope('filter_B'):
            zero_state = tf.zeros(dtype=tf.float32, shape=(tf.shape(encoder_output_A)[0],1,tf.shape(encoder_output_A)[-1]))#zero_state=[batch_size,1,hidden_size]; here, zero_state likes a inner placeholder（it is permitted, like the zero initial state of RNN）; because zero_state is not a variable (tf.zeros has not trainable), it can't affect optimization                                      
            print(zero_state)
            encoder_output = tf.concat([zero_state,encoder_output_A], axis=1)#encoder_output=[batch_size,timestamp_A+1,hidden_size]
            print(encoder_output)
            select_output_A = tf.gather_nd(encoder_output,pos_B)#select_output_A=[batch_size,timestamp_B,hidden_size]
            print(select_output_A)
            filter_input_B = tf.concat([encoder_output_B,select_output_A], axis=-1)#filter_input_B=[batch_size,timestamp_B,hidden_size+hidden_size]
            print(filter_input_B)
            member_embedding_B = tf.get_variable(dtype=tf.float32, name='member_embedding_B', shape=[num_members,embedding_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            print(member_embedding_B)
            filter_cell_B = tf.contrib.rnn.MultiRNNCell([self.get_filter_cell(hidden_size, num_members, member_embedding_B, keep_prob) for _ in range(num_layers)])
            filter_output_B,filter_state_B = tf.nn.dynamic_rnn(filter_cell_B, filter_input_B, sequence_length=len_B, dtype=tf.float32)#filter_output_B=[batch_size,timestamp_B,hidden_size]，filter_state_B=[batch_size,hidden_size]            
            print(filter_output_B)
            print(filter_state_B)            
        return filter_output_B,filter_state_B
        
    def transfer_B(self, filter_output_B, num_members, len_B, hidden_size, num_layers, keep_prob):
        transfer_output_B = []
        transfer_state_B = []
        for i in range(num_members):
            with tf.variable_scope('transfer_B', reuse = i>0):#share the transfer rnn for all members
                filter_output_B_i = filter_output_B[:,:,i,:]#filter_output_B_i=[batch_size,timestamp_B,hidden_size]
                transfer_cell_B = tf.contrib.rnn.MultiRNNCell([self.get_gru_cell(hidden_size, keep_prob) for _ in range(num_layers)])
                transfer_output_B_i,transfer_state_B_i = tf.nn.dynamic_rnn(transfer_cell_B, filter_output_B_i, sequence_length=len_B, dtype=tf.float32)#transfer_output_B=[batch_size,timestamp_B,hidden_size], transfer_state_B=([batch_size,hidden_size]*num_layers)     
                transfer_output_B.append(transfer_output_B_i)
                transfer_state_B.append(transfer_state_B_i[-1])
        print(transfer_output_B)#transfer_output_B=[num_members,batch_size,timestamp_B,hidden_size]
        print(transfer_state_B)#transfer_state_B=[num_members,batch_size,hidden_size]
        return transfer_output_B,transfer_state_B
    
    def attention_A2B(self, encoder_output_A, transfer_output_B, mask_A2B, mask_A, num_members):
        context_B_list = []
        for i in range(num_members):
            with tf.variable_scope('attention_A2B_1', reuse = i>0):
                context_B = self.attention_1(encoder_output_A, transfer_output_B[i], mask_A2B)#context_B=[batch_size,1,hidden_size]
                context_B_list.append(context_B)
        context_B = tf.concat(context_B_list, axis=1)#context_B=[batch_size,num_members,hidden_size]
        print(context_B)
        with tf.variable_scope('attention_A2B_2'):
            context_B = self.attention_2(encoder_output_A, context_B, mask_A)#context_B=[batch_size,hidden_size]
        return context_B
    
    def prediction_A(self, num_items_A, encoder_state_A, context_B, keep_prob, hidden_size):
        with tf.variable_scope('prediction_A'):
            concat_output = tf.concat([encoder_state_A[-1], context_B], axis=-1)#concat_output=[batch_size,hidden_size*2]                                                                                                          
            print(concat_output)
            concat_output = tf.nn.dropout(concat_output, keep_prob)#concat_output=[batch_size,hidden_size*2]
            output_weight_A = tf.get_variable(dtype=tf.float32, name='output_weight_A', shape=[hidden_size*2, num_items_A], initializer=tf.contrib.layers.xavier_initializer(uniform=False))            
            output_bias_A = tf.get_variable(dtype=tf.float32, name='output_bias_A', shape=[num_items_A,], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            pred_A = tf.matmul(concat_output, output_weight_A)+output_bias_A#pred_A=[batch_size,num_items_A]
            print(pred_A)
        return pred_A
    
    def filter_A(self, encoder_output_B, encoder_output_A, len_A, pos_A, num_members, embedding_size, hidden_size, num_layers, keep_prob):
        with tf.variable_scope('filter_A'):
            zero_state = tf.zeros(dtype=tf.float32, shape=(tf.shape(encoder_output_B)[0],1,tf.shape(encoder_output_B)[-1]))#zero_state=[batch_size,1,hidden_size]; here, zero_state likes a inner placeholder（it is permitted, like the zero initial state of RNN）; because zero_state is not a variable (tf.zeros has not trainable), it can't affect optimization                                      
            print(zero_state)
            encoder_output = tf.concat([zero_state,encoder_output_B], axis=1)#encoder_output=[batch_size,timestamp_B+1,hidden_size]
            print(encoder_output)
            select_output_B = tf.gather_nd(encoder_output,pos_A)#select_output_B=[batch_size,timestamp_A,hidden_size]
            print(select_output_B)
            filter_input_A = tf.concat([encoder_output_A,select_output_B], axis=-1)#filter_input_A=[batch_size,timestamp_A,hidden_size+hidden_size]
            print(filter_input_A)
            member_embedding_A = tf.get_variable(dtype=tf.float32, name='member_embedding_A', shape=[num_members,embedding_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            print(member_embedding_A)
            filter_cell_A = tf.contrib.rnn.MultiRNNCell([self.get_filter_cell(hidden_size, num_members, member_embedding_A, keep_prob) for _ in range(num_layers)])
            filter_output_A,filter_state_A = tf.nn.dynamic_rnn(filter_cell_A, filter_input_A, sequence_length=len_A, dtype=tf.float32)#filter_output_A=[batch_size,timestamp_A,hidden_size]，filter_state_A=[batch_size,hidden_size]            
            print(filter_output_A)
            print(filter_state_A)            
        return filter_output_A,filter_state_A
        
    def transfer_A(self, filter_output_A, num_members, len_A, hidden_size, num_layers, keep_prob):
        transfer_output_A = []
        transfer_state_A = []
        for i in range(num_members):
            with tf.variable_scope('transfer_A', reuse = i>0):#share the transfer rnn for all members
                filter_output_A_i = filter_output_A[:,:,i,:]#filter_output_A_i=[batch_size,timestamp_A,hidden_size]
                transfer_cell_A = tf.contrib.rnn.MultiRNNCell([self.get_gru_cell(hidden_size, keep_prob) for _ in range(num_layers)])
                transfer_output_A_i,transfer_state_A_i = tf.nn.dynamic_rnn(transfer_cell_A, filter_output_A_i, sequence_length=len_A, dtype=tf.float32)#transfer_output_A=[batch_size,timestamp_A,hidden_size], transfer_state_A=([batch_size,hidden_size]*num_layers)     
                transfer_output_A.append(transfer_output_A_i)
                transfer_state_A.append(transfer_state_A_i[-1])
        print(transfer_output_A)#transfer_output_A=[num_user,batch_size,timestamp_A,hidden_size]
        print(transfer_state_A)#transfer_state_A=[num_user,batch_size,hidden_size]
        return transfer_output_A,transfer_state_A
    
    def attention_B2A(self, encoder_output_B, transfer_output_A, mask_B2A, mask_B, num_members):
        context_A_list = []
        for i in range(num_members):
            with tf.variable_scope('attention_B2A_1', reuse = i>0):
                context_A = self.attention_1(encoder_output_B, transfer_output_A[i], mask_B2A)#context_A=[batch_size,1,hidden_size]
                context_A_list.append(context_A)
        context_A = tf.concat(context_A_list, axis=1)#context_A=[batch_size,num_members,hidden_size]
        print(context_A)
        with tf.variable_scope('attention_B2A_2'):
            context_A = self.attention_2(encoder_output_B, context_A, mask_B)#context_A=[batch_size,hidden_size]
        return context_A
    
    def prediction_B(self, num_items_B, encoder_state_B, context_A, keep_prob, hidden_size):
        with tf.variable_scope('prediction_B'):
            concat_output = tf.concat([encoder_state_B[-1], context_A], axis=-1)#concat_output=[batch_size,hidden_size*2]                                                                                                           
            print(concat_output)
            concat_output = tf.nn.dropout(concat_output, keep_prob)#concat_output=[batch_size,hidden_size*2]  
            output_weight_B = tf.get_variable(dtype=tf.float32, name='output_weight_B', shape=[hidden_size*2, num_items_B], initializer=tf.contrib.layers.xavier_initializer(uniform=False))            
            output_bias_B = tf.get_variable(dtype=tf.float32, name='output_bias_B', shape=[num_items_B,], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            pred_B = tf.matmul(concat_output, output_weight_B)+output_bias_B#pred_B=[batch_size,num_items_B]
            print(pred_B)
        return pred_B
    
    def cal_loss(self, target_A, pred_A, target_B, pred_B):
        loss_A = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_A, logits=pred_A)#Because we use sparse_softmax_cross_entropy_with_logits to calculate loss, we don't use softmax(pred).
        #loss_A = tf.contrib.keras.losses.sparse_categorical_crossentropy(target_A, pred_A)#If pred_A has been softmax (target_A=[batch_size,1])        
        loss_A = tf.reduce_mean(loss_A, name='loss_A')
        loss_B = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_B, logits=pred_B)
        #loss_B = tf.contrib.keras.losses.sparse_categorical_crossentropy(target_B, pred_B)#If pred_B has been softmax (target_B=[batch_size,1])        
        loss_B = tf.reduce_mean(loss_B, name='loss_B')
        loss = loss_A+loss_B
        return loss
    
    def optimizer(self, loss, learning_rate):
        optimizer = tf.train.AdamOptimizer(learning_rate)                                
        gradients = optimizer.compute_gradients(loss)
        capped_gradients = [(tf.clip_by_value(grad,-5.,5.),var) for grad,var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
        return train_op

    def train_recommender(self, sess, seq_A, seq_B, pos_A, pos_B, len_A, len_B, target_A, target_B, mask_A, mask_B,
                          mask_A2B, mask_B2A, learning_rate, keep_prob):
        feed_dict = {self.seq_A: seq_A, self.seq_B: seq_B, self.pos_A: pos_A,
                     self.pos_B: pos_B, self.len_A: len_A, self.len_B: len_B,
                     self.target_A: target_A, self.target_B: target_B,
                     self.mask_A: mask_A, self.mask_B: mask_B,
                     self.mask_A2B: mask_A2B, self.mask_B2A: mask_B2A,
                     self.learning_rate: learning_rate,
                     self.keep_prob: keep_prob}

        return sess.run([self.train_op, self.loss], feed_dict)

    def eval_recommender(self, sess, seq_A, seq_B, pos_A, pos_B, len_A, len_B, target_A, target_B, mask_A, mask_B,
                          mask_A2B, mask_B2A, learning_rate):
        feed_dict = {self.seq_A: seq_A, self.seq_B: seq_B, self.pos_A: pos_A,
                     self.pos_B: pos_B, self.len_A: len_A, self.len_B: len_B,
                     self.target_A: target_A, self.target_B: target_B,
                     self.mask_A: mask_A, self.mask_B: mask_B,
                     self.mask_A2B: mask_A2B, self.mask_B2A: mask_B2A,
                     self.learning_rate: learning_rate, self.keep_prob: 1.0}
        return sess.run([self.pred_A, self.pred_B], feed_dict)