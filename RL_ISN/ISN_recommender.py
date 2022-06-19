# _*_coding: UTF_8 _*_
from __future__ import absolute_import
from __future__ import division

import os

import tensorflow as tf
from RL_ISN.ISN_configuration import *

args = setting()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# The definition of recommender network for 'RL_ISN' model
class RecommenderNetwork(object):
    # Initialize the recommender network in 'ISN_main.py' before the training starts.
    def __init__(self, items_max_num_A, items_max_num_B, args, padding_A, padding_B, dict_A, dict_B, max_len_pos):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
        self.recommender_lr = args.recommender_pre_lr
        self.recommender_tau = args.recommender_pre_tau
        self.is_training = True
        self.items_max_num_A = items_max_num_A
        self.items_max_num_B = items_max_num_B

        self.padding_A = padding_A
        self.padding_B = padding_B

        self.dict_A = dict_A
        self.dict_B = dict_B

        self.embedding_size = args.recommender_embedding_size
        self.account_size_A = args.account_num_A
        self.account_size_B = args.account_num_B
        self.weight_size = self.embedding_size

        self.beta = args.beta
        self.regs = args.regs
        self.lambda_bilinear = args.regs[0]
        self.gamma_bilinear = args.regs[1]
        self.eta_bilinear = args.regs[2]
        self.sess = None
        self.dropout_rate = args.dropout_rate

        self.max_len_pos = max_len_pos
        self.delta = args.delta

        self.graph = tf.Graph()

        with self.graph.as_default():
            with tf.name_scope('inputs'):
                self.global_step = tf.Variable(0, trainable=False, name="Attention_step")
                self.num_other_variables = len(tf.trainable_variables())
                self.get_recommender_inputs()
                # Activate the recommender
                self.origin_seq_A, self.origin_len_A, self.target_item_A, \
                self.output_prediction_A, _, _, _, self.origin_seq_B, self.origin_len_B, self.target_item_B, \
                self.output_prediction_B, _, _, _ = self.create_recommender_network("Active")
                # Train the recommender
                self.train_origin_seq_A, self.train_origin_len_A, self.train_target_item_A, self.train_output_A, \
                self.train_ebd_user_A, self.train_ebd_item_A, self.train_weights_A, self.train_origin_seq_B, self.train_origin_len_B, self.train_target_item_B, self.train_output_B, \
                self.train_ebd_user_B, self.train_ebd_item_B, self.train_weights_B = self.create_recommender_network(
                    "Train")

                print("len_network_params_A:{}".format(len(self.network_params_A)))
                print("len_train_network_params_A:{}".format(len(self.train_network_params_A)))
                # % delayed updating recommender network ops
                self.assign_params_with_tau_A = [
                    self.train_network_params_A[i].assign(tf.multiply(self.network_params_A[i], self.recommender_tau) +
                                                          tf.multiply(self.train_network_params_A[i],
                                                                      1 - self.recommender_tau))
                    for i in range(len(self.train_network_params_A))]

                self.assign_params_with_tau_B = [self.train_network_params_B[i].assign(
                    tf.multiply(self.network_params_B[i], self.recommender_tau) +
                    tf.multiply(self.train_network_params_B[i], 1 - self.recommender_tau))
                    for i in range(len(self.train_network_params_B))]

                # % network parameters --> target network parameters
                self.assign_params_without_tau_A = [self.train_network_params_A[i].assign(self.network_params_A[i])
                                                    for i in range(len(self.train_network_params_A))]

                self.assign_params_without_tau_B = [self.train_network_params_B[i].assign(self.network_params_B[i])
                                                    for i in range(len(self.train_network_params_B))]

                # % target network parameters -->  network parameters
                self.assign_active_network_params_A = [self.network_params_A[i].assign(self.train_network_params_A[i])
                                                       for i in range(len(self.network_params_A))]

                self.assign_active_network_params_B = [self.network_params_B[i].assign(self.train_network_params_B[i])
                                                       for i in range(len(self.network_params_B))]

            with tf.name_scope("reward"):
                # for Domain_A
                self.one_minus_output_A = 1.0 - self.train_output_A
                self.reward_output_concat_A = tf.concat([self.one_minus_output_A, self.train_output_A], 1)
                self.classes_A = tf.constant(2)
                self.labels_reduce_dim_A = tf.reduce_sum(self.labels_A, 1)
                self.one_hot_A = tf.one_hot(self.labels_reduce_dim_A, self.classes_A)
                self.reward_B_2_A = tf.log(tf.reduce_sum((self.reward_output_concat_A * self.one_hot_A + 1e-15), 1))
                # for Domain_B
                self.one_minus_output_B = 1.0 - self.train_output_B
                self.reward_output_concat_B = tf.concat([self.one_minus_output_B, self.train_output_B], 1)
                self.classes_B = tf.constant(2)
                self.labels_reduce_dim_B = tf.reduce_sum(self.labels_B, 1)
                self.one_hot_B = tf.one_hot(self.labels_reduce_dim_B, self.classes_B)
                self.reward_A_2_B = tf.log(tf.reduce_sum((self.reward_output_concat_B * self.one_hot_B + 1e-15), 1))

            with tf.name_scope('loss'):
                ####loss A####
                self.l2loss_A = 0
                self.l2loss_A = self.lambda_bilinear * tf.reduce_sum(tf.square(self.train_ebd_item_A)) + \
                                self.gamma_bilinear * tf.reduce_sum(tf.square(self.train_ebd_user_A)) + \
                                self.eta_bilinear * tf.reduce_sum(tf.square(self.train_weights_A))
                self.loss_A = tf.losses.log_loss(self.labels_A, self.train_output_A)
                self.loss_A += self.l2loss_A
                self.gradients_A = tf.gradients(self.loss_A, self.train_network_params_A)
                self.optimizer_A = tf.train.AdagradOptimizer(self.recommender_lr,
                                                             initial_accumulator_value=1e-8).apply_gradients(
                    zip(self.gradients_A, self.network_params_A), global_step=self.global_step)

                ####loss B####
                self.l2loss_B = 0
                self.l2loss_B = self.lambda_bilinear * tf.reduce_sum(tf.square(self.train_ebd_item_B)) + \
                                self.gamma_bilinear * tf.reduce_sum(tf.square(self.train_ebd_user_B)) + \
                                self.eta_bilinear * tf.reduce_sum(tf.square(self.train_weights_B))
                self.loss_B = tf.losses.log_loss(self.labels_B, self.train_output_B)
                self.loss_B += self.l2loss_B
                self.gradients_B = tf.gradients(self.loss_B, self.train_network_params_B)
                self.optimizer_B = tf.train.AdagradOptimizer(self.recommender_lr,
                                                             initial_accumulator_value=1e-8).apply_gradients(
                    zip(self.gradients_B, self.network_params_B), global_step=self.global_step)

    def create_variables_A(self, scope):
        with tf.name_scope(scope):
            self.c1_A_temp = tf.Variable(
                tf.truncated_normal(shape=[self.items_max_num_A, self.embedding_size], mean=0.0, stddev=0.01),
                name='c1_A',
                dtype=tf.float32, trainable=True)
            self.c1_A = tf.concat((self.c1_A_temp, tf.zeros(shape=[1, self.embedding_size])), 0)
            self.c2_A = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2_A')
            self.embedding_user_A = tf.concat([self.c1_A, self.c2_A], 0, name='embedding_user_A')

            self.embedding_item_A_temp = tf.Variable(
                tf.truncated_normal(shape=[self.items_max_num_A, self.embedding_size], mean=0.0, stddev=0.01),
                name='embedding_item_A_temp', dtype=tf.float32, trainable=True)
            self.embedding_item_A = tf.concat((self.embedding_item_A_temp, tf.zeros(shape=[1, self.embedding_size])), 0,
                                              name='embedding_item_A')

            self.bias_A_ = tf.Variable(tf.zeros(self.items_max_num_A), name='bias_A', trainable=True)
            self.bias_A = tf.concat((self.bias_A_, tf.zeros(1, )), 0)

            self.mlp_weights_A = tf.Variable(
                tf.truncated_normal(shape=[self.embedding_size, self.weight_size], mean=0.0,
                                    stddev=tf.sqrt(
                                        tf.div(2.0, self.weight_size + self.embedding_size))),
                name='Weights_for_Attetion_A', dtype=tf.float32, trainable=True)

            self.b_A = tf.Variable(tf.truncated_normal(shape=[1, self.embedding_size], mean=0.0, stddev=tf.sqrt(
                tf.div(2.0, self.weight_size + self.embedding_size))), name='Bias_for_MLP_A', dtype=tf.float32,
                                   trainable=True)
            self.h_A = tf.Variable(tf.ones([self.weight_size, 1]), name='H_for_MLP', dtype=tf.float32)

    def create_variables_B(self, scope):
        with tf.name_scope(scope):
            self.c1_B_ = tf.Variable(
                tf.truncated_normal(shape=[self.items_max_num_B, self.embedding_size], mean=0.0, stddev=0.01),
                name='c1_B',
                dtype=tf.float32, trainable=True)
            self.c1_B = tf.concat((self.c1_B_, tf.zeros(shape=[1, self.embedding_size])), 0)
            self.c2_B = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2_B')
            self.embedding_user_B = tf.concat([self.c1_B, self.c2_B], 0, name='embedding_user_B')
            # self.embedding_position_B = tf.concat([self.c1_B, self.c2_B], 0, name='embedding_pos_B')
            self.embedding_item_B_temp = tf.Variable(
                tf.truncated_normal(shape=[self.items_max_num_B, self.embedding_size], mean=0.0, stddev=0.01),
                name='embedding_item_B_temp', dtype=tf.float32, trainable=True)
            self.embedding_item_B = tf.concat((self.embedding_item_B_temp, tf.zeros(shape=[1, self.embedding_size])), 0,
                                              name='embedding_item_B')
            self.bias_B_ = tf.Variable(tf.zeros(self.items_max_num_B), name='bias_B', trainable=True)
            self.bias_B = tf.concat((self.bias_B_, tf.zeros(1, )), 0)

            # Variables for attention
            self.mlp_weights_B = tf.Variable(
                tf.truncated_normal(shape=[self.embedding_size, self.weight_size], mean=0.0, stddev=tf.sqrt(
                    tf.div(2.0, self.weight_size + self.embedding_size))),
                name='Weights_for_MLP_B', dtype=tf.float32, trainable=True)
            self.b_B = tf.Variable(tf.truncated_normal(shape=[1, self.weight_size], mean=0.0, stddev=tf.sqrt(
                tf.div(2.0, self.weight_size + self.embedding_size))), name='Bias_for_MLP_B', dtype=tf.float32,
                                   trainable=True)
            self.h_B = tf.Variable(tf.ones([self.weight_size, 1]), name='H_for_MLP_B', dtype=tf.float32)

    def network_preprocessing_A(self, scope):
        with tf.name_scope(scope):
            ###############  A  ###########################
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                self.pos_table_embedding_A = tf.get_variable('position_table_A',
                                                           dtype=tf.float32,
                                                           shape=[self.max_len_pos, self.embedding_size],
                                                           initializer=tf.contrib.layers.xavier_initializer())
            self.processed_ebd_user_A = tf.nn.embedding_lookup(params=self.embedding_user_A, ids=self.origin_seq_A)
            self.processed_ebd_pos_A = tf.nn.embedding_lookup(params=self.pos_table_embedding_A,
                                                              ids=self.origin_pos_A)  # positional Encoding
            self.processed_ebd_user_A = (1-self.delta) * self.processed_ebd_user_A + \
                                        self.delta * self.processed_ebd_pos_A  # new

            self.processed_ebd_user_A = tf.layers.dropout(self.processed_ebd_user_A, self.dropout_rate,
                                                          training=self.is_training)

            ###############  B  ###########################
            self.selected_ebd_user_B = tf.nn.embedding_lookup(params=self.embedding_user_B, ids=self.selected_seq_B)
            self.selected_ebd_pos_B = tf.nn.embedding_lookup(params=self.pos_table_embedding_A, ids=self.origin_pos_B)
            self.selected_ebd_user_B = (1-self.delta) * self.selected_ebd_user_B + \
                                       self.delta * self.selected_ebd_pos_B  # new

            self.selected_ebd_user_B = tf.layers.dropout(self.selected_ebd_user_B, self.dropout_rate,
                                                         training=self.is_training)

            # target_A
            self.processed_ebd_item_A = tf.nn.embedding_lookup(params=self.embedding_item_A, ids=self.target_item_A)
            # Attention_network_A for both seqs
            calculated_ebd_A = self.attention_network_A(self.processed_ebd_user_A * self.processed_ebd_item_A, scope,
                                                        True)
            calculated_ebd_B = self.attention_network_A(self.selected_ebd_user_B * self.processed_ebd_item_A, scope,
                                                        False)
            self.sum_embedding_A = tf.concat([calculated_ebd_A, calculated_ebd_B], axis=1)
            self.processed_ebd_item_A = tf.reduce_sum(self.processed_ebd_item_A, 1)
            self.temp_processed_ebd_item_A = tf.concat([self.processed_ebd_item_A, self.processed_ebd_item_A],
                                                       axis=1)

            self.bias_i_A = tf.nn.embedding_lookup(self.bias_A, self.target_item_A)
            self.coeff_A = tf.pow(self.origin_len_A + self.selected_len_B, tf.constant(1, tf.float32, [1]))

            self.output_prediction_A = tf.sigmoid(
                self.coeff_A * tf.expand_dims(tf.reduce_sum(self.sum_embedding_A * self.temp_processed_ebd_item_A, 1),
                                              1) + self.bias_i_A)

    def network_preprocessing_B(self, scope):
        with tf.name_scope(scope):
            ############# B #################
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                self.pos_table_embedding_B = tf.get_variable('position_table_B',
                                                           dtype=tf.float32,
                                                           shape=[self.max_len_pos, self.embedding_size],
                                                           initializer=tf.contrib.layers.xavier_initializer())
            self.processed_ebd_user_B = tf.nn.embedding_lookup(self.embedding_user_B, self.origin_seq_B)
            self.processed_ebd_pos_B = tf.nn.embedding_lookup(self.pos_table_embedding_B,
                                                              self.origin_pos_B)  # positional Encoding
            self.processed_ebd_user_B = (1-self.delta) * self.processed_ebd_user_B + \
                                        self.delta * self.processed_ebd_pos_B  # new

            self.processed_ebd_user_B = tf.layers.dropout(self.processed_ebd_user_B, self.dropout_rate,
                                                          training=self.is_training)

            ############# A #################
            self.selected_ebd_user_A = tf.nn.embedding_lookup(self.embedding_user_A, self.selected_seq_A)
            self.selected_ebd_pos_A = tf.nn.embedding_lookup(self.pos_table_embedding_B,
                                                             self.origin_pos_A)  # positional Encoding
            self.selected_ebd_user_A = (1-self.delta) * self.selected_ebd_user_A + \
                                       self.delta * self.selected_ebd_pos_A  # new

            self.selected_ebd_user_A = tf.layers.dropout(self.selected_ebd_user_A, self.dropout_rate,
                                                         training=self.is_training)

            # target_B
            self.processed_ebd_item_B = tf.nn.embedding_lookup(self.embedding_item_B, self.target_item_B)

            calculated_ebd_A = self.attention_network_B(self.selected_ebd_user_A * self.processed_ebd_item_B, scope,
                                                        True)
            calculated_ebd_B = self.attention_network_B(self.processed_ebd_user_B * self.processed_ebd_item_B,
                                                        scope)
            self.sum_embedding_B = tf.concat([calculated_ebd_B, calculated_ebd_A], axis=1)
            self.processed_ebd_item_B = tf.reduce_sum(self.processed_ebd_item_B, 1)
            self.temp_processed_ebd_item_B = tf.concat([self.processed_ebd_item_B, self.processed_ebd_item_B],
                                                       axis=1)

            self.bias_i_B = tf.nn.embedding_lookup(self.bias_B, self.target_item_B)
            self.coeff_B = tf.pow(self.origin_len_B + self.selected_len_A,
                                  tf.constant(1, tf.float32, [1]))
            self.output_prediction_B = tf.sigmoid(
                self.coeff_B * tf.expand_dims(tf.reduce_sum(self.sum_embedding_B * self.temp_processed_ebd_item_B, 1),
                                              1) + self.bias_i_B)

    def attention_network_A(self, input_embedding, scope, isA=False):
        with tf.name_scope(scope):
            if scope is "Train":
                input_embedding = tf.layers.dense(input_embedding, self.account_size_A)
                input_embedding = tf.layers.dense(input_embedding, self.embedding_size)
            origin_shape_row = tf.shape(input_embedding)[0]
            origin_shape_column = tf.shape(input_embedding)[1]

            mlp_output_A = tf.nn.relu(tf.matmul(tf.reshape(input_embedding, [-1, self.embedding_size]),
                                                self.mlp_weights_A) + self.b_A)  # (b*n, e or 2*e) * (e or 2*e, w) + (1, w)
            self.A__A = tf.reshape(tf.matmul(mlp_output_A, self.h_A),
                                   [origin_shape_row, origin_shape_column])  # (b*n, w) * (w, 1) => (None, 1) => (b, n)

            exp_A_ = tf.exp(self.A__A)
            num_idx = tf.reduce_sum(self.origin_len_A, 1)
            mask_mat = tf.sequence_mask(num_idx, maxlen=origin_shape_column, dtype=tf.float32)  # (b, n)
            exp_A_ = mask_mat * exp_A_
            exp_sum = tf.reduce_sum(exp_A_, 1, keepdims=True)  # (b, 1) #keep_dims is deprecated, use keepdims instead
            exp_sum = tf.pow(exp_sum, tf.constant(self.beta, tf.float32, [1]))

            self.A_A = tf.expand_dims(tf.div(exp_A_, exp_sum), 2)  # (b, n, 1)
            if isA:
                return tf.reduce_sum(self.A_A * self.processed_ebd_user_A, 1)
            else:
                return tf.reduce_sum(self.A_A * self.selected_ebd_user_B, 1)

    def attention_network_B(self, input_embedding, scope, isA=False):
        with tf.name_scope(scope):
            if scope is "Train":
                input_embedding = tf.layers.dense(input_embedding, self.account_size_B)
                input_embedding = tf.layers.dense(input_embedding, self.embedding_size)
            origin_shape_row = tf.shape(input_embedding)[0]
            input_column = tf.shape(input_embedding)[1]

            self.embedding_B = input_embedding

            mlp_output_B = tf.nn.relu(tf.matmul(tf.reshape(input_embedding, [-1, self.embedding_size]),
                                                self.mlp_weights_B) + self.b_B)  # (b*n, e or 2*e) * (e or 2*e, w) + (1, w)

            self.A__B = tf.reshape(tf.matmul(mlp_output_B, self.h_B),
                                   [origin_shape_row, input_column])  # (b*n, w) * (w, 1) => (None, 1) => (b, n)

            # softmax for not mask features
            exp_A_ = tf.exp(self.A__B)
            num_idx = tf.reduce_sum(self.origin_len_B, 1)
            mask_mat = tf.sequence_mask(num_idx, maxlen=input_column, dtype=tf.float32)  # (b, n)
            exp_A_ = mask_mat * exp_A_
            exp_sum = tf.reduce_sum(exp_A_, 1, keepdims=True)  # (b, 1) #keep_dims is deprecated, use keepdims instead
            exp_sum = tf.pow(exp_sum, tf.constant(self.beta, tf.float32, [1]))

            self.A_B = tf.expand_dims(tf.div(exp_A_, exp_sum), 2)  # (b, n, 1)
            if isA:
                return tf.reduce_sum(self.A_B * self.selected_ebd_user_A, 1)
            else:
                return tf.reduce_sum(self.A_B * self.processed_ebd_user_B, 1)

    def create_recommender_network(self, scope):
        # At May, 2022, we have already format this block by diff the scope of "Active" and "train"
        if scope == "Active":
            self.create_variables_A(scope)
            self.network_params_A = tf.trainable_variables()[self.num_other_variables:]
            self.create_variables_B(scope)
            self.network_params_B = tf.trainable_variables()[self.num_other_variables + len(self.network_params_A):]
            self.network_preprocessing_A(scope)
            self.network_other_param_temp_active_A = tf.trainable_variables()[self.num_other_variables +
                                                                              len(self.network_params_A) + len(
                self.network_params_B):]
            self.network_preprocessing_B(scope)
            self.network_other_param_temp_active_B = tf.trainable_variables()[self.num_other_variables +
                                                                              len(self.network_params_A) +
                                                                              len(self.network_params_B) + len(
                self.network_other_param_temp_active_A):]
        else:  # scope == 'train'
            self.create_variables_A(scope)
            self.train_network_params_A = tf.trainable_variables()[self.num_other_variables +
                                                                   len(self.network_params_A) +
                                                                   len(self.network_params_B) + len(
                self.network_other_param_temp_active_A)
                                                                   + len(self.network_other_param_temp_active_B):]
            self.create_variables_B(scope)
            self.train_network_params_B = tf.trainable_variables()[self.num_other_variables +
                                                                   len(self.network_params_A) +
                                                                   len(self.network_params_B) + len(
                self.network_other_param_temp_active_A)
                                                                   + len(self.network_other_param_temp_active_B)
                                                                   + len(self.train_network_params_A):]
            self.network_preprocessing_A(scope)
            self.network_other_param_temp_train_A = tf.trainable_variables()[self.num_other_variables +
                                                                             len(self.network_params_A) +
                                                                             len(self.network_params_B) + len(
                self.network_other_param_temp_active_A)
                                                                             + len(
                self.network_other_param_temp_active_B)
                                                                             + len(self.train_network_params_A)
                                                                             + len(self.train_network_params_B):]
            self.network_preprocessing_B(scope)
            self.network_other_param_temp_train_B = tf.trainable_variables()[self.num_other_variables +
                                                                             len(self.network_params_A) +
                                                                             len(self.network_params_B) + len(
                self.network_other_param_temp_active_A)
                                                                             + len(
                self.network_other_param_temp_active_B)
                                                                             + len(self.train_network_params_A)
                                                                             + len(self.train_network_params_B)
                                                                             + len(
                self.network_other_param_temp_train_A):]
            # When the scope is "train", we have to sum up all the parameter for both domain-A and B respectively. May, 2022
            self.network_params_A += self.network_other_param_temp_active_A
            self.network_params_B += self.network_other_param_temp_active_B
            self.train_network_params_A += self.network_other_param_temp_active_A
            self.train_network_params_B += self.network_other_param_temp_active_B

        return self.origin_seq_A, self.origin_len_A, self.target_item_A, self.output_prediction_A, \
               self.processed_ebd_user_A, self.temp_processed_ebd_item_A, self.mlp_weights_A, \
               self.origin_seq_B, self.origin_len_B, self.target_item_B, self.output_prediction_B, \
               self.processed_ebd_user_B, self.temp_processed_ebd_item_B, self.mlp_weights_B

    def assign_train_params_with_tau_A(self):
        self.sess.run(self.assign_params_with_tau_A)

    def assign_train_params_with_tau_B(self):
        self.sess.run(self.assign_params_with_tau_B)

    def assign_train_params_without_tau_A(self):
        self.sess.run(self.assign_params_without_tau_A)

    def assign_train_params_without_tau_B(self):
        self.sess.run(self.assign_params_without_tau_B)

    def assign_active_network_A(self):
        self.sess.run(self.assign_active_network_params_A)

    def assign_active_network_B(self):
        self.sess.run(self.assign_active_network_params_B)

    def getloss_A(self, user_input, num_idx, item_input, labels):
        feed_dict = {self.train_origin_seq_A: user_input, self.train_origin_len_A: num_idx,
                     self.train_target_item_A: item_input,
                     self.labels_A: labels}
        return self.sess.run(self.loss_A, feed_dict)

    def getloss_B(self, user_input, num_idx, item_input, labels):
        feed_dict = {self.train_origin_seq_B: user_input, self.train_origin_len_B: num_idx,
                     self.train_target_item_B: item_input,
                     self.labels_B: labels}
        return self.sess.run(self.loss_B, feed_dict)

    def train_recommender_A(self, origin_seq_A, origin_seq_B, origin_length_A, origin_length_B,
                            target_item_A, target_item_B, labels_A, labels_B, selected_user_input_B,
                            selected_num_idx_B, is_training, pos_A, pos_B):
        feed_dict = {self.train_origin_seq_A: origin_seq_A, self.train_origin_seq_B: origin_seq_B,
                     self.train_origin_len_A: origin_length_A, self.train_origin_len_B: origin_length_B,
                     self.train_target_item_A: target_item_A, self.train_target_item_B: target_item_B,
                     self.labels_A: labels_A, self.labels_B: labels_B,
                     self.selected_seq_B: selected_user_input_B, self.selected_len_B: selected_num_idx_B,
                     self.is_training: is_training, self.origin_pos_A: pos_A, self.origin_pos_B: pos_B}

        return self.sess.run([self.loss_A, self.optimizer_A], feed_dict)

    def train_recommender_B(self, origin_seq_A, origin_seq_B, origin_length_A, origin_length_B,
                            target_item_A, target_item_B, labels_A, labels_B, selected_user_input_A,
                            selected_num_idx_A, is_training, pos_A, pos_B):
        feed_dict = {self.train_origin_seq_A: origin_seq_A, self.train_origin_seq_B: origin_seq_B,
                     self.train_origin_len_A: origin_length_A, self.train_origin_len_B: origin_length_B,
                     self.train_target_item_A: target_item_A, self.train_target_item_B: target_item_B,
                     self.labels_A: labels_A, self.labels_B: labels_B,
                     self.selected_seq_A: selected_user_input_A, self.selected_len_A: selected_num_idx_A,
                     self.is_training: is_training,
                     self.origin_pos_A: pos_A, self.origin_pos_B: pos_B}

        return self.sess.run([self.loss_B, self.optimizer_B], feed_dict)

    def predict_A(self, test_user_input_A, test_user_input_B, test_num_idx_A, test_num_idx_B, test_target_A,
                  test_target_B, test_label_A, test_label_B, selected_user_input_B, selected_num_idx_B, is_training,
                  pos_A, pos_B):
        feed_dict = {self.origin_seq_A: test_user_input_A, self.origin_seq_B: test_user_input_B,
                     self.origin_len_A: test_num_idx_A, self.origin_len_B: test_num_idx_B,
                     self.target_item_A: test_target_A, self.target_item_B: test_target_B,
                     self.labels_A: test_label_A, self.labels_B: test_label_B,
                     self.selected_seq_B: selected_user_input_B, self.selected_len_B: selected_num_idx_B,
                     self.is_training: is_training, self.origin_pos_A: pos_A, self.origin_pos_B: pos_B}

        return self.sess.run([self.train_output_A, self.loss_A], feed_dict)

    def predict_B(self, test_user_input_A, test_user_input_B, test_num_idx_A, test_num_idx_B, test_target_A,
                  test_target_B, test_label_A, test_label_B, selected_user_input_A, selected_num_idx_A, is_training,
                  pos_A, pos_B):

        feed_dict = {self.origin_seq_A: test_user_input_A, self.origin_seq_B: test_user_input_B,
                     self.origin_len_A: test_num_idx_A, self.origin_len_B: test_num_idx_B,
                     self.target_item_A: test_target_A, self.target_item_B: test_target_B,
                     self.labels_A: test_label_A, self.labels_B: test_label_B,
                     self.selected_seq_A: selected_user_input_A, self.selected_len_A: selected_num_idx_A,
                     self.is_training: is_training,
                     self.origin_pos_A: pos_A, self.origin_pos_B: pos_B}

        return self.sess.run([self.train_output_B, self.loss_B], feed_dict)

    def get_data_embedding_A(self):
        data_embedding_user_A = self.sess.run(self.embedding_user_A)
        data_embedding_item_A = self.sess.run(self.embedding_item_A)
        return np.array(data_embedding_user_A), np.array(data_embedding_item_A)

    def get_data_embedding_B(self):
        data_embedding_user_B = self.sess.run(self.embedding_user_B)
        data_embedding_item_B = self.sess.run(self.embedding_item_B)
        return np.array(data_embedding_user_B), np.array(data_embedding_item_B)

    def reward_from_recommender(self, selected_from_domain, seq_A, len_A, seq_B, len_B, s_seq_A, s_len_A, s_seq_B,
                                s_len_B, target_A, target_B, label_A, label_B, is_training, pos_A, pos_B):
        feed_dict = {self.origin_seq_A: seq_A, self.origin_len_A: len_A,
                     self.selected_seq_B: s_seq_B, self.selected_len_B: s_len_B,
                     self.origin_seq_B: seq_B, self.origin_len_B: len_B,
                     self.selected_seq_A: s_seq_A, self.selected_len_A: s_len_A,
                     self.target_item_A: target_A, self.target_item_B: target_B,
                     self.labels_A: label_A, self.labels_B: label_B, self.is_training: is_training,
                     self.origin_pos_A: pos_A, self.origin_pos_B: pos_B}

        if selected_from_domain == "A":
            return self.sess.run([self.reward_A_2_B, self.labels_B, self.output_prediction_B], feed_dict)
        else:
            return self.sess.run([self.reward_B_2_A, self.labels_A, self.output_prediction_A], feed_dict)

    def get_origin_rewards(self, dataset, selected_from_domain):
        seq_A, seq_B, pos_A, pos_B, len_A, len_B, target_A, target_B, labels_A, labels_B, \
        num_batch = dataset[0], dataset[1], dataset[2], dataset[3], dataset[4], \
                    dataset[5], dataset[6], dataset[7], dataset[8], dataset[9], dataset[10]

        batch_reward_sum = []

        item_sets_A = list(self.dict_A.values())
        item_sets_B = list(self.dict_B.values())

        for batch_index in range(num_batch):
            batched_seq_A = np.array([seq for seq in seq_A[batch_index]])
            batched_target_A = np.reshape(target_A[batch_index], (-1, 1))
            batched_label_A = np.reshape(labels_A[batch_index], (-1, 1))
            batched_len_A = np.reshape(len_A[batch_index], (-1, 1))
            batched_pos_A = np.array([pos for pos in pos_A[batch_index]])
            batched_seq_B = np.array([seq for seq in seq_B[batch_index]])
            batched_target_B = np.reshape(target_B[batch_index], (-1, 1))
            batched_label_B = np.reshape(labels_B[batch_index], (-1, 1))
            batched_len_B = np.reshape(len_B[batch_index], (-1, 1))
            batched_pos_B = np.array([pos for pos in pos_B[batch_index]])

            s_batched_seq_A = np.zeros(batched_seq_A.shape) + np.random.choice(item_sets_A, 1, replace=True)[0]
            s_len_A = batched_len_A
            s_batched_seq_B = np.zeros(batched_seq_B.shape) + np.random.choice(item_sets_B, 1, replace=True)[0]
            s_len_B = batched_len_B

            batch_origin_reward, _, _ = \
                self.reward_from_recommender(selected_from_domain=selected_from_domain,
                                             seq_A=batched_seq_A, len_A=batched_len_A,
                                             seq_B=batched_seq_B, len_B=batched_len_B,
                                             s_seq_A=s_batched_seq_A, s_len_A=s_len_A,
                                             s_seq_B=s_batched_seq_B, s_len_B=s_len_B,
                                             target_A=batched_target_A, target_B=batched_target_B,
                                             label_A=batched_label_A, label_B=batched_label_B, is_training=True,
                                             pos_A=batched_pos_A, pos_B=batched_pos_B)

            batch_reward_sum.append(batch_origin_reward)

        return np.array(batch_reward_sum)

    def get_recommender_inputs(self):
        self.origin_seq_A = tf.placeholder(dtype=tf.int32, shape=[None, None], name='origin_seq_A')
        self.origin_seq_B = tf.placeholder(dtype=tf.int32, shape=[None, None], name='origin_seq_B')
        self.origin_len_A = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='origin_len_A')
        self.target_item_A = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='target_item_A')
        self.selected_seq_B = tf.placeholder(dtype=tf.int32, shape=[None, None], name='selected_seq_B')
        self.selected_len_B = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='selected_len_B')
        self.origin_len_B = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='origin_len_B')
        self.target_item_B = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='target_item_B')
        # We have added the origin_pos_A and B for injecting the postional encoding for both domains. May, 2022
        self.selected_seq_A = tf.placeholder(dtype=tf.int32, shape=[None, None], name='selected_seq_A')
        self.selected_len_A = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='selected_len_A')
        self.origin_pos_A = tf.placeholder(dtype=tf.int32, shape=[None, None], name="pos_A")
        self.origin_pos_B = tf.placeholder(dtype=tf.int32, shape=[None, None], name="pos_B")
        self.labels_A = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='labels_A')
        self.labels_B = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='labels_B')
        self.is_training = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')

    def update_tau_4_joint(self, joint_tau):
        self.recommender_tau = joint_tau

    def update_lr_4_joint(self, joint_lr):
        self.recommender_lr = joint_lr

