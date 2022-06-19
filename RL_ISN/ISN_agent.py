# _*_coding: UTF_8 _*_
import tensorflow as tf
from RL_ISN.ISN_configuration import *
# Read the Hyper-parameters from 'ISN_parameters.py'
args = setting()


# The definition of Agent network for 'RL_ISN' model
class AgentNetwork(object):
    # Initialize the Agent network in 'ISN_main.py' before the training starts.
    def __init__(self, sess):
        self.global_step = tf.Variable(0, trainable=False, name="AgentStep")
        self.sess = sess
        self.agent_pretrain_lr_low = args.agent_pretrain_lr_low
        self.agent_pretrain_lr_high = args.agent_pretrain_lr_high
        self.learning_rate_l = tf.train.exponential_decay(self.agent_pretrain_lr_low, self.global_step,
                                                          decay_steps=args.decay_steps,
                                                          decay_rate=args.decay_rate, staircase=args.staircase)
        self.learning_rate_h = tf.train.exponential_decay(self.agent_pretrain_lr_high, self.global_step,
                                                          decay_steps=args.decay_steps,
                                                          decay_rate=args.decay_rate, staircase=args.staircase)
        self.agent_pretrain_tau_high = args.agent_pretrain_tau_high
        self.agent_pretrain_tau_low = args.agent_pretrain_tau_low
        self.high_state_size = args.high_state_size
        self.A_state_size = args.A_state_size
        self.B_state_size = args.B_state_size
        self.weight_size = args.agent_weight_size
        self.optimizer = tf.train.AdagradOptimizer(self.learning_rate_l)
        self.optimizer_h = tf.train.AdagradOptimizer(self.learning_rate_h)
        self.num_other_variables = len(tf.trainable_variables())

        '''
        high-level network  A
        '''

        # Agent network(updating)
        self.high_input_state_A, self.high_prob_A, _, _, _, _, _ = self.create_agent_network("Activate/high_A",
                                                                                             self.high_state_size)
        self.high_network_params_A = tf.trainable_variables()[self.num_other_variables:]

        # Agent network(delayed updating)
        self.target_high_input_state_A, self.target_high_prob_A, self.high_W_A, self.high_b_A, self.high_h_A, \
        self.high_out_A, self.high_keep_pro_A = self.create_agent_network("Target/high_A", self.high_state_size)
        self.target_high_network_params_A = tf.trainable_variables()[self.num_other_variables +
                                                                     len(self.high_network_params_A):]

        '''
        high-level network  B
        '''
        # Agent network(updating)
        self.high_input_state_B, self.high_prob_B, _, _, _, _, _ = self.create_agent_network("Activate/high_B",
                                                                                             self.high_state_size)
        self.high_network_params_B = tf.trainable_variables()[
                                     self.num_other_variables + len(self.high_network_params_A) + len(
                                         self.target_high_network_params_A):]

        # Agent network(delayed updating)
        self.target_high_input_state_B, self.target_high_prob_B, self.high_W_B, self.high_b_B, self.high_h_B, \
        self.high_out_B, self.high_keep_pro_B = self.create_agent_network("Target/high_B", self.high_state_size)
        self.target_high_network_params_B = tf.trainable_variables()[
                                            self.num_other_variables + len(self.high_network_params_A)
                                            + len(self.target_high_network_params_A) + len(self.high_network_params_B):]

        '''
        Agent A network
        '''
        # Agent network(updating)
        self.A_input_state, self.A_prob, _, _, _, _, _ = self.create_agent_network("Activate/A", self.A_state_size)
        self.A_network_params = tf.trainable_variables()[self.num_other_variables + len(self.high_network_params_A)
                                                         + len(self.target_high_network_params_A) + len(
            self.high_network_params_B)
                                                         + len(self.target_high_network_params_B):]

        # Agent network(delayed updating)
        self.target_A_input_state, self.target_low_prob_A, self.W_A, self.b_A, self.h_A, self.out_A, self.keep_pro_A \
            = self.create_agent_network("Target/A", self.A_state_size)
        self.target_A_network_params = tf.trainable_variables()[
                                       self.num_other_variables + len(self.high_network_params_A)
                                       + len(self.target_high_network_params_A) + len(self.high_network_params_B)
                                       + len(self.target_high_network_params_B) + len(self.A_network_params):]

        '''
        Agent B network
        '''
        # Agent network(updating)
        self.B_input_state, self.B_prob, _, _, _, _, _ = self.create_agent_network("Activate/B", self.B_state_size)
        self.B_network_params = tf.trainable_variables()[self.num_other_variables + len(self.high_network_params_A)
                                                         + len(self.target_high_network_params_A) + len(
            self.high_network_params_B)
                                                         + len(self.target_high_network_params_B) + len(
            self.A_network_params)
                                                         + len(self.target_A_network_params):]

        # Agent network(delayed updating)
        self.target_B_input_state, self.target_low_prob_B, self.W_B, self.b_B, self.h_B, self.out_B, self.keep_pro_B = \
            self.create_agent_network("Target/B", self.B_state_size)
        self.target_B_network_params = tf.trainable_variables()[
                                       self.num_other_variables + len(self.high_network_params_A)
                                       + len(self.target_high_network_params_A) + len(self.high_network_params_B)
                                       + len(self.target_high_network_params_B) + len(self.A_network_params)
                                       + len(self.target_A_network_params) + len(self.B_network_params):]

        # delayed updating Agent network
        self.update_target_high_network_params_A = \
            [self.target_high_network_params_A[i].assign(
                tf.multiply(self.high_network_params_A[i], self.agent_pretrain_tau_high) + \
                tf.multiply(self.target_high_network_params_A[i], 1 - self.agent_pretrain_tau_high)) \
                for i in range(len(self.target_high_network_params_A))]

        self.update_target_high_network_params_B = \
            [self.target_high_network_params_B[i].assign(
                tf.multiply(self.high_network_params_B[i], self.agent_pretrain_tau_high) + \
                tf.multiply(self.target_high_network_params_B[i], 1 - self.agent_pretrain_tau_high)) \
                for i in range(len(self.target_high_network_params_B))]

        self.update_target_A_network_params = \
            [self.target_A_network_params[i].assign(
                tf.multiply(self.A_network_params[i], self.agent_pretrain_tau_low) + \
                tf.multiply(self.target_A_network_params[i], 1 - self.agent_pretrain_tau_low)) \
                for i in range(len(self.target_A_network_params))]

        self.update_target_B_network_params = \
            [self.target_B_network_params[i].assign(
                tf.multiply(self.B_network_params[i], self.agent_pretrain_tau_low) + \
                tf.multiply(self.target_B_network_params[i], 1 - self.agent_pretrain_tau_low)) \
                for i in range(len(self.target_B_network_params))]

        self.assign_active_high_network_params_A = \
            [self.high_network_params_A[i].assign(self.target_high_network_params_A[i]) for i in
             range(len(self.high_network_params_A))]

        self.assign_active_high_network_params_B = \
            [self.high_network_params_B[i].assign(
                self.target_high_network_params_B[i]) for i in range(len(self.high_network_params_B))]

        self.assign_active_A_network_params = \
            [self.A_network_params[i].assign(self.target_A_network_params[i]) for i in
             range(len(self.A_network_params))]

        self.assign_active_B_network_params = \
            [self.B_network_params[i].assign(
                self.target_B_network_params[i]) for i in range(len(self.B_network_params))]

        self.high_reward_holder_A = tf.placeholder(shape=[None], dtype=tf.float32)
        self.high_action_holder_A = tf.placeholder(shape=[None], dtype=tf.float32)
        self.high_pi_A = self.high_action_holder_A * self.target_high_prob_A + (1 - self.high_action_holder_A) * (
                1 - self.target_high_prob_A)
        self.high_loss_A = -tf.reduce_sum(tf.log(self.high_pi_A) * self.high_reward_holder_A)
        self.high_gradients_A = tf.gradients(self.high_loss_A, self.target_high_network_params_A)

        self.high_reward_holder_B = tf.placeholder(shape=[None], dtype=tf.float32)
        self.high_action_holder_B = tf.placeholder(shape=[None], dtype=tf.float32)
        self.high_pi_B = self.high_action_holder_B * self.target_high_prob_B + (1 - self.high_action_holder_B) * (
                1 - self.target_high_prob_B)
        self.high_loss_B = -tf.reduce_sum(tf.log(self.high_pi_B) * self.high_reward_holder_B)
        self.high_gradients_B = tf.gradients(self.high_loss_B, self.target_high_network_params_B)

        self.A_reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.A_action_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.A_pi = self.A_action_holder * self.target_low_prob_A + (1 - self.A_action_holder) * (
                1 - self.target_low_prob_A)
        self.A_loss = -tf.reduce_sum(tf.log(self.A_pi) * self.A_reward_holder)
        self.A_gradients = tf.gradients(self.A_loss, self.target_A_network_params)

        self.B_reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.B_action_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.B_pi = self.B_action_holder * self.target_low_prob_B + (1 - self.B_action_holder) * (
                1 - self.target_low_prob_B)
        self.B_loss = -tf.reduce_sum(tf.log(self.B_pi) * self.B_reward_holder)
        self.B_gradients = tf.gradients(self.B_loss, self.target_B_network_params)

        self.high_gradient_holders_A = []
        for idx, var in enumerate(self.high_network_params_A):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.high_gradient_holders_A.append(placeholder)
        self.high_optimize_A = self.optimizer_h.apply_gradients(
            zip(self.high_gradient_holders_A, self.high_network_params_A),
            global_step=self.global_step)

        self.high_gradient_holders_B = []
        for idx, var in enumerate(self.high_network_params_B):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.high_gradient_holders_B.append(placeholder)
        self.high_optimize_B = self.optimizer_h.apply_gradients(
            zip(self.high_gradient_holders_B, self.high_network_params_B),
            global_step=self.global_step)

        # update parameters using gradient
        self.A_gradient_holders = []
        for idx, var in enumerate(self.A_network_params):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.A_gradient_holders.append(placeholder)
        self.A_optimize = self.optimizer.apply_gradients(zip(self.A_gradient_holders, self.A_network_params),
                                                         global_step=self.global_step)

        self.B_gradient_holders = []
        for idx, var in enumerate(self.B_network_params):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.B_gradient_holders.append(placeholder)
        self.B_optimize = self.optimizer.apply_gradients(zip(self.B_gradient_holders, self.B_network_params),
                                                         global_step=self.global_step)

    def udpate_agent_tau_4_joint(self, joint_agent_tau_low, joint_agent_tau_high):
        self.agent_pretrain_tau_high = joint_agent_tau_high
        self.agent_pretrain_tau_low = joint_agent_tau_low

    def update_agent_lr_4_joint(self, joint_agent_lr_low, joint_agent_lr_high):
        self.agent_pretrain_lr_low = joint_agent_lr_low
        self.agent_pretrain_lr_high = joint_agent_lr_high

    def create_agent_network(self, scope, state_size):
        with tf.variable_scope(scope):
            keep_pro = tf.placeholder(dtype=tf.float32, shape=[], name='keep_pro')
            input_state = tf.placeholder(shape=[None, state_size], dtype=tf.float32)
            embedding_size = state_size
            W = tf.Variable(tf.truncated_normal(shape=[embedding_size, self.weight_size], mean=0.0,
                                                stddev=tf.sqrt(tf.div(2.0, self.weight_size + embedding_size))),
                            name='Weights_for_MLP', dtype=tf.float32, trainable=True)
            b = tf.Variable(tf.constant(0, shape=[1, self.weight_size], dtype=tf.float32), name='Bias_for_MLP',
                            dtype=tf.float32, trainable=True)
            h = tf.Variable(
                tf.truncated_normal(shape=[self.weight_size, 1], mean=0.0,
                                    stddev=tf.sqrt(tf.div(2.0, self.weight_size))),
                name='H_for_MLP', dtype=tf.float32, trainable=True)
            MLP_output = tf.matmul(input_state, W) + b  # (b, e) * (e, w) + (1, w)
            MLP_output = tf.nn.relu(MLP_output)
            prob = tf.nn.sigmoid(
                tf.reduce_sum(tf.matmul(MLP_output, h), 1) + 1e-15)  # (b, w) * (w,1 ) => (b, 1)
            prob = tf.clip_by_value(prob, 1e-5, 1 - 1e-5)
            return input_state, prob, W, b, h, MLP_output, keep_pro

    def init_high_gradbuffer_A(self):
        gradBuffer = self.sess.run(self.target_high_network_params_A)
        for index, grad in enumerate(gradBuffer):
            gradBuffer[index] = grad * 0
        return gradBuffer

    def init_high_gradbuffer_B(self):
        gradBuffer = self.sess.run(self.target_high_network_params_B)
        for index, grad in enumerate(gradBuffer):
            gradBuffer[index] = grad * 0
        return gradBuffer

    def init_A_gradbuffer(self):
        gradBuffer = self.sess.run(self.target_A_network_params)
        for index, grad in enumerate(gradBuffer):
            gradBuffer[index] = grad * 0
        return gradBuffer

    def train_high_A(self, high_gradbuffer, high_grads):
        for index, grad in enumerate(high_grads):
            high_gradbuffer[index] += grad
        feed_dict = dict(zip(self.high_gradient_holders_A, high_gradbuffer))
        self.sess.run(self.high_optimize_A, feed_dict=feed_dict)

    def train_high_B(self, high_gradbuffer, high_grads):
        for index, grad in enumerate(high_grads):
            high_gradbuffer[index] += grad
        feed_dict = dict(zip(self.high_gradient_holders_B, high_gradbuffer))
        self.sess.run(self.high_optimize_B, feed_dict=feed_dict)

    def train_low_A(self, A_gradbuffer, A_grads):
        for index, grad in enumerate(A_grads):
            A_gradbuffer[index] += grad
        feed_dict = dict(zip(self.A_gradient_holders, A_gradbuffer))
        self.sess.run(self.A_optimize, feed_dict=feed_dict)

    def init_B_gradbuffer(self):
        gradBuffer = self.sess.run(self.target_B_network_params)
        for index, grad in enumerate(gradBuffer):
            gradBuffer[index] = grad * 0
        return gradBuffer

    def train_low_B(self, B_gradbuffer, B_grads):
        for index, grad in enumerate(B_grads):
            B_gradbuffer[index] += grad
        feed_dict = dict(zip(self.B_gradient_holders, B_gradbuffer))
        return self.sess.run(self.B_optimize, feed_dict=feed_dict)

    def predict_high_target_A(self, high_state, keep_pro):
        return self.sess.run([self.target_high_prob_A, self.high_W_A, self.high_b_A, self.high_h_A, self.high_out_A],
                             feed_dict={
                                 self.target_high_input_state_A: high_state,
                                 self.high_keep_pro_A: keep_pro})

    def predict_high_target_B(self, high_state, keep_pro):
        return self.sess.run([self.target_high_prob_B, self.high_W_B, self.high_b_B, self.high_h_B, self.high_out_B],
                             feed_dict={
                                 self.target_high_input_state_B: high_state,
                                 self.high_keep_pro_B: keep_pro})

    def predict_low_target_A(self, low_state_A, keep_pro):
        return self.sess.run([self.target_low_prob_A, self.W_A, self.b_A, self.h_A, self.out_A], feed_dict={
            self.target_A_input_state: low_state_A,
            self.keep_pro_A: keep_pro})

    def predict_low_target_B(self, low_state_B, keep_pro):
        return self.sess.run([self.target_low_prob_B, self.W_B, self.b_B, self.h_B, self.out_B], feed_dict={
            self.target_B_input_state: low_state_B,
            self.keep_pro_B: keep_pro})

    def get_high_gradient_A(self, high_state, high_reward, high_action, keep_pro):
        return self.sess.run([self.high_gradients_A, self.high_loss_A], feed_dict={
            self.target_high_input_state_A: high_state,
            self.high_reward_holder_A: high_reward,
            self.high_action_holder_A: high_action,
            self.high_keep_pro_A: keep_pro})

    def get_high_gradient_B(self, high_state, high_reward, high_action, keep_pro):
        return self.sess.run([self.high_gradients_B, self.high_loss_B], feed_dict={
            self.target_high_input_state_B: high_state,
            self.high_reward_holder_B: high_reward,
            self.high_action_holder_B: high_action,
            self.high_keep_pro_B: keep_pro})

    def get_A_gradient(self, A_state, A_reward, A_action, keep_pro):
        return self.sess.run([self.A_gradients, self.A_loss], feed_dict={
            self.target_A_input_state: A_state,
            self.A_reward_holder: A_reward,
            self.A_action_holder: A_action,
            self.keep_pro_A: keep_pro})

    def get_B_gradient(self, B_state, B_reward, B_action, keep_pro):
        return self.sess.run([self.B_gradients, self.B_loss], feed_dict={
            self.target_B_input_state: B_state,
            self.B_reward_holder: B_reward,
            self.B_action_holder: B_action,
            self.keep_pro_B: keep_pro})

    def update_target_high_network_A(self):
        self.sess.run(self.update_target_high_network_params_A)

    def update_target_high_network_B(self):
        self.sess.run(self.update_target_high_network_params_B)

    def update_target_A_network(self):
        self.sess.run(self.update_target_A_network_params)

    def update_target_B_network(self):
        self.sess.run(self.update_target_B_network_params)

    def assign_active_high_network_A(self):
        self.sess.run(self.assign_active_high_network_params_A)

    def assign_active_high_network_B(self):
        self.sess.run(self.assign_active_high_network_params_B)

    def assign_active_A_network(self):
        self.sess.run(self.assign_active_A_network_params)

    def assign_active_B_network(self):
        self.sess.run(self.assign_active_B_network_params)
