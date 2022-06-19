import tensorflow as tf


class setting:

    def __init__(self):

        self.learning_rate = 0.01
        self.hidden_dims = 10
        self.Optimizer = tf.train.GradientDescentOptimizer
        self.reg = 0.0001
        self.top_K = 10
        self.init_mean = 0
        self.init_stdev = 0.01
        self.maxIter = 100
        self.batch_size = 32
        self.gpu_num = '0'