import time
import random
import numpy as np
import os
import heapq
from collections import defaultdict
from bpr_settings import *

args = setting()


def evaluate_model(model, testRatings, K):
    global _model
    global _testRatings
    global _K
    _model = model
    _testRatings = testRatings
    _K = K
    num_rating = len(testRatings)

    res = list(map(eval_one_rating, range(num_rating)))

    hits = [r[0] for r in res]
    ndcgs = [r[1] for r in res]

    return hits, ndcgs


def eval_one_rating(idx):
    rating = _testRatings[idx]
    hr = ndcg = 0.0
    u, gtItem = rating[0], rating[1]
    map_item_score = {}
    # calculate the score of the ground truth item
    maxScore = _model.score(u, gtItem)

    # early stopping if there are K items larger than maxScore
    countLarger = 0
    early_stop = False
    for i in range(1, _model.num_item + 1):
        score_ui = _model.score(u, i)
        map_item_score[i] = score_ui

        if score_ui > maxScore:
            countLarger += 1
        if countLarger > _K:
            early_stop = True
            break
    if early_stop == False:
        ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
        hr = getHitRatio(ranklist, gtItem)
        ndcg = getNDCG(ranklist, gtItem)

    return hr, ndcg


def getHitRatio(ranklist, gtItem):
    if gtItem in ranklist:
        return 1
    else:
        return 0


def getNDCG(ranklist, gtItem):
    for i in range(_K):
        item = ranklist[i]
        if item == gtItem:
            return np.log(2) / np.log(i + 2)
    return 0


class BPR_MF(object):

    def __init__(self, train_data, test_data, num_user, num_item, hidden_dims=args.hidden_dims,
                 Optimizer=args.Optimizer, learning_rate=args.learning_rate, reg=args.reg, topK=args.top_K,
                 init_mean=args.init_mean, init_stdev=args.init_stdev):
        self.train = train_data
        self.test = test_data
        self.num_user = num_user
        self.num_item = num_item
        self.hidden_dims = hidden_dims
        self.Optimizer = Optimizer
        self.learning_rate = learning_rate
        self.reg = reg
        self.topK = topK

        self.items_of_user = defaultdict(set)
        self.num_rating = 0
        for u in range(1, self.num_user + 1):
            for inta in train_data[u]:  # interaction
                self.items_of_user[u].add(inta[0])
                self.num_rating += 1

        self.u = tf.placeholder(tf.int32, [None])
        self.i = tf.placeholder(tf.int32, [None])
        self.j = tf.placeholder(tf.int32, [None])

        # latent matrices of users and items
        '''
        Since users and items are numbered consecutively from 1, 
        the first dimension of latent matrices is set to (num + 1), 
        and the first row doesn't do anything.
        '''
        self.user_emb_w = tf.get_variable('user_emb_w', [self.num_user + 1, self.hidden_dims],
                                          initializer=tf.random_normal_initializer(init_mean, init_stdev))
        self.item_emb_w = tf.get_variable('item_emb_w', [self.num_item + 1, self.hidden_dims],
                                          initializer=tf.random_normal_initializer(init_mean, init_stdev))

        self.u_emb = tf.nn.embedding_lookup(self.user_emb_w, self.u)
        self.i_emb = tf.nn.embedding_lookup(self.item_emb_w, self.i)
        self.j_emb = tf.nn.embedding_lookup(self.item_emb_w, self.j)

        # calculate loss of the sample
        y_ui = tf.reduce_sum(tf.multiply(self.u_emb, self.i_emb), axis=1, keep_dims=True)
        y_uj = tf.reduce_sum(tf.multiply(self.u_emb, self.j_emb), axis=1, keep_dims=True)
        l2_reg = self.reg * tf.add_n([tf.reduce_sum(tf.multiply(self.u_emb, self.u_emb)),
                                      tf.reduce_sum(tf.multiply(self.i_emb, self.i_emb)),
                                      tf.reduce_sum(tf.multiply(self.j_emb, self.j_emb))])
        bprloss = l2_reg - tf.reduce_mean(tf.log(tf.sigmoid(y_ui - y_uj)))

        # optimization
        self.sgd_step = self.Optimizer(self.learning_rate).minimize(bprloss)

    def build_model(self, maxIter=100, batch_size=32):
        self.maxIter = maxIter
        self.batch_size = batch_size
        print('Training MF-BPR model with: learning_rate={}, reg={}, hidden_dims={}, #epoch={}, batch_size={}.'.format(
            self.learning_rate, self.reg, self.hidden_dims, self.maxIter, self.batch_size))
        GPU_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=GPU_options)) as sess:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
            sess.run(tf.global_variables_initializer())
            # training process
            # each training epoch
            for iteration in range(maxIter):
                t1 = time.time()
                for _ in range(self.num_rating // self.batch_size):
                    uij_train = self.get_batch()
                    sess.run([self.sgd_step], feed_dict={  # optimization
                        self.u: uij_train[:, 0],
                        self.i: uij_train[:, 1],
                        self.j: uij_train[:, 2]})
                print('Have finished epoch {}.'.format(iteration + 1))

                # check performance
                t2 = time.time()
                variable_names = [v.name for v in tf.trainable_variables()]
                self.parameters = sess.run(variable_names)
                # self.parameters[0] ==> latent matrix for users
                # self.parameters[1] ==> latent matrix for items
                hits, ndcgs = evaluate_model(self, self.test, self.topK)
                print('Iter: {} [{:.2f} s] HitRatio@{} = {:.4f}, NDCG@{} = {:.4f} [{:.2f} s]'.format(
                    iteration + 1, t2 - t1, self.topK, np.array(hits).mean(), self.topK, np.array(ndcgs).mean(),
                    time.time() - t2))

    def score(self, u, i):
        return np.inner(self.parameters[0][u], self.parameters[1][i])

    def get_batch(self):
        t = []
        for _ in range(self.batch_size):
            # sample a user
            _u = random.sample(range(1, self.num_user + 1), 1)[0]
            # sample a positive item
            _i = random.sample(self.items_of_user[_u], 1)[0]
            # sample a negative item
            _j = random.sample(range(1, self.num_item + 1), 1)[0]
            while _j in self.items_of_user[_u]:
                _j = random.sample(range(1, self.num_item + 1), 1)[0]
            t.append([_u, _i, _j])
        return np.asarray(t)
