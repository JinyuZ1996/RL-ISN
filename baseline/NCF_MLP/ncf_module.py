# encoding: utf-8
'''
 MLP++ (sharing the same user embedding matrix P between E-domain and V-domain)
'''

import numpy as np

import theano
import theano.tensor as T
import keras
from keras import backend as K
from keras import initializations
from keras.regularizers import l2, activity_l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, Dropout
from keras.constraints import maxnorm
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from ncf_evaluation import evaluate_model
from ncf_dataset import Dataset
from time import time
import sys
import argparse
import multiprocessing as mp
import random
import os

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--path', nargs='?', default='',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each layer")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()

def init_normal(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)

def build_model(num_users,num_edus,num_vods,layers=[20,10],reg_layers=[0,0]):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers)  # Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')
    item_input2 = Input(shape=(1,), dtype='int32', name='item_input2')

    MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=layers[0] / 2, name='user_embedding',
                                   init=init_normal, W_regularizer=l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim=num_edus, output_dim=layers[0] / 2, name='item_embedding',
                                   init=init_normal, W_regularizer=l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item2 = Embedding(input_dim=num_vods, output_dim=layers[0] / 2, name='item_embedding2',
                             init=init_normal, W_regularizer=l2(reg_layers[0]), input_length=1)
    # Crucial to flatten an embedding vector
    user_latent = Flatten()(MLP_Embedding_User(user_input))
    item_latent = Flatten()(MLP_Embedding_Item(item_input))
    item_latent2 = Flatten()(MLP_Embedding_Item2(item_input2))

    # The 0-th layer is the concatenation of embedding layers
    vector = merge([user_latent, item_latent], mode='concat')
    vector2 = merge([user_latent, item_latent2], mode='concat')  # 64

    for idx in xrange(1, num_layer):
        layer = Dense(layers[idx], W_regularizer=l2(reg_layers[idx]), activation='relu', name='layere%d' % idx)
        vector = layer(vector)
        layer2 = Dense(layers[idx], W_regularizer=l2(reg_layers[idx]), activation='relu', name='layere2%d' % idx)
        vector2 = layer2(vector2)

    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name='prediction')(vector)
    prediction2 = Dense(1, activation='sigmoid', init='lecun_uniform', name='prediction2')(vector2)

    model = Model(input=[user_input, item_input, item_input2], output=[prediction, prediction2])

    return model

def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [], [], []
    num_users, num_items = train.shape
    choose_list = []
    user = set()
    for (u, i) in train.keys():
        if u not in user:
            user.add(u)
    print("user:" + str(len(user)))

    for u in user:
        ui_list = []
        for (u1, i1) in train.keys():
            if u == u1:
                for i in range(train[u1, i1]):
                    ui_list.append((u1, i1))
        # print(str(u)+":\t"+str(len(ui_list)))
        ui_list2 = random.sample(ui_list, 1)
        for item in ui_list2:
            choose_list.append(item)
    # print(str(len(choose_list))+"\n")

    for (u, i) in choose_list:
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in xrange(num_negatives):
            j = np.random.randint(num_items)
            while train.has_key((u, j)):
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    args = parse_args()
    path = args.path
    dataset = args.dataset
    layers = eval(args.layers)
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    verbose = args.verbose

    topK = ''
    evaluation_threads = 1  # mp.cpu_count()
    print("MLP arguments: %s " % (args))
    model_out_file = 'NCF-master/neural_collaborative_filtering-master/Pretrain/%s_MLP++_%s_%d.h5' % (args.dataset, args.layers, time())

    # Loading data
    t1 = time()
    dataset = Dataset(args.path + '') # input E-domain data
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    dataset2 = Dataset(args.path + '') # input V-domain data
    train2, testRatings2, testNegatives2 = dataset2.trainMatrix, dataset2.testRatings, dataset2.testNegatives
    num_users2, num_items2 = train2.shape

    print(str(num_users)+'\t'+str(num_users2)+'\t'+str(num_items)+'\t'+str(num_items2)+'\t') # [u,u,iA,iB]
    assert num_users== num_users2

    print("Load data done [%.1f s]. #user=%d, #item_edu=%d, #train_edu=%d, #test_edu=%d,"
        "#user=%d,#item_vod=%d,#train_vod=%d, #test_vod=%d"
         % (time() - t1, num_users, num_items, train.nnz, len(testRatings),
            num_users2,num_items2, train2.nnz, len(testRatings2)))
    # print(str(num_users) + "\t" + str(num_items) + "\t" + str(num_users2) + "\t" + str(num_items2))

    # Build model
    model= build_model(num_users, num_items ,num_items2, layers, reg_layers)
    losses = {'prediction':'binary_crossentropy','prediction2':'binary_crossentropy'}
    loss_weights = {'prediction':1.0,'prediction2':1.0}

    if learner.lower() == "adagrad":
        model.compile(optimizer=Adagrad(lr=learning_rate), loss=losses,loss_weights=loss_weights)
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss=losses,loss_weights=loss_weights)
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss=losses,loss_weights=loss_weights)
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss=losses,loss_weights=loss_weights)

    # Check Init performance
    t1 = time()
    (hits, ndcgs,hits2, ndcgs2) = evaluate_model(model, testRatings, testNegatives,testRatings2,
                                                 testNegatives2, topK, evaluation_threads)
    hr, ndcg,hr2, ndcg2 = np.array(hits).mean(), np.array(ndcgs).mean(),np.array(hits2).mean(), np.array(ndcgs2).mean()
    print('edu-Init: Recall = %.4f, Mrr = %.4f [%.1f]' % (hr, ndcg, time() - t1))
    print('vod-Init: Recall = %.4f, Mrr = %.4f [%.1f]' % (hr2, ndcg2, time() - t1))

    # # Train model
    best_hr, best_ndcg,best_hr2, best_ndcg2,best_iter = hr, ndcg, hr2, ndcg2, -1

    for epoch in xrange(epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(train, num_negatives)
        user_input2, item_input2, labels2 = get_train_instances(train2, num_negatives)

        # Training
        hist = model.fit([np.array(user_input), np.array(item_input),np.array(item_input2)],  # input
                         [np.array(labels),np.array(labels2)],  # labels
                         batch_size=batch_size, nb_epoch=1, verbose=0, shuffle=True)
        t2 = time()


        # Evaluation
        num_test_sequence = ''
        if epoch % verbose == 0:
            (hits, ndcgs, hits2, ndcgs2) = evaluate_model(model, testRatings, testNegatives, testRatings2,
                                                          testNegatives2, topK, evaluation_threads)
            hr, ndcg, hr2, ndcg2 = np.array(hits).mean(), np.array(ndcgs).mean(), np.array(hits2).mean(), np.array(ndcgs2).mean()
            hr, ndcg, hr2, ndcg2 = hr / num_test_sequence, ndcg / num_test_sequence, hr2 / num_test_sequence, ndcg2 / num_test_sequence
            print('edu-Iteration %d [%.1f s]: Recall = %.4f, Mrr = %.4f, loss = %.4f [%.1f s]'
                  % (epoch, t2 - t1, hr, ndcg, loss, time() - t2))
            print('vod-Iteration %d [%.1f s]: Recall = %.4f, Mrr = %.4f, loss = %.4f [%.1f s]'
                  % (epoch, t2 - t1, hr2, ndcg2, loss, time() - t2))
