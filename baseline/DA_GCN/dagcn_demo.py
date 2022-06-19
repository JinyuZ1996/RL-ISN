#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os
import random
import time
import numpy as np
import pandas as pd
import collections
import tensorflow as tf
import scipy.sparse as sp
import dagcn_module as DA_GCN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# In[ ]:

random.seed(1)
np.random.seed(1)


# In[ ]:

def getdict(r_path):
    itemdict = {}
    with open(r_path, 'r') as f:
        items = f.readlines()
    for item in items:
        item = item.strip().split('\t')
        itemdict[item[1]] = int(item[0])
    return itemdict

# In[ ]:


def getdata(datapath, itemE, itemV):
    with open(datapath, 'r') as f:
        sessions = []
        for line in f.readlines():
            session = []
            line = line.strip().split('\t')
            user = line[0]
            session.append(userU[user])
            for item in line[1:]:
                #if item[0] == 'E':
                item = item.split('|')[0]
                if item in itemE:
                    session.append(itemE[item])
                else:
                    session.append(itemV[item] + len(itemE))
            sessions.append(session)
        return sessions
# In[ ]:

##
def processdata(dataset):
    sessions = []
    for session in dataset:
        temp = []
        seq1 = []
        seq2 = []
        pos1 = []
        pos2 = []
        len1 = 0
        len2 = 0
        seq1.append(session[0])
        seq2.append(session[0])
        for item in session[1:-2]:
            if item < len(itemE):
                seq1.append(item)
                pos1.append(len2)
                len1 += 1
            else:
                seq2.append(item - len(itemE))
                pos2.append(len1)
                len2 += 1
        temp.append(seq1)
        temp.append(seq2)
        temp.append(pos1)
        temp.append(pos2)
        temp.append(len1)
        temp.append(len2)
        temp.append(session[-2])
        temp.append(session[-1]-len(itemE))
        sessions.append(temp)
    return sessions

def load_ratings(dataset):
    A = list()
    B = list()
    C = list()
    D = list()
    E = list()
    F = list()
    G = list()
    for session in dataset:
        a = session[0]
        b = session[1]
        items_A = [int(i) for i in a[1:]]
        items_B = [int(j) for j in b[1:]]
        uid = int(a[0])
        G.append([uid, uid])
        for i in items_A:
            A.append([uid, i])
            C.append([i, uid])
        for j in items_B:
            B.append([uid, j])
            D.append([j, uid])
        for ids in range(0, len(items_A) - 1):
            i = items_A[ids]
            m = items_A[ids+1]
            E.append([i, i])
            E.append([i, m])
        for id in range(0, len(items_B) - 1):
            j = items_B[id]
            n = items_B[id+1]
            F.append([j, j])
            F.append([j, n])
    A = pd.DataFrame(A, columns=['reviewerID', 'asinA'])
    B = pd.DataFrame(B, columns=['reviewerID', 'asinB'])
    C = pd.DataFrame(C, columns=['asinA', 'reviewerID'])
    D = pd.DataFrame(D, columns=['asinB', 'reviewerID'])
    E = pd.DataFrame(E, columns=['asinA', 'asinA'])
    F = pd.DataFrame(F, columns=['asinB', 'asinB'])
    G = pd.DataFrame(G, columns=['reviewerID', 'reviewerID'])
    A.duplicated()
    A.drop_duplicates(inplace=True)
    B.duplicated()
    B.drop_duplicates(inplace=True)
    C.duplicated()
    C.drop_duplicates(inplace=True)
    D.duplicated()
    D.drop_duplicates(inplace=True)
    E.duplicated()
    E.drop_duplicates(inplace=True)
    F.duplicated()
    F.drop_duplicates(inplace=True)
    G.duplicated()
    G.drop_duplicates(inplace=True)
    A = A.values.tolist()
    B = B.values.tolist()
    C = C.values.tolist()
    D = D.values.tolist()
    E = E.values.tolist()
    F = F.values.tolist()
    G = G.values.tolist()
    return np.array(A), np.array(B), np.array(C), np.array(D), np.array(E), np.array(F), np.array(G)


def _get_relational_adj_list():
    adj_mat_list = []
    n_items_A = len(itemE)
    n_items_B = len(itemV)
    n_users = len(userU)
    n_all = n_items_A + n_users + n_items_B
    def _np_mat2sp_adj(np_mat, row_pre, col_pre):
        a_rows = np_mat[:, 0] + row_pre
        a_cols = np_mat[:, 1] + col_pre
        a_vals = [1.] * len(a_rows)
        a_adj = sp.coo_matrix((a_vals, (a_rows, a_cols)), shape=(n_all, n_all))
        return a_adj

    R_A = _np_mat2sp_adj(A, row_pre=n_items_A, col_pre=0)
    R_B = _np_mat2sp_adj(B, row_pre=n_items_A, col_pre=n_items_A + n_users)
    R_C = _np_mat2sp_adj(C, row_pre=0, col_pre=n_items_A)
    R_D = _np_mat2sp_adj(D, row_pre=n_items_A+n_users, col_pre=n_items_A)
    R_E = _np_mat2sp_adj(E, row_pre=0, col_pre=0)
    R_F = _np_mat2sp_adj(F, row_pre=n_items_A+n_users, col_pre=n_items_A+n_users)
    R_G = _np_mat2sp_adj(G, row_pre=n_items_A, col_pre=n_items_A)
    print('\tconvert ratings into adj mat done.')
    adj_mat_list.append(R_A)
    adj_mat_list.append(R_B)
    adj_mat_list.append(R_C)
    adj_mat_list.append(R_D)
    adj_mat_list.append(R_E)
    adj_mat_list.append(R_F)
    adj_mat_list.append(R_G)
    #lap_list = [_si_norm_lap(adj+sp.eye(adj.shape[0])) for adj in adj_mat_list]
    lap_list = [adj.tocoo() for adj in adj_mat_list]
    #lap_list = [_si_norm_lap(adj) for adj in adj_mat_list]
    print('\tgenerate si-normalized adjacency matrix.')
    return lap_list

def _si_norm_lap(adj):
    rowsum = np.array(adj.sum(1))

    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)

    norm_adj = d_mat_inv.dot(adj)
    return norm_adj.tocoo()

def _get_all__data( ):
    def _reorder_list(org_list, order):
        new_list = np.array(org_list)
        new_list = new_list[order]
        return new_list

    all_h_list, all_t_list = [], []
    all_v_list = []
    for l_id, lap in enumerate(lap_list):
        all_h_list += list(lap.row)
        all_t_list += list(lap.col)
        all_v_list += list(lap.data)

    assert len(all_h_list) == sum([len(lap.data) for lap in lap_list])
    print('\treordering indices...')
    org_h_dict = dict()

    for idx, h in enumerate(all_h_list):
        if h not in org_h_dict.keys():
            org_h_dict[h] = [[], []]
        org_h_dict[h][0].append(all_t_list[idx])
        org_h_dict[h][1].append(all_v_list[idx])
    print('\treorganize head_tail_value data done.')

    sorted_h_dict = dict()
    for h in org_h_dict.keys():
        org_t_list, org_v_list = org_h_dict[h]
        sort_t_list = np.array(org_t_list)
        sort_order = np.argsort(sort_t_list)

        sort_t_list = _reorder_list(org_t_list, sort_order)
        sort_v_list = _reorder_list(org_v_list, sort_order)

        sorted_h_dict[h] = [sort_t_list, sort_v_list]
    print('\tsort triple-data done.')

    od = collections.OrderedDict(sorted(sorted_h_dict.items()))
    new_h_list, new_t_list, new_v_list = [], [], []

    for h, vals in od.items():
        new_h_list += [h] * len(vals[0])
        new_t_list += list(vals[0])
        new_v_list += list(vals[1])

    assert sum(new_h_list) == sum(all_h_list)
    assert sum(new_t_list) == sum(all_t_list)
    return new_h_list, new_t_list, new_v_list


# In[ ]:
# 批处理
''''''


def getbatches(dataset, batch_size, pad_int):
    random.shuffle(dataset)
    for batch_i in range(0, len(dataset) // batch_size + 1):
        start_i = batch_i * batch_size
        batch = dataset[start_i:start_i + batch_size]
        yield batchtoinput(batch, pad_int)

def batchtoinput(batch, pad_int):
    uid = []
    seq_A = []
    seq_B = []
    len_A = []
    len_B = []
    target_A = []
    target_B = []
    for session in batch:
       len_A.append(session[4])
       len_B.append(session[5])
    maxlen_A = max(len_A)
    maxlen_B = max(len_B)
    i = 0
    for session in batch:
        a = session[0]
        b = session[1]
        c = int(a[0])
        uid.append(c)
        seq_A.append(a[1:] + [pad_int] * (maxlen_A - len_A[i]))
        seq_B.append(b[1:] + [pad_int] * (maxlen_B - len_B[i]))
        target_A.append(session[6])
        target_B.append(session[7])
        i += 1
    return np.array(uid), np.array(seq_A), np.array(seq_B), np.array(target_A), np.array(target_B)


# In[ ]:
''''''

def get_eval(predlist, truelist, klist):
    recall = []
    mrr = []
    predlist = predlist.argsort()   # 获取predict_list
    for k in klist:
        recall.append(0)
        mrr.append(0)
        templist = predlist[:, -k:]
        i = 0
        while i < len(truelist):
            pos = np.argwhere(
                templist[i] == truelist[i])
            if len(pos) > 0:
                recall[-1] += 1
                mrr[-1] += 1 / (k - pos[0][0])
            else:
                recall[-1] += 0
                mrr[-1] += 0
            i += 1
    return recall, mrr  # they are sum instead of mean；

path = 'finalcontruth_info/Elist.txt'
itemE = getdict(path)
path = 'finalcontruth_info/Vlist.txt'
itemV = getdict(path)
path = 'finalcontruth_info/userlist.txt'
userU = getdict(path)

# In[ ]:

traindatapath = 'finalcontruth_info/traindata_sess.txt'
validdatapath = 'finalcontruth_info/validdata_sess.txt'
testdatapath = 'finalcontruth_info/testdata_sess.txt'
alldatapath = 'finalcontruth_info/alldata_sess.txt'
traindata = getdata(traindatapath, itemE, itemV)
validdata = getdata(validdatapath, itemE, itemV)
testdata = getdata(testdatapath, itemE, itemV)
alldata = getdata(alldatapath, itemE, itemV)

# In[ ]:

traindata = processdata(traindata)
validdata = processdata(validdata)
testdata = processdata(testdata)
alldata = processdata(alldata)

A, B, C, D, E, F, G = load_ratings(alldata)

lap_list = _get_relational_adj_list()

A_in = sum(lap_list)

all_h_list, all_t_list, all_v_list = _get_all__data()




# 参数的定义
learning_rate = 0.001
keep_prob = 0.8
dropout_rate = 0.001
pad_int = 0
batch_size = 128
epochs = 50

model = DA_GCN.DA_GCN(n_items_A=len(itemE), n_items_B=len(itemV), n_users=len(userU), A_in=A_in,
                    all_h_list=all_h_list, all_t_list=all_t_list, all_v_list=all_v_list)

# In[ ]:

print(time.localtime())
checkpoint = 'checkpoint/trained_model.ckpt'
with tf.Session(graph=model.graph, config=model.config) as sess:
    #writer = tf.summary.FileWriter('checkpoint/', sess.graph)
    #saver = tf.train.Saver(max_to_keep=20)
    saver = tf.train.Saver(max_to_keep=epochs)
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        loss = 0
        step = 0
        for _, (uid, seq_A, seq_B, target_A, target_B) in enumerate(getbatches(traindata, batch_size, pad_int)):
            _, l, t1, t2, loss1, loss2 = sess.run([model.train_op, model.loss, model.target_A, model.target_B, model.loss1, model.loss2], {model.uid: uid, model.seq_A: seq_A, model.seq_B: seq_B, model.target_A: target_A, model.target_B: target_B,model.learning_rate: learning_rate, model.dropout_rate: dropout_rate, model.keep_prob: keep_prob})
            start_time = time.time()
            loss += l
            step += 1
            if step % 1000 == 0:
                print(loss / step)
        model.update_attentive_A(sess)
        print('Epoch {}/{} - Training Loss: {:.3f}'.format(epoch + 1, epochs, loss / step))
        saver.save(sess, checkpoint, global_step=epoch + 1, write_meta_graph=False)
        #if epoch>=20:
            #print('Epoch {}/{} - Training Loss: {:.3f}'.format(epoch + 1, epochs, loss / step))
        #print(time.localtime())
        #start_time = time.time()
        r5_a = 0
        m5_a = 0
        r10_a = 0
        m10_a = 0
        r20_a = 0
        m20_a = 0
        r5_b = 0
        m5_b = 0
        r10_b = 0
        m10_b = 0
        r20_b = 0
        m20_b = 0

        for _, (uid, seq_A, seq_B, target_A, target_B) in enumerate(getbatches(validdata, batch_size, pad_int)):
            pa, pb = sess.run([model.pred_A, model.pred_B], {model.uid: uid, model.seq_A: seq_A, model.seq_B: seq_B, model.target_A: target_A, model.target_B: target_B, model.learning_rate: learning_rate,model.dropout_rate: dropout_rate, model.keep_prob: 1.0})
            recall, mrr = get_eval(pa, target_A, [5, 10, 20])
            r5_a += recall[0]
            m5_a += mrr[0]
            r10_a += recall[1]
            m10_a += mrr[1]
            r20_a += recall[2]
            m20_a += mrr[2]
            recall, mrr = get_eval(pb, target_B, [5, 10, 20])
            r5_b += recall[0]
            m5_b += mrr[0]
            r10_b += recall[1]
            m10_b += mrr[1]
            r20_b += recall[2]
            m20_b += mrr[2]
        print('Recall5: {:.5f}; Mrr5: {:.5f}'.format(r5_a / len(validdata), m5_a / len(validdata)))
        print('Recall10: {:.5f}; Mrr10: {:.5f}'.format(r10_a / len(validdata), m10_a / len(validdata)))
        print('Recall20: {:.5f}; Mrr20: {:.5f}'.format(r20_a / len(validdata), m20_a / len(validdata)))
        print('Recall5: {:.5f}; Mrr5: {:.5f}'.format(r5_b / len(validdata), m5_b / len(validdata)))
        print('Recall10: {:.5f}; Mrr10: {:.5f}'.format(r10_b / len(validdata), m10_b / len(validdata)))
        print('Recall20: {:.5f}; Mrr20: {:.5f}'.format(r20_b / len(validdata), m20_b / len(validdata)))
        print("Epoch %d, finish training " % (epoch+1) + "took " +
              time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time)) + ';')

# # Test

# In[ ]:


print(time.localtime())
with tf.Session(graph=model.graph, config=model.config) as sess:
    saver = tf.train.Saver()
    saver.restore(sess, 'checkpoint/trained_model.ckpt-40')  # a和b的未必要取同一轮，可以取各自最优的一轮;
    r5_a = 0
    m5_a = 0
    r10_a = 0
    m10_a = 0
    r20_a = 0
    m20_a = 0
    r5_b = 0
    m5_b = 0
    r10_b = 0
    m10_b = 0
    r20_b = 0
    m20_b = 0
    for _, (uid, seq_A, seq_B, target_A, target_B) in enumerate(getbatches(testdata, batch_size, pad_int)):
        pa, pb = sess.run([model.pred_A, model.pred_B], {model.uid: uid, model.seq_A: seq_A, model.seq_B: seq_B, model.target_A: target_A, model.target_B: target_B, model.learning_rate: learning_rate, model.dropout_rate: dropout_rate, model.keep_prob: 1.0})
        recall, mrr = get_eval(pa, target_A, [5, 10, 20])
        r5_a += recall[0]
        m5_a += mrr[0]
        r10_a += recall[1]
        m10_a += mrr[1]
        r20_a += recall[2]
        m20_a += mrr[2]
        recall, mrr = get_eval(pb, target_B, [5, 10, 20])
        r5_b += recall[0]
        m5_b += mrr[0]
        r10_b += recall[1]
        m10_b += mrr[1]
        r20_b += recall[2]
        m20_b += mrr[2]
    print('Recall5: {:.5f}; Mrr5: {:.5f}'.format(r5_a / len(testdata), m5_a / len(testdata)))
    print('Recall10: {:.5f}; Mrr10: {:.5f}'.format(r10_a / len(testdata), m10_a / len(testdata)))
    print('Recall20: {:.5f}; Mrr20: {:.5f}'.format(r20_a / len(testdata), m20_a / len(testdata)))
    print('Recall5: {:.5f}; Mrr5: {:.5f}'.format(r5_b / len(testdata), m5_b / len(testdata)))
    print('Recall10: {:.5f}; Mrr10: {:.5f}'.format(r10_b / len(testdata), m10_b / len(testdata)))
    print('Recall20: {:.5f}; Mrr20: {:.5f}'.format(r20_b / len(testdata), m20_b / len(testdata)))
    print(time.localtime())
