#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
import time
import numpy as np
import tensorflow as tf
import psj_module as PsiNet


# In[ ]:


random.seed(1)
np.random.seed(1)


# In[ ]:


def getdict(r_path):
    itemdict = {}
    with open(r_path,'r') as f:
        items =  f.readlines()
    for item in items:
        item = item.strip().split('\t')
        itemdict[item[1]] = int(item[0])
    return itemdict


# In[ ]:


def getdata(datapath, itemA, itemB):
    with open(datapath, 'r') as f:
        sessions = []
        for line in f.readlines():
            session = []
            line = line.strip().split('\t')
            for item in line[1:]:#从1开始为去除user
                item = item.split('|')[0]
                if item in itemA:
                    session.append(itemA[item])
                else:
                    session.append(itemB[item]+len(itemA))#为了区分序列中的A与B物品
            sessions.append(session)
    return sessions


# In[ ]:


def processdata(dataset):
    sessions = []
    for session in dataset:
        temp = []#1->A，2->B
        seq1 = []
        seq2 = []
        pos1 = []
        pos2 = []    
        len1 = 0
        len2 = 0
        for item in session[:-2]:
            if item < len(itemA):
                seq1.append(item)
                pos1.append(len2)
                len1 += 1
            else:
                seq2.append(item)
                pos2.append(len1)
                len2 += 1
        temp.append(seq1)
        temp.append(seq2)
        temp.append(pos1)
        temp.append(pos2)
        temp.append(len1)
        temp.append(len2)
        temp.append(session[-2])#处理数据保证这一个就是target_A
        temp.append(session[-1])#处理数据保证这一个就是target_B
        sessions.append(temp)
    return sessions


# In[ ]:


def getbatches(dataset, batch_size, pad_int, itemA):
    random.shuffle(dataset)
    for batch_i in range(0,len(dataset)//batch_size+1):
        start_i = batch_i*batch_size
        batch = dataset[start_i:start_i+batch_size]
        yield batchtoinput(batch, pad_int, itemA)


# In[ ]:


'''
def batchtoinput(batch, pad_int, itemA):#如果没有processdata
    seq_A = []#1->A，2->B
    seq_B = []
    len_A = []
    len_B = []
    pos_A = []
    pos_B = []
    target_A = []
    target_B = []
    i = 0
    for session in batch:
        seq1 = []
        seq2 = []
        pos1 = []
        pos2 = []    
        len1 = 0
        len2 = 0
        for item in session[:-2]:
            if item < len(itemA):
                seq1.append(item)
                pos1.append([i,len2])
                len1 += 1
            else:
                seq2.append(item)
                pos2.append([i,len1])
                len2 += 1
        seq_A.append(seq1)
        seq_B.append(seq2)
        pos_A.append(pos1)
        pos_B.append(pos2)
        len_A.append(len1)
        len_B.append(len2)
        target_A.append(session[-2])
        target_B.append(session[-1])
        i += 1
    maxlen_A = max(len_A)
    maxlen_B = max(len_B)
    mask_A = np.ones((len(len_A), maxlen_A))
    mask_B = np.ones((len(len_B), maxlen_B))
    #adjust_A = np.zeros((len(len_A), max(len_A)))#针对PsiNet_II_V1
    #adjust_B = np.zeros((len(len_B), max(len_B)))
    mask_A2B = np.ones((len(len_A), maxlen_A, maxlen_B))
    i = 0
    while i < len(batch):
        seq_A[i].extend([0]*(maxlen_A-len(seq_A[i])))
        seq_B[i].extend([0]*(maxlen_B-len(seq_B[i])))
        pos_A[i].extend([[i,0]]*(maxlen_A-len(pos_A[i])))
        pos_B[i].extend([[i,0]]*(maxlen_B-len(pos_B[i])))  
        mask_A[i,len_A[i]:] = 0
        mask_B[i,len_B[i]:] = 0
        #adjust_A[i,len_A[i]:] = 1
        #adjust_B[i,len_B[i]:] = 1  
        mask_A2B[i,len_A[i]:,:] = 0
        mask_A2B[i,:,len_B[i]:] = 0
        i += 1
    mask_B2A = np.transpose(mask_A2B, (0,2,1))
    return np.array(seq_A), np.clip(np.array(seq_B)-len(itemA), a_min=0, a_max=None), np.array(pos_A), np.array(pos_B), np.array(len_A), np.array(len_B), np.array(target_A), np.clip(np.array(target_B)-len(itemA), a_min=0, a_max=None), mask_A, mask_B, mask_A2B, mask_B2A#对于B中的物品需要还原它的id；np.clip是要把padding的id修正回0  
'''


# In[ ]:


def batchtoinput(batch, pad_int, itemA):
    seq_A = []
    seq_B = []
    len_A = []
    len_B = []
    pos_A = []
    pos_B = []
    target_A = []
    target_B = []
    for session in batch:
        len_A.append(session[4])
        len_B.append(session[5])
    maxlen_A = max(len_A)
    maxlen_B = max(len_B) 
    mask_A = np.ones((len(len_A), maxlen_A))
    mask_B = np.ones((len(len_B), maxlen_B))
    #adjust_A = np.zeros((len(len_A), max(len_A)))#针对PsiNet_II_V1
    #adjust_B = np.zeros((len(len_B), max(len_B)))
    mask_A2B = np.ones((len(len_A), maxlen_A, maxlen_B))
    i = 0
    for session in batch:
        seq_A.append(session[0]+[pad_int]*(maxlen_A-len_A[i]))#注意不要用session[0].extend([pad_int]*(maxlen_A-len_A[i]))，因为这里的session其实是实参（list、set和dict都是可改变变量，默认是传实参，里面的就是外面的），extend会直接修改dataset里的session（之前的方法并不改变session，而是根据session动态地产生样例），而+操作会返回一个新的list，从而不会改变session      
        seq_B.append(session[1]+[pad_int]*(maxlen_B-len_B[i]))
        pos_A.append(session[2]+[0]*(maxlen_A-len_A[i]))
        pos_B.append(session[3]+[0]*(maxlen_B-len_B[i]))
        target_A.append(session[6])
        target_B.append(session[7])   
        mask_A[i,len_A[i]:] = 0
        mask_B[i,len_B[i]:] = 0
        mask_A2B[i,len_A[i]:,:] = 0
        mask_A2B[i,:,len_B[i]:] = 0  
        i += 1
    mask_B2A = np.transpose(mask_A2B, (0,2,1))   
    index = np.arange(len(batch))
    index = np.expand_dims(index, axis=-1)
    index_p = np.repeat(index, maxlen_A, axis=1)
    pos_A = np.stack([index_p, np.array(pos_A)], axis=-1)
    index_p = np.repeat(index, maxlen_B, axis=1)
    pos_B = np.stack([index_p, np.array(pos_B)], axis=-1)
    return np.array(seq_A), np.clip(np.array(seq_B)-len(itemA), a_min=0, a_max=None), pos_A, pos_B, np.array(len_A), np.array(len_B), np.array(target_A), np.clip(np.array(target_B)-len(itemA), a_min=0, a_max=None),\
           mask_A, mask_B, mask_A2B, mask_B2A#对于B中的物品需要还原它的id；np.clip是要把padding的id修正回0


# In[ ]:


def get_eval(predlist, truelist, klist):#return recall@k and mrr@k
    recall = []
    mrr = []
    predlist = predlist.argsort()
    for k in klist:
        recall.append(0)
        mrr.append(0)
        templist = predlist[:,-k:]#the result of argsort is in ascending 
        i = 0
        while i < len(truelist):
            pos = np.argwhere(templist[i]==truelist[i])#pos is a list of positions whose values are all truelist[i]
            if len(pos) > 0:
                recall[-1] += 1
                mrr[-1] += 1/(k-pos[0][0])
            else:
                recall[-1] += 0
                mrr[-1] += 0
            i += 1
    return recall, mrr#they are sum instead of mean


# In[ ]:


'''
def get_eval(predlist, truelist, klist):#return recall@k and mrr@k
    recall = []
    mrr = []
    predlist = predlist.argsort()
    for k in klist:
        recall.append([])
        mrr.append([])
        templist = predlist[:,-k:]#the result of argsort is in ascending 
        i = 0
        while i < len(truelist):
            pos = np.argwhere(templist[i]==truelist[i])#pos is a list of positions whose values are all truelist[i]
            if len(pos) > 0:
                recall[-1].append(1)
                mrr[-1].append(1/(k-pos[0][0]))
            else:
                recall[-1].append(0)
                mrr[-1].append(0)
            i += 1
    return recall, mrr#return lists
'''


# In[ ]:


path = 'data/Alist.txt'
itemA = getdict(path)
path = 'data/Blist.txt'
itemB = getdict(path)


# In[ ]:


traindatapath = 'data/traindata.txt'
validdatapath = 'data/validdata.txt'
testdatapath = 'data/testdata.txt'
traindata = getdata(traindatapath, itemA, itemB)
validdata = getdata(validdatapath, itemA, itemB)
testdata = getdata(testdatapath, itemA, itemB)


# In[ ]:


traindata = processdata(traindata)
validdata = processdata(validdata)
testdata = processdata(testdata)


# In[ ]:


learning_rate = 0.001
keep_prob = 0.8
pad_int = 0#注意第一个物品的id和padding的id都是0，但是这个无碍，因为我们有mask
batch_size = 64
epochs = 50
model = PsiNet.PSJ_Net(items_max_num_A=len(itemA), items_max_num_B=len(itemB))


# In[ ]:


print(time.localtime())
checkpoint = 'checkpoint/trained_model.ckpt'
with tf.Session(graph=model.graph,config=model.config) as sess:
    writer = tf.summary.FileWriter('checkpoint/',sess.graph)
    saver = tf.train.Saver(max_to_keep=epochs)
    sess.run(tf.global_variables_initializer())
    #saver.restore(sess, 'checkpoint/trained_model.ckpt-30')
    for epoch in range(epochs):
        loss = 0
        step = 0
        for _,(seq_A, seq_B, pos_A, pos_B, len_A, len_B, target_A, target_B, mask_A, mask_B, mask_A2B, mask_B2A) in enumerate(getbatches(traindata, batch_size, pad_int, itemA)):
            _, l = sess.run([model.train_op, model.loss],{model.seq_A:seq_A, model.seq_B:seq_B, model.pos_A:pos_A, model.pos_B:pos_B, model.len_A:len_A, model.len_B:len_B, model.target_A:target_A, model.target_B:target_B, model.mask_A:mask_A, model.mask_B:mask_B, model.mask_A2B:mask_A2B, model.mask_B2A:mask_B2A, model.learning_rate:learning_rate, model.keep_prob:keep_prob}) 
            loss += l
            step += 1
            if step%1000 == 0:
                print(loss/step)
        print('Epoch {}/{} - Training Loss: {:.3f}'.format(epoch+1,epochs,loss/step))
        saver.save(sess, checkpoint, global_step=epoch+1)
        print(time.localtime())
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
        for _,(seq_A, seq_B, pos_A, pos_B, len_A, len_B, target_A, target_B, mask_A, mask_B, mask_A2B, mask_B2A) in enumerate(getbatches(validdata, batch_size, pad_int, itemA)):
            pa,pb = sess.run([model.pred_A, model.pred_B],{model.seq_A:seq_A, model.seq_B:seq_B, model.pos_A:pos_A, model.pos_B:pos_B, model.len_A:len_A, model.len_B:len_B, model.target_A:target_A, model.target_B:target_B, model.mask_A:mask_A, model.mask_B:mask_B, model.mask_A2B:mask_A2B, model.mask_B2A:mask_B2A, model.learning_rate:learning_rate, model.keep_prob:1.0}) 
            recall,mrr = get_eval(pa, target_A, [5,10,20])#应该更关注mrr
            r5_a += recall[0]
            m5_a += mrr[0]
            r10_a += recall[1]
            m10_a += mrr[1]
            r20_a += recall[2]
            m20_a += mrr[2]
            recall,mrr = get_eval(pb, target_B, [5,10,20])
            r5_b += recall[0]
            m5_b += mrr[0]
            r10_b += recall[1]
            m10_b += mrr[1]
            r20_b += recall[2]
            m20_b += mrr[2]
        print('Recall5: {:.5f}; Mrr5: {:.5f}'.format(r5_a/len(validdata),m5_a/len(validdata)))
        print('Recall10: {:.5f}; Mrr10: {:.5f}'.format(r10_a/len(validdata),m10_a/len(validdata)))
        print('Recall20: {:.5f}; Mrr20: {:.5f}'.format(r20_a/len(validdata),m20_a/len(validdata)))
        print('Recall5: {:.5f}; Mrr5: {:.5f}'.format(r5_b/len(validdata),m5_b/len(validdata)))
        print('Recall10: {:.5f}; Mrr10: {:.5f}'.format(r10_b/len(validdata),m10_b/len(validdata)))
        print('Recall20: {:.5f}; Mrr20: {:.5f}'.format(r20_b/len(validdata),m20_b/len(validdata)))
        print(time.localtime()) 


# In[ ]:


print(time.localtime())
with tf.Session(graph=model.graph,config=model.config) as sess:
    saver = tf.train.Saver()
    saver.restore(sess, 'checkpoint/trained_model.ckpt-30')#a和b的未必要取同一轮，可以取各自最优的一轮
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
    for _,(seq_A, seq_B, pos_A, pos_B, len_A, len_B, target_A, target_B, mask_A, mask_B, mask_A2B, mask_B2A) in enumerate(getbatches(testdata, batch_size, pad_int, itemA)):
        pa,pb = sess.run([model.pred_A, model.pred_B],{model.seq_A:seq_A, model.seq_B:seq_B, model.pos_A:pos_A, model.pos_B:pos_B, model.len_A:len_A, model.len_B:len_B, model.target_A:target_A, model.target_B:target_B, model.mask_A:mask_A, model.mask_B:mask_B, model.mask_A2B:mask_A2B, model.mask_B2A:mask_B2A, model.learning_rate:learning_rate, model.keep_prob:1.0}) 
        recall,mrr = get_eval(pa, target_A, [5,10,20])
        r5_a += recall[0]
        m5_a += mrr[0]
        r10_a += recall[1]
        m10_a += mrr[1]
        r20_a += recall[2]
        m20_a += mrr[2]
        recall,mrr = get_eval(pb, target_B, [5,10,20])
        r5_b += recall[0]
        m5_b += mrr[0]
        r10_b += recall[1]
        m10_b += mrr[1]
        r20_b += recall[2]
        m20_b += mrr[2]
    print('Recall5: {:.5f}; Mrr5: {:.5f}'.format(r5_a/len(testdata),m5_a/len(testdata)))
    print('Recall10: {:.5f}; Mrr10: {:.5f}'.format(r10_a/len(testdata),m10_a/len(testdata)))
    print('Recall20: {:.5f}; Mrr20: {:.5f}'.format(r20_a/len(testdata),m20_a/len(testdata)))
    print('Recall5: {:.5f}; Mrr5: {:.5f}'.format(r5_b/len(testdata),m5_b/len(testdata)))
    print('Recall10: {:.5f}; Mrr10: {:.5f}'.format(r10_b/len(testdata),m10_b/len(testdata)))
    print('Recall20: {:.5f}; Mrr20: {:.5f}'.format(r20_b/len(testdata),m20_b/len(testdata)))
    print(time.localtime()) 


# In[ ]:


e_a_o = model.graph.get_tensor_by_name('encoder_A/encoder_A/rnn/transpose:0')
e_b_o = model.graph.get_tensor_by_name('encoder_B/encoder_B/rnn/transpose:0')
s_b_o_1 = model.graph.get_tensor_by_name('segregation_B/transfer_B/rnn/transpose:0')
s_b_o_2 = model.graph.get_tensor_by_name('segregation_B/transfer_B_1/rnn/transpose:0')
s_b_o_3 = model.graph.get_tensor_by_name('segregation_B/transfer_B_2/rnn/transpose:0')
s_b_o_4 = model.graph.get_tensor_by_name('segregation_B/transfer_B_3/rnn/transpose:0')
i_b = model.graph.get_tensor_by_name('integration_B/concat:0')
s_a_o_1 = model.graph.get_tensor_by_name('segregation_A/transfer_A/rnn/transpose:0')
s_a_o_2 = model.graph.get_tensor_by_name('segregation_A/transfer_A_1/rnn/transpose:0')
s_a_o_3 = model.graph.get_tensor_by_name('segregation_A/transfer_A_2/rnn/transpose:0')
s_a_o_4 = model.graph.get_tensor_by_name('segregation_A/transfer_A_3/rnn/transpose:0')
i_a = model.graph.get_tensor_by_name('integration_A/concat:0')


# In[ ]:


print(time.localtime())
checkpoint = 'checkpoint/trained_model.ckpt'
with tf.Session(graph=model.graph,config=model.config) as sess:
    writer = tf.summary.FileWriter('checkpoint/',sess.graph)
    saver = tf.train.Saver(max_to_keep=epochs)
    sess.run(tf.global_variables_initializer())
    #saver.restore(sess, 'checkpoint/trained_model.ckpt-24')
    for epoch in range(epochs):
        loss = 0
        step = 0
        for k,(seq_A, seq_B, pos_A, pos_B, len_A, len_B, target_A, target_B, mask_A, mask_B, mask_A2B, mask_B2A) in enumerate(getbatches(traindata, batch_size, pad_int, itemA)):
            l, pa, pb, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12 = sess.run([model.loss, model.pred_A, model.pred_B, e_a_o, e_b_o, s_b_o_1, s_b_o_2, s_b_o_3, s_b_o_4, i_b, s_a_o_1, s_a_o_2, s_a_o_3, s_a_o_4, i_a],{model.seq_A:seq_A, model.seq_B:seq_B, model.pos_A:pos_A, model.pos_B:pos_B, model.len_A:len_A, model.len_B:len_B, model.target_A:target_A, model.target_B:target_B, model.mask_A:mask_A, model.mask_B:mask_B, model.mask_A2B:mask_A2B, model.mask_B2A:mask_B2A, model.learning_rate:learning_rate, model.keep_prob:keep_prob})             
            if np.isnan(l):#如果出现了nan错误，用这种方式来检查
                saver.save(sess, 'checkpoint_for_error/trained_model.ckpt')
                print(k)
                print(l)
                print(pa)
                print(pb)
                print(v1)
                print(v2)
                print(v3)
                print(v4)
                print(v5)
                print(v6)
                print(v7)
                print(v8)
                print(v9)
                print(v10)
                print(v11)
                print(v12)        
                print(error)
            _, l = sess.run([model.train_op, model.loss],{model.seq_A:seq_A, model.seq_B:seq_B, model.pos_A:pos_A, model.pos_B:pos_B, model.len_A:len_A, model.len_B:len_B, model.target_A:target_A, model.target_B:target_B, model.mask_A:mask_A, model.mask_B:mask_B, model.mask_A2B:mask_A2B, model.mask_B2A:mask_B2A, model.learning_rate:learning_rate, model.keep_prob:keep_prob})
            loss += l
            step += 1
            if step%1000 == 0:
                print(loss/step)
        print('Epoch {}/{} - Training Loss: {:.3f}'.format(epoch+1,epochs,loss/step))
        saver.save(sess, checkpoint, global_step=epoch+25)
        print(time.localtime())

