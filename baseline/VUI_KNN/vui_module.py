# encoding: utf-8

import numpy as np
import pandas as pd
import operator
import  random
import os
class ItemKNN:
    '''
    ItemKNN(n_sims = 100, lmbd = 20, alpha = 0.5, session_key = 'SessionId', item_key = 'ItemId', time_key = 'Time')

    Item-to-item predictor that computes the the similarity to all items to the given item.
    Similarity of two items is given by:

    .. math::
        s_{i,j}=\sum_{s}I\{(s,i)\in D & (s,j)\in D\} / (supp_i+\\lambda)^{\\alpha}(supp_j+\\lambda)^{1-\\alpha}

    Parameters
    --------
    n_sims : int
        Only give back non-zero scores to the N most similar items. Should be higher or equal than the cut-off of your evaluation. (Default value: 100)

    lmbd : float
        Regularization. Discounts the similarity of rare items (incidental co-occurrences). (Default value: 20)

    alpha : float
        Balance between normalizing with the supports of the two items. 0.5 gives cosine similarity, 1.0 gives confidence (as in association rules).

    session_key : string
        header of the session ID column in the input file (default: 'SessionId')

    item_key : string
        header of the item ID column in the input file (default: 'ItemId')

    time_key : string
        header of the timestamp column in the input file (default: 'Time')

    '''

    def __init__(self, n_sims=50, lmbd=20, alpha=0.5, session_key='SessionId', item_key='ItemId', time_key='Time'):
        self.n_sims = n_sims
        self.lmbd = lmbd
        self.alpha = alpha
        self.item_key = item_key
        self.session_key = session_key
        self.time_key = time_key

    def fit(self, data):

        '''
        Trains the predictor.

        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        '''
        data.set_index(np.arange(len(data)), inplace=True)
        itemids = data[self.item_key].unique()
        n_items = len(itemids)
        data = pd.merge(data, pd.DataFrame({self.item_key: itemids, 'ItemIdx': np.arange(len(itemids))}),
                        on=self.item_key, how='inner')

        sessionids = data[self.session_key].unique()
        self.n_sessions = len(sessionids)
        data = pd.merge(data, pd.DataFrame({self.session_key: sessionids, 'SessionIdx': np.arange(len(sessionids))}),
                        on=self.session_key, how='inner')

        supp = data.groupby('SessionIdx').size()
        session_offsets = np.zeros(len(supp) + 1, dtype=np.int32)
        session_offsets[1:] = supp.cumsum()

        index_by_sessions = data.sort_values(['SessionIdx', self.time_key]).index.values

        supp = data.groupby('ItemIdx').size()
        item_offsets = np.zeros(n_items + 1, dtype=np.int32)
        item_offsets[1:] = supp.cumsum()

        index_by_items = data.sort_values(['ItemIdx', self.time_key]).index.values

        self.sims = dict()
        for i in range(n_items):
            iarray = np.zeros(n_items)
            start = item_offsets[i]
            end = item_offsets[i + 1]
            for e in index_by_items[start:end]:
                uidx = data.SessionIdx.values[e]
                ustart = session_offsets[uidx]
                uend = session_offsets[uidx + 1]
                user_events = index_by_sessions[ustart:uend]
                iarray[data.ItemIdx.values[user_events]] += 1

            iarray[i] = len(user_events)
            norm = np.power((supp[i] + self.lmbd), self.alpha) * np.power((supp.values + self.lmbd), (1.0 - self.alpha))
            norm[norm == 0] = 1
            iarray = iarray / norm
            indices = np.argsort(iarray)[-1:-1 - self.n_sims:-1]
            # self.sims[itemids[i]] = pd.Series(data=iarray[indices], index=itemids[indices])
            self.sims[itemids[i]] = {'itemId': itemids[indices], 'sim_score': iarray[indices]}
        print("fit over!")

    def predict2(self,path,path2,K):

        last_user_id = 0
        last_peroid_id = 6
        user_list = []
        list = []
        data = dict()
        userId = []
        user_peroid = 0
        with open(path, 'r')as f, open(path2, 'w')as w:
            for line in f:
                line = line.strip("\n").split('\t')
                user_id = line[0]
                peroid_id = line[2]
                peroid_id2 = line[3]
                if last_user_id == 0 and last_peroid_id == 6:
                    last_user_id = user_id
                    last_peroid_id = peroid_id
                    userId.append(last_user_id)
                    list.append(last_peroid_id)
                    list.append(peroid_id2)
                    list.append(line[1])
                    continue
                if last_user_id == user_id and last_peroid_id != peroid_id:
                    user_list.append(list)
                    list = []
                    user_peroid += 1
                    last_peroid_id = peroid_id
                    list.append(last_peroid_id)
                    list.append(peroid_id2)
                    list.append(line[1])
                    continue
                if last_user_id != user_id:
                    user_list.append(list)
                    list = []
                    user_peroid += 1
                    data[last_user_id] = user_list
                    user_list = []
                    last_user_id = user_id
                    last_peroid_id = peroid_id
                    userId.append(last_user_id)
                    list.append(last_peroid_id)
                    list.append(peroid_id2)
                    list.append(line[1])
                    continue
                list.append(line[1])
            user_list.append(list)
            data[last_user_id] = user_list

            print(user_peroid)
            # prediction
            self.predict_item_list = {}
            for i in range(len(userId)):
                item_unique = []
                item_sims = {}
                user_list = data[userId[i]]
                for j in range(len(user_list)):
                    item_list = user_list[j]
                    for id in range(len(item_list) - 2):
                        itemId = item_list[id + 2]
                        itemId_list = self.sims[itemId]['itemId']
                        sims_list = self.sims[itemId]['sim_score']
                        for m in range(len(itemId_list)):
                            item_id = itemId_list[m]
                            item_id_sim = sims_list[m]
                            if item_id in item_unique:
                                item_sims[item_id] = item_id_sim + item_sims[item_id]
                            if item_id not in item_unique:
                                item_unique.append(item_id)
                                item_sims[item_id] = item_id_sim
                    sort_list = sorted(item_sims.items(), key=operator.itemgetter(1))
                    w.write(userId[i] + "\t" + user_list[j][0] + "\t" + user_list[j][1] + "\t")
                    predict_list = []
                    for jj in range(len(sort_list) - K, len(sort_list)):
                        predict_list.append(sort_list[jj][0])
                        w.write(sort_list[jj][0] + "\t")
                    self.predict_item_list[userId[i]] = predict_list
                    w.write("\n")
        print('predict over!')

    def target2(self,path,path2):
        self.predict_item_list = {}
        last_user_id = 0
        list2 = []
        userId = set()
        l = 0
        with open(path, 'r')as f:
            for line in f:
                line = line.strip("\n").split('\t')
                l += 1
                user_id = line[0]
                if user_id not in userId:
                    userId.add(user_id)
                if last_user_id == 0:
                    last_user_id = user_id
                    list = []
                    for i in range(1, len(line)):
                        list.append(line[i])
                    list2.append(list)
                    continue
                if last_user_id != user_id:
                    self.predict_item_list[last_user_id] = list2
                    list2 = []
                    last_user_id = user_id
                    list = []
                    for i in range(1, len(line)):
                        list.append(line[i])
                    list2.append(list)
                    continue
                list = []
                for i in range(1, len(line)):
                    list.append(line[i])
                list2.append(list)
            self.predict_item_list[last_user_id] = list2
        print("read predict over!")

        last_user_id = 0
        last_peroid_id = 6
        item_num = 0
        recall = 0
        recall_sum = 0
        mrr = 0
        mrr_sum = 0
        item_r = 0
        user_num = 0
        with open(path2, 'r')as f:
            for line in f:
                line = line.strip("\n").split('\t')
                user_id = line[0]
                peroid_id = line[2]
                if user_id not in userId:
                    continue
                item_r = item_r + 1
                if last_user_id != user_id:
                    list = self.predict_item_list[user_id]
                if last_user_id == 0 and last_peroid_id == 6:
                    last_user_id = user_id
                    last_peroid_id = peroid_id
                    item_num = 1
                    user_num = 1
                    print(len(list))
                    for i in range(len(list)):
                        if peroid_id == list[i][0]:
                            if line[1] in list[i]:
                                recall = 1
                                mrr = mrr + 1.0 / (list[i].index(line[1]) - 1)
                                break
                    continue
                if user_id != last_user_id or peroid_id != last_peroid_id:
                    user_num = user_num + 1
                    recall_sum = recall_sum + recall / item_num
                    mrr_sum = mrr_sum + mrr / item_num
                    if user_num % 1000 == 0:
                        print(str(user_num) + ",recall:" + str((recall_sum + 0.0) / user_num) + ",mrr:" + str(mrr_sum / user_num))
                    last_user_id = user_id
                    last_peroid_id = peroid_id
                    item_num = 1
                    recall = 0
                    mrr = 0
                    for i in range(len(list)):
                        if peroid_id == list[i][0]:
                            if line[1] in list[i]:
                                recall += 1
                                mrr = mrr + 1.0 / (list[i].index(line[1]) + 1)
                                break
                    continue
                for j in range(len(list)):
                    if peroid_id == list[j][1]:
                        if line[1] in list[j]:
                            recall += 1
                            mrr = mrr + 1.0 / (list[j].index(line[1]) + 1)
                            break
                item_num = item_num + 1

        # print("recall:" + str((recall_sum + 0.0) / num_testdata))  # Recall
        # print("mrr" + str(mrr_sum / num_testdata))  # MRR
        # print(user_num)  # users

def data_handle2(path):
    user_id = []
    item_id = []
    Time = []
    with open(path, 'r')as f:
        for line in f:
            line = line.strip("\n").split('\t')
            user_id.append(line[0])
            item_id.append(line[1])
            Time.append(line[2])
    data = {'SessionId': user_id, 'ItemId': item_id, 'Time': Time}
    data_last = pd.DataFrame(data)
    return data_last

def data_sort(path,path2):
    data=[]
    with open(path,'r')as f:
        for line in f:
            line=line.strip('\n').split('\t')
            data.append(line)
        data = sorted(data,key=lambda d: d[2])
        data=sorted(data,key=lambda d: d[0])
    with open(path2,'w')as w:
        for i in range(len(data)):
            w.write(data[i][0]+"\t"+data[i][1]+"\t"+data[i][2]+"\n")


if __name__ == '__main__':
    # VUI-KNN
    print('edu-VUI-KNN')
    path1 = 'NewData/edu/mid_data/test_period_merge.txt'
    path2 = 'NewData/edu/mid_data/test_period_sorted.txt'
    path = 'NewData/edu/mid_data/train_sorted_merge.txt'
    # data_sort(path1,path2)
    data_list = data_handle2(path)
    Item_knn = ItemKNN()
    Item_knn.fit(data_list)
    K = 20 # top_K metrics
    path_K = 'NewData/edu/mid_data/predict_VUI-KNN@K.txt'
    Item_knn.predict2(path, path_K, K)
    Item_knn.target2(path_K, path2)

