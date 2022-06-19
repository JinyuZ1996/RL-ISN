# _*_coding: UTF_8 _*_
import numpy as np
import random
from RL_ISN.ISN_parameters import *

args = setting()

'''
    TO generate the proper shape of training data or testing data.
'''


#  get dictionary of dataset
def get_dict(dict_path):
    item_dict = {}
    with open(dict_path, 'r') as file_object:
        lines = file_object.readlines()
        for item in lines:
            item = item.strip().split('\t')
            item_dict[item[1]] = int(item[0])
    return item_dict


# get mixed_data from hvideo dataset by dictionary
def get_mixedData_video(data_path, dict_E, dict_V):
    with open(data_path, 'r') as file_object:
        mixed_data = []
        lines = file_object.readlines()
        for line in lines:
            temp = []
            line = line.strip().split('\t')
            for item in line[1:]:
                if item[0] == 'E':
                    temp.append(dict_E[item])
                else:
                    temp.append(dict_V[item] + len(dict_E))
            mixed_data.append(temp)
        return mixed_data


# get mixed_data from hamazon dataset by dictionary
def get_mixedData_amazon(data_path, dict_A, dict_B, isTrain=True):
    with open(data_path, 'r') as file_object:
        mixed_data = []
        lines = file_object.readlines()
        jump_time = 0
        for line in lines:
            sign_A, sign_B = 0, 0
            temp = []
            line = line.strip().split('\t')
            if len(line) <= 3 and isTrain:
                continue
            else:
                for item in line[1:]:
                    if item in dict_A:
                        temp.append(dict_A[item])
                        sign_A += 1
                    elif item in dict_B:
                        temp.append(dict_B[item] + len(dict_A))
                        sign_B += 1
                if (sign_A <= 1 or sign_B <= 1) and isTrain:
                    jump_time += 1
                    print("jump {} times".format(jump_time))
                    continue
            mixed_data.append(temp)
    return mixed_data


#  process data with target items
def process_data(origin_train_data, dict_A, origin_test_data):
    list_with_target = []
    count = -1
    for i in origin_train_data:
        count += 1
        temp = []
        len_mix = 0
        seq_A, seq_B = [], []
        pos_A, pos_B = [], []
        len_A, len_B = 0, 0
        for item in i[:-2]:
            if item < len(dict_A):
                seq_A.append(item)
                pos_A.append(len_mix)     # 以B的len定位item_A在原始序列里的位置
                len_mix += 1
                len_A += 1
            else:
                seq_B.append(item - len(dict_A))
                pos_B.append(len_mix)
                len_mix += 1
                len_B += 1
        temp.append(seq_A)  # [0]
        temp.append(seq_B)  # [1]
        temp.append(pos_A)  # new 2
        temp.append(pos_B)  # new 3
        temp.append(len_A)  # 4
        temp.append(len_B)  # 5
        if origin_test_data is None:        # 训练集生成方法
            temp.append(i[-2])  # 6
            temp.append(i[-1] - len(dict_A))    # 7
        else:   # 测试集生成方法
            temp.append(origin_test_data[count][-2])    # 6
            temp.append(origin_test_data[count][-1] - len(dict_A))  # 7
        list_with_target.append(temp)   # 将一条数据加入list
    return list_with_target


#  generate the negs for training data
def get_train_with_neg(data_with_target, num_negs, dict_A, dict_B):
    train_data_with_neg = []
    for i in range(len(data_with_target)):
        data = data_with_target[i]
        neg_list_A, neg_list_B, target_list_A, target_list_B = [], [], [], []
        target_list_A.append(data[4])
        target_list_B.append(data[5])
        for t in range(num_negs):
            random_item_A = np.random.randint(len(dict_A))
            random_item_B = np.random.randint(len(dict_B))
            while random_item_A in data[0] or random_item_A in neg_list_A or random_item_A in target_list_A:
                random_item_A = np.random.randint(len(dict_A))
            while random_item_B in data[1] or random_item_B in neg_list_B or random_item_B in target_list_B:
                random_item_B = np.random.randint(len(dict_B))
            neg_list_A.append(random_item_A)
            neg_list_B.append(random_item_B)
            seq_A = data[0]
            seq_B = data[1]
            pos_A = data[2]
            pos_B = data[3]
            len_A = data[4]
            len_B = data[5]
            if t + 1 == num_negs:
                target_A = data[6]
                target_B = data[7]
                label_A = 1
                label_B = 1
            else:
                target_A = random_item_A
                target_B = random_item_B
                label_A = 0
                label_B = 0
            train_data_with_neg.append((seq_A, seq_B, pos_A, pos_B, len_A, len_B, target_A, target_B, label_A, label_B))
    return train_data_with_neg


# load the test_negs from files
def load_negative_file(neg_path):
    neg_list = []
    with open(neg_path, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.rstrip().split("\t")
            negatives = []
            for x in arr[0:]:
                negatives.append(x)
            neg_list.append(negatives)
            line = f.readline()
    return neg_list


# generate the testing data with negs
def get_test_with_neg(data_with_target, num_negs, dict_A, dict_B):
    test_negs_A = load_negative_file(neg_path=args.test_negative_A)
    test_negs_B = load_negative_file(neg_path=args.test_negative_B)
    test_data_with_neg = []
    for i in range(len(data_with_target)):
        data = data_with_target[i]
        for t in range(num_negs):
            seq_A = data[0]
            seq_B = data[1]
            pos_A = data[2]
            pos_B = data[3]
            len_A = data[4]
            len_B = data[5]
            if t + 1 == num_negs:
                target_A = data[6]
                target_B = data[7]
                label_A = 1
                label_B = 1
            else:
                target_A = dict_A[test_negs_A[i][t]]
                target_B = dict_B[test_negs_B[i][t]]
                label_A = 0
                label_B = 0
            test_data_with_neg.append((seq_A, seq_B, pos_A, pos_B, len_A, len_B, target_A, target_B, label_A, label_B))

    return test_data_with_neg


# process batch data
def batch_to_input(batch, pad_int_A, pad_int_B, dict_A):
    seq_A, seq_B, pos_A, pos_B, len_A, len_B, target_A, target_B, \
    label_A, label_B = [], [], [], [], [], [], [], [], [], []
    for data_index in batch:
        len_A.append(data_index[4])
        len_B.append(data_index[5])
    maxlen_A = max(len_A)
    maxlen_B = max(len_B)
    i = 0
    for data_index in batch:
        seq_A.append(data_index[0] + [pad_int_A] * (maxlen_A - len_A[i]))
        seq_B.append(data_index[1] + [pad_int_B] * (maxlen_B - len_B[i]))
        pos_A.append(data_index[2] + [pad_int_A] * (maxlen_A - len_A[i]))
        pos_B.append(data_index[3] + [pad_int_B] * (maxlen_B - len_B[i]))
        target_A.append(data_index[6])
        target_B.append(data_index[7])
        label_A.append(data_index[8])
        label_B.append(data_index[9])
        i += 1
    # index = np.arange(len(batch))
    # index = np.expand_dims(index, axis=-1)
    # index_p = np.repeat(index, maxlen_A, axis=1)
    # pos_A = np.stack([index_p, np.array(pos_A)], axis=-1)
    # index_p = np.repeat(index, maxlen_B, axis=1)
    # pos_B = np.stack([index_p, np.array(pos_B)], axis=-1)


    return np.array(seq_A), np.clip(np.array(seq_B) - len(dict_A), a_min=0, a_max=None), np.array(pos_A), np.array(pos_B), np.array(len_A), \
           np.array(len_B), np.array(target_A), np.clip(np.array(target_B) - len(dict_A), a_min=0, a_max=None), \
           np.array(label_A), np.array(label_B), maxlen_A + maxlen_B


# generate the final batches of training and testing
def get_batches(dataset, batch_size, pad_int_A, pad_int_B, dict_A, shuffle=True):
    seq_A_list, seq_B_list, pos_A_list, pos_B_list, len_A_list, len_B_list, target_A_list, target_B_list = [], [], [], \
                                                                                                           [], [], [], \
                                                                                                           [], []
    label_A_list, label_B_list = [], []

    if shuffle:
        random.shuffle(dataset)
    max_len_mix = 0
    num_batch = int(len(dataset) / batch_size)
    for batch_index in range(0, num_batch):
        start_index = batch_index * batch_size
        batch = dataset[start_index: start_index + batch_size]
        seq_A, seq_B, pos_A, pos_B, len_A, len_B, target_A, target_B, label_A, label_B, max_len= batch_to_input(
            batch=batch, pad_int_A=pad_int_A, pad_int_B=pad_int_B, dict_A=dict_A)
        seq_A_list.append(seq_A)
        seq_B_list.append(seq_B)
        pos_A_list.append(pos_A)
        pos_B_list.append(pos_B)
        len_A_list.append(len_A)
        len_B_list.append(len_B)
        target_A_list.append(target_A)
        target_B_list.append(target_B)
        label_A_list.append(label_A)
        label_B_list.append(label_B)
        if max_len > max_len_mix:
            max_len_mix = max_len

    return list((seq_A_list, seq_B_list, pos_A_list, pos_B_list, len_A_list, len_B_list, target_A_list, target_B_list, label_A_list,
                 label_B_list, num_batch)), max_len_mix
