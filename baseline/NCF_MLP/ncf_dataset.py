# encoding: utf-8
'''
Created on Aug 8, 2016
Processing datasets. 

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import scipy.sparse as sp
import numpy as np


class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        '''

        self.testRatings = self.load_rating_file_as_list(path + "_test0.txt")
        self.testNegatives = self.load_negative_file(path + "_negative1.txt")
        # print('neg length:',len(self.testNegatives))
        # print('test length:',len(self.testRatings))
        assert len(self.testRatings) == len(self.testNegatives)
        self.trainMatrix = self.load_rating_file_as_matrix(path + "_train0.txt")
        self.num_users, self.num_items = self.trainMatrix.shape
        self.trainNeg = self.load_trainneg(path + "_trainneg.txt")
        self.trainui = self.load_trainui(path+'_train0.txt')

    def load_rating_file_as_list(self, filename):#test的数据
        ratingList = []
        i=0
        with open(filename, "r") as f:
            line = f.readline()
            list = []
            while line != None and line != "":
                i=i=+1
                arr = line.strip().split("\t")
                user= int(arr[0])
                list.append(user)
                for i in range(len(arr)-1):
                    list.append(int(arr[i+1]))
                ratingList.append(list)
                line = f.readline()
                list=[]
        print('test load over...')
        return ratingList

    def load_negative_file(self, filename):
        negativeList = []
        f = open(filename,'r')
        print(filename)
        while True:
            line = f.readline()
            if line:
                arr = line.strip().split('\t')
                negativeList.append(arr[1:len(arr)])
            else:
                break
        f.close()
        # negativeList = []
        # with open(filename, "r") as f:
        #     line = f.readline()
        #     while line != None and line != "":
        #         arr = line.strip().split('\t')
        #         # negatives = []
        #         # for i in range(len(arr)-2):
        #         #     negatives.append(int(arr[i+1]))
        #         # negativeList.append(negatives)
        #         negativeList.append(arr[1:len(arr)])
        #         line = f.readline()
        print('negative load over...')
        return negativeList

    def load_rating_file_as_matrix(self, filename):#train数据
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.strip().split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                mat[user, item] = 1.0 + mat[user, item]
                line = f.readline()
        print('train load over...')
        return mat

    def load_trainneg(self,filename):
        trainnegArr = []
        f = open(filename,'r')
        while True:
            line = f.readline()
            if line:
                line = line.strip().split('\t')
                trainnegArr.append(line)
            else:
                break
        f.close()
        return trainnegArr

    def load_trainui(self,filename):
        trainui = []
        f = open(filename,'r')
        while True:
            line = f.readline()
            itemtmp = []
            if line:
                linelist = line.strip().split('\t')
                userID = linelist[0]
                itemID = linelist[1]
                itemtmp.append(itemID)
                line1 = f.readline()
                itemID1 = 0
                while line1:
                    linelist1 = line1.strip().split('\t')
                    userID1 = linelist1[0]
                    itemID1 = linelist1[1]
                    if userID1 == userID:
                        itemtmp.append(itemID1)
                    else:
                        userID = userID1
                        trainui.append(itemtmp)
                        itemtmp = []
                        itemtmp.append(itemID1)
                    line1 = f.readline()
                itemtmp.append(itemID1)
                trainui.append(itemtmp)
                break
            else:
                break
        f.close()
        return trainui
