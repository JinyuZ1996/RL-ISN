'''
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Recall and MRR
'''
import math
import heapq  # for retrieval topK
import multiprocessing
import numpy as np
from time import time

# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_testRatings2 = None
_testNegatives2 = None
_K = None


def evaluate_model(model, testRatings, testNegatives,testRatings2, testNegatives2, K, num_thread):
    """
    Evaluate the performance (Recall, MRR) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _testRatings2
    global _testNegatives2

    global _K
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _testRatings2 = testRatings2
    _testNegatives2 = testNegatives2
    _K = K

    recalls, mrrs,recalls2, mrrs2 = [], [],[],[]
    if (num_thread > 1):  # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()
        recalls = [r[0] for r in res]
        mrrs = [r[1] for r in res]
        return (recalls, mrrs)
    # Single thread
    print('test len:',len(_testRatings))
    for idx in xrange(len(_testRatings)):
        (recall, mrr,recall2, mrr2) = eval_one_rating(idx)
        recalls.append(recall)
        mrrs.append(mrr)
        recalls2.append(recall2)
        mrrs2.append(mrr2)
        # print(str(idx)+';recall:'+str(recalls)+';mrr:'+str(mrrs))
        # print(str(idx) + ';recall2:' + str(recalls2) + ';mrr:' + str(mrrs2))

    return (recalls, mrrs,recalls2, mrrs2)


def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    rating2 = _testRatings2[idx]
    items2 = _testNegatives2[idx]
    u = rating[0]

    # Get prediction scores
    users = np.full(len(items), u, dtype='int32')
    [predictions,predictions2] = _model.predict([users, np.array(items),np.array(items2)],
                                 batch_size=100, verbose=0)
    # print('predict:',len(predictions))
    # print('predict2:',len(predictions2))
    # Evaluate A domain
    map_item_score = {}
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    truth1 = rating[1:len(rating)]
    # print('ranklist1:',ranklist)
    # print('truth1:',truth1)
    Recall,MRR = get_eval(ranklist,truth1)
    # print('recall1',Recall)
    # print('mrr1', MRR)

    # evaluate B domain
    map_item_score2 = {}
    for i in range(len(items2)):
        item = items2[i]
        map_item_score2[item] = predictions2[i]
    ranklist2 = heapq.nlargest(_K, map_item_score2, key=map_item_score2.get)
    truth2 = rating2[1:len(rating2)]
    # print('ranklist2:', ranklist2)
    # print('truth2:',truth2)
    Recall2,MRR2 = get_eval(ranklist2,truth2)
    # print('recall2', Recall2)
    # print('mrr2', MRR2)

    return Recall,MRR,Recall2,MRR2

def get_eval(ranklist,truth):
    recall = 0.0
    mrr = 0.0
    for item in truth:
        recall += getHitRatio(ranklist, item)
        mrr += getMRR(ranklist, item)
    return recall,mrr

def getHitRatio(ranklist, gtItem):
    ranklist = map(int, ranklist)
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getMRR(ranklist, gtItem):
    ranklist = map(int, ranklist)
    if gtItem in ranklist:
        index = ranklist.index(gtItem)+1
        re = 1.0 /index
        return re
    else :
        return 0

