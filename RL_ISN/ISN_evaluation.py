# _*_coding: UTF_8 _*_
import heapq
import math

from RL_ISN.ISN_configuration import *

args = setting()


def eval_rating(recommender, seq_A, seq_B, pos_A, pos_B, len_A, len_B, target_A, target_B, label_A, label_B,
                s_seq_A, s_seq_B, s_len_A, s_len_B, test_batch_num):
    hits5_A, ndcgs5_A, hits10_A, ndcgs10_A, maps_A, mrrs_A, losses_A = [], [], [], [], [], [], []
    hits5_B, ndcgs5_B, hits10_B, ndcgs10_B, maps_B, mrrs_B, losses_B = [], [], [], [], [], [], []

    for batch in range(test_batch_num):


        test_user_input_A = seq_A[batch]
        test_user_input_B = seq_B[batch]

        test_pos_A = pos_A[batch]
        test_pos_B = pos_B[batch]

        test_num_idx_A = np.reshape(len_A[batch], (-1, 1))
        test_num_idx_B = np.reshape(len_B[batch], (-1, 1))

        test_target_A = target_A[batch]
        test_target_B = target_B[batch]

        test_label_A = np.reshape(label_A[batch], (-1, 1))
        test_label_B = np.reshape(label_B[batch], (-1, 1))

        selected_user_input_B = s_seq_B[batch]
        selected_user_input_A = s_seq_A[batch]

        selected_num_idx_B = np.reshape(s_len_B[batch], (-1, 1))
        selected_num_idx_A = np.reshape(s_len_A[batch], (-1, 1))

        predictions_A, loss_A = recommender.predict_A(test_user_input_A=test_user_input_A,
                                                      test_user_input_B=test_user_input_B,
                                                      pos_A=test_pos_A, pos_B=test_pos_B,
                                                      test_num_idx_A=test_num_idx_A, test_num_idx_B=test_num_idx_B,
                                                      test_target_A=np.reshape(test_target_A, (-1, 1)),
                                                      test_target_B=np.reshape(test_target_B, (-1, 1)),
                                                      test_label_A=test_label_A, test_label_B=test_label_B,
                                                      selected_user_input_B=selected_user_input_B,
                                                      selected_num_idx_B=selected_num_idx_B, is_training=False)

        map_item_score_A = {test_target_A[i]: predictions_A[i] for i in range(len(test_target_A))}
        gtItem = test_target_A[-1]
        if test_label_A[-1] == 0:
            print('A evaluation error')
        ranklist5 = heapq.nlargest(5, map_item_score_A, key=map_item_score_A.get)
        ranklist10 = heapq.nlargest(10, map_item_score_A, key=map_item_score_A.get)
        ranklist100 = heapq.nlargest(100, map_item_score_A, key=map_item_score_A.get)
        hr5 = getHitRatio(ranklist5, gtItem)
        ndcg5 = getNDCG(ranklist5, gtItem)
        hr10 = getHitRatio(ranklist10, gtItem)
        ndcg10 = getNDCG(ranklist10, gtItem)
        ap = getAP(ranklist100, gtItem)
        mrr = getMRR(ranklist100, gtItem)
        hits5_A.append(hr5)
        ndcgs5_A.append(ndcg5)
        hits10_A.append(hr10)
        ndcgs10_A.append(ndcg10)
        maps_A.append(ap)
        mrrs_A.append(mrr)
        losses_A.append(loss_A)

        predictions_B, loss_B = recommender.predict_B(test_user_input_A, test_user_input_B, test_num_idx_A,
                                                      test_num_idx_B,
                                                      np.reshape(test_target_A, (-1, 1)),
                                                      np.reshape(test_target_B, (-1, 1)), test_label_A, test_label_B,
                                                      selected_user_input_A, selected_num_idx_A, is_training=False,
                                                      pos_A=test_pos_A, pos_B=test_pos_B)

        map_item_score_B = {test_target_B[i]: predictions_B[i] for i in range(len(test_target_B))}
        gtItem = test_target_B[-1]
        if test_label_B[-1] == 0:
            print('B evaluation error')
        ranklist5 = heapq.nlargest(5, map_item_score_B, key=map_item_score_B.get)
        ranklist10 = heapq.nlargest(10, map_item_score_B, key=map_item_score_B.get)
        ranklist100 = heapq.nlargest(100, map_item_score_B, key=map_item_score_B.get)
        hr5 = getHitRatio(ranklist5, gtItem)
        ndcg5 = getNDCG(ranklist5, gtItem)
        hr10 = getHitRatio(ranklist10, gtItem)
        ndcg10 = getNDCG(ranklist10, gtItem)
        ap = getAP(ranklist100, gtItem)
        mrr = getMRR(ranklist100, gtItem)
        hits5_B.append(hr5)
        ndcgs5_B.append(ndcg5)
        hits10_B.append(hr10)
        ndcgs10_B.append(ndcg10)
        maps_B.append(ap)
        mrrs_B.append(mrr)
        losses_B.append(loss_B)

    final_hr5_A, final_ndcg5_A, final_hr10_A, final_ndcg10_A, final_map_A, final_mrr_A, final_test_loss_A, \
    final_hr5_B, final_ndcg5_B, final_hr10_B, final_ndcg10_B, final_map_B, final_mrr_B, final_test_loss_B \
        = np.array(hits5_A).mean(), np.array(ndcgs5_A).mean(), np.array(hits10_A).mean(), np.array(ndcgs10_A).mean(), \
          np.array(maps_A).mean(), np.array(mrrs_A).mean(), np.array(losses_A).mean(), \
          np.array(hits5_B).mean(), np.array(ndcgs5_B).mean(), np.array(hits10_B).mean(), np.array(ndcgs10_B).mean(), \
          np.array(maps_B).mean(), np.array(mrrs_B).mean(), np.array(losses_B).mean(),

    return (final_hr5_A, final_ndcg5_A, final_hr10_A, final_ndcg10_A, final_map_A, final_mrr_A, final_test_loss_A,
            final_hr5_B, final_ndcg5_B, final_hr10_B, final_ndcg10_B, final_map_B, final_mrr_B, final_test_loss_B)


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0


def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0


def getAP(ranklist, gtItem):
    hits = 0
    sum_precs = 0
    for n in range(len(ranklist)):
        if ranklist[n] == gtItem:
            hits += 1
            sum_precs += hits / (n + 1.0)
    if hits > 0:
        return sum_precs / 1
    else:
        return 0


def getMRR(ranklist, gtItem):
    for index, item in enumerate(ranklist):
        if item == gtItem:
            return 1.0 / (index + 1.0)
    return 0
