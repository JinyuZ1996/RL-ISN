import numpy as np
from pop_settings import *

args = setting()


class Pop_model:
    def __init__(self, support_by_key=None):
        self.top_n = args.top_N
        # self.user_A = user  # item_key
        # self.user_B = user
        self.support_by_key = support_by_key

    def train_recommender(self, items_matrix_A, items_matrix_B):
        # Domain_A
        grp_A = items_matrix_A.groupby('user')
        self.pop_list_A = grp_A.size() if self.support_by_key is None else grp_A[self.support_by_key].nunique()
        self.pop_list_A = self.pop_list_A / (self.pop_list_A + 1)
        self.pop_list_A.sort_values(ascending=False, inplace=True)
        self.pop_list_A = self.pop_list_A.head(self.top_n)

        # Domain_B
        grp_B = items_matrix_B.groupby('user')
        self.pop_list_B = grp_B.size() if self.support_by_key is None else grp_B[self.support_by_key].nunique()
        self.pop_list_B = self.pop_list_B / (self.pop_list_B + 1)
        self.pop_list_B.sort_values(ascending=False, inplace=True)
        self.pop_list_B = self.pop_list_B.head(self.top_n)

    def prediction_A(self, target_A):
        # calculated the prediction of Domain-A
        self.preds_A = np.zeros(len(target_A))
        mask = np.in1d(target_A, self.pop_list_A.index)
        self.preds_A[mask] = self.pop_list_A[target_A[mask]]

        return self.preds_A

    def prediction_B(self, target_B):
        # calculated the prediction of Domain-B
        self.preds_B = np.zeros(len(target_B))
        mask = np.in1d(target_B, self.pop_list_B.index)
        self.preds_B[mask] = self.pop_list_A[target_B[mask]]

        return self.preds_B
