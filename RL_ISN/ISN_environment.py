# _*_coding: UTF_8 _*_
import os
from RL_ISN.ISN_configuration import *

olderr = np.seterr(all='ignore')  # to avoid：RuntimeWarning: invalid value encountered in true_divide

args = setting()


def get_cos_similarity(user_input, item_input):
    norm_user = np.sqrt(np.sum(np.multiply(user_input, user_input), axis=1))
    norm_item = np.sqrt(np.sum(np.multiply(item_input, item_input), axis=1))
    norm = np.multiply(norm_user, norm_item)
    dot_prod = np.sum(np.multiply(user_input, item_input), 1)
    cos_similarity = np.where(norm != 0, dot_prod / norm, dot_prod)
    return np.reshape(cos_similarity, (-1, 1))


class Environment(object):
    def __init__(self, recommender, agent=None):

        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
        self.sess = None
        self.recommender = recommender
        self.agent = agent
        self.test_item_ebd_A = None
        self.test_item_ebd_B = None
        self.train_item_ebd_B = None
        self.train_item_ebd_A = None
        self.inner_reward_mask = args.inner_reward_mask
        self.Random = False

    def initilize_state(self, recommender, train_data, test_data, high_state_size, A_state_size, B_state_size,
                        padding_number_A, padding_number_B):
        self.high_state_size = high_state_size
        self.low_state_size_A = A_state_size
        self.low_state_size_B = B_state_size
        self.padding_number_A = padding_number_A
        self.padding_number_B = padding_number_B
        self.data_embedding_user_A, self.data_embedding_item_A = recommender.get_data_embedding_A()
        self.data_embedding_user_B, self.data_embedding_item_B = recommender.get_data_embedding_B()

        # if 'selected_from_domain' is "A" ,the reward should be given to Agent_A
        self.origin_train_rewards_A = recommender.get_origin_rewards(train_data, selected_from_domain="A")
        self.origin_train_rewards_B = recommender.get_origin_rewards(train_data, selected_from_domain="B")
        self.origin_test_rewards_A = recommender.get_origin_rewards(test_data, selected_from_domain="A")
        self.origin_test_rewards_B = recommender.get_origin_rewards(test_data, selected_from_domain="B")

        self.embedding_size_A = len(self.data_embedding_user_A[0])
        self.embedding_size_B = len(self.data_embedding_user_B[0])

        self.set_train_original_rewards()

    def set_train_original_rewards(self):
        self.origin_rewards_4_Agent_A = self.origin_train_rewards_A
        self.origin_rewards_4_Agent_B = self.origin_train_rewards_B

    def set_test_original_rewards(self):
        self.origin_rewards_4_Agent_A = self.origin_test_rewards_A
        self.origin_rewards_4_Agent_B = self.origin_test_rewards_B

    def reset_A_state(self, user_input_A, user_input_B, num_idx_A, num_idx_B, target_A, target_B, label_A, label_B,
                      batch_size, max_item_num_A, max_item_num_B, batch_index):
        self.user_input_A = user_input_A
        self.user_input_B = user_input_B
        self.num_idx_A = num_idx_A
        self.num_idx_B = num_idx_B
        self.item_input_A = np.reshape(target_A, (-1,))
        self.item_input_B = np.reshape(target_B, (-1,))
        self.label_A = label_A
        self.label_B = label_B
        self.batch_size = batch_size
        self.max_item_num_A = max_item_num_A
        self.max_item_num_B = max_item_num_B
        self.batch_index_A = batch_index
        self.cos_sum_A = np.zeros((self.batch_size, 1), dtype=np.float32)
        self.cos_mean_A = np.zeros((self.batch_size, 1), dtype=np.float32)
        self.element_wise_mean_A = np.zeros((self.batch_size, self.embedding_size_A), dtype=np.float32)
        self.element_wise_sum_A = np.zeros((self.batch_size, self.embedding_size_A), dtype=np.float32)
        self.vector_sum_A = np.zeros((self.batch_size, self.embedding_size_A), dtype=np.float32)
        self.vector_mean_A = np.zeros((self.batch_size, self.embedding_size_A), dtype=np.float32)
        self.num_selected_A = np.zeros(self.batch_size, dtype=np.int)
        self.action_matrix_A = np.zeros((self.batch_size, self.max_item_num_A), dtype=np.int)
        self.state_matrix_A = np.zeros((self.batch_size, self.max_item_num_A, self.low_state_size_A), dtype=np.float32)
        self.selected_input_A = np.full((self.batch_size, self.max_item_num_A), self.padding_number_A)

    def reset_B_state(self, user_input_B, user_input_A, num_idx_B, num_idx_A, target_A, target_B, label_A, label_B,
                      batch_size, max_item_num_B, max_item_num_A, batch_index):
        self.user_input_B = user_input_B
        self.user_input_A = user_input_A
        self.num_idx_B = num_idx_B
        self.num_idx_A = num_idx_A
        self.item_input_A = np.reshape(target_A, (-1,))
        self.item_input_B = np.reshape(target_B, (-1,))
        self.label_A = label_A
        self.label_B = label_B
        self.batch_size = batch_size
        self.max_item_num_B = max_item_num_B
        self.max_item_num_A = max_item_num_A
        self.batch_index_B = batch_index
        self.cos_sum_B = np.zeros((self.batch_size, 1), dtype=np.float32)
        self.cos_mean_B = np.zeros((self.batch_size, 1), dtype=np.float32)
        self.element_wise_mean_B = np.zeros((self.batch_size, self.embedding_size_B), dtype=np.float32)
        self.element_wise_sum_B = np.zeros((self.batch_size, self.embedding_size_B), dtype=np.float32)
        self.vector_sum_B = np.zeros((self.batch_size, self.embedding_size_B), dtype=np.float32)
        self.vector_mean_B = np.zeros((self.batch_size, self.embedding_size_B), dtype=np.float32)
        self.num_selected_B = np.zeros(self.batch_size, dtype=np.int)
        self.action_matrix_B = np.zeros((self.batch_size, self.max_item_num_B), dtype=np.int)
        self.state_matrix_B = np.zeros((self.batch_size, self.max_item_num_B, self.low_state_size_B), dtype=np.float32)
        self.selected_input_B = np.full((self.batch_size, self.max_item_num_B), self.padding_number_B)

    def get_high_state_A(self):
        def _mask(i):
            return [True] * i[0] + [False] * (self.max_item_num_A - i[0])
        origin_prob = np.reshape(self.origin_rewards_4_Agent_A[self.batch_index_A], (-1, 1))  # (batch_size, 1)
        self.num_idx_A = np.reshape(self.num_idx_A, (-1, 1))
        cos_similarity = self.rank_cos_by_matrix_A(self.user_input_A, self.item_input_B)
        element_wise = self.rank_element_wise_by_matrix_A(self.user_input_A, self.item_input_B)
        mask_mat = np.array(list(map(_mask, np.reshape(self.num_idx_A, (self.batch_size, 1)))))
        cos_similarity = np.reshape(np.sum(cos_similarity * mask_mat, 1), (-1, 1)) / self.num_idx_A
        mask_mat = np.repeat(np.reshape(mask_mat, (self.batch_size, self.max_item_num_A, 1)), self.embedding_size_A, 2)
        element_wise = np.sum(element_wise * mask_mat, 1) / self.num_idx_A
        return np.concatenate((cos_similarity, element_wise, origin_prob), 1)

    def get_high_state_B(self):
        def _mask(i):
            return [True] * i[0] + [False] * (self.max_item_num_B - i[0])

        origin_prob = np.reshape(self.origin_rewards_4_Agent_B[self.batch_index_B], (-1, 1))  # (batch_size, 1)
        self.num_idx_B = np.reshape(self.num_idx_B, (-1, 1))
        cos_similarity = self.rank_cos_by_matrix_B(self.user_input_B, self.item_input_A)
        element_wise = self.rank_element_wise_by_matrix_B(self.user_input_B, self.item_input_A)
        mask_mat = np.array(list(map(_mask, np.reshape(self.num_idx_B, (self.batch_size, 1)))))
        cos_similarity = np.reshape(np.sum(cos_similarity * mask_mat, 1), (-1, 1)) / self.num_idx_B
        mask_mat = np.repeat(np.reshape(mask_mat, (self.batch_size, self.max_item_num_B, 1)), self.embedding_size_B, 2)
        element_wise = np.sum(element_wise * mask_mat, 1) / self.num_idx_B
        return np.concatenate((cos_similarity, element_wise, origin_prob), 1)

    def get_high_action(self, prob, Random):
        batch_size = prob.shape[0]
        if Random:
            random_number = np.random.rand(batch_size)
            return np.where(random_number < prob, np.ones(batch_size, dtype=np.int), np.zeros(batch_size, dtype=np.int))
        else:
            return np.where(prob >= 0.5, np.ones(batch_size, dtype=np.int), np.zeros(batch_size, dtype=np.int))

    def get_low_action(self, prob, user_input_column, padding_number, Random):

        batch_size = prob.shape[0]
        if Random:
            random_number = np.random.rand(batch_size)
            return np.where((random_number < prob) & (user_input_column != padding_number),
                            np.ones(batch_size, dtype=np.int),
                            np.zeros(batch_size, dtype=np.int))
        else:
            return np.where((prob >= 0.5) & (user_input_column != padding_number),
                            np.ones(batch_size, dtype=np.int),
                            np.zeros(batch_size, dtype=np.int))

    def get_low_state_A(self, item_index):
        self.cos_similarity_A = self.rank_cos_A(user_input_A=self.user_input_A, item_input_B=self.item_input_B,
                                                item_index=item_index)
        self.element_wise_current_A = self.rank_element_wise_A(user_input_A=self.user_input_A,
                                                               item_input_B=self.item_input_B, item_index=item_index)
        self.vector_current_A = self.data_embedding_user_A[self.user_input_A[:, item_index]]
        self.vector_item_B = self.data_embedding_item_B[self.item_input_B]
        self.vector_current_A = np.abs(self.vector_current_A - self.vector_item_B)
        return np.concatenate((self.vector_mean_A,
                               self.vector_current_A,
                               self.cos_similarity_A,
                               self.cos_mean_A
                               ), 1)  # 16+16+1+1

    def get_low_state_B(self, item_index):
        self.cos_similarity_B = self.rank_cos_B(user_input_B=self.user_input_B, item_input_A=self.item_input_A,
                                                item_index=item_index)
        self.element_wise_current_B = self.rank_element_wise_B(user_input_B=self.user_input_B,
                                                               item_input_A=self.item_input_A, item_index=item_index)
        self.vector_current_B = self.data_embedding_item_B[self.user_input_B[:, item_index]]  # 来源于item_embedding
        self.vector_item_A = self.data_embedding_item_A[self.item_input_A]
        self.vector_current_B = np.abs(self.vector_current_B - self.vector_item_A)
        return np.concatenate((self.vector_mean_B,
                               self.vector_current_B,
                               self.cos_similarity_B,
                               self.cos_mean_B
                               ), 1)  # 16+16+1+1

    def rank_element_wise_A(self, user_input_A, item_input_B, item_index):
        self.train_item_ebd_A = self.data_embedding_user_A[user_input_A[:, item_index]]
        self.test_item_ebd_B = np.reshape(self.data_embedding_item_B[item_input_B],
                                          (self.batch_size, self.embedding_size_B))
        return np.multiply(self.train_item_ebd_A, self.test_item_ebd_B)  # (batch_size, embedding_size_A)

    def rank_element_wise_B(self, user_input_B, item_input_A, item_index):
        self.train_item_ebd_B = self.data_embedding_user_B[user_input_B[:, item_index]]
        self.test_item_ebd_A = np.reshape(self.data_embedding_item_A[item_input_A],
                                          (self.batch_size, self.embedding_size_A))
        return np.multiply(self.train_item_ebd_B, self.test_item_ebd_A)  # (batch_size, embedding_size_B)

    def rank_cos_A(self, user_input_A, item_input_B, item_index):
        self.train_item_ebd_A = self.data_embedding_user_A[user_input_A[:, item_index]]
        self.test_item_ebd_B = np.reshape(self.data_embedding_item_B[item_input_B],
                                          (self.batch_size, self.embedding_size_B))
        cos_similarity = get_cos_similarity(user_input=self.train_item_ebd_A, item_input=self.test_item_ebd_B)
        return cos_similarity

    def rank_cos_B(self, user_input_B, item_input_A, item_index):
        self.train_item_ebd_B = self.data_embedding_user_B[user_input_B[:, item_index]]
        self.test_item_ebd_A = np.reshape(self.data_embedding_item_A[item_input_A],
                                          (self.batch_size, self.embedding_size_A))
        cos_similarity = get_cos_similarity(user_input=self.train_item_ebd_B, item_input=self.test_item_ebd_A)
        return cos_similarity


    def rank_element_wise_by_matrix_A(self, user_input_A, item_input_B):
        self.train_item_ebd_A = self.data_embedding_user_A[np.reshape(user_input_A, (-1, 1))]
        self.test_item_ebd_B = self.data_embedding_item_B[
            np.reshape(np.tile(item_input_B, (1, self.max_item_num_A)), (-1, 1))]
        return np.reshape(np.multiply(self.train_item_ebd_A, self.test_item_ebd_B),
                          (-1, self.max_item_num_A, self.embedding_size_A))

    def rank_element_wise_by_matrix_B(self, user_input_B, item_input_A):
        self.train_item_ebd_B = self.data_embedding_user_B[np.reshape(user_input_B, (-1, 1))]
        self.test_item_ebd_A = self.data_embedding_item_A[
            np.reshape(np.tile(item_input_A, (1, self.max_item_num_B)), (-1, 1))]
        return np.reshape(np.multiply(self.train_item_ebd_B, self.test_item_ebd_A),
                          (-1, self.max_item_num_B, self.embedding_size_B))


    def rank_cos_by_matrix_A(self, user_input_A, item_input_B):
        self.train_item_ebd_A = self.data_embedding_user_A[np.reshape(user_input_A, (-1,))]
        self.test_item_ebd_B = self.data_embedding_item_B[
            np.reshape(np.tile(item_input_B, (1, self.max_item_num_A)), (-1,))]
        norm_user = np.sqrt(np.sum(np.multiply(self.train_item_ebd_A, self.train_item_ebd_A), 1))
        norm_item = np.sqrt(np.sum(np.multiply(self.test_item_ebd_B, self.test_item_ebd_B), 1))
        norm = np.multiply(norm_user, norm_item)
        dot_prod = np.sum(np.multiply(self.train_item_ebd_A, self.test_item_ebd_B), 1)
        cos_similarity = np.where(norm != 0, dot_prod / norm, dot_prod)
        return np.reshape(cos_similarity, (-1, self.max_item_num_A))  # (batch_size, 1)

    def rank_cos_by_matrix_B(self, user_input_B, item_input_A):
        self.train_item_ebd_B = self.data_embedding_user_B[
            np.reshape(user_input_B, (-1,))]  # (batch_size, embedding_size)
        self.test_item_ebd_A = self.data_embedding_item_A[
            np.reshape(np.tile(item_input_A, (1, self.max_item_num_B)), (-1,))]  # (batch_size, embedding_size)
        norm_user = np.sqrt(np.sum(np.multiply(self.train_item_ebd_B, self.train_item_ebd_B), 1))
        norm_item = np.sqrt(np.sum(np.multiply(self.test_item_ebd_A, self.test_item_ebd_A), 1))
        norm = np.multiply(norm_user, norm_item)
        dot_prod = np.sum(np.multiply(self.train_item_ebd_B, self.test_item_ebd_A), 1)
        cos_similarity = np.where(norm != 0, dot_prod / norm, dot_prod)
        return np.reshape(cos_similarity, (-1, self.max_item_num_B))  # (batch_size, 1)

    def update_low_state_A(self, low_action_A, low_state_A, item_index):
        self.action_matrix_A[:, item_index] = low_action_A
        self.state_matrix_A[:, item_index] = low_state_A

        self.num_selected_A = self.num_selected_A + low_action_A
        self.vector_sum_A = self.vector_sum_A + np.multiply(np.reshape(low_action_A, (-1, 1)), self.vector_current_A)
        self.element_wise_sum_A = self.element_wise_sum_A + np.multiply(np.reshape(low_action_A, (-1, 1)),
                                                                        self.element_wise_current_A)
        self.cos_sum_A = self.cos_sum_A + np.multiply(np.reshape(low_action_A, (-1, 1)),
                                                      self.cos_similarity_A)
        num_selected_array_A = np.reshape(self.num_selected_A, (-1, 1))
        self.element_wise_mean_A = np.where(num_selected_array_A != 0, self.element_wise_sum_A / num_selected_array_A,
                                            self.element_wise_sum_A)
        self.vector_mean_A = np.where(num_selected_array_A != 0, self.vector_sum_A / num_selected_array_A,
                                      self.vector_sum_A)
        self.cos_mean_A = np.where(num_selected_array_A != 0, self.cos_sum_A / num_selected_array_A,
                                   self.cos_sum_A)

    def update_low_state_B(self, low_action_B, low_state_B, item_index):
        self.action_matrix_B[:, item_index] = low_action_B
        self.state_matrix_B[:, item_index] = low_state_B
        self.num_selected_B = self.num_selected_B + low_action_B
        self.vector_sum_B = self.vector_sum_B + np.multiply(np.reshape(low_action_B, (-1, 1)), self.vector_current_B)
        self.cos_sum_B = self.cos_sum_B + np.multiply(np.reshape(low_action_B, (-1, 1)),
                                                      self.cos_similarity_B)
        num_selected_array_B = np.reshape(self.num_selected_B, (-1, 1))
        self.element_wise_mean_B = np.where(num_selected_array_B != 0, self.element_wise_sum_B / num_selected_array_B,
                                            self.element_wise_sum_B)
        self.vector_mean_B = np.where(num_selected_array_B != 0, self.vector_sum_B / num_selected_array_B,
                                      self.vector_sum_B)
        self.cos_mean_B = np.where(num_selected_array_B != 0, self.cos_sum_B / num_selected_array_B,
                                   self.cos_sum_B)

    def get_action_matrix(self, domain):
        if domain == "A":
            return self.action_matrix_A
        else:
            return self.action_matrix_B

    def get_state_matrix(self, domain):
        if domain == "A":
            return self.state_matrix_A
        else:
            return self.state_matrix_B

    # 决定删除与否 更新于 2022-04-29
    def get_selected_items_A(self, high_action):
        notrevised_index = []
        revised_index = []
        delete_index = []
        keep_index = []
        select_user_input = np.zeros((self.batch_size, self.max_item_num_A), dtype=np.int)

        for index in range(self.batch_size):

            selected = []
            for item_index in range(self.max_item_num_A):
                if self.action_matrix_A[index, item_index] == 1:  # 1
                    selected.append(self.user_input_A[index, item_index])

            # revise
            if high_action[index] == 1:
                #  只需要让这里把整句都delete就行了
                if args.inner_reward_mask == 0:
                    delete_index.append(index)
                else:
                    # delete
                    if len(selected) == 0:
                        delete_index.append(index)
                    # keep
                    if len(selected) == self.num_idx_A[index]:
                        keep_index.append(index)
                revised_index.append(index)
            # not revise
            if high_action[index] == 0:
                notrevised_index.append(index)
            if len(selected) == 0:
                original_item_set = list(set(self.user_input_A[index]))
                if self.padding_number_A in original_item_set:
                    original_item_set.remove(self.padding_number_A)
                random_item = np.random.choice(original_item_set, 1)[0]
                selected.append(random_item)
                self.num_selected_A[index] = 1

            for item_index in range(self.max_item_num_A - len(selected)):
                selected.append(self.padding_number_A)
            select_user_input[index, :] = np.array(selected)

        nochanged = notrevised_index + keep_index
        select_user_input[nochanged] = self.user_input_A[nochanged]
        self.num_selected_A[nochanged] = np.reshape(self.num_idx_A[nochanged], (-1,))
        return select_user_input, self.num_selected_A, notrevised_index, revised_index, delete_index, keep_index

    def get_selected_items_B(self, high_action):
        notrevised_index = []
        revised_index = []
        delete_index = []
        keep_index = []
        select_user_input = np.zeros((self.batch_size, self.max_item_num_B), dtype=np.int)

        for index in range(self.batch_size):

            selected = []
            for item_index in range(self.max_item_num_B):
                if self.action_matrix_B[index, item_index] == 1:
                    selected.append(self.user_input_B[index, item_index])
            # revise
            if high_action[index] == 1:
                # delete
                if len(selected) == 0:
                    delete_index.append(index)
                # keep
                if len(selected) == self.num_idx_B[index]:
                    keep_index.append(index)
                revised_index.append(index)
            # not revise
            if high_action[index] == 0:
                notrevised_index.append(index)
            if len(selected) == 0:
                original_item_set = list(set(self.user_input_B[index]))
                if self.padding_number_B in original_item_set:
                    original_item_set.remove(self.padding_number_B)
                random_item = np.random.choice(original_item_set, 1)[0]
                selected.append(random_item)
                self.num_selected_B[index] = 1

            for item_index in range(self.max_item_num_B - len(selected)):
                selected.append(self.padding_number_B)
            select_user_input[index, :] = np.array(selected)

        nochanged = notrevised_index + keep_index
        select_user_input[nochanged] = self.user_input_B[nochanged]
        self.num_selected_B[nochanged] = np.reshape(self.num_idx_B[nochanged], (-1,))
        return select_user_input, self.num_selected_B, notrevised_index, revised_index, delete_index, keep_index


    def get_env_reward(self, recommender, batch_index, high_actions, seq_A, seq_B, pos_A, pos_B, len_A, len_B, target_A, target_B,
                       s_seq_A, s_seq_B, label_A, label_B, domain):

        batch_size_A = s_seq_A.shape[0]
        batch_size_B = s_seq_B.shape[0]

        # while domain is "A", the reward should be given to agent_A, so that we need to caculate the effect of s_seq_A
        if domain == "A":
            recommender_reward, _, _ = \
                recommender.reward_from_recommender(selected_from_domain="A",
                                                    seq_A=seq_A, len_A=len_A,
                                                    seq_B=seq_B, len_B=len_B,
                                                    s_seq_A=s_seq_A, s_len_A=np.reshape(self.num_selected_A, (-1, 1)),
                                                    s_seq_B=s_seq_B, s_len_B=np.reshape(self.num_selected_B, (-1, 1)),
                                                    target_A=target_A, target_B=target_B,
                                                    label_A=label_A, label_B=label_B, is_training=True, pos_A = pos_A,
                                                    pos_B = pos_B)

            recommender_origin_reward = self.origin_rewards_4_Agent_A[batch_index]
            reward_diff = recommender_reward - recommender_origin_reward
            reward_diff = np.where(high_actions == 1, reward_diff, np.zeros(batch_size_A))
            cos_similarity = self.rank_cos_by_matrix_A(s_seq_A, target_B)
            new_dot_product = np.sum(np.multiply(cos_similarity, self.action_matrix_A),
                                     1) / self.num_selected_A
            old_dot_product = np.sum(cos_similarity, 1) / np.reshape(len_B, (-1,))

        # while domain is "B", the reward should be given to agent_B, so that we need to caculate the effect of s_seq_B
        else:
            recommender_reward, _, _ = \
                recommender.reward_from_recommender(selected_from_domain="B",
                                                    seq_A=seq_A, len_A=len_A,
                                                    seq_B=seq_B, len_B=len_B,
                                                    s_seq_A=s_seq_A, s_len_A=np.reshape(self.num_selected_A, (-1, 1)),
                                                    s_seq_B=s_seq_B, s_len_B=np.reshape(self.num_selected_B, (-1, 1)),
                                                    target_A=target_A, target_B=target_B,
                                                    label_A=label_A, label_B=label_B, is_training=True, pos_A=pos_A,
                                                    pos_B=pos_B)
            recommender_origin_reward = self.origin_rewards_4_Agent_B[batch_index]
            reward_diff = recommender_reward - recommender_origin_reward
            reward_diff = np.where(high_actions == 1, reward_diff, np.zeros(batch_size_B))
            cos_similarity = self.rank_cos_by_matrix_B(s_seq_B, target_A)
            new_dot_product = np.sum(np.multiply(cos_similarity, self.action_matrix_B), 1) / self.num_selected_B
            old_dot_product = np.sum(cos_similarity, 1) / np.reshape(len_A, (-1,))

        reward_sum = np.reshape(new_dot_product, (-1,)) - old_dot_product
        reward_output = reward_diff + self.inner_reward_mask * reward_sum

        return reward_diff, old_dot_product, reward_output
