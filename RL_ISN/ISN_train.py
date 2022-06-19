# _*_coding: UTF_8 _*_
import logging
import os

import numpy as np

from RL_ISN.ISN_evaluation import *

global env

args = setting()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num


def sampling_RL(environment, recommender, agent, seq_to_revise, seq_to_improve, len_A, len_B, target_A, target_B,
                label_A, label_B, batch_index, Random=args.random, domain="B", keep_pro=1.0):
    batch_size_reviseSeq = seq_to_revise.shape[0]
    max_item_num_reviseSeq = seq_to_revise.shape[1]
    max_item_num_improveSeq = seq_to_improve.shape[1]
    environment.Random = Random

    if domain is "A":
        environment.reset_A_state(user_input_A=seq_to_revise, user_input_B=seq_to_improve,
                                  num_idx_A=len_A, num_idx_B=len_B,
                                  target_A=target_A, target_B=target_B,
                                  label_A=label_A, label_B=label_B,
                                  batch_size=batch_size_reviseSeq,
                                  max_item_num_A=max_item_num_reviseSeq,
                                  max_item_num_B=max_item_num_improveSeq,
                                  batch_index=batch_index)

        high_state_A = environment.get_high_state_A()

        high_prob_A, _, _, _, _ = agent.predict_high_target_A(high_state=high_state_A, keep_pro=keep_pro)

        high_action_A = environment.get_high_action(prob=high_prob_A, Random=Random)

        for index in range(max_item_num_reviseSeq):


            low_state_A = environment.get_low_state_A(item_index=index)

            low_prob_A, _, _, _, _ = agent.predict_low_target_A(low_state_A=low_state_A, keep_pro=keep_pro)
            low_action_A = environment.get_low_action(prob=low_prob_A, user_input_column=seq_to_revise[:, index],
                                                      padding_number=recommender.padding_A,
                                                      Random=Random)
            environment.update_low_state_A(low_action_A=low_action_A, low_state_A=low_state_A, item_index=index)

        select_user_input, select_num_idx, notrevised_index, revised_index, delete_index, keep_index = \
            environment.get_selected_items_A(high_action=high_action_A)

        high_state = high_state_A
        high_action = high_action_A
    else:
        environment.reset_B_state(user_input_B=seq_to_revise, user_input_A=seq_to_improve,
                                  num_idx_B=len_B, num_idx_A=len_A,
                                  target_A=target_A, target_B=target_B,
                                  label_A=label_A, label_B=label_B,
                                  batch_size=batch_size_reviseSeq,
                                  max_item_num_B=max_item_num_reviseSeq,
                                  max_item_num_A=max_item_num_improveSeq,
                                  batch_index=batch_index)

        high_state_B = environment.get_high_state_B()

        high_prob_B, _, _, _, _ = agent.predict_high_target_B(high_state=high_state_B, keep_pro=keep_pro)

        high_action_B = environment.get_high_action(prob=high_prob_B, Random=Random)

        for index in range(max_item_num_reviseSeq):
            low_state_B = environment.get_low_state_B(item_index=index)

            low_prob_B, _, _, _, _ = agent.predict_low_target_B(low_state_B=low_state_B, keep_pro=keep_pro)
            low_action_B = environment.get_low_action(prob=low_prob_B, user_input_column=seq_to_revise[:, index],
                                                      padding_number=recommender.padding_B,
                                                      Random=Random)
            environment.update_low_state_B(low_action_B=low_action_B, low_state_B=low_state_B, item_index=index)
        select_user_input, select_num_idx, notrevised_index, revised_index, delete_index, keep_index = environment.get_selected_items_B(
            high_action_B)
        high_state = high_state_B
        high_action = high_action_B

    return high_action, high_state, select_user_input, select_num_idx, label_A, label_B, \
           notrevised_index, revised_index, delete_index, keep_index


def train(env, agent, recommender, batches_train, args, dict_A, dict_B, recommender_trainable=True,
          agent_trainable=True):
    seq_A_all, seq_B_all, pos_A_all, pos_B_all, len_A_all, len_B_all, target_A_all, target_B_all, label_A_all, \
    label_B_all, train_batch_num = (batches_train[0], batches_train[1], batches_train[2], batches_train[3],
                                    batches_train[4], batches_train[5], batches_train[6], batches_train[7],
                                    batches_train[8], batches_train[9], batches_train[10])

    sample_times = 3
    high_state_size = args.high_state_size  # 18
    low_state_size_A = args.A_state_size  # 34
    low_state_size_B = args.B_state_size  # 34

    avg_loss_A = 0
    avg_loss_B = 0

    avg_A_high_loss = 0
    avg_B_high_loss = 0
    avg_A_low_loss = 0
    avg_B_low_loss = 0

    shuffled_batch_indexes = np.random.permutation(train_batch_num)

    for batch_index in shuffled_batch_indexes:

        origin_seq_A = np.array([seq for seq in seq_A_all[batch_index]])
        origin_seq_B = np.array([seq for seq in seq_B_all[batch_index]])

        pos_A = np.array([pos for pos in pos_A_all[batch_index]])
        pos_B = np.array([pos for pos in pos_B_all[batch_index]])

        target_items_A = np.reshape(target_A_all[batch_index], (-1, 1))
        target_items_B = np.reshape(target_B_all[batch_index], (-1, 1))

        label_A = np.reshape(label_A_all[batch_index], (-1, 1))
        label_B = np.reshape(label_B_all[batch_index], (-1, 1))

        origin_length_A = np.reshape(len_A_all[batch_index], (-1, 1))
        origin_length_B = np.reshape(len_B_all[batch_index], (-1, 1))

        batch_size_A = origin_seq_A.shape[0]
        batch_size_B = origin_seq_B.shape[0]

        max_item_num_A = origin_seq_A.shape[1]
        max_item_num_B = origin_seq_B.shape[1]

        agent.assign_active_high_network_A()
        agent.assign_active_high_network_B()
        agent.assign_active_A_network()
        agent.assign_active_B_network()

        recommender.assign_active_network_A()
        recommender.assign_active_network_B()

        train_loss_A = 0
        train_loss_B = 0

        if agent_trainable is True:

            sampled_high_states_A = np.zeros((sample_times, batch_size_A, high_state_size), dtype=np.float32)
            sampled_high_actions_A = np.zeros((sample_times, batch_size_A), dtype=np.int)
            sampled_low_states_A = np.zeros((sample_times, batch_size_A, max_item_num_A, low_state_size_A),
                                            dtype=np.float32)
            sampled_low_actions_A = np.zeros((sample_times, batch_size_A, max_item_num_A), dtype=np.float32)

            sampled_high_rewards_A = np.zeros((sample_times, batch_size_A), dtype=np.float32)
            sampled_low_rewards_A = np.zeros((sample_times, batch_size_A), dtype=np.float32)

            sampled_high_states_B = np.zeros((sample_times, batch_size_B, high_state_size), dtype=np.float32)
            sampled_high_actions_B = np.zeros((sample_times, batch_size_B), dtype=np.int)
            sampled_low_states_B = np.zeros((sample_times, batch_size_B, max_item_num_B, low_state_size_B),
                                            dtype=np.float32)
            sampled_low_actions_B = np.zeros((sample_times, batch_size_B, max_item_num_B), dtype=np.float32)
            sampled_high_rewards_B = np.zeros((sample_times, batch_size_B), dtype=np.float32)
            sampled_low_rewards_B = np.zeros((sample_times, batch_size_B), dtype=np.float32)

            sampled_revise_index_A, sampled_revise_index_B = [], []

            avg_high_reward_A = np.zeros((batch_size_A), dtype=np.float32)
            avg_high_reward_B = np.zeros((batch_size_B), dtype=np.float32)
            avg_low_reward_A = np.zeros((batch_size_A), dtype=np.float32)
            avg_low_reward_B = np.zeros((batch_size_B), dtype=np.float32)

            for sample_time in range(sample_times):
                high_action_A, high_state_A, selected_seq_A, selected_len_A, label_input_A, label_input_B, \
                notrevised_index_A, revised_index_A, delete_index_A, keep_index_A = \
                    sampling_RL(env, recommender, agent, seq_to_revise=origin_seq_A, seq_to_improve=origin_seq_B,
                                len_A=origin_length_A, len_B=origin_length_B,
                                target_A=target_items_A, target_B=target_items_B,
                                label_A=label_A, label_B=label_B, batch_index=batch_index,
                                Random=True, domain="A", keep_pro=args.agent_keep_prob)

                sampled_high_actions_A[sample_time, :] = high_action_A
                sampled_high_states_A[sample_time, :] = high_state_A
                sampled_revise_index_A.append(revised_index_A)
                high_action_B, high_state_B, selected_seq_B, selected_len_B, label_input_A, label_input_B, \
                notrevised_index_B, revised_index_B, delete_index_B, keep_index_B = \
                    sampling_RL(env, recommender, agent, seq_to_revise=origin_seq_B, seq_to_improve=origin_seq_A,
                                len_B=origin_length_B, len_A=origin_length_A,
                                target_A=target_items_A, target_B=target_items_B,
                                label_A=label_A, label_B=label_B, batch_index=batch_index,
                                Random=True, domain="B", keep_pro=args.agent_keep_prob)

                sampled_high_actions_B[sample_time, :] = high_action_B
                sampled_high_states_B[sample_time, :] = high_state_B
                sampled_revise_index_B.append(revised_index_B)

                _, _, reward_A_4_Agent_B = \
                    env.get_env_reward(recommender, batch_index, high_actions=high_action_A, seq_A=origin_seq_A,
                                       seq_B=origin_seq_B, pos_A = pos_A, pos_B = pos_B,
                                       len_A=origin_length_A, len_B=origin_length_B, target_A=target_items_A,
                                       target_B=target_items_B,
                                       s_seq_A=selected_seq_A, s_seq_B=selected_seq_B,
                                       label_A=label_A, label_B=label_B,
                                       domain="A")

                _, _, reward_B_4_Agent_A = \
                    env.get_env_reward(recommender, batch_index, high_actions=high_action_B, seq_A=origin_seq_A,
                                       seq_B=origin_seq_B, pos_A=pos_A, pos_B=pos_B,
                                       len_A=origin_length_A, len_B=origin_length_B, target_A=target_items_A,
                                       target_B=target_items_B,
                                       s_seq_A=selected_seq_A, s_seq_B=selected_seq_B,
                                       label_A=label_A, label_B=label_B,
                                       domain="B")

                avg_low_reward_A += reward_A_4_Agent_B
                avg_high_reward_A += reward_A_4_Agent_B
                sampled_high_rewards_A[sample_time, :] = reward_A_4_Agent_B
                sampled_low_rewards_A[sample_time, :] = reward_A_4_Agent_B
                sampled_low_actions_A[sample_time, :] = env.get_action_matrix("A")
                sampled_low_states_A[sample_time, :] = env.get_state_matrix("A")

                avg_low_reward_B += reward_B_4_Agent_A
                avg_high_reward_B += reward_B_4_Agent_A
                sampled_high_rewards_B[sample_time, :] = reward_B_4_Agent_A
                sampled_low_rewards_B[sample_time, :] = reward_B_4_Agent_A
                sampled_low_actions_B[sample_time, :] = env.get_action_matrix("B")
                sampled_low_states_B[sample_time, :] = env.get_state_matrix("B")

            avg_high_reward_A = avg_high_reward_A / sample_times
            avg_high_reward_B = avg_high_reward_B / sample_times

            avg_low_reward_A = avg_low_reward_A / sample_times
            avg_low_reward_B = avg_low_reward_B / sample_times

            high_gradbuffer_A = agent.init_high_gradbuffer_A()
            high_gradbuffer_B = agent.init_high_gradbuffer_B()

            low_gradbuffer_A = agent.init_A_gradbuffer()
            low_gradbuffer_B = agent.init_B_gradbuffer()

            for sample_time in range(sample_times):
                high_reward_A = np.subtract(sampled_high_rewards_A[sample_time], avg_high_reward_A)
                high_gradient_A, A_high_loss = agent.get_high_gradient_A(sampled_high_states_A[sample_time],
                                                                         high_reward_A,
                                                                         sampled_high_actions_A[sample_time],
                                                                         args.agent_keep_prob)
                agent.train_high_A(high_gradbuffer_A, high_gradient_A)
                avg_A_high_loss += A_high_loss

                revised_index_A = sampled_revise_index_A[sample_time]
                low_reward_A = np.subtract(sampled_low_rewards_A[sample_time], avg_low_reward_A)  # 减去均值变好
                low_reward_row_A = np.tile(np.reshape(low_reward_A[revised_index_A], (-1, 1)), max_item_num_A)
                low_gradient_A, A_low_loss = agent.get_A_gradient(
                    np.reshape(sampled_low_states_A[sample_time][revised_index_A], (-1, low_state_size_A)),
                    np.reshape(low_reward_row_A, (-1,)),
                    np.reshape(sampled_low_actions_A[sample_time][revised_index_A], (-1,)),
                    args.agent_keep_prob
                )
                agent.train_low_A(low_gradbuffer_A, low_gradient_A)
                avg_A_low_loss += A_low_loss

                high_reward_B = np.subtract(sampled_high_rewards_B[sample_time], avg_high_reward_B)
                high_gradient_B, B_high_loss = agent.get_high_gradient_B(sampled_high_states_B[sample_time],
                                                                         high_reward_B,
                                                                         sampled_high_actions_B[sample_time],
                                                                         args.agent_keep_prob)
                agent.train_high_B(high_gradbuffer_B, high_gradient_B)
                avg_B_high_loss += B_high_loss

                revised_index_B = sampled_revise_index_B[sample_time]
                low_reward_B = np.subtract(sampled_low_rewards_B[sample_time], avg_low_reward_B)
                low_reward_row_B = np.tile(np.reshape(low_reward_B[revised_index_B], (-1, 1)), max_item_num_B)
                low_gradient_B, B_low_loss = agent.get_B_gradient(
                    np.reshape(sampled_low_states_B[sample_time][revised_index_B], (-1, low_state_size_B)),
                    np.reshape(low_reward_row_B, (-1,)),
                    np.reshape(sampled_low_actions_B[sample_time][revised_index_B], (-1,)),
                    args.agent_keep_prob
                )
                agent.train_low_B(low_gradbuffer_B, low_gradient_B)
                avg_B_low_loss += B_low_loss
            if recommender_trainable:
                _, _, selected_seq_A, selected_len_A, label_input_A, label_input_B, notrevised_index_A, \
                revised_index_A, delete_index_A, keep_index_A = \
                    sampling_RL(env, recommender, agent, seq_to_revise=origin_seq_A, seq_to_improve=origin_seq_B,
                                len_A=origin_length_A, len_B=origin_length_B,
                                target_A=target_items_A, target_B=target_items_B,
                                label_A=label_A, label_B=label_B, batch_index=batch_index,
                                Random=False, domain="A", keep_pro=1.0)

                _, _, selected_seq_B, selected_len_B, label_input_A, label_input_B, notrevised_index_B, \
                revised_index_B, delete_index_B, keep_index_B = \
                    sampling_RL(env, recommender, agent, seq_to_revise=origin_seq_B, seq_to_improve=origin_seq_A,
                                len_B=origin_length_B, len_A=origin_length_A,
                                target_A=target_items_A, target_B=target_items_B,
                                label_A=label_A, label_B=label_B, batch_index=batch_index,
                                Random=False, domain="B", keep_pro=1.0)

                selected_len_A = np.reshape(selected_len_A, (-1, 1))
                selected_len_B = np.reshape(selected_len_B, (-1, 1))

                train_loss_A, _ = recommender.train_recommender_A(origin_seq_A=origin_seq_A,
                                                                  origin_seq_B=origin_seq_B,
                                                                  origin_length_A=origin_length_A,
                                                                  origin_length_B=origin_length_B,
                                                                  selected_user_input_B=selected_seq_B,
                                                                  selected_num_idx_B=selected_len_B,
                                                                  target_item_A=target_items_A,
                                                                  target_item_B=target_items_B,
                                                                  labels_A=label_A, labels_B=label_B, is_training=True,
                                                                  pos_A=pos_A, pos_B=pos_B)

                train_loss_B, _ = recommender.train_recommender_B(origin_seq_A=origin_seq_A,
                                                                  origin_seq_B=origin_seq_B,
                                                                  origin_length_A=origin_length_A,
                                                                  origin_length_B=origin_length_B,
                                                                  selected_user_input_A=selected_seq_A,
                                                                  selected_num_idx_A=selected_len_A,
                                                                  target_item_A=target_items_A,
                                                                  target_item_B=target_items_B,
                                                                  labels_A=label_A, labels_B=label_B, is_training=True,
                                                                  pos_A=pos_A, pos_B=pos_B)
        else:
            # item_sets_A = list(dict_A.values())  # item A set
            # # s_seq_A = np.zeros(origin_seq_A.shape) + np.random.choice(item_sets_A, 1)[0]
            # # s_len_A = np.reshape(np.zeros(origin_seq_A.shape[0]) + max_item_num_A, (-1, 1))
            #
            # item_sets_B = list(dict_B.values())  # item B set
            # # s_seq_B = np.zeros(origin_seq_B.shape) + np.random.choice(item_sets_B, 1)[0]
            # # s_len_B = np.reshape(np.zeros(origin_seq_B.shape[0]) + max_item_num_B, (-1, 1))
            train_loss_A, _ = recommender.train_recommender_A(origin_seq_A=origin_seq_A, origin_seq_B=origin_seq_B,
                                                              origin_length_A=origin_length_A,
                                                              origin_length_B=origin_length_B,
                                                              target_item_A=target_items_A,
                                                              target_item_B=target_items_B,
                                                              labels_A=label_A, labels_B=label_B,
                                                              selected_user_input_B=origin_seq_B,
                                                              selected_num_idx_B=origin_length_B, is_training=True,
                                                              pos_A=pos_A, pos_B=pos_B)
            train_loss_B, _ = recommender.train_recommender_B(origin_seq_A=origin_seq_A, origin_seq_B=origin_seq_B,
                                                              origin_length_A=origin_length_A,
                                                              origin_length_B=origin_length_B,
                                                              target_item_A=target_items_A,
                                                              target_item_B=target_items_B,
                                                              labels_A=label_A, labels_B=label_B,
                                                              selected_user_input_A=origin_seq_A,
                                                              selected_num_idx_A=origin_length_A, is_training=True,
                                                              pos_A=pos_A, pos_B=pos_B)

        avg_loss_A += train_loss_A
        avg_loss_B += train_loss_B

        if agent_trainable:
            agent.update_target_high_network_A()
            agent.update_target_high_network_B()
            agent.update_target_A_network()
            agent.update_target_B_network()
            if recommender_trainable:
                recommender.assign_train_params_with_tau_A()
                recommender.assign_train_params_with_tau_B()
        else:
            recommender.assign_train_params_with_tau_A()
            recommender.assign_train_params_with_tau_B()

    rec_loss_A = avg_loss_A / train_batch_num
    rec_loss_B = avg_loss_B / train_batch_num
    agent_loss_high_A = avg_A_high_loss / train_batch_num
    agent_loss_high_B = avg_B_high_loss / train_batch_num
    agent_loss_low_A = avg_A_low_loss / train_batch_num
    agent_loss_low_B = avg_B_low_loss / train_batch_num

    return rec_loss_A, rec_loss_B, agent_loss_high_A, agent_loss_high_B, agent_loss_low_A, agent_loss_low_B


def get_avg_reward(env, recommender, agent, trainset):
    seq_A_all, seq_B_all, pos_A_all, pos_B_all, len_A_all, len_B_all, target_A_all, target_B_all, label_A_all, label_B_all, train_batch_num = \
        (trainset[0], trainset[1], trainset[2], trainset[3], trainset[4], trainset[5], trainset[6], trainset[7],
         trainset[8], trainset[9], trainset[10])
    avg_reward_A, total_selected_courses_A, total_deleted_instances_A, total_keep_instances_A, total_revised_instances_A, total_notrevised_instances_A = 0, 0, 0, 0, 0, 0
    avg_reward_B, total_selected_courses_B, total_deleted_instances_B, total_keep_instances_B, total_revised_instances_B, total_notrevised_instances_B = 0, 0, 0, 0, 0, 0
    total_instances_A = 0
    total_instances_B = 0
    for batch_index in range(train_batch_num):
        origin_seq_A = np.array([u for u in seq_A_all[batch_index]])
        origin_seq_B = np.array([u for u in seq_B_all[batch_index]])
        pos_A = np.array([u for u in pos_A_all[batch_index]])
        pos_B = np.array([u for u in pos_B_all[batch_index]])
        target_items_A = np.reshape(target_A_all[batch_index], (-1, 1))
        target_items_B = np.reshape(target_B_all[batch_index], (-1, 1))

        origin_len_A = np.reshape(len_A_all[batch_index], (-1, 1))
        origin_len_B = np.reshape(len_B_all[batch_index], (-1, 1))

        label_A = np.reshape(label_A_all[batch_index], (-1, 1))
        label_B = np.reshape(label_B_all[batch_index], (-1, 1))

        high_action_A, high_state_A, select_user_input_A, select_num_idx_A, label_input_A, label_input_B, \
        notrevised_index_A, revised_index_A, delete_index_A, keep_index_A \
            = sampling_RL(env, recommender, agent, seq_to_revise=origin_seq_A, seq_to_improve=origin_seq_B,
                          len_A=origin_len_A, len_B=origin_len_B,
                          target_A=target_items_A, target_B=target_items_B,
                          label_A=label_A, label_B=label_B, batch_index=batch_index,
                          Random=False, domain="A", keep_pro=1.0)

        high_action_B, high_state_B, select_user_input_B, select_num_idx_B, label_input_A, label_input_B, \
        notrevised_index_B, revised_index_B, delete_index_B, keep_index_B \
            = sampling_RL(env, recommender, agent, seq_to_revise=origin_seq_B, seq_to_improve=origin_seq_A,
                          len_B=origin_len_B, len_A=origin_len_A,
                          target_A=target_items_A, target_B=target_items_B,
                          label_A=label_A, label_B=label_B, batch_index=batch_index,
                          Random=False, domain="B", keep_pro=1.0)

        reward_A, _, reward_HRL_A = \
            env.get_env_reward(recommender, batch_index, high_actions=high_action_A, seq_A=origin_seq_A,
                               seq_B=origin_seq_B, pos_A=pos_A, pos_B=pos_B,
                               len_A=origin_len_A, len_B=origin_len_B, target_A=target_items_A, target_B=target_items_B,
                               s_seq_A=select_user_input_A, s_seq_B=select_user_input_B,
                               label_A=label_A, label_B=label_B, domain="A")

        reward_B, _, reward_HRL_B, = \
            env.get_env_reward(recommender, batch_index, high_actions=high_action_B, seq_A=origin_seq_A,
                               seq_B=origin_seq_B, pos_A=pos_A, pos_B=pos_B,
                               len_A=origin_len_A, len_B=origin_len_B, target_A=target_items_A, target_B=target_items_B,
                               s_seq_A=select_user_input_A, s_seq_B=select_user_input_B,
                               label_A=label_A, label_B=label_B, domain="B")

        avg_reward_A += np.sum(reward_HRL_A)
        total_selected_courses_A += np.sum(select_num_idx_A)
        total_revised_instances_A += len(revised_index_A)
        total_notrevised_instances_A += len(notrevised_index_A)
        total_deleted_instances_A += len(delete_index_A)
        total_keep_instances_A += len(keep_index_A)
        total_instances_A += origin_seq_A.shape[0]

        avg_reward_B += np.sum(reward_HRL_B)
        total_selected_courses_B += np.sum(select_num_idx_B)
        total_revised_instances_B += len(revised_index_B)
        total_notrevised_instances_B += len(notrevised_index_B)
        total_deleted_instances_B += len(delete_index_B)
        total_keep_instances_B += len(keep_index_B)
        total_instances_B += origin_seq_B.shape[0]

    avg_reward_A = avg_reward_A / total_instances_A
    avg_reward_B = avg_reward_B / total_instances_B

    return avg_reward_A, total_selected_courses_A, total_notrevised_instances_A, total_revised_instances_A, total_deleted_instances_A, total_keep_instances_A, \
           avg_reward_B, total_selected_courses_B, total_notrevised_instances_B, total_revised_instances_B, total_deleted_instances_B, total_keep_instances_B


def evalute(env, agent, recommender, testset, noAgent=False):
    seq_A_all, seq_B_all, pos_A_all, pos_B_all, len_A_all, len_B_all, target_A_all, target_B_all, label_A_all, label_B_all, test_batch_num \
        = (testset[0], testset[1], testset[2], testset[3], testset[4], testset[5], testset[6], testset[7], testset[8], testset[9], testset[10])
    if noAgent:
        return eval_rating(recommender, seq_A=seq_A_all, seq_B=seq_B_all,
                           len_A=len_A_all, len_B=len_B_all,
                           target_A=target_A_all, target_B=target_B_all,
                           label_A=label_A_all, label_B=label_B_all,
                           s_seq_A=seq_A_all, s_seq_B=seq_B_all, s_len_A=len_A_all, s_len_B=len_B_all,
                           test_batch_num=test_batch_num, pos_A=pos_A_all, pos_B=pos_B_all)
    else:
        env.set_test_original_rewards()
        s_seq_A_list, select_len_A_list, select_target_A_list, select_label_A_list = [], [], [], []
        s_seq_B_list, select_len_B_list, select_target_B_list, select_label_B_list = [], [], [], []
        pos_A_list, pos_B_list = [], []
        s_len_A_list, s_len_B_list = [], []

        for batch_index in range(test_batch_num):
            origin_seq_A = np.array([seq for seq in seq_A_all[batch_index]])
            origin_seq_B = np.array([seq for seq in seq_B_all[batch_index]])

            target_items_A = target_A_all[batch_index]
            target_items_B = target_B_all[batch_index]
            label_A = label_A_all[batch_index]
            label_B = label_B_all[batch_index]
            origin_len_A = len_A_all[batch_index]
            origin_len_B = len_B_all[batch_index]

            _, _, select_seq_A, select_len_A, label_input_A, label_input_B, notrevised_index_A, revised_index_A, \
            delete_index_A, keep_index_A = \
                sampling_RL(env, recommender, agent, seq_to_revise=origin_seq_A, seq_to_improve=origin_seq_B,
                            len_A=origin_len_A, len_B=origin_len_B,
                            target_A=target_items_A, target_B=target_items_B,
                            label_A=label_A, label_B=label_B, batch_index=batch_index,
                            Random=False, domain="A", keep_pro=args.agent_keep_prob)

            _, _, select_seq_B, select_len_B, label_input_A, label_input_B, notrevised_index_B, revised_index_B, \
            delete_index_B, keep_index_B = \
                sampling_RL(env, recommender, agent, seq_to_revise=origin_seq_B, seq_to_improve=origin_seq_A,
                            len_B=origin_len_B, len_A=origin_len_A,
                            target_A=target_items_A, target_B=target_items_B,
                            batch_index=batch_index, label_A=label_A, label_B=label_B,
                            Random=False, domain="B", keep_pro=args.agent_keep_prob)

            s_seq_A_list.append(select_seq_A)
            s_seq_B_list.append(select_seq_B)
            s_len_A_list.append(select_len_A)
            s_len_B_list.append(select_len_B)
        env.set_train_original_rewards()

        return eval_rating(recommender, seq_A=seq_A_all, seq_B=seq_B_all, len_A=len_A_all, len_B=len_B_all,
                           target_A=target_A_all, target_B=target_B_all,
                           label_A=label_A_all, label_B=label_B_all,
                           s_seq_A=s_seq_A_list, s_seq_B=s_seq_B_list,
                           s_len_A=s_len_A_list, s_len_B=s_len_B_list,
                           test_batch_num=test_batch_num, pos_A=pos_A_all, pos_B= pos_B_all)
