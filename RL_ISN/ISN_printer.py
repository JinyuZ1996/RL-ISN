# _*_coding: UTF_8 _*_
import logging
from ISN_parameters import *

args = setting()


def print_global_config():
    print("Current parameters are as followed:")
    print("Datasets: {}".format(args.dataset))
    print("Rec_Pretrain:{}, Rec_pre_lr:{}, Rec_pre_tau:{}, Rec_pre_epochs:{}, Beta:{}, K-value-A:{}, K-value-B:{}, Delta:{} .".format(
        args.recommender_pretrain, args.recommender_pre_lr, args.recommender_pre_tau, args.recommender_epochs,
        args.beta, args.account_num_A, args.account_num_B, args.delta))
    print("Agent_Pretrain:{}, Agent_pre_lr:{}, Agent_pre_tau:{}, Agent_pre_epochs:{} .".format(
        args.agent_pretrain, args.agent_pretrain_lr_low, args.agent_pretrain_tau_low, args.agent_pretrain_epochs))
    print("Joint epochs:{}, embedding-size:{}, isFastRunning:{}".format(args.joint_train_epochs,
                                                                        args.recommender_embedding_size,
                                                                        args.fast_running))


def print_rec_message(epoch, rec_loss_A, rec_loss_B, rec_pre_time, joint=False):
    if joint is False:
        print('Epoch {} - Training Loss_A: {:.5f} - Training time: {:.3}'.format(epoch, rec_loss_A,
                                                                                 rec_pre_time))
        logging.info('Epoch {} - Training Loss_A: {:.5f}'.format(epoch, rec_loss_A))
        print('Epoch {} - Training Loss_B: {:.5f} - Training time: {:.3}'.format(epoch, rec_loss_B,
                                                                                 rec_pre_time))
        logging.info('Epoch {} - Training Loss_B: {:.5f}'.format(epoch, rec_loss_B))
    else:
        print('Joint Epoch {} - Recommender Loss_A: {:.5f} - Training time: {:.3}'.format(epoch, rec_loss_A,
                                                                                          rec_pre_time))
        logging.info('Joint Epoch {} - Recommender Loss_A: {:.5f} - Training time: {:.3}'.format(epoch, rec_loss_A,
                                                                                                 rec_pre_time))
        print('Joint Epoch {} - Recommender Loss_B: {:.5f} - Training time: {:.3}'.format(epoch, rec_loss_B,
                                                                                          rec_pre_time))
        logging.info('Joint Epoch {} - Recommender Loss_B: {:.5f} - Training time: {:.3}'.format(epoch, rec_loss_B,
                                                                                                 rec_pre_time))


def print_recommender_train(epoch, hr5_A, ndcg5_A, hr10_A, ndcg10_A, map_A, mrr_A, hr5_B, ndcg5_B,
                            hr10_B, ndcg10_B, map_B, mrr_B, test_loss_A, test_loss_B, rec_test_time, pretrain=True):
    if pretrain is True:
        print(
            "Evaluate recommender, Epoch %d : HR5 = %.4f, NDCG5 = %.4f, HR10 = %.4f, NDCG10 = %.4f, MAP = %.4f, MRR = %.4f, test_loss = %.4f" % (
                epoch, hr5_A, ndcg5_A, hr10_A, ndcg10_A, map_A, mrr_A, test_loss_A))
        print(
            "Evaluate recommender, Epoch %d : HR5 = %.4f, NDCG5 = %.4f, HR10 = %.4f, NDCG10 = %.4f, MAP = %.4f, MRR = %.4f, test_loss = %.4f" % (
                epoch, hr5_B, ndcg5_B, hr10_B, ndcg10_B, map_B, mrr_B, test_loss_B))
        logging.info(
            "Evaluate recommender, Epoch %d : HR5 = %.4f, NDCG5 = %.4f, HR10 = %.4f, NDCG10 = %.4f, MAP = %.4f, MRR = %.4f, test_loss = %.4f" % (
                epoch, hr5_A, ndcg5_A, hr10_A, ndcg10_A, map_A, mrr_A, test_loss_A))
        logging.info(
            "Evaluate recommender, Epoch %d : HR5 = %.4f, NDCG5= %.4f, HR10 = %.4f, NDCG10 = %.4f, MAP = %.4f, MRR = %.4f, test_loss = %.4f" % (
                epoch, hr5_B, ndcg5_B, hr10_B, ndcg10_B, map_B, mrr_B, test_loss_B))
        print("Epoch: {}, Recommender evaluate time: {:.3f}".format(epoch, rec_test_time))
        logging.info("Epoch: {}, Recommender evaluate time: {:.3f}".format(epoch, rec_test_time))
    else:
        print(
            "Evaluate recommender : HR5 = %.4f, NDCG5 = %.4f, HR10 = %.4f, NDCG10 = %.4f, MAP = %.4f, MRR = %.4f, test_loss = %.4f" % (
                hr5_A, ndcg5_A, hr10_A, ndcg10_A, map_A, mrr_A, test_loss_A))
        print(
            "Evaluate recommender : HR5 = %.4f, NDCG5 = %.4f, HR10 = %.4f, NDCG10 = %.4f, MAP = %.4f, MRR = %.4f, test_loss = %.4f" % (
                hr5_B, ndcg5_B, hr10_B, ndcg10_B, map_B, mrr_B, test_loss_B))
        logging.info(
            "Evaluate recommender : HR5 = %.4f, NDCG5 = %.4f, HR10 = %.4f, NDCG10 = %.4f, MAP = %.4f, MRR = %.4f, test_loss = %.4f" % (
                hr5_A, ndcg5_A, hr10_A, ndcg10_A, map_A, mrr_A, test_loss_A))
        logging.info(
            "Evaluate recommender : HR5 = %.4f, NDCG5= %.4f, HR10 = %.4f, NDCG10 = %.4f, MAP = %.4f, MRR = %.4f, test_loss = %.4f" % (
                hr5_B, ndcg5_B, hr10_B, ndcg10_B, map_B, mrr_B, test_loss_B))
        print("Recommender evaluate time: {:.3f}".format(rec_test_time))
        logging.info("Recommender evaluate time: {:.3f}".format(rec_test_time))


def print_eval_agent(hr5_A, ndcg5_A, hr10_A, ndcg10_A, map_A, mrr_A, hr5_B, ndcg5_B, hr10_B, ndcg10_B, map_B, mrr_B,
                     test_loss_A, test_loss_B, rec_test_time):
    print(
        "Evaluate with agent: HR5 = %.4f, NDCG5 = %.4f, HR10 = %.4f, NDCG10 = %.4f, MAP = %.4f, MRR = %.4f, test_loss = %.4f" % (
            hr5_A, ndcg5_A, hr10_A, ndcg10_A, map_A, mrr_A, test_loss_A))
    print(
        "Evaluate with agent: HR5 = %.4f, NDCG5 = %.4f, HR10 = %.4f, NDCG10 = %.4f, MAP = %.4f, MRR = %.4f, test_loss = %.4f" % (
            hr5_B, ndcg5_B, hr10_B, ndcg10_B, map_B, mrr_B, test_loss_B))
    logging.info(
        "Evaluate with agent: HR5 = %.4f, NDCG5 = %.4f, HR10 = %.4f, NDCG10 = %.4f, MAP = %.4f, MRR = %.4f, test_loss = %.4f" % (
            hr5_A, ndcg5_A, hr10_A, ndcg10_A, map_A, mrr_A, test_loss_A))
    logging.info(
        "Evaluate with agent: HR5 = %.4f, NDCG5= %.4f, HR10 = %.4f, NDCG10 = %.4f, MAP = %.4f, MRR = %.4f, test_loss = %.4f" % (
            hr5_B, ndcg5_B, hr10_B, ndcg10_B, map_B, mrr_B, test_loss_B))
    print("Recommender evaluate time: {:.3f}".format(rec_test_time))
    logging.info("Recommender evaluate time: {:.3f}".format(rec_test_time))


def print_agent_train(epoch, high_loss_A, high_loss_B, low_loss_A, low_loss_B, train_time, joint=False):
    if joint is False:
        print("Agent training Epoch : {} - Training time:{:.3f}".format(epoch, train_time))
        logging.info("Agent training Epoch : {} - Training time:{:.3f}".format(epoch, train_time))
    else:
        print("Joint Agent training Epoch : {} - Training time:{:.3f}".format(epoch, train_time))
        logging.info("Joint Agent training Epoch : {} - Training time:{:.3f}".format(epoch, train_time))

    print('Domain-A high level task training loss: {:.3f}'.format(high_loss_A))
    logging.info('Domain-A high level task training loss: {:.3f}'.format(high_loss_A))
    print('Domain-B high level task training loss: {:.3f}'.format(high_loss_B))
    logging.info('Domain-B high level task training loss: {:.3f}'.format(high_loss_B))
    print('Domain-A low level task training loss: {:.3f}'.format(low_loss_A))
    logging.info('Domain-A low level task training loss: {:.3f}'.format(low_loss_A))
    print('Domain-B low level task training loss: {:.3f}'.format(low_loss_B))
    logging.info('Domain-B low level task training loss: {:.3f}'.format(low_loss_B))


def print_agent_message(domain, epoch, avg_reward, total_selected_courses, total_revised_instances,
                        total_notrevised_instances, total_deleted_instances, total_keep_instances,
                        test_time, train_time):
    partial_revised = total_revised_instances - total_deleted_instances - total_keep_instances
    if domain is "A":
        logging.info(
            "Evaluate agent at epoch %d : reward for Agent_A = %.4f, items (selected = %d), instances (revised = %d, notrevise = %d, deleted = %d, keep = %d, partial revise = %d), test time = %.1fs, train_time = %.1fs" % (
                epoch, avg_reward, total_selected_courses, total_revised_instances, total_notrevised_instances,
                total_deleted_instances, total_keep_instances, partial_revised, test_time, train_time))
        print(
            "Evaluate agent at epoch %d : reward for Agent_A = %.4f, items (selected = %d), instances (revised = %d, notrevise = %d, deleted = %d, keep = %d, partial revise = %d), test time = %.1fs, train_time = %.1fs" % (
                epoch, avg_reward, total_selected_courses, total_revised_instances, total_notrevised_instances,
                total_deleted_instances, total_keep_instances, partial_revised, test_time, train_time))
    else:
        logging.info(
            "Evaluate agent at epoch %d : reward for Agent_B = %.4f, items (selected = %d), instances (revised = %d, notrevise = %d, deleted = %d, keep = %d, partial revise = %d), test time = %.1fs, train_time = %.1fs" % (
                epoch, avg_reward, total_selected_courses, total_revised_instances, total_notrevised_instances,
                total_deleted_instances, total_keep_instances, partial_revised, test_time, train_time))
        print(
            "Evaluate agent at epoch %d : reward for Agent_B = %.4f, items (selected = %d), instances (revised = %d, notrevise = %d, deleted = %d, keep = %d, partial revise = %d), test time = %.1fs, train_time = %.1fs" % (
                epoch, avg_reward, total_selected_courses, total_revised_instances, total_notrevised_instances,
                total_deleted_instances, total_keep_instances, partial_revised, test_time, train_time))
