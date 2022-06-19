# _*_coding: UTF_8 _*_
from RL_ISN.ISN_train import *
from RL_ISN.ISN_agent import AgentNetwork
from RL_ISN.ISN_environment import Environment
from RL_ISN.ISN_recommender import RecommenderNetwork
from RL_ISN.ISN_printer import *
from time import *
import tensorflow as tf

if __name__ == '__main__':

    '''
        Initialize the log record file, 
        which is saved in the corresponding dataset file directory,
        and only the latest log is retained
    '''
    logging.basicConfig(level=logging.INFO, filename=args.log, filemode='w',
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    # print the Global configurations of training here.
    print_global_config()

    '''
        We provide two available datasets here: ['hvideo', 'hamazon']
    '''
    if args.dataset is "hvideo":  # Video datasets
        dict_A = get_dict(dict_path=args.path_E)
        dict_B = get_dict(dict_path=args.path_V)
        origin_train_data = get_mixedData_video(data_path=args.train_path,
                                                dict_E=dict_A, dict_V=dict_B)
        origin_test_data = get_mixedData_video(data_path=args.test_path,
                                               dict_E=dict_A, dict_V=dict_B)
    else:  # Amazon datasets
        dict_A = get_dict(dict_path=args.path_A)
        dict_B = get_dict(dict_path=args.path_B)
        origin_train_data = get_mixedData_amazon(data_path=args.train_path,
                                                 dict_A=dict_A, dict_B=dict_B, isTrain=True)
        origin_test_data = get_mixedData_amazon(data_path=args.test_path,
                                                dict_A=dict_A, dict_B=dict_B, isTrain=False)

    '''
        If 'args.fast_running' is set and only partial data will be used, 
        please execute the following code.
    '''
    if args.fast_running:
        origin_train_data = origin_train_data[:int(len(origin_train_data)*args.fr_num)]

    # To set the padding number of your chosen dataset.
    padding_number_A = len(dict_A)
    padding_number_B = len(dict_B)
    padding_number_sum = len(dict_A) + len(dict_B)

    train_with_target = process_data(origin_train_data=origin_train_data, dict_A=dict_A,
                                     origin_test_data=None)
    test_with_target = process_data(origin_train_data=origin_train_data, dict_A=dict_A,
                                    origin_test_data=origin_test_data)

    train_data = get_train_with_neg(data_with_target=train_with_target, num_negs=1, dict_A=dict_A,
                                    dict_B=dict_B)
    train_data_with_neg = get_train_with_neg(data_with_target=train_data, num_negs=args.num_negatives,
                                             dict_A=dict_A,
                                             dict_B=dict_B)
    print("Already load the trainData...")

    test_data_with_neg = get_test_with_neg(data_with_target=test_with_target, num_negs=args.test_batch_size,
                                           dict_A=dict_A, dict_B=dict_B)
    print("Already load the testData...")

    train_batches, _ = get_batches(dataset=train_data, batch_size=args.batch_size, pad_int_A=padding_number_A,
                                pad_int_B=padding_number_sum, dict_A=dict_A, shuffle=True)
    train_batches_neg, max_len_train = get_batches(dataset=train_data_with_neg, batch_size=args.batch_size,
                                    pad_int_A=padding_number_A, pad_int_B=padding_number_sum,
                                    dict_A=dict_A, shuffle=True)
    print("Already get the train_batches.")
    test_batches_neg, max_len_test = get_batches(dataset=test_data_with_neg, batch_size=args.test_batch_size,
                                   pad_int_A=padding_number_A,
                                   pad_int_B=padding_number_sum,
                                   dict_A=dict_A, shuffle=False)
    print("Already get the test_batches.")

    max_len_pos = 0
    if max_len_train > max_len_test:
        max_len_pos = max_len_train
    else:
        max_len_pos = max_len_test

    # Initialize the recommender_network (Only the '__init__' function is called here)
    recommender = RecommenderNetwork(items_max_num_A=len(dict_A), items_max_num_B=len(dict_B), args=args,
                                     padding_A=padding_number_A, padding_B=padding_number_B,
                                     dict_A=dict_A, dict_B=dict_B, max_len_pos=max_len_pos)
    print("Recommender initialize finished.")

    # Initialize the environment (Only the '__init__' function is called here)
    env = Environment(recommender=recommender)
    print("Environment Initialize finished.")

    '''
        The training starts here. The 'GPU_options' is used to control the CUDA
        and allow the auto_growth of the GPU memories usage.
    '''
    GPU_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(graph=recommender.graph, config=tf.ConfigProto(gpu_options=GPU_options)) as sess:

        # Initialize the agent_network (Only the '__init__' function is called here)
        agent = AgentNetwork(sess=sess)
        print("Agent Initialize finished.")

        # deliver the 'session' to recommender agent and environment
        recommender.sess = sess
        env.sess = sess
        env.agent = agent

        '''
            Initialize the model savers. 
            When using the savers, all variables defined in the model will be restored
        '''
        pre_recommender_saver = tf.train.Saver()
        pre_agent_saver = tf.train.Saver()
        recommender_saver = tf.train.Saver()
        agent_saver = tf.train.Saver()

        '''
            Block 1_1: Recommender_pretrain task
            Before using the joint training, basic recommender must be pretrained for at least one time
            Relevant parameters was defined at 'ISN_parameters.py -block_1'
        '''
        if args.recommender_pretrain is True:

            best_score_domain_A = 0.0
            best_score_domain_B = 0.0

            sess.run(tf.global_variables_initializer())

            for epoch in range(args.recommender_epochs):
                rec_pre_begin_time = time()
                rec_loss_A, rec_loss_B, _, _, _, _ = train(env, agent, recommender, train_batches_neg, args,
                                                           dict_A=dict_A, dict_B=dict_B,
                                                           recommender_trainable=True, agent_trainable=False)
                rec_pre_time = time() - rec_pre_begin_time

                epoch_to_print = epoch + 1
                print_rec_message(epoch=epoch_to_print, rec_loss_A=rec_loss_A, rec_loss_B=rec_loss_B,
                                  rec_pre_time=rec_pre_time)

                # The parameter verbose controls how many training sessions to evaluate once.
                if epoch_to_print % args.recommender_verbose == 0:
                    rec_test_begin_time = time()
                    (hr5_A, ndcg5_A, hr10_A, ndcg10_A, map_A, mrr_A, test_loss_A, hr5_B, ndcg5_B, hr10_B, ndcg10_B,
                     map_B, mrr_B, test_loss_B) = evalute(env, agent, recommender, test_batches_neg, noAgent=True)
                    rec_test_time = time() - rec_test_begin_time

                    print_recommender_train(epoch_to_print, hr5_A, ndcg5_A, hr10_A, ndcg10_A, map_A, mrr_A, hr5_B,
                                            ndcg5_B, hr10_B, ndcg10_B, map_B, mrr_B, test_loss_A, test_loss_B,
                                            rec_test_time, pretrain=True)

                    if hr5_A >= best_score_domain_A or hr5_B >= best_score_domain_B:
                        best_score_domain_A = hr5_A
                        best_score_domain_B = hr5_B
                        pre_recommender_saver.save(sess, args.pre_recommender, global_step=epoch_to_print)
                        print("Recommender performs better, saving current model....")
                        logging.info("Recommender performs better, saving current model....")

                # regenerate the Negative_instance and fold them into batches for the next training
                train_data_with_neg = get_train_with_neg(train_data, args.num_negatives, dict_A, dict_B)
                train_batches_neg, _ = get_batches(train_data_with_neg, args.batch_size, padding_number_A,
                                                padding_number_sum, dict_A)

            print("Recommender pretrain finished.")
            logging.info("Recommender pretrain finished.")

        '''
            Block 1_2: Load the best pretrained recommender
            Before using the joint training, basic recommender must be pretrained for at least one time
        '''
        print("Load the best pre-trained recommender from ", args.pre_recommender)
        logging.info("Load the best pre-trained recommender from %s " % args.pre_recommender)
        pre_recommender_saver.restore(sess, tf.train.get_checkpoint_state(
            os.path.dirname(args.pre_recommender + 'checkpoint')).model_checkpoint_path)

        '''
            Block 1_3: Without pretrain task, just evaluate the former module
            If the 'args.recommender_pretrain' is 'False', then the loaded model will be evaluated directly
        '''
        if args.recommender_pretrain is False:
            print("Evaluate the best pre-trained recommender model...")
            logging.info("Evaluate the best pre-trained recommender model...")
            rec_test_begin_time = time()
            (hr5_A, ndcg5_A, hr10_A, ndcg10_A, map_A, mrr_A, test_loss_A, hr5_B, ndcg5_B, hr10_B, ndcg10_B, map_B,
             mrr_B, test_loss_B) = evalute(env, agent, recommender, test_batches_neg, noAgent=True)
            rec_test_time = time() - rec_test_begin_time
            print_recommender_train(-1, hr5_A, ndcg5_A, hr10_A, ndcg10_A,
                                    map_A, mrr_A, hr5_B, ndcg5_B, hr10_B, ndcg10_B, map_B, mrr_B, test_loss_A,
                                    test_loss_B, rec_test_time, pretrain=False)
        else:  # while the pretrain was just finished
            print("Already load the best pre-trained recommender.")
            logging.info("Already load the best pre-trained recommender.")

        print("Before agent training, the Environment needs pretrained recommender to initialize state...")

        env.initilize_state(recommender=recommender, train_data=train_batches, test_data=test_batches_neg,
                            high_state_size=args.high_state_size, A_state_size=args.A_state_size,
                            B_state_size=args.B_state_size, padding_number_A=padding_number_A,
                            padding_number_B=padding_number_B)

        print("agent: environment initialize ok")
        logging.info("agent: environment initialize ok")

        '''
           Block 2_1: Agent_pretrain task
           Before using the joint training, agent_network must be pretrained for at least one time
           Relevant parameters was defined at 'ISN_parameters.py -block_3'
        '''
        if args.agent_pretrain is True:

            best_reward_4_agent_A = -1000
            best_reward_4_agent_B = -1000

            for epoch in range(args.agent_pretrain_epochs):
                agent_pre_begin_time = time()
                _, _, agent_A_high_loss, agent_B_high_loss, agent_A_low_loss, agent_B_low_loss = \
                    train(env=env, agent=agent, recommender=recommender, batches_train=train_batches, args=args,
                          dict_A=dict_A, dict_B=dict_B, recommender_trainable=False, agent_trainable=True)
                agent_pre_time = time() - agent_pre_begin_time

                epoch_to_print = epoch + 1
                print_agent_train(epoch_to_print, agent_A_high_loss, agent_B_high_loss,
                                  agent_A_low_loss, agent_B_low_loss, agent_pre_time)
                # The parameter verbose controls how many training sessions to evaluate once.
                if epoch_to_print % args.agent_verbose == 0:

                    agent_test_begin_time = time()
                    reward_A, selected_items_A, notrevised_instances_A, revised_instances_A, deleted_instances_A, \
                    keep_instances_A, reward_B, selected_items_B, notrevised_instances_B, revised_instances_B, \
                    deleted_instances_B, keep_instances_B = get_avg_reward(env, recommender, agent, train_batches)
                    agent_test_time = time() - agent_test_begin_time
                    # the rewards for agent_A shows the effect of 's_seq_A' to domain B
                    print_agent_message("A", epoch_to_print, reward_A, selected_items_A, revised_instances_A,
                                        notrevised_instances_A, deleted_instances_A, keep_instances_A,
                                        agent_test_time, agent_pre_time)
                    print_agent_message("B", epoch_to_print, reward_B, selected_items_B, revised_instances_B,
                                        notrevised_instances_B, deleted_instances_B, keep_instances_B,
                                        agent_test_time, agent_pre_time)

                    if reward_A >= best_reward_4_agent_A or reward_B >= best_reward_4_agent_B:
                        best_reward_4_agent_B = reward_B
                        best_reward_4_agent_A = reward_A
                        pre_agent_saver.save(sess, args.pre_agent, global_step=epoch_to_print)
                        print("Agent performs better, saving current model....")
                        logging.info("Agent performs better, saving current model....")

            print("Agent pretrain finished.")
            logging.info("Agent pretrain finished.")

        '''
           Block 2_2: Load the best pretrained agent here
           Before using the joint training, agent_network must be pretrained for at least one time
        '''
        print("Load best pre-trained agent from", args.pre_agent)
        logging.info("Load best pre-trained agent from %s" % args.pre_agent)
        pre_agent_saver.restore(sess, tf.train.get_checkpoint_state(
            os.path.dirname(args.pre_agent + 'checkpoint')).model_checkpoint_path)

        '''
            Block 2_3: Evaluate the pretrained agent
            By using agent on pretrained recommender to predict on the testing sets
        '''
        print("Evaluate recommender based on the selected test instances by the pre-trained agent")
        logging.info("Evaluate recommender based on the selected test instances by the pre-trained agent")
        rec_test_begin_time = time()
        (hr5_A, ndcg5_A, hr10_A, ndcg10_A, map_A, mrr_A, test_loss_A, hr5_B, ndcg5_B, hr10_B, ndcg10_B, map_B, mrr_B,
         test_loss_B) = evalute(env, agent, recommender, test_batches_neg, noAgent=False)
        rec_test_time = time() - rec_test_begin_time
        print_eval_agent(hr5_A, ndcg5_A, hr10_A, ndcg10_A, map_A, mrr_A, hr5_B, ndcg5_B, hr10_B, ndcg10_B, map_B,
                         mrr_B, test_loss_A, test_loss_B, rec_test_time)

        '''
           Block 3_1: Joint training task
           Before using the joint training, both agent and recommender must be pretrained for at least one time
           Relevant parameters was defined at 'ISN_parameters.py -block_5'
        '''
        print("Recommender and Agent pretrained finished. Begin to jointly train.")
        logging.info("Recommender and Agent pretrained finished. Begin to jointly train.")

        best_score_domain_A = 0.0
        best_score_domain_B = 0.0
        best_reward_4_agent_A = -1000
        best_reward_4_agent_B = -1000

        # using different parameters to update the joint training models
        recommender.update_tau_4_joint(joint_tau=args.recommender_joint_tau)
        recommender.update_lr_4_joint(joint_lr=args.recommender_joint_lr)
        agent.udpate_agent_tau_4_joint(joint_agent_tau_low=args.agent_joint_tau_low,
                                       joint_agent_tau_high=args.agent_joint_tau_high)
        agent.update_agent_lr_4_joint(joint_agent_lr_low=args.agent_joint_lr_low,
                                      joint_agent_lr_high=args.agent_joint_lr_high)

        for epoch in range(args.joint_train_epochs):
            # reset the original rewards for joint training from best recommender pretrained results
            env.set_train_original_rewards()
            joint_begin_time = time()
            rec_loss_A, rec_loss_B, agent_A_high_loss, agent_B_high_loss, agent_A_low_loss, agent_B_low_loss \
                = train(env, agent, recommender, train_batches, args, dict_A=dict_A, dict_B=dict_B,
                        recommender_trainable=True, agent_trainable=True)
            joint_train_time = time() - joint_begin_time
            epoch_to_print = epoch + 1
            # print the message of joint training
            print_rec_message(epoch_to_print, rec_loss_A, rec_loss_B, joint_train_time, joint=True)
            print_agent_train(epoch_to_print, agent_A_high_loss, agent_B_high_loss,
                              agent_A_low_loss, agent_B_low_loss, joint_train_time, joint=True)

            # get agent rewards and save better agent for training
            if epoch_to_print % args.joint_verbose == 0:
                # env.set_train_original_rewards()
                joint_test_begin_time = time()
                # get the joint rewards from recommender to agent
                reward_A, selected_items_A, notrevised_instances_A, revised_instances_A, deleted_instances_A, \
                keep_instances_A, reward_B, selected_items_B, notrevised_instances_B, revised_instances_B, \
                deleted_instances_B, keep_instances_B = get_avg_reward(env, recommender, agent, train_batches)
                joint_test_time = time() - joint_test_begin_time
                # print the message of rewards
                print_agent_message("A", epoch_to_print, reward_A, selected_items_A, revised_instances_A,
                                    notrevised_instances_A, deleted_instances_A, keep_instances_A,
                                    joint_test_time, joint_train_time)
                print_agent_message("B", epoch_to_print, reward_B, selected_items_B, revised_instances_B,
                                    notrevised_instances_B, deleted_instances_B, keep_instances_B,
                                    joint_test_time, joint_train_time)

                if reward_A >= best_reward_4_agent_A or reward_B >= best_reward_4_agent_B:
                    best_reward_4_agent_A = reward_A
                    best_reward_4_agent_B = reward_B
                    agent_saver.save(sess, args.joint_agent, global_step=epoch_to_print)
                    print("Joint agent performs better, saving current model....")
                    logging.info("Joint agent performs better, saving current model....")

            # get recommender evaluation and save better recommender for training
            if epoch_to_print % args.joint_verbose == 0:
                joint_eval_begin_time = time()
                (hr5_A, ndcg5_A, hr10_A, ndcg10_A, map_A, mrr_A, test_loss_A, hr5_B, ndcg5_B, hr10_B, ndcg10_B,
                 map_B, mrr_B, test_loss_B) = evalute(env, agent, recommender, test_batches_neg, noAgent=False)

                joint_eval_time = time() - joint_eval_begin_time
                print_recommender_train(epoch_to_print, hr5_A, ndcg5_A, hr10_A, ndcg10_A, map_A, mrr_A, hr5_B, ndcg5_B,
                                        hr10_B, ndcg10_B, map_B, mrr_B, test_loss_A, test_loss_B, joint_eval_time)
                if hr5_A >= best_score_domain_A or hr5_B >= best_score_domain_B:
                    best_score_domain_A = hr5_A
                    best_score_domain_B = hr5_B
                    recommender_saver.save(sess, args.joint_recommender, global_step=epoch_to_print)
                    print("Joint recommender performs better, saving current model....")
                    logging.info("Joint recommender performs better, saving current model....")

            env.initilize_state(recommender, train_batches, test_batches_neg, args.high_state_size, args.A_state_size,
                                args.B_state_size, padding_number_A, padding_number_B)

        print("Jointly training finished, activate the final evaluation.")
        logging.info("Jointly training finished, activate the final evaluation.")
        print("Load best trained agent from", args.joint_agent)
        logging.info("Load best trained agent from %s" % args.joint_agent)
        agent_saver.restore(sess, tf.train.get_checkpoint_state(
            os.path.dirname(args.joint_agent + 'checkpoint')).model_checkpoint_path)
        print("Load best trained recommender from ", args.joint_recommender)
        logging.info("Load best trained recommender from %s " % args.joint_recommender)
        recommender_saver.restore(sess, tf.train.get_checkpoint_state(
            os.path.dirname(args.joint_recommender + 'checkpoint')).model_checkpoint_path)

        print("Evaluate jointly-trained recommender based on the selected test instances by the jointly-trained agent")
        logging.info(
            "Evaluate jointly-trained recommender based on the selected test instances by the jointly-trained agent")
        final_test_begin_time = time()
        (hr5_A, ndcg5_A, hr10_A, ndcg10_A, map_A, mrr_A, test_loss_A,
         hr5_B, ndcg5_B, hr10_B, ndcg10_B, map_B, mrr_B, test_loss_B) \
            = evalute(env, agent, recommender, test_batches_neg, noAgent=False)
        final_test_time = time() - final_test_begin_time
        print_recommender_train(epoch_to_print, hr5_A, ndcg5_A, hr10_A, ndcg10_A, map_A, mrr_A, hr5_B, ndcg5_B, hr10_B,
                                ndcg10_B, map_B, mrr_B, test_loss_A, test_loss_B, final_test_time, False)
        print("All training process has been finished.")
