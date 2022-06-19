# _*_coding: UTF_8 _*_
# To reach the best performance of RL-ISN on Hvideo, please follow this settings.
class setting:

    def __init__(self):
        self.our_code_path = "https://gitee.com/leiguo01/rl-net-hr"

        '''
            Block_1: parameters for the Recommender Pretrain Task
        '''
        self.recommender_pre_lr = 0.01  # Learning rate of recommender pretrain
        self.recommender_pre_tau = 0.9  # Model Conversion rate
        self.recommender_epochs = 30    # 20
        self.recommender_verbose = 15   # 5
        self.recommender_pretrain = True  # If this is the first time you run this model, this option must be 'True'

        '''
            Block_2: hyper-parameters for basic recommender configuration
        '''
        self.recommender_weight_size = 16
        self.recommender_embedding_size = 16
        self.account_num_A = 2  # hyper-param: numbers of latent user of a Single Account for domain_A
        self.account_num_B = 4  # hyper-param: numbers of latent user of a Single Account for domain_B
        self.regs = [1e-6, 1e-6, 1e-6]  # param to prevent gradient disappearance
        self.beta = 0.5  # hyper-param: to make the output smoothly in basic recommender
        self.dropout_rate = 0.1
        self.keep_prob = 1 - self.dropout_rate
        self.delta = 0.3

        '''
            Block_3: parameters for the Agent Pretrain Task
        '''
        self.agent_pretrain_lr_low = 0.05      # 0.05
        self.agent_pretrain_lr_high = 0.05    # 0.05
        self.decay_steps = 10000
        self.decay_rate = 0.95
        self.staircase = True
        self.agent_pretrain_tau_low = 0.05
        self.agent_pretrain_tau_high = self.agent_pretrain_tau_low
        self.agent_pretrain_epochs = 30
        self.agent_verbose = 10
        self.agent_pretrain = True  # If this is the first time you run this model, this option must be 'True'
        '''
            Block_4: parameters for the Agent Configuration
        '''
        self.agent_dropout = 0.1
        self.agent_keep_prob = 1 - self.agent_dropout
        self.high_state_size = self.recommender_weight_size + 2
        self.A_state_size = self.recommender_weight_size * 2 + 2
        self.B_state_size = self.recommender_weight_size * 2 + 2
        self.agent_weight_size = 8
        self.agent_hidden_size = 18
        self.random = False
        self.sample_times = 3

        '''
            Block_5: parameters for the joint training task
        '''
        self.recommender_joint_lr = 0.0001
        self.recommender_joint_tau = 0.0005
        self.agent_joint_lr_low = 0.0005
        self.agent_joint_lr_high = 0.0001
        self.agent_joint_tau_low = 0.0005
        self.agent_joint_tau_high = 0.0001

        self.joint_train_epochs = 5
        self.joint_verbose = 5

        '''
            Block_6: parameters for code global configuration
        '''
        self.fast_running = False  # set 'True' to use partial data for training
        self.fr_num = 0.2  # to set how many data will you use
        self.test_batch_size = 100  # number of test batch_size (99 negs + 1 target)
        self.num_negatives = 4  # to generate how many negs for training
        self.batch_size = 256  # number of train batch_size
        self.gpu_num = '0'

        '''
            Block_7: parameters for model_saver and checkpoint configuration
        '''
        self.pre_agent = "../Checkpoint/pre-agent/"
        self.pre_recommender = "../Checkpoint/pre-recommender/"
        self.joint_agent = "../Checkpoint/agent/"
        self.joint_recommender = "../Checkpoint/recommender/"
        self.checkpoint = 'checkpoint/trained_model.ckpt'

        '''
            Block_8: parameters for data_path and log_files configuration
        '''
        self.dataset = 'hvideo'  # 'hamazon' or 'hvideo'
        self.log = "../data/" + self.dataset + "/RL_ISN_logs.txt"
        self.path_E = '../data/' + self.dataset + '/Elist.txt'
        self.path_V = '../data/' + self.dataset + '/Vlist.txt'
        self.path_A = '../data/' + self.dataset + '/Alist.txt'
        self.path_B = '../data/' + self.dataset + '/Blist.txt'
        self.train_path = '../data/' + self.dataset + '/train_new.txt'
        self.test_path = '../data/' + self.dataset + '/test_new.txt'
        self.test_negative_A = '../data/' + self.dataset + '/test_new_Anegatives.txt'
        self.test_negative_B = '../data/' + self.dataset + '/test_new_Bnegatives.txt'

        '''
            Additional parameters for one-level testing.
        '''
        self.inner_reward_mask = 0  # if this param is set to 0, it means only the high-level tasks are worked


