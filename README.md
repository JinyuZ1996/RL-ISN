# **RL-ISN** 

<p align="left">
  <img src='https://img.shields.io/badge/python-3.6+-blue'>
  <img src='https://img.shields.io/badge/Tensorflow-1.12+-blue'>
  <img src='https://img.shields.io/badge/NumPy-1.16-brightgreen'>
  <img src='https://img.shields.io/badge/pandas-0.22.0-brightgreen'>
  <img src='https://img.shields.io/badge/scipy-1.5.3-brightgreen'>
</p> 

## **Overall description** 
- Here presents the code of RL-ISN and several baselines, the datasets of RL-ISN are located in the 'datasets' branch. You should download them before training the RL-ISN. Note that, this version of the code is attached to our paper: " **Reinforcement Learning-enhanced Shared-account Cross-domain Sequential Recommendation" (TKDE 2022)** ". If you want to use our codes and datasets in your research, please cite our paper. Note that, our paper is still in the state of 'Early Access', if you wang to see the preview version (on Arxiv), please visit: [http://arxiv.org/abs/2206.08088](http://arxiv.org/abs/2206.08088). As the dataset is too big for GitHub, we upload two datasets (i.e., Hvideo and Hamazon) on Bitbucket: [https://bitbucket.org/jinyuz1996/rl-isn-data/src/main/](https://bitbucket.org/jinyuz1996/rl-isn-data/src/main/ "https://bitbucket.org/jinyuz1996/rl-isn-data/src/main/")
## **Code description** 
### **Vesion of implements and tools**
1. python 3.6
2. tensorflow 1.12.0
3. scipy 1.5.3
4. numpy 1.16.0
5. pandas 0.22.0
6. matplotlib 3.3.4
7. Keras 1.0.7
8. tqdm 4.60.0
### **List of baselines**
1. Item-KNN, BPR-MF see: baseline/, which is quoted from: https://github.com/hidasib/GRU4Rec
2. Conet see: baseline/CO_NET/, which is quoted from "Conet: Collaborative Cross Networks for Cross-Domain Recommendation"
3. VUI-KNN see: baseline/VUI_KNN/, vui_process.py is the pre-processing code that generates the input data needed by code vui-module.py. 
4. NCF-MLP (NCF) see: baseline/NCF_MLP/, which is quoted from "Neural Collaborative Filtering". The different between 'NCF' and 'NCF-MLP++' is that we use it for CR task by sharing the collaborative filtering in different domains.
5. GRU4REC see: baseline/GRU4REC/, which is quoted from "session-based recommendations with recurrent neural networks"
6. HGRU see: baseline/HGRU/, which is implemented in Tensorflow
    * when your are trying to recurrent above two GRU-based network, you need to install numpy=1.16.4, pandas=0.24.2, CUDA=9.2 with CUDNN,theano=1.0.4 or newer version.
7. NAIS see: baseline/NAIS/, which is quoted from "NAIS: Neural Attentive Item Similarity Model for Recommendation"
8. DA-GCN see: baseline/DA_GCN/, which is quoted from "DA-GCN: A Domain-aware Attentive Graph Convolution Network for Shared-account Cross-domain Sequential Recommendation"
9. π-net see: baseline/PI_NET/, which is quoted from "π-Net: A Parallel Information-sharing Network for Shared-account Cross-domain Sequential Recommendations"
10. PSJ-net see: baseline/PSJ_NET/, which is quoted from "Parallel Split-Join Networks for Shared Account Cross-domain Sequential Recommendations"
11. Light-GCN see: baseline/LIGHT_GCN/, which is quoted from "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation".
### **Source code of RL-ISN**
1. the definition of Basic recommender see: RL_ISN/ISN_recommender.py
2. the definition of Agent see: RL_ISN/ISN_agent.py
3. the definition of Enviroment see: RL_ISN/ISN_environment.py
4. the definition of Training process see: RL_ISN/ISN_train.py
5. the definition of Evaluating process see: RL_ISN/ISN_evaluation.py
6. the preprocess of dataset see: RL_ISN/ISN_configuration.py
7. the hyper-parameters of RL-ISN see: RL_ISN/ISN_parameters.py
8. to run the training method see: RL_ISN/ISN_main.py and the training log printer was defined in: RL_ISN/ISN_printer.py
    * The directory named Checkpoint is used to save the trained recommenders ,agents and joint-training models.
    * Note that, for different dataset (i.e., HVIDEO and HAMAZON), we set two different settings for the haper-parameters as 'ISN_parameters_Hamazon.py' and 'ISN_parameters_Hvideo.py'
    * Note that, the 'userlist.txt' in both dataset are only provided for the baselines which needs it (i.e., the 'DA-GCN').
