import os
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import tqdm
import torch.nn.functional as F
import json
import torch.nn as nn
import torch.utils.data as Data
from sklearn import metrics

import plot_lstm_feature
import mmd
from model20 import CNN, DANN
from myutils import cal_class_weight, load_data, mkdir, load_data_strict, npy_loader
from pytorchtools import SaveBest
from collections import Counter
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE, ADASYN, RandomOverSampler

def run(config):
    
    # 指定计算设备
    DEVICE = torch.device('cuda:{}'.format(config['cuda'])) if torch.cuda.is_available() else torch.device('cpu')
    
    # 保存输入参数
    with open(config['dic_path'] + '/saved_params.json', 'w') as fp:
        json.dump(config, fp, indent=4)
    
    # 选择基础模型
    if (config['model'] in ['CNN', 'DDC', 'DAN']):
        model = CNN(DEVICE, config['srcclassnum']).to(DEVICE)
        print("CNN is null")
    elif (config['model'] in ['DANN', 'DANN-DDC', 'DANN-DAN']):
        model = DANN(DEVICE, config['srcclassnum']).to(DEVICE)
        train_dann(model, config, DEVICE)
    else:
        pass

def train_dann(model, config, DEVICE):

    # 定义重新初始化权重的函数
    def weights_init(m):
        if isinstance(m, nn.Linear):
            m.reset_parameters()
    
    def linear_increment(counter_origin: Counter, total_epoch: int, now_epoch: int) -> dict:
        return_dict = {}
        major_class = counter_origin.most_common(1)[0][0]
        num_major_class = counter_origin.most_common(1)[0][1]
        return_dict[major_class] = num_major_class
        for i in counter_origin.keys():
            if (i == major_class): continue
            return_dict[i] = int((counter_origin[i] + now_epoch * (num_major_class - counter_origin[i]) / total_epoch))
        logging.info("counter of data:{}".format(counter_origin))
        logging.info("counter of resampled data:{}".format(return_dict))
        return return_dict
    
    def dict_adapter(counter_batch: Counter, counter_all: Counter, counter_incre: dict) -> dict:
        result_dict = {}
        add_dict = {}
        for i in counter_batch.keys():
            result_dict[i] = counter_incre[i] - counter_batch[i] + counter_all[i]
            add_dict[i] = counter_incre[i] - counter_batch[i]
        logging.info("counter of added data:{}".format(add_dict))
        logging.info("counter of result data:{}".format(result_dict))
        return result_dict
    
    def full_sample(counter_origin: Counter) -> dict:
        return_dict = {}
        num_major_class = counter_origin.most_common(1)[0][1]
        for i in counter_origin.keys():
            return_dict[i] = num_major_class
        logging.info("counter of data:{}".format(counter_origin))
        logging.info("counter of resampled data:{}".format(return_dict))
        return return_dict

    def quadratic_func_increment(counter_origin: Counter, total_epoch: int, now_epoch: int) -> dict:
        return_dict = {}
        major_class = counter_origin.most_common(1)[0][0]
        num_major_class = counter_origin.most_common(1)[0][1]
        return_dict[major_class] = num_major_class
        for i in counter_origin.keys():
            if (i == major_class): continue
            return_dict[i] = int((counter_origin[i] + now_epoch * now_epoch * (num_major_class - counter_origin[i]) / (total_epoch * total_epoch)))
        logging.info("counter of data:{}".format(counter_origin))
        logging.info("counter of resampled data:{}".format(return_dict))
        return return_dict

    def train_stage1(model, optimizer_feature, optimizer_classifier, optimizer_domain, scheduler_feature, scheduler_classifier, src_train_feature, src_train_label, tar_train_feature, tar_train_label, config, DEVICE):
        # config: 目录配置
        # 定义和建立保存实验所有内容的实验文件夹
        dic_path_origin = config['dic_path']
        # 在实验文件夹下，定义和建立保存训练过程数据的训练文件夹
        dic_path = dic_path_origin + '/trainperiod'   
        mkdir(dic_path)
        # 在实验文件夹下，定义和建立保存训练过程中过采样数据的训练文件夹
        dic_path_oversample = '{}/oversample'.format(dic_path_origin)
        mkdir(dic_path_oversample)

        # config: 损失配置
        if (config['w_ceclass'] == True):
            # 计算或导入非均衡权重，通过cal_class_weight或直接指定numpy.array格式的weight序列，直接指定时，最好序列长度同类别数目一致
            weight = cal_class_weight(config['src_train_label'])
            weight = torch.from_numpy(weight).float()
            logging.info('The weight of classes:{}'.format(weight))
            loss_class = torch.nn.CrossEntropyLoss(weight=weight).to(DEVICE)  # 定义含权重的交叉熵损失
        else:
            loss_class = torch.nn.CrossEntropyLoss().to(DEVICE)
        loss_pseudoclass = torch.nn.CrossEntropyLoss().to(DEVICE) # 定义计算目标域损失的函数，用于观察目标域的分类损失

        # config: 定义作图用的list，list在训练过程中接收参数
        if (config['model'] == 'DANN'):
            train_loss_list = []
            valid_src_epoch = []
            valid_tar_epoch = []
            ce_list_epoch = []
            ce_domain_epoch = []
            ce_tar_epoch = []
        else:
            pass

        # 设置不同情况下的最优模型保存器
        best_for_trainsrc = SaveBest(ptpath = dic_path_origin)
        best_for_testtar = SaveBest(ptpath = dic_path_origin)

        dataloader_src, dataloader_tar = load_data(src_train_feature, src_train_label, tar_train_feature, tar_train_label, config['batchsize'])
        dataloader_oversampling, dataloader_tarfeature = None, None
        
        # 接受过采样向量的变量
        x_oversampled_embedding = torch.ones(0, 64, 1).to(DEVICE)
        y_oversampled_embedding = torch.ones(0, 1).to(DEVICE)
        
        for epoch in range(config['nepoch']):
            
            # 等预训练结束后，应当修改特征提取器和优化器
            if (epoch == config['preDANN']):
                logging.info("now model fixed")
                for k, v in model.feature.named_parameters():
                    if 'conv1' in k:
                        v.requires_grad = False
                new_parameters = filter(lambda p: p.requires_grad, model.feature.parameters())
                #重置特征提取器和分类器的学习率以及优化器
                lr_origin = config['lr']
                lambda_dctln = lambda epoch: 1 / ((1 + 10 * epoch / config['nepoch']) ** 0.75)
                # model.classifier.apply(weights_init)
                optimizer_feature = optim.AdamW(new_parameters, lr=lr_origin)
                optimizer_classifier = optim.AdamW(model.classifier.parameters(), lr=lr_origin)
                scheduler_feature = optim.lr_scheduler.LambdaLR(optimizer_feature, lr_lambda=lambda_dctln)
                scheduler_classifier = optim.lr_scheduler.LambdaLR(optimizer_classifier, lr_lambda=lambda_dctln)

            # 汇报当前epoch的学习率
            logging.info('feature_learning-rate:{}'.format(optimizer_feature.param_groups[0]['lr']))
            logging.info('classifier_learning-rate:{}'.format(optimizer_classifier.param_groups[0]['lr'])) 
            i = 1  # 初始化batch计数器

            # 定义接收器接收loss来观察
            total_err = 0
            total_celoss = 0
            total_tarpseudoloss = 0
            total_domainloss = 0
            
            # 计算源域和目标域数据最小batch数，用来循环
            len_dataloader = len(dataloader_src)
            logging.info("length of src dataloader:{}".format(len(dataloader_src)))
            logging.info("length of tar dataloader:{}".format(len(dataloader_tar)))
            
            reoversampling_epoch_list = [0] + [i * 20 for i in range(50, 100)]
            
            if epoch in reoversampling_epoch_list:
                model.classifier.apply(weights_init)
                logging.info("{} epoch is resampling epoch".format(epoch))
                # 过采样
                model.eval()
                with torch.no_grad():
                    src_train_feature_gpu = torch.from_numpy(np.load(src_train_feature)).to(DEVICE)
                    tar_train_feature_gpu = torch.from_numpy(np.load(tar_train_feature)).to(DEVICE)
                    _, _, src_full_embedding_gpu, _, _, _ = model(input_data=src_train_feature_gpu, alpha=0, source=True)
                    _, _, tar_full_embedding_gpu, _, _, _ = model(input_data=tar_train_feature_gpu, alpha=0, source=True)
                    src_full_embedding = src_full_embedding_gpu.clone().detach().cpu().numpy()
                    tar_full_embedding = tar_full_embedding_gpu.clone().detach().cpu()
                    src_full_label = np.load(src_train_label)
                    tar_full_label = torch.from_numpy(np.load(tar_train_label))

                    # 解耦向量和梯度
                    # 计算需要过采样的数量
                    src_temp_all_length = len(src_full_embedding)
                    # thisbatch_dict = linear_increment(Counter(src_full_label.reshape((-1))), config['nepoch'], epoch)
                    thisbatch_dict = full_sample(Counter(src_full_label.reshape((-1))))
                    # ROS
                    # x_src_reshape_resampled, y_src_reshape_resampled = RandomOverSampler(sampling_strategy=thisbatch_dict).fit_resample(src_full_feature.reshape((-1, 64)), src_full_label.reshape((-1)))
                    # SMOTE过采样
                    # regular-SMOTE
                    x_src_reshape_resampled, y_src_reshape_resampled = SMOTE(sampling_strategy=thisbatch_dict).fit_resample(src_full_embedding.reshape((-1, 64)), src_full_label.reshape((-1)))
                    # Borderline1-SMOTE
                    # x_src_reshape_resampled, y_src_reshape_resampled = BorderlineSMOTE(sampling_strategy=thisepoch_dict, kind='borderline-2').fit_resample(src_temp_feature_origin.reshape((-1, 64)), src_temp_y_origin.reshape((-1)))
                    # Borderline2-SMOTE
                    # x_src_reshape_resampled, y_src_reshape_resampled = BorderlineSMOTE(sampling_strategy=thisepoch_dict).fit_resample(src_temp_feature_origin.reshape((-1, 64)), src_temp_y_origin.reshape((-1)))
                    # ADASYN error: No samples will be generated with the provided ratio settings.
                    # x_src_reshape_resampled, y_src_reshape_resampled = ADASYN(sampling_strategy=thisepoch_dict).fit_resample(src_temp_feature_origin.reshape((-1, 64)), src_temp_y_origin.reshape((-1)))
                    # SVMSMOTE error: Found array with 0 sample(s) (shape=(0, 64)) while a minimum of 1 is required.
                    # x_src_reshape_resampled, y_src_reshape_resampled = SVMSMOTE(sampling_strategy=thisepoch_dict).fit_resample(src_temp_feature_origin.reshape((-1, 64)), src_temp_y_origin.reshape((-1)))
                    # KMeansSMOTE error: No clusters found with sufficient samples of class 5. Try lowering the cluster_balance_threshold or or increasing the number of clusters.
                    # x_src_reshape_resampled, y_src_reshape_resampled = KMeansSMOTE(sampling_strategy=thisepoch_dict).fit_resample(src_temp_feature_origin.reshape((-1, 64)), src_temp_y_origin.reshape((-1)))
                    # 记录过采样结果
                    x_oversampled_embedding = torch.from_numpy(x_src_reshape_resampled.reshape((-1, 64, 1))).to(DEVICE)
                    y_oversampled_embedding = torch.from_numpy(y_src_reshape_resampled.reshape((-1, 1))).to(DEVICE)
                    # 将过采样新生成的样本单独输入模型并计算损失
                    # X_resampled = torch.from_numpy(x_src_reshape_resampled.reshape((-1, 64, 1))[src_temp_all_length:])
                    # y_resampled = torch.from_numpy(y_src_reshape_resampled.reshape((-1, 1))[src_temp_all_length:])
                    X_resampled = torch.from_numpy(x_src_reshape_resampled.reshape((-1, 64, 1)))
                    y_resampled = torch.from_numpy(y_src_reshape_resampled.reshape((-1, 1)))

                    # 对过采样后的样本做batch
                    dataset_oversampling = Data.TensorDataset(X_resampled, y_resampled)
                    batchsize_oversampling = int(len(X_resampled) / len_dataloader)
                    dataloader_oversampling = torch.utils.data.DataLoader(
                        dataset=dataset_oversampling,
                        batch_size=batchsize_oversampling,
                        shuffle=True,
                        drop_last=True,
                    )
                    dataset_tar_feature = Data.TensorDataset(tar_full_embedding, tar_full_label)
                    dataloader_tarfeature = torch.utils.data.DataLoader(
                        dataset=dataset_tar_feature,
                        batch_size=config['batchsize'],
                        shuffle=True,
                        drop_last=True,
                    )
                    dataset_src_feature = Data.TensorDataset(torch.from_numpy(src_full_embedding), torch.from_numpy(src_full_label))
                    dataloader_srcfeature = torch.utils.data.DataLoader(
                        dataset=dataset_src_feature,
                        batch_size=config['batchsize'],
                        shuffle=True,
                        drop_last=True,
                    )
            
            if (epoch >= 0 and epoch < config['preDANN']):
                # 每个batch使用优化器更新参数
                for (data_src, data_tar) in tqdm.tqdm(zip(enumerate(dataloader_src), enumerate(dataloader_tar)), total=len_dataloader, leave=False):
                    # 将模型调整成训练模式
                    model.train()  

                    # 读入每个batch数据
                    _, (x_src, y_src) = data_src
                    _, (x_tar, y_tar) = data_tar
                    x_src, y_src, x_tar, y_tar = x_src.to(DEVICE), y_src.to(DEVICE), x_tar.to(DEVICE), y_tar.to(DEVICE)
                    x_src = x_src.float()
                    x_tar = x_tar.float()
                    y_src = y_src.squeeze(1).long()
                    y_tar = y_tar.squeeze(1).long()

                    # alpha，DANN的反转层参数，用于在训练中不断调整
                    p = float(i + epoch * len_dataloader) / config['nepoch'] / len_dataloader
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1

                    # 输入源域训练数据，得到分类器fc层输出、域损失、特征、域分类结果
                    class_output_src, err_s_domain, src_temp_feature, src_dan2, src_dan3, domain_pred_src = model(input_data=x_src, alpha=alpha, source=True)                    

                    # 计算训练数据源域分类损失
                    err_s_label = loss_class(class_output_src, y_src)
                    
                    # 输入目标域训练数据，得到分类器fc层输出、域损失、特征、域分类结果
                    class_output_tar, err_t_domain, tar_temp_feature, tar_dan2, tar_dan3, domain_pred_tar = model(input_data=x_tar, alpha=alpha, source=False)

                    # 压缩处理
                    domain_pred_src = domain_pred_src.squeeze(1)
                    domain_pred_tar = domain_pred_tar.squeeze(1)

                    # 计算总的域损失
                    err_domain = err_t_domain + err_s_domain

                    # 计算类伪标签及损失，用于观察目标域的损失情况
                    class_out_vt_detach = class_output_tar.detach()
                    err_t_pseudo = loss_pseudoclass(class_out_vt_detach, y_tar)

                    # 定义loss，计算总损失
                    if (config['model'] == 'DANN'):
                        err = err_s_label + config['w_gamma'] * err_domain
                        total_domainloss += err_domain.item() * config['w_gamma']
                        total_celoss += err_s_label.item()
                        total_tarpseudoloss += err_t_pseudo.item()
                        total_err += err.item()
                    else:
                        pass

                    optimizer_feature.zero_grad()
                    optimizer_classifier.zero_grad()
                    optimizer_domain.zero_grad()
                    err.backward()
                    optimizer_feature.step()
                    optimizer_classifier.step()
                    optimizer_domain.step()
                    i += 1
            
            else:
                # 每个batch使用优化器更新参数
                for (data_src, data_srcover, data_tar) in tqdm.tqdm(zip(enumerate(dataloader_srcfeature), enumerate(dataloader_oversampling), enumerate(dataloader_tarfeature)), total=len_dataloader, leave=False):
                    # 将模型调整成训练模式
                    model.train()  

                    # 读入每个batch数据
                    _, (x_src, y_src) = data_src
                    _, (x_tar, y_tar) = data_tar
                    x_src, y_src, x_tar, y_tar = x_src.to(DEVICE), y_src.to(DEVICE), x_tar.to(DEVICE), y_tar.to(DEVICE)
                    x_src = x_src.float()
                    x_tar = x_tar.float()
                    y_src = y_src.squeeze(1).long()
                    y_tar = y_tar.squeeze(1).long()
                    
                    _, (x_srcover, y_srcover) = data_srcover
                    x_srcover, y_srcover = x_srcover.to(DEVICE), y_srcover.to(DEVICE)
                    x_srcover = x_srcover.float()
                    y_srcover = y_srcover.squeeze(1).long()

                    # alpha，DANN的反转层参数，用于在训练中不断调整
                    p = float(i + epoch * len_dataloader) / config['nepoch'] / len_dataloader
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1

                    # 输入源域训练数据，得到域损失
                    _, err_s_domain, _, _, _, _ = model(input_data=x_src, alpha=alpha, source=True, only_resampled=True)
                    
                    # 输入源域过采样训练数据，得到分类器fc层输出、域损失、特征、域分类结果
                    class_output_srcover, err_sover_domain, _, _, _, domain_pred_src = model(input_data=x_srcover, alpha=alpha, source=True, only_resampled=True)
                    
                    # 计算训练数据源域分类损失
                    err_s_label = loss_class(class_output_srcover, y_srcover)
                    
                    # 输入目标域训练数据，得到分类器fc层输出、域损失、特征、域分类结果
                    class_output_tar, err_t_domain, tar_temp_feature, tar_dan2, tar_dan3, domain_pred_tar = model(input_data=x_tar, alpha=alpha, source=False, only_resampled=True)

                    # 压缩处理
                    domain_pred_src = domain_pred_src.squeeze(1)
                    domain_pred_tar = domain_pred_tar.squeeze(1)

                    # 计算总的域损失
                    err_domain = err_t_domain + err_s_domain

                    # 计算类伪标签及损失，用于观察目标域的损失情况
                    class_out_vt_detach = class_output_tar.detach()
                    err_t_pseudo = loss_pseudoclass(class_out_vt_detach, y_tar)

                    # 定义loss，计算总损失
                    if (config['model'] == 'DANN'):
                        err = err_s_label + config['w_gamma'] * err_domain
                        total_domainloss += err_domain.item() * config['w_gamma']
                        total_celoss += err_s_label.item()
                        total_tarpseudoloss += err_t_pseudo.item()
                        total_err += err.item()
                    else:
                        pass

                    optimizer_feature.zero_grad()
                    optimizer_classifier.zero_grad()
                    optimizer_domain.zero_grad()
                    err.backward()
                    optimizer_feature.step()
                    optimizer_classifier.step()
                    optimizer_domain.step()
                    i += 1
                    
            scheduler_feature.step()
            scheduler_classifier.step()

            # 保存和输出模型训练时的loss
            if (config['model'] == 'DANN'):
                item_pr = 'Epoch: [{}/{}], classify_loss: {:.4f}, domain_loss: {:.4f}, total_loss: {:.4f}, batchnum: {}, lr: {:.4f}'.format(
                    epoch, config['nepoch'], total_celoss/len_dataloader, total_domainloss/len_dataloader, total_err/len_dataloader, i, optimizer_classifier.param_groups[0]['lr'])
            else:
                pass
            print(item_pr)
            logging.info(item_pr)
            fp = open(os.getcwd() + '/' + dic_path + '/' + config['resultpath'], 'a')
            fp.write(item_pr + '\n')

            # validate，并使用验证集loss去判定early_stopping
            f1_train_src, f1_train_tar = modelvalid(model, config, 'train')
            f1_test_src, f1_test_tar = modelvalid(model, config, 'test')
            train_info = 'Train Source f1-score: {:.4f}, Train Target f1-score: {:.4f}'.format(f1_train_src, f1_train_tar)
            test_info = 'Test Source f1-score: {:.4f}, Test Target f1-score: {:.4f}'.format(f1_test_src, f1_test_tar)
            print(best_for_trainsrc.best_score, best_for_trainsrc.renew)
            print(best_for_testtar.best_score, best_for_testtar.renew)
            best_for_trainsrc(f1_train_src, model, 'besttrainsrc', epoch, x_oversampled_embedding, y_oversampled_embedding)
            best_for_testtar(f1_test_tar, model, 'besttesttar', epoch, x_oversampled_embedding, y_oversampled_embedding)
            logging.info('Trainsrc_bestepoch:{}, Testtar_bestepoch:{}'.format(best_for_trainsrc.renew, best_for_testtar.renew))
            fp.write(train_info + ", " + test_info +  '\n')
            print(train_info + ", " + test_info)
            logging.info(train_info + ", " + test_info)
            if (config['model'] == 'DANN'):
                valid_src_epoch.append(f1_test_src)
                valid_tar_epoch.append(f1_test_tar)
                train_loss_list.append(total_err/len_dataloader)
                ce_list_epoch.append(total_celoss/len_dataloader)
                ce_domain_epoch.append(total_domainloss/len_dataloader)
                ce_tar_epoch.append(total_tarpseudoloss/len_dataloader)
            else:
                pass
            fp.close()
            
            if (epoch % 10 == 0):
                modeltest_light(dic_path_origin, model, config, epoch, 'besttrainsrc_train')
                modeltest_light(dic_path_origin, model, config, epoch, 'besttrainsrc_test')

            # 训练结束
            if (epoch == config['nepoch']-1):
                print("Stop Training")
                logging.info("Stop Training")
                lastepoch = epoch
                torch.save(model.state_dict(), '{}/endpoint.pt'.format(dic_path_origin))
                # 过采样数据的存储和预测情况判断
                oversampling_save(dic_path_origin, x_oversampled_embedding, y_oversampled_embedding)
                # testtar_checkpoint模型读取
                model_besttesttar = DANN(DEVICE, config['srcclassnum']).to(DEVICE)
                model_besttesttar.load_state_dict(torch.load('{}/besttesttar_checkpoint.pt'.format(dic_path_origin)))
                x_besttesttar = torch.from_numpy(np.load('{}/oversample/x_oversample_besttesttar.npy'.format(dic_path_origin))).to(DEVICE)
                y_besttesttar = torch.from_numpy(np.load('{}/oversample/y_oversample_besttesttar.npy'.format(dic_path_origin))).to(DEVICE)
                oversampling_determine(dic_path_origin, model_besttesttar, config, x_besttesttar, y_besttesttar, 'besttesttar')
                modeltest(dic_path_origin, model_besttesttar, config, 'besttesttar_train')
                modeltest(dic_path_origin, model_besttesttar, config, 'besttesttar_test')       
                # trainsrc_checkpoint模型读取
                model_besttrainsrc = DANN(DEVICE, config['srcclassnum']).to(DEVICE)
                model_besttrainsrc.load_state_dict(torch.load('{}/besttrainsrc_checkpoint.pt'.format(dic_path_origin)))
                x_besttrainsrc = torch.from_numpy(np.load('{}/oversample/x_oversample_besttrainsrc.npy'.format(dic_path_origin))).to(DEVICE)
                y_besttrainsrc = torch.from_numpy(np.load('{}/oversample/y_oversample_besttrainsrc.npy'.format(dic_path_origin))).to(DEVICE)
                oversampling_determine(dic_path_origin, model_besttrainsrc, config, x_besttrainsrc, y_besttrainsrc, 'besttrainsrc')
                modeltest(dic_path_origin, model_besttrainsrc, config, 'besttrainsrc_train')
                modeltest(dic_path_origin, model_besttrainsrc, config, 'besttrainsrc_test')   
                break

        # 画图
        x1 = range(lastepoch+1)
        x2 = range(lastepoch+1)
        x3 = range(lastepoch+1)
        y1 = valid_tar_epoch
        y2 = train_loss_list
        y3 = valid_src_epoch
        plt.subplot(3, 1, 1)
        plt.plot(x1, y1, 'o-')
        plt.title('Test f1-score vs. epoches')
        plt.ylabel('Test f1-score')
        plt.subplot(3, 1, 2)
        plt.plot(x2, y2, '.-', label='Training Loss')
        plt.plot(x2, valid_src_epoch, label='Validation f1-score')
        # 绘制checkpoint
        minposs = valid_src_epoch.index(max(valid_src_epoch)) + 1
        plt.axvline(minposs, linestyle='--', color='r',
                    label='Early Stopping Checkpoint')

        plt.xlabel('total loss vs. epoches')
        plt.ylabel('total loss')
        plt.subplot(3, 1, 3)
        plt.plot(x3, y3, 'o-')
        plt.xlabel('train f1-score vs. epoches')
        plt.ylabel('train f1-score')
        plt.tight_layout()
        plt.show()
        plt.savefig("{}/f1-score_loss.jpg".format(dic_path))
        plt.clf()

        # 损失图像
        if (config['model'] == 'DANN'):
            plt.figure(figsize=(20, 10))
            x4 = range(lastepoch+1)
            x5 = range(lastepoch+1)
            x6 = range(lastepoch+1)
            y4 = ce_list_epoch
            y5 = ce_tar_epoch
            y6 = ce_domain_epoch
            plt.subplot(3, 1, 1)
            plt.plot(x4, y4, '.-')
            plt.title('loss vs. epoches')
            plt.xlabel('epoches')
            plt.ylabel('CEloss')
            plt.subplot(3, 1, 2)
            plt.plot(x5, y5, '.-')
            plt.xlabel('epoches')
            plt.ylabel('pseudoloss')
            plt.subplot(3, 1, 3)
            plt.plot(x6, y6, '.-')
            plt.xlabel('epoches')
            plt.ylabel('domainloss')
            plt.tight_layout()
            plt.savefig("{}/loss.jpg".format(dic_path))
            print('plot feature')
            logging.info('plot DANN feature')
        else:
            pass
    
    def oversampling_save(dic_path_origin, x_oversampling_emb, y_oversampling_emb):
        dic_path = '{}/oversample'.format(dic_path_origin)
        mkdir(dic_path)
        np.save('{}/x_oversample_final.npy'.format(dic_path), x_oversampling_emb.cpu().detach())
        np.save('{}/y_oversample_final.npy'.format(dic_path), y_oversampling_emb.cpu().detach())

    def oversampling_determine(dic_path_origin, model, config, x_oversampling_emb, y_oversampling_emb, datamode):
        dic_path = '{}/oversample'.format(dic_path_origin)
        mkdir(dic_path)
        model.eval()
        test_alpha = 0
        with torch.no_grad():
            class_output_oversample, _, _, _, _, _ = model(input_data=x_oversampling_emb, alpha=test_alpha, source=True, only_resampled=True)
            prob_oversample = F.softmax(class_output_oversample, dim=1)
            class_prob_oversample, class_output_oversample = torch.max(prob_oversample.data, dim=1)
            np.save('{}/y_oversample_predict_{}.npy'.format(dic_path, datamode), class_output_oversample.cpu().detach())
            average_list = ['micro', 'macro', 'weighted', None]
            target_names = ['']
            for i in range(config['srcclassnum']):
                target_names += ['class_{}'.format(i)]
            f1_score = [metrics.f1_score(y_oversampling_emb.cpu().detach(), class_output_oversample.cpu().detach(), average=ways) for ways in average_list]
            recall_score = [metrics.recall_score(y_oversampling_emb.cpu().detach(), class_output_oversample.cpu().detach(), average=ways) for ways in average_list]
            precision_score = [metrics.precision_score(y_oversampling_emb.cpu().detach(), class_output_oversample.cpu().detach(), average=ways) for ways in average_list]
            confusion_matrix = metrics.confusion_matrix(y_oversampling_emb.cpu().detach(), class_output_oversample.cpu().detach())
            with open(dic_path + '/{}_metrics.txt'.format(datamode), 'w') as f:
                f.write('---------- for target blew: \n')
                # f1_score
                f.write('f1_score : \n')
                [f.write('{} : {} \n'.format(average_list[i], f1_score[i])) for i in range(len(average_list))]
                f.write('\n')
                # recall_score
                f.write('recall_score : \n')
                [f.write('{} : {} \n'.format(average_list[i], recall_score[i])) for i in range(len(average_list))]
                f.write('\n')
                # precision_score
                f.write('precision_score : \n')
                [f.write('{} : {} \n'.format(average_list[i], precision_score[i])) for i in range(len(average_list))]
                f.write('\n')
                f.write('confusion_matrix : \n')
                for item in target_names:
                    f.write('{:<10s} '.format(item))
                f.write('\n')
                for i in range(len(confusion_matrix)):
                    f.write('{:<10s} '.format(target_names[i + 1]))
                    for item in confusion_matrix[i]:
                        f.write('{:<10d} '.format(item))
                    f.write('\n')



    def modelvalid(model, config, datamode = 'train'):
        #data preparing
        if (datamode == 'train'):
            feature_motor_src = torch.from_numpy(np.load(config['src_train_feature']))
            label_motor_src = torch.from_numpy(np.load(config['src_train_label']))
            feature_motor_tar = torch.from_numpy(np.load(config['tar_train_feature']))
            label_motor_tar = torch.from_numpy(np.load(config['tar_train_label']))
        elif (datamode == 'test'):
            feature_motor_src = torch.from_numpy(np.load(config['src_test_feature']))
            label_motor_src = torch.from_numpy(np.load(config['src_test_label']))
            feature_motor_tar = torch.from_numpy(np.load(config['tar_test_feature']))
            label_motor_tar = torch.from_numpy(np.load(config['tar_test_label']))
        else:
            feature_motor_src = None
            label_motor_src = None
            feature_motor_tar = None
            label_motor_tar = None
            print('Fatal Error: In def modeltest there is an error in datamode')
            logging.error('Fatal Error: In def modeltest there is an error in datamode')
        feature_motor_src, label_motor_src = feature_motor_src.to(DEVICE).float(), label_motor_src.to(DEVICE)
        feature_motor_tar, label_motor_tar = feature_motor_tar.to(DEVICE).float(), label_motor_tar.to(DEVICE)
        model.eval()
        test_alpha = 0
        with torch.no_grad():
            #src 输出可以作为验证集的信息辅助训练
            class_out_vs, domain_out_vs, vs_temp_feature, vs_dan2, vs_dan3, domain_pred_vs = model(input_data=feature_motor_src, alpha=test_alpha, source=True)
            prob_src = F.softmax(class_out_vs, dim=1)
            class_prob_src, class_out_src = torch.max(prob_src.data, dim=1)

            src_y_true = label_motor_src.cpu().detach()
            src_y_pred = class_out_src.cpu().detach()

            f1_score_src = metrics.f1_score(src_y_true, src_y_pred, average='macro')

            class_out_vt, domain_out_vt, vt_temp_feature, vt_dan2, vt_dan3, domain_pred_vt = model(input_data=feature_motor_tar, alpha=test_alpha, source=False)
            prob_tar = F.softmax(class_out_vt, dim=1)
            class_prob_tar, class_out_tar = torch.max(prob_tar.data, dim=1)

            tar_y_true = label_motor_tar.cpu().detach()
            tar_y_pred = class_out_tar.cpu().detach()

            f1_score_tar = metrics.f1_score(tar_y_true, tar_y_pred, average='macro')

            label_motor_src = label_motor_src.squeeze(1).long()
            label_motor_tar = label_motor_tar.squeeze(1).long()

            return f1_score_src, f1_score_tar
    
    def modeltest(dic_path_origin, model, config, datamode='train'):
        #data preparing
        if (datamode == 'besttrainsrc_train' or datamode == 'besttesttar_train'):
            feature_motor_src = torch.from_numpy(np.load(config['src_train_feature']))
            label_motor_src = torch.from_numpy(np.load(config['src_train_label']))
            feature_motor_tar = torch.from_numpy(np.load(config['tar_train_feature']))
            label_motor_tar = torch.from_numpy(np.load(config['tar_train_label'])) 
        elif (datamode == 'besttrainsrc_test' or datamode == 'besttesttar_test'):
            feature_motor_src = torch.from_numpy(np.load(config['src_test_feature']))
            label_motor_src = torch.from_numpy(np.load(config['src_test_label']))
            feature_motor_tar = torch.from_numpy(np.load(config['tar_test_feature']))
            label_motor_tar = torch.from_numpy(np.load(config['tar_test_label']))
        else:
            feature_motor_src = None
            label_motor_src = None
            feature_motor_tar = None
            label_motor_tar = None
            print('Fatal Error: In def modeltest there is an error in datamode')
            logging.error('Fatal Error: In def modeltest there is an error in datamode')
        feature_motor_src, label_motor_src = feature_motor_src.to(DEVICE).float(), label_motor_src.to(DEVICE)
        feature_motor_tar, label_motor_tar = feature_motor_tar.to(DEVICE).float(), label_motor_tar.to(DEVICE)
        #feature_motor_src = feature_motor_src.unsqueeze(1)
        #feature_motor_tar = feature_motor_tar.unsqueeze(1)

        dic_path = '{}/{}_period'.format(dic_path_origin, datamode)
        mkdir(dic_path)
        model.eval()
        # alpha 
        test_alpha = 0
        with torch.no_grad():
            class_out_vs, _, vs_temp_feature, _, _, _ = model(input_data=feature_motor_src, alpha=test_alpha, source=True)
            class_out_vt, _, vt_temp_feature, _, _, _ = model(input_data=feature_motor_tar, alpha=test_alpha, source=False)

            prob_src = F.softmax(class_out_vs, dim=1)
            class_prob_src, class_out_src = torch.max(prob_src.data, dim=1)
            prob_tar = F.softmax(class_out_vt, dim=1)
            class_prob_tar, class_out_tar = torch.max(prob_tar.data, dim=1)

            raw_src_y_true = class_prob_src.cpu().detach()
            src_y_pred_vector = class_out_vs.cpu().detach()
            src_y_true = label_motor_src.cpu().detach()
            src_y_pred = class_out_src.cpu().detach()

            raw_tar_y_true = class_prob_tar.cpu().detach()
            tar_y_pred_vector = class_out_vt.cpu().detach()
            tar_y_true = label_motor_tar.cpu().detach()
            tar_y_pred = class_out_tar.cpu().detach()

            np.save('{}/src_feature.npy'.format(dic_path), vs_temp_feature.cpu().detach())
            np.save('{}/tar_feature.npy'.format(dic_path), vt_temp_feature.cpu().detach())
            np.save('{}/tar_predict_result.npy'.format(dic_path), tar_y_pred_vector)
            np.save('{}/src_predict_result.npy'.format(dic_path), src_y_pred_vector)
            np.save('{}/tar_predict_result_inverse.npy'.format(dic_path), tar_y_pred)
            np.save('{}/src_predict_result_inverse.npy'.format(dic_path), src_y_pred)
            np.save('{}/src_label.npy'.format(dic_path), src_y_true)
            np.save('{}/tar_label.npy'.format(dic_path), tar_y_true)

            average_list = ['micro', 'macro', 'weighted', None]
            target_names = ['']
            for i in range(config['srcclassnum']):
                target_names += ['class_{}'.format(i)]
            # 记录目标域结果
            f1_score = [metrics.f1_score(tar_y_true, tar_y_pred, average=ways) for ways in average_list]
            recall_score = [metrics.recall_score(tar_y_true, tar_y_pred, average=ways) for ways in average_list]
            precision_score = [metrics.precision_score(tar_y_true, tar_y_pred, average=ways) for ways in average_list]
            confusion_matrix = metrics.confusion_matrix(tar_y_true, tar_y_pred)
            # 记录源域结果
            f1_score_train = [metrics.f1_score(src_y_true, src_y_pred, average=ways) for ways in average_list]
            recall_score_train = [metrics.recall_score(src_y_true, src_y_pred, average=ways) for ways in average_list]
            precision_score_train = [metrics.precision_score(src_y_true, src_y_pred, average=ways) for ways in average_list]
            confusion_matrix_train = metrics.confusion_matrix(src_y_true, src_y_pred)
            with open(dic_path + '/metrics.txt', 'w') as f:
                f.write('---------- for target blew: \n')
                # f1_score
                f.write('f1_score : \n')
                [f.write('{} : {} \n'.format(average_list[i], f1_score[i])) for i in range(len(average_list))]
                f.write('\n')
                # recall_score
                f.write('recall_score : \n')
                [f.write('{} : {} \n'.format(average_list[i], recall_score[i])) for i in range(len(average_list))]
                f.write('\n')
                # precision_score
                f.write('precision_score : \n')
                [f.write('{} : {} \n'.format(average_list[i], precision_score[i])) for i in range(len(average_list))]
                f.write('\n')
                f.write('confusion_matrix : \n')
                for item in target_names:
                    f.write('{:<10s} '.format(item))
                f.write('\n')
                for i in range(len(confusion_matrix)):
                    f.write('{:<10s} '.format(target_names[i + 1]))
                    for item in confusion_matrix[i]:
                        f.write('{:<10d} '.format(item))
                    f.write('\n')
                # for the source
                f.write('\n\n')
                f.write('---------- for source blew: \n')
                # f1_score
                f.write('f1_score : \n')
                [f.write('{} : {} \n'.format(average_list[i], f1_score_train[i])) for i in range(len(average_list))]
                f.write('\n')
                # recall_score
                f.write('recall_score : \n')
                [f.write('{} : {} \n'.format(average_list[i], recall_score_train[i])) for i in range(len(average_list))]
                f.write('\n')
                # precision_score
                f.write('precision_score : \n')
                [f.write('{} : {} \n'.format(average_list[i], precision_score_train[i])) for i in range(len(average_list))]
                f.write('\n')
                f.write('confusion_matrix : \n')
                for item in target_names:
                    f.write('{:<10s} '.format(item))
                f.write('\n')
                for i in range(len(confusion_matrix_train)):
                    f.write('{:<10s} '.format(target_names[i + 1]))
                    for item in confusion_matrix_train[i]:
                        f.write('{:<10d} '.format(item))
                    f.write('\n')
            if (datamode == 'besttesttar_test' or datamode == 'besttrainsrc_test'):
                plot_lstm_feature.plot_oversampling(dic_path_origin, config['source'], config['target'], datamode, True)
    
    def modeltest_light(dic_path_origin, model, config, nowepoch, datamode='train'):
        #data preparing
        if (datamode == 'besttrainsrc_train' or datamode == 'besttesttar_train'):
            feature_motor_src = torch.from_numpy(np.load(config['src_train_feature']))
            label_motor_src = torch.from_numpy(np.load(config['src_train_label']))
            feature_motor_tar = torch.from_numpy(np.load(config['tar_train_feature']))
            label_motor_tar = torch.from_numpy(np.load(config['tar_train_label']))
            mode = 'train' 
        elif (datamode == 'besttrainsrc_test' or datamode == 'besttesttar_test'):
            feature_motor_src = torch.from_numpy(np.load(config['src_test_feature']))
            label_motor_src = torch.from_numpy(np.load(config['src_test_label']))
            feature_motor_tar = torch.from_numpy(np.load(config['tar_test_feature']))
            label_motor_tar = torch.from_numpy(np.load(config['tar_test_label']))
            mode = 'test'
        else:
            feature_motor_src = None
            label_motor_src = None
            feature_motor_tar = None
            label_motor_tar = None
            print('Fatal Error: In def modeltest there is an error in datamode')
            logging.error('Fatal Error: In def modeltest there is an error in datamode')
        feature_motor_src, label_motor_src = feature_motor_src.to(DEVICE).float(), label_motor_src.to(DEVICE)
        feature_motor_tar, label_motor_tar = feature_motor_tar.to(DEVICE).float(), label_motor_tar.to(DEVICE)
        #feature_motor_src = feature_motor_src.unsqueeze(1)
        #feature_motor_tar = feature_motor_tar.unsqueeze(1)

        dic_path = '{}/trainperiod'.format(dic_path_origin)
        mkdir(dic_path)
        model.eval()
        # alpha 
        test_alpha = 0
        with torch.no_grad():
            class_out_vs, _, vs_temp_feature, _, _, _ = model(input_data=feature_motor_src, alpha=test_alpha, source=True)
            class_out_vt, _, vt_temp_feature, _, _, _ = model(input_data=feature_motor_tar, alpha=test_alpha, source=False)

            prob_src = F.softmax(class_out_vs, dim=1)
            class_prob_src, class_out_src = torch.max(prob_src.data, dim=1)
            prob_tar = F.softmax(class_out_vt, dim=1)
            class_prob_tar, class_out_tar = torch.max(prob_tar.data, dim=1)

            raw_src_y_true = class_prob_src.cpu().detach()
            src_y_pred_vector = class_out_vs.cpu().detach()
            src_y_true = label_motor_src.cpu().detach()
            src_y_pred = class_out_src.cpu().detach()

            raw_tar_y_true = class_prob_tar.cpu().detach()
            tar_y_pred_vector = class_out_vt.cpu().detach()
            tar_y_true = label_motor_tar.cpu().detach()
            tar_y_pred = class_out_tar.cpu().detach()

            average_list = ['micro', 'macro', 'weighted', None]
            target_names = ['']
            for i in range(config['srcclassnum']):
                target_names += ['class_{}'.format(i)]
            # 记录目标域结果
            f1_score = [metrics.f1_score(tar_y_true, tar_y_pred, average=ways) for ways in average_list]
            recall_score = [metrics.recall_score(tar_y_true, tar_y_pred, average=ways) for ways in average_list]
            precision_score = [metrics.precision_score(tar_y_true, tar_y_pred, average=ways) for ways in average_list]
            confusion_matrix = metrics.confusion_matrix(tar_y_true, tar_y_pred)
            # 记录源域结果
            f1_score_train = [metrics.f1_score(src_y_true, src_y_pred, average=ways) for ways in average_list]
            recall_score_train = [metrics.recall_score(src_y_true, src_y_pred, average=ways) for ways in average_list]
            precision_score_train = [metrics.precision_score(src_y_true, src_y_pred, average=ways) for ways in average_list]
            confusion_matrix_train = metrics.confusion_matrix(src_y_true, src_y_pred)
            
            metric_name = '{}_{}_metrics.txt'.format(nowepoch, mode)
            with open(dic_path + '/' + metric_name, 'w') as f:
                f.write('---------- for target blew: \n')
                # f1_score
                f.write('f1_score : \n')
                [f.write('{} : {} \n'.format(average_list[i], f1_score[i])) for i in range(len(average_list))]
                f.write('\n')
                # recall_score
                f.write('recall_score : \n')
                [f.write('{} : {} \n'.format(average_list[i], recall_score[i])) for i in range(len(average_list))]
                f.write('\n')
                # precision_score
                f.write('precision_score : \n')
                [f.write('{} : {} \n'.format(average_list[i], precision_score[i])) for i in range(len(average_list))]
                f.write('\n')
                f.write('confusion_matrix : \n')
                for item in target_names:
                    f.write('{:<10s} '.format(item))
                f.write('\n')
                for i in range(len(confusion_matrix)):
                    f.write('{:<10s} '.format(target_names[i + 1]))
                    for item in confusion_matrix[i]:
                        f.write('{:<10d} '.format(item))
                    f.write('\n')
                # for the source
                f.write('\n\n')
                f.write('---------- for source blew: \n')
                # f1_score
                f.write('f1_score : \n')
                [f.write('{} : {} \n'.format(average_list[i], f1_score_train[i])) for i in range(len(average_list))]
                f.write('\n')
                # recall_score
                f.write('recall_score : \n')
                [f.write('{} : {} \n'.format(average_list[i], recall_score_train[i])) for i in range(len(average_list))]
                f.write('\n')
                # precision_score
                f.write('precision_score : \n')
                [f.write('{} : {} \n'.format(average_list[i], precision_score_train[i])) for i in range(len(average_list))]
                f.write('\n')
                f.write('confusion_matrix : \n')
                for item in target_names:
                    f.write('{:<10s} '.format(item))
                f.write('\n')
                for i in range(len(confusion_matrix_train)):
                    f.write('{:<10s} '.format(target_names[i + 1]))
                    for item in confusion_matrix_train[i]:
                        f.write('{:<10d} '.format(item))
                    f.write('\n')

    optimizer_feature = optim.AdamW(model.feature.parameters(), lr=config['lr'])
    optimizer_classifier = optim.AdamW(model.classifier.parameters(), lr=config['lr'])
    optimizer_domain = optim.AdamW(model.domain_classifier.parameters(), lr=config['lr'])
    lambda_dctln = lambda epoch: 1 / ((1 + 10 * epoch / config['nepoch']) ** 0.75)
    scheduler_feature = optim.lr_scheduler.LambdaLR(optimizer_feature, lr_lambda=lambda_dctln)
    scheduler_classifier = optim.lr_scheduler.LambdaLR(optimizer_classifier, lr_lambda=lambda_dctln)

    # data loading
    src_train_feature = config['src_train_feature']
    src_train_label = config['src_train_label']
    tar_train_feature = config['tar_train_feature']
    tar_train_label = config['tar_train_label']
    
    train_stage1(model, optimizer_feature, optimizer_classifier, optimizer_domain, scheduler_feature, scheduler_classifier, src_train_feature, src_train_label, tar_train_feature, tar_train_label, config, DEVICE)