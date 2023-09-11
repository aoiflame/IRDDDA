import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from bhtsne import tsne

global_classnum = 6
global_samplesize = 300

def get_random_block_from_data_and_label(data, label, size):
    sample = np.random.randint(len(data), size=size)
    return data[sample], label[sample]


def get_random_sample_from_data(data, size):
    sample = np.random.randint(len(data), size=size)
    return sample


def plot_tsne_for_different_motor_testsrctrue(x, y, ax, motor):
    ax.scatter(x[:, 0], x[:, 1], c=y, s=10)
    # ax.legend(loc="upper right")
    ax.set_title("t-sne for feature of motor {}".format(motor))

    colors = [plt.cm.jet(float(i) / global_classnum) for i in range(global_classnum)]
    for i in range(global_classnum):
        colors[i] = np.array(colors[i])
        colors[i] = np.reshape(colors[i], (1, len(colors[i])))

        ax.scatter(x[y == i][:, 0],
                   x[y == i][:, 1],
                   c=colors[i],
                   s=30,
                   label='class_' + str(i))

def plot_tsne_for_different_motor_testsrcpre(x, y, ax, motor):
    ax.scatter(x[:, 0], x[:, 1], c=y, s=10)
    # ax.legend(loc="upper right")
    ax.set_title("t-sne for feature of motor {} with predict".format(motor))

    colors = [plt.cm.jet(float(i) / global_classnum) for i in range(global_classnum)]
    for i in range(global_classnum):
        colors[i] = np.array(colors[i])
        colors[i] = np.reshape(colors[i], (1, len(colors[i])))

        ax.scatter(x[y == i][:, 0],
                   x[y == i][:, 1],
                   c=colors[i],
                   s=30,
                   label='class_' + str(i))

def plot_tsne_for_different_motor_testtartrue(x, y, ax, motor):
    ax.scatter(x[:, 0], x[:, 1], c=y, s=10)
    ax.legend(loc="upper right")
    ax.set_title("t-sne for feature of motor {}".format(motor))

    colors = [plt.cm.jet(float(i) / global_classnum) for i in range(global_classnum)]
    for i in range(global_classnum):
        colors[i] = np.array(colors[i])
        colors[i] = np.reshape(colors[i], (1, len(colors[i])))

        ax.scatter(x[y == i][:, 0],
                   x[y == i][:, 1],
                   c=colors[i],
                   s=30,
                   label='class_' + str(i))
    # ax.legend(loc='upper left', bbox_to_anchor=(1., 0.7))


# plot predict
def plot_tsne_for_different_motor_testtarpre(x, y, ax, motor):
    ax.scatter(x[:, 0], x[:, 1], c=y, s=10)
    # ax.legend(loc="upper right")
    ax.set_title("t-sne for feature of motor {} with predict".format(motor))

    colors = [plt.cm.jet(float(i) / global_classnum) for i in range(global_classnum)]
    for i in range(global_classnum):
        colors[i] = np.array(colors[i])
        colors[i] = np.reshape(colors[i], (1, len(colors[i])))

        ax.scatter(x[y == i][:, 0],
                   x[y == i][:, 1],
                   c=colors[i],
                   s=30,
                   label='class_' + str(i))
    ax.legend(loc='upper left', bbox_to_anchor=(1., 0.7))


def plot(path, train_motor, test_motor, ifPlot=False, path_test='', path_pic=''):
    path_train = path + '/train_step1_period'
    if (path_test == ''):
        path_test = path + '/test_step1_period'
    else:
        path_test = path + path_test
    if (path_pic == ''):
        path_pic = path
    else:
        path_pic = path + path_pic
    sample_size = global_samplesize
    # train
    train_src_feature = np.load('{}/src_feature.npy'.format(path_train))
    train_src_feature = np.reshape(train_src_feature, (train_src_feature.shape[0], -1))
    train_tar_feature = np.load('{}/tar_feature.npy'.format(path_train))
    train_tar_feature = np.reshape(train_tar_feature, (train_tar_feature.shape[0], -1))
    train_src_labels = np.load('{}/src_label.npy'.format(path_train))
    train_tar_labels = np.load('{}/tar_label.npy'.format(path_train))
    train_srcpredict_labels = np.load('{}/src_predict_result_inverse.npy'.format(path_train))
    train_tarpredict_labels = np.load('{}/tar_predict_result_inverse.npy'.format(path_train))

    train_src_feature = pd.DataFrame(train_src_feature)
    train_src_feature = train_src_feature.fillna(0)
    train_src_feature = train_src_feature.astype('float64')

    train_tar_feature = pd.DataFrame(train_tar_feature)
    train_tar_feature = train_tar_feature.fillna(0)
    train_tar_feature = train_tar_feature.astype('float64')

    train_src_sample = random.sample(range(0, train_src_feature.shape[0]), sample_size)
    train_tar_sample = random.sample(range(0, train_tar_feature.shape[0]), sample_size)
    # test
    test_src_feature = np.load('{}/src_feature.npy'.format(path_test))
    test_src_feature = np.reshape(test_src_feature, (test_src_feature.shape[0], -1))
    test_tar_feature = np.load('{}/tar_feature.npy'.format(path_test))
    test_tar_feature = np.reshape(test_tar_feature, (test_tar_feature.shape[0], -1))
    test_src_labels = np.load('{}/src_label.npy'.format(path_test))
    test_tar_labels = np.load('{}/tar_label.npy'.format(path_test))
    test_srcpredict_labels = np.load('{}/src_predict_result_inverse.npy'.format(path_test))
    test_tarpredict_labels = np.load('{}/tar_predict_result_inverse.npy'.format(path_test))

    test_src_feature = pd.DataFrame(test_src_feature)
    test_src_feature = test_src_feature.fillna(0)
    test_src_feature = test_src_feature.astype('float64')

    test_tar_feature = pd.DataFrame(test_tar_feature)
    test_tar_feature = test_tar_feature.fillna(0)
    test_tar_feature = test_tar_feature.astype('float64')

    test_src_sample = random.sample(range(0, test_src_feature.shape[0]), sample_size)
    test_tar_sample = random.sample(range(0, test_tar_feature.shape[0]), sample_size)

    fit_feature = pd.concat((train_src_feature.iloc[train_src_sample],
                            train_tar_feature.iloc[train_tar_sample],
                            test_src_feature.iloc[test_src_sample],
                            test_tar_feature.iloc[test_tar_sample]),
                            axis=0)
    print(fit_feature.shape)
    tsne_result = tsne(fit_feature)
    
    # train plot
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    plot_tsne_for_different_motor_testsrctrue(
        tsne_result[:sample_size],
        np.reshape(train_src_labels[train_src_sample], (sample_size)), axes[0],
        train_motor)
    plot_tsne_for_different_motor_testsrcpre(
        tsne_result[:sample_size],
        np.reshape(train_srcpredict_labels[train_src_sample], (sample_size)), axes[1],
        train_motor)
    plot_tsne_for_different_motor_testtartrue(
        tsne_result[sample_size:sample_size*2],
        np.reshape(train_tar_labels[train_tar_sample], (sample_size)), axes[2],
        test_motor)
    plot_tsne_for_different_motor_testtarpre(
        tsne_result[sample_size:sample_size*2],
        np.reshape(train_tarpredict_labels[train_tar_sample], (sample_size)), axes[3],
        test_motor)
    plt.savefig('{}/train_feature_tsne.jpg'.format(path_pic))
    if ifPlot:
        plt.show()
    # test plot
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    plot_tsne_for_different_motor_testsrctrue(
        tsne_result[sample_size*2:sample_size*3],
        np.reshape(test_src_labels[test_src_sample], (sample_size)), axes[0],
        train_motor)
    plot_tsne_for_different_motor_testsrcpre(
        tsne_result[sample_size*2:sample_size*3],
        np.reshape(test_srcpredict_labels[test_src_sample], (sample_size)), axes[1],
        train_motor)
    plot_tsne_for_different_motor_testtartrue(
        tsne_result[sample_size*3:],
        np.reshape(test_tar_labels[test_tar_sample], (sample_size)), axes[2],
        test_motor)
    plot_tsne_for_different_motor_testtarpre(
        tsne_result[sample_size*3:],
        np.reshape(test_tarpredict_labels[test_tar_sample], (sample_size)), axes[3],
        test_motor)
    plt.savefig('{}/test_feature_tsne.jpg'.format(path_pic))
    if ifPlot:
        plt.show()
    

def plot_for_test(path, train_motor, test_motor, ifPlot=False):

    def mkdir(path):
        folder = os.path.exists(path)

        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径

    path_train = path + '/besttesttar_train_period'
    path_test1 = path + '/besttesttar_test_period'
    path_test2 = path + '/besttrainsrc_test_period'
    path_test3 = path + '/besttrainsrc_train_period'
    path_pic = path + '/picstore'
    mkdir(path_pic)
    sample_size = 1000

    def get_sample(path_train):
        # train
        train_src_feature = np.load('{}/src_feature.npy'.format(path_train))
        train_src_feature = np.reshape(train_src_feature, (train_src_feature.shape[0], -1))
        train_tar_feature = np.load('{}/tar_feature.npy'.format(path_train))
        train_tar_feature = np.reshape(train_tar_feature, (train_tar_feature.shape[0], -1))
        train_src_labels = np.load('{}/src_label.npy'.format(path_train))
        train_tar_labels = np.load('{}/tar_label.npy'.format(path_train))
        train_srcpredict_labels = np.load('{}/src_predict_result_inverse.npy'.format(path_train))
        train_tarpredict_labels = np.load('{}/tar_predict_result_inverse.npy'.format(path_train))

        train_src_feature = pd.DataFrame(train_src_feature)
        train_src_feature = train_src_feature.fillna(0)
        train_src_feature = train_src_feature.astype('float64')

        train_tar_feature = pd.DataFrame(train_tar_feature)
        train_tar_feature = train_tar_feature.fillna(0)
        train_tar_feature = train_tar_feature.astype('float64')

        train_src_sample = random.sample(range(0, train_src_feature.shape[0]), sample_size)
        train_tar_sample = random.sample(range(0, train_tar_feature.shape[0]), sample_size)
        
        return train_src_feature, train_tar_feature, train_src_labels, train_tar_labels, train_srcpredict_labels, train_tarpredict_labels, train_src_sample, train_tar_sample
    
    train_src_feature, train_tar_feature, train_src_labels, train_tar_labels, train_srcpredict_labels, train_tarpredict_labels, train_src_sample, train_tar_sample = get_sample(path_train)
    test1_src_feature, test1_tar_feature, test1_src_labels, test1_tar_labels, test1_srcpredict_labels, test1_tarpredict_labels, test1_src_sample, test1_tar_sample = get_sample(path_test1)
    test2_src_feature, test2_tar_feature, test2_src_labels, test2_tar_labels, test2_srcpredict_labels, test2_tarpredict_labels, test2_src_sample, test2_tar_sample = get_sample(path_test2)
    test3_src_feature, test3_tar_feature, test3_src_labels, test3_tar_labels, test3_srcpredict_labels, test3_tarpredict_labels, test3_src_sample, test3_tar_sample = get_sample(path_test3)

    fit_feature = pd.concat((train_src_feature.iloc[train_src_sample],
                            train_tar_feature.iloc[train_tar_sample],
                            test1_src_feature.iloc[test1_src_sample],
                            test1_tar_feature.iloc[test1_tar_sample],
                            test2_src_feature.iloc[test2_src_sample],
                            test2_tar_feature.iloc[test2_tar_sample],
                            test3_src_feature.iloc[test3_src_sample],
                            test3_tar_feature.iloc[test3_tar_sample]),
                            axis=0)
    tsne_result = tsne(fit_feature)
    
    # train plot
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    plot_tsne_for_different_motor_testsrctrue(
        tsne_result[:sample_size],
        np.reshape(train_src_labels[train_src_sample], (sample_size)), axes[0],
        train_motor)
    plot_tsne_for_different_motor_testsrcpre(
        tsne_result[:sample_size],
        np.reshape(train_srcpredict_labels[train_src_sample], (sample_size)), axes[1],
        train_motor)
    plot_tsne_for_different_motor_testtartrue(
        tsne_result[sample_size:sample_size*2],
        np.reshape(train_tar_labels[train_tar_sample], (sample_size)), axes[2],
        test_motor)
    plot_tsne_for_different_motor_testtarpre(
        tsne_result[sample_size:sample_size*2],
        np.reshape(train_tarpredict_labels[train_tar_sample], (sample_size)), axes[3],
        test_motor)
    plt.savefig('{}/train_feature_tsne.jpg'.format(path_pic))
    plt.close()
    # test1 plot
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    plot_tsne_for_different_motor_testsrctrue(
        tsne_result[sample_size*2:sample_size*3],
        np.reshape(test1_src_labels[test1_src_sample], (sample_size)), axes[0],
        train_motor)
    plot_tsne_for_different_motor_testsrcpre(
        tsne_result[sample_size*2:sample_size*3],
        np.reshape(test1_srcpredict_labels[test1_src_sample], (sample_size)), axes[1],
        train_motor)
    plot_tsne_for_different_motor_testtartrue(
        tsne_result[sample_size*3:sample_size*4],
        np.reshape(test1_tar_labels[test1_tar_sample], (sample_size)), axes[2],
        test_motor)
    plot_tsne_for_different_motor_testtarpre(
        tsne_result[sample_size*3:sample_size*4],
        np.reshape(test1_tarpredict_labels[test1_tar_sample], (sample_size)), axes[3],
        test_motor)
    plt.savefig('{}/test1_feature_tsne.jpg'.format(path_pic))
    plt.close()
    #test2 plot
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    plot_tsne_for_different_motor_testsrctrue(
        tsne_result[sample_size*4:sample_size*5],
        np.reshape(test2_src_labels[test2_src_sample], (sample_size)), axes[0],
        train_motor)
    plot_tsne_for_different_motor_testsrcpre(
        tsne_result[sample_size*4:sample_size*5],
        np.reshape(test2_srcpredict_labels[test2_src_sample], (sample_size)), axes[1],
        train_motor)
    plot_tsne_for_different_motor_testtartrue(
        tsne_result[sample_size*5:sample_size*6],
        np.reshape(test2_tar_labels[test2_tar_sample], (sample_size)), axes[2],
        test_motor)
    plot_tsne_for_different_motor_testtarpre(
        tsne_result[sample_size*5:sample_size*6],
        np.reshape(test2_tarpredict_labels[test2_tar_sample], (sample_size)), axes[3],
        test_motor)
    plt.savefig('{}/test2_feature_tsne.jpg'.format(path_pic))
    plt.close()
    #test3 plot
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    plot_tsne_for_different_motor_testsrctrue(
        tsne_result[sample_size*6:sample_size*7],
        np.reshape(test3_src_labels[test3_src_sample], (sample_size)), axes[0],
        train_motor)
    plot_tsne_for_different_motor_testsrcpre(
        tsne_result[sample_size*6:sample_size*7],
        np.reshape(test3_srcpredict_labels[test3_src_sample], (sample_size)), axes[1],
        train_motor)
    plot_tsne_for_different_motor_testtartrue(
        tsne_result[sample_size*7:],
        np.reshape(test3_tar_labels[test3_tar_sample], (sample_size)), axes[2],
        test_motor)
    plot_tsne_for_different_motor_testtarpre(
        tsne_result[sample_size*7:],
        np.reshape(test3_tarpredict_labels[test3_tar_sample], (sample_size)), axes[3],
        test_motor)
    plt.savefig('{}/test3_feature_tsne.jpg'.format(path_pic))
    plt.close()

def plot_tensamples(path, train_motor, test_motor, ratio, ifPlot=False):

    def mkdir(path):
        folder = os.path.exists(path)

        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径

    #定义需要读取的1个训练集和10个测试集的路径，并设置画图文件夹的路径。    
    path_train = path + '/train_period'
    path_test0 = path + '/testperiod_{}_{}_0'.format(ratio, ratio)
    path_test1 = path + '/testperiod_{}_{}_1'.format(ratio, ratio)
    path_test2 = path + '/testperiod_{}_{}_2'.format(ratio, ratio)
    path_test3 = path + '/testperiod_{}_{}_3'.format(ratio, ratio)
    path_test4 = path + '/testperiod_{}_{}_4'.format(ratio, ratio)
    path_test5 = path + '/testperiod_{}_{}_5'.format(ratio, ratio)
    path_test6 = path + '/testperiod_{}_{}_6'.format(ratio, ratio)
    path_test7 = path + '/testperiod_{}_{}_7'.format(ratio, ratio)
    path_test8 = path + '/testperiod_{}_{}_8'.format(ratio, ratio)
    path_test9 = path + '/testperiod_{}_{}_9'.format(ratio, ratio)
    path_pic = path + '/picstore'
    mkdir(path_pic)
    sample_size = 2000
    
    #该函数从定义好的路径中取出中间文件，进行采样，做降维和画图准备
    def get_sample(path_train):
        # train
        train_src_feature = np.load('{}/src_feature.npy'.format(path_train))
        train_src_feature = np.reshape(train_src_feature, (train_src_feature.shape[0], -1))
        train_tar_feature = np.load('{}/tar_feature.npy'.format(path_train))
        train_tar_feature = np.reshape(train_tar_feature, (train_tar_feature.shape[0], -1))
        train_src_labels = np.load('{}/src_label.npy'.format(path_train))
        train_tar_labels = np.load('{}/tar_label.npy'.format(path_train))
        train_srcpredict_labels = np.load('{}/src_predict_result_inverse.npy'.format(path_train))
        train_tarpredict_labels = np.load('{}/tar_predict_result_inverse.npy'.format(path_train))

        train_src_feature = pd.DataFrame(train_src_feature)
        train_src_feature = train_src_feature.fillna(0)
        train_src_feature = train_src_feature.astype('float64')

        train_tar_feature = pd.DataFrame(train_tar_feature)
        train_tar_feature = train_tar_feature.fillna(0)
        train_tar_feature = train_tar_feature.astype('float64')

        train_src_sample = random.sample(range(0, train_src_feature.shape[0]), sample_size)
        train_tar_sample = random.sample(range(0, train_tar_feature.shape[0]), sample_size)
        
        return train_src_feature, train_tar_feature, train_src_labels, train_tar_labels, train_srcpredict_labels, train_tarpredict_labels, train_src_sample, train_tar_sample
    
    #getsample函数执行
    train_src_feature, train_tar_feature, train_src_labels, train_tar_labels, train_srcpredict_labels, train_tarpredict_labels, train_src_sample, train_tar_sample = get_sample(path_train)
    test0_src_feature, test0_tar_feature, test0_src_labels, test0_tar_labels, test0_srcpredict_labels, test0_tarpredict_labels, test0_src_sample, test0_tar_sample = get_sample(path_test0)
    test1_src_feature, test1_tar_feature, test1_src_labels, test1_tar_labels, test1_srcpredict_labels, test1_tarpredict_labels, test1_src_sample, test1_tar_sample = get_sample(path_test1)
    test2_src_feature, test2_tar_feature, test2_src_labels, test2_tar_labels, test2_srcpredict_labels, test2_tarpredict_labels, test2_src_sample, test2_tar_sample = get_sample(path_test2)
    test3_src_feature, test3_tar_feature, test3_src_labels, test3_tar_labels, test3_srcpredict_labels, test3_tarpredict_labels, test3_src_sample, test3_tar_sample = get_sample(path_test3)
    test4_src_feature, test4_tar_feature, test4_src_labels, test4_tar_labels, test4_srcpredict_labels, test4_tarpredict_labels, test4_src_sample, test4_tar_sample = get_sample(path_test4)
    test5_src_feature, test5_tar_feature, test5_src_labels, test5_tar_labels, test5_srcpredict_labels, test5_tarpredict_labels, test5_src_sample, test5_tar_sample = get_sample(path_test5)
    test6_src_feature, test6_tar_feature, test6_src_labels, test6_tar_labels, test6_srcpredict_labels, test6_tarpredict_labels, test6_src_sample, test6_tar_sample = get_sample(path_test6)
    test7_src_feature, test7_tar_feature, test7_src_labels, test7_tar_labels, test7_srcpredict_labels, test7_tarpredict_labels, test7_src_sample, test7_tar_sample = get_sample(path_test7)
    test8_src_feature, test8_tar_feature, test8_src_labels, test8_tar_labels, test8_srcpredict_labels, test8_tarpredict_labels, test8_src_sample, test8_tar_sample = get_sample(path_test8)
    test9_src_feature, test9_tar_feature, test9_src_labels, test9_tar_labels, test9_srcpredict_labels, test9_tarpredict_labels, test9_src_sample, test9_tar_sample = get_sample(path_test9)

    #t-sne 降维
    fit_feature = pd.concat((train_src_feature.iloc[train_src_sample],
                            train_tar_feature.iloc[train_tar_sample],
                            test0_src_feature.iloc[test0_src_sample],
                            test0_tar_feature.iloc[test0_tar_sample],
                            test1_src_feature.iloc[test1_src_sample],
                            test1_tar_feature.iloc[test1_tar_sample],
                            test2_src_feature.iloc[test2_src_sample],
                            test2_tar_feature.iloc[test2_tar_sample],
                            test3_src_feature.iloc[test3_src_sample],
                            test3_tar_feature.iloc[test3_tar_sample],
                            test4_src_feature.iloc[test4_src_sample],
                            test4_tar_feature.iloc[test4_tar_sample],
                            test5_src_feature.iloc[test5_src_sample],
                            test5_tar_feature.iloc[test5_tar_sample],
                            test6_src_feature.iloc[test6_src_sample],
                            test6_tar_feature.iloc[test6_tar_sample],
                            test7_src_feature.iloc[test7_src_sample],
                            test7_tar_feature.iloc[test7_tar_sample],
                            test8_src_feature.iloc[test8_src_sample],
                            test8_tar_feature.iloc[test8_tar_sample],
                            test9_src_feature.iloc[test9_src_sample],
                            test9_tar_feature.iloc[test9_tar_sample],),
                            axis=0)
    tsne_result = tsne(fit_feature)

    #统一作图
    # train plot
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    plot_tsne_for_different_motor_testsrctrue(
        tsne_result[:sample_size],
        np.reshape(train_src_labels[train_src_sample], (sample_size)), axes[0],
        train_motor)
    plot_tsne_for_different_motor_testsrcpre(
        tsne_result[:sample_size],
        np.reshape(train_srcpredict_labels[train_src_sample], (sample_size)), axes[1],
        train_motor)
    plot_tsne_for_different_motor_testtartrue(
        tsne_result[sample_size:sample_size*2],
        np.reshape(train_tar_labels[train_tar_sample], (sample_size)), axes[2],
        test_motor)
    plot_tsne_for_different_motor_testtarpre(
        tsne_result[sample_size:sample_size*2],
        np.reshape(train_tarpredict_labels[train_tar_sample], (sample_size)), axes[3],
        test_motor)
    plt.savefig('{}/train_feature_tsne.jpg'.format(path_pic))
    plt.close()
    # test0 plot
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    plot_tsne_for_different_motor_testsrctrue(
        tsne_result[sample_size*2:sample_size*3],
        np.reshape(test0_src_labels[test0_src_sample], (sample_size)), axes[0],
        train_motor)
    plot_tsne_for_different_motor_testsrcpre(
        tsne_result[sample_size*2:sample_size*3],
        np.reshape(test0_srcpredict_labels[test0_src_sample], (sample_size)), axes[1],
        train_motor)
    plot_tsne_for_different_motor_testtartrue(
        tsne_result[sample_size*3:sample_size*4],
        np.reshape(test0_tar_labels[test0_tar_sample], (sample_size)), axes[2],
        test_motor)
    plot_tsne_for_different_motor_testtarpre(
        tsne_result[sample_size*3:sample_size*4],
        np.reshape(test0_tarpredict_labels[test0_tar_sample], (sample_size)), axes[3],
        test_motor)
    plt.savefig('{}/test0_feature_tsne.jpg'.format(path_pic))
    plt.close()
    # test1 plot
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    plot_tsne_for_different_motor_testsrctrue(
        tsne_result[sample_size*4:sample_size*5],
        np.reshape(test1_src_labels[test1_src_sample], (sample_size)), axes[0],
        train_motor)
    plot_tsne_for_different_motor_testsrcpre(
        tsne_result[sample_size*4:sample_size*5],
        np.reshape(test1_srcpredict_labels[test1_src_sample], (sample_size)), axes[1],
        train_motor)
    plot_tsne_for_different_motor_testtartrue(
        tsne_result[sample_size*5:sample_size*6],
        np.reshape(test1_tar_labels[test1_tar_sample], (sample_size)), axes[2],
        test_motor)
    plot_tsne_for_different_motor_testtarpre(
        tsne_result[sample_size*5:sample_size*6],
        np.reshape(test1_tarpredict_labels[test1_tar_sample], (sample_size)), axes[3],
        test_motor)
    plt.savefig('{}/test1_feature_tsne.jpg'.format(path_pic))
    plt.close()
    # test2 plot
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    plot_tsne_for_different_motor_testsrctrue(
        tsne_result[sample_size*6:sample_size*7],
        np.reshape(test2_src_labels[test2_src_sample], (sample_size)), axes[0],
        train_motor)
    plot_tsne_for_different_motor_testsrcpre(
        tsne_result[sample_size*6:sample_size*7],
        np.reshape(test2_srcpredict_labels[test2_src_sample], (sample_size)), axes[1],
        train_motor)
    plot_tsne_for_different_motor_testtartrue(
        tsne_result[sample_size*7:sample_size*8],
        np.reshape(test2_tar_labels[test2_tar_sample], (sample_size)), axes[2],
        test_motor)
    plot_tsne_for_different_motor_testtarpre(
        tsne_result[sample_size*7:sample_size*8],
        np.reshape(test2_tarpredict_labels[test2_tar_sample], (sample_size)), axes[3],
        test_motor)
    plt.savefig('{}/test2_feature_tsne.jpg'.format(path_pic))
    plt.close()
    # test3 plot
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    plot_tsne_for_different_motor_testsrctrue(
        tsne_result[sample_size*8:sample_size*9],
        np.reshape(test3_src_labels[test3_src_sample], (sample_size)), axes[0],
        train_motor)
    plot_tsne_for_different_motor_testsrcpre(
        tsne_result[sample_size*8:sample_size*9],
        np.reshape(test3_srcpredict_labels[test3_src_sample], (sample_size)), axes[1],
        train_motor)
    plot_tsne_for_different_motor_testtartrue(
        tsne_result[sample_size*9:sample_size*10],
        np.reshape(test3_tar_labels[test3_tar_sample], (sample_size)), axes[2],
        test_motor)
    plot_tsne_for_different_motor_testtarpre(
        tsne_result[sample_size*9:sample_size*10],
        np.reshape(test3_tarpredict_labels[test3_tar_sample], (sample_size)), axes[3],
        test_motor)
    plt.savefig('{}/test3_feature_tsne.jpg'.format(path_pic))
    plt.close()
    # test4 plot
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    plot_tsne_for_different_motor_testsrctrue(
        tsne_result[sample_size*10:sample_size*11],
        np.reshape(test4_src_labels[test4_src_sample], (sample_size)), axes[0],
        train_motor)
    plot_tsne_for_different_motor_testsrcpre(
        tsne_result[sample_size*10:sample_size*11],
        np.reshape(test4_srcpredict_labels[test4_src_sample], (sample_size)), axes[1],
        train_motor)
    plot_tsne_for_different_motor_testtartrue(
        tsne_result[sample_size*11:sample_size*12],
        np.reshape(test4_tar_labels[test4_tar_sample], (sample_size)), axes[2],
        test_motor)
    plot_tsne_for_different_motor_testtarpre(
        tsne_result[sample_size*11:sample_size*12],
        np.reshape(test4_tarpredict_labels[test4_tar_sample], (sample_size)), axes[3],
        test_motor)
    plt.savefig('{}/test4_feature_tsne.jpg'.format(path_pic))
    plt.close()
    # test5 plot
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    plot_tsne_for_different_motor_testsrctrue(
        tsne_result[sample_size*12:sample_size*13],
        np.reshape(test5_src_labels[test5_src_sample], (sample_size)), axes[0],
        train_motor)
    plot_tsne_for_different_motor_testsrcpre(
        tsne_result[sample_size*12:sample_size*13],
        np.reshape(test5_srcpredict_labels[test5_src_sample], (sample_size)), axes[1],
        train_motor)
    plot_tsne_for_different_motor_testtartrue(
        tsne_result[sample_size*13:sample_size*14],
        np.reshape(test5_tar_labels[test5_tar_sample], (sample_size)), axes[2],
        test_motor)
    plot_tsne_for_different_motor_testtarpre(
        tsne_result[sample_size*13:sample_size*14],
        np.reshape(test5_tarpredict_labels[test5_tar_sample], (sample_size)), axes[3],
        test_motor)
    plt.savefig('{}/test5_feature_tsne.jpg'.format(path_pic))
    plt.close()
    # test6 plot
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    plot_tsne_for_different_motor_testsrctrue(
        tsne_result[sample_size*14:sample_size*15],
        np.reshape(test6_src_labels[test6_src_sample], (sample_size)), axes[0],
        train_motor)
    plot_tsne_for_different_motor_testsrcpre(
        tsne_result[sample_size*14:sample_size*15],
        np.reshape(test6_srcpredict_labels[test6_src_sample], (sample_size)), axes[1],
        train_motor)
    plot_tsne_for_different_motor_testtartrue(
        tsne_result[sample_size*15:sample_size*16],
        np.reshape(test6_tar_labels[test6_tar_sample], (sample_size)), axes[2],
        test_motor)
    plot_tsne_for_different_motor_testtarpre(
        tsne_result[sample_size*15:sample_size*16],
        np.reshape(test6_tarpredict_labels[test6_tar_sample], (sample_size)), axes[3],
        test_motor)
    plt.savefig('{}/test6_feature_tsne.jpg'.format(path_pic))
    plt.close()
    # test7 plot
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    plot_tsne_for_different_motor_testsrctrue(
        tsne_result[sample_size*16:sample_size*17],
        np.reshape(test7_src_labels[test7_src_sample], (sample_size)), axes[0],
        train_motor)
    plot_tsne_for_different_motor_testsrcpre(
        tsne_result[sample_size*16:sample_size*17],
        np.reshape(test7_srcpredict_labels[test7_src_sample], (sample_size)), axes[1],
        train_motor)
    plot_tsne_for_different_motor_testtartrue(
        tsne_result[sample_size*17:sample_size*18],
        np.reshape(test7_tar_labels[test7_tar_sample], (sample_size)), axes[2],
        test_motor)
    plot_tsne_for_different_motor_testtarpre(
        tsne_result[sample_size*17:sample_size*18],
        np.reshape(test7_tarpredict_labels[test7_tar_sample], (sample_size)), axes[3],
        test_motor)
    plt.savefig('{}/test7_feature_tsne.jpg'.format(path_pic))
    plt.close()
    # test8 plot
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    plot_tsne_for_different_motor_testsrctrue(
        tsne_result[sample_size*18:sample_size*19],
        np.reshape(test8_src_labels[test8_src_sample], (sample_size)), axes[0],
        train_motor)
    plot_tsne_for_different_motor_testsrcpre(
        tsne_result[sample_size*18:sample_size*19],
        np.reshape(test8_srcpredict_labels[test8_src_sample], (sample_size)), axes[1],
        train_motor)
    plot_tsne_for_different_motor_testtartrue(
        tsne_result[sample_size*19:sample_size*20],
        np.reshape(test8_tar_labels[test8_tar_sample], (sample_size)), axes[2],
        test_motor)
    plot_tsne_for_different_motor_testtarpre(
        tsne_result[sample_size*19:sample_size*20],
        np.reshape(test8_tarpredict_labels[test8_tar_sample], (sample_size)), axes[3],
        test_motor)
    plt.savefig('{}/test8_feature_tsne.jpg'.format(path_pic))
    plt.close()
    # test9 plot
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    plot_tsne_for_different_motor_testsrctrue(
        tsne_result[sample_size*20:sample_size*21],
        np.reshape(test9_src_labels[test9_src_sample], (sample_size)), axes[0],
        train_motor)
    plot_tsne_for_different_motor_testsrcpre(
        tsne_result[sample_size*20:sample_size*21],
        np.reshape(test9_srcpredict_labels[test9_src_sample], (sample_size)), axes[1],
        train_motor)
    plot_tsne_for_different_motor_testtartrue(
        tsne_result[sample_size*21:sample_size*22],
        np.reshape(test9_tar_labels[test9_tar_sample], (sample_size)), axes[2],
        test_motor)
    plot_tsne_for_different_motor_testtarpre(
        tsne_result[sample_size*21:sample_size*22],
        np.reshape(test9_tarpredict_labels[test9_tar_sample], (sample_size)), axes[3],
        test_motor)
    plt.savefig('{}/test9_feature_tsne.jpg'.format(path_pic))
    plt.close()

def plot_oversampling(path, train_motor, test_motor, datamode, ifPlot=False, path_test='', path_pic=''):
    datamode_temp = datamode.split("_")[0]
    path_train = path + '/{}_train_period'.format(datamode_temp)
    if (path_test == ''):
        path_test = path + '/{}_test_period'.format(datamode_temp)
    else:
        path_test = path + path_test
    if (path_pic == ''):
        path_pic = path
    else:
        path_pic = path + path_pic
    sample_size = global_samplesize
    print(path_train, path_test)
    # train
    train_src_feature = np.load('{}/src_feature.npy'.format(path_train))
    train_src_feature = np.reshape(train_src_feature, (train_src_feature.shape[0], -1))
    train_tar_feature = np.load('{}/tar_feature.npy'.format(path_train))
    train_tar_feature = np.reshape(train_tar_feature, (train_tar_feature.shape[0], -1))
    train_src_labels = np.load('{}/src_label.npy'.format(path_train))
    train_tar_labels = np.load('{}/tar_label.npy'.format(path_train))
    train_srcpredict_labels = np.load('{}/src_predict_result_inverse.npy'.format(path_train))
    train_tarpredict_labels = np.load('{}/tar_predict_result_inverse.npy'.format(path_train))

    train_src_feature = pd.DataFrame(train_src_feature)
    train_src_feature = train_src_feature.fillna(0)
    train_src_feature = train_src_feature.astype('float64')

    train_tar_feature = pd.DataFrame(train_tar_feature)
    train_tar_feature = train_tar_feature.fillna(0)
    train_tar_feature = train_tar_feature.astype('float64')

    train_src_sample = random.sample(range(0, train_src_feature.shape[0]), sample_size)
    train_tar_sample = random.sample(range(0, train_tar_feature.shape[0]), sample_size)
    # test
    test_src_feature = np.load('{}/src_feature.npy'.format(path_test))
    test_src_feature = np.reshape(test_src_feature, (test_src_feature.shape[0], -1))
    test_tar_feature = np.load('{}/tar_feature.npy'.format(path_test))
    test_tar_feature = np.reshape(test_tar_feature, (test_tar_feature.shape[0], -1))
    test_src_labels = np.load('{}/src_label.npy'.format(path_test))
    test_tar_labels = np.load('{}/tar_label.npy'.format(path_test))
    test_srcpredict_labels = np.load('{}/src_predict_result_inverse.npy'.format(path_test))
    test_tarpredict_labels = np.load('{}/tar_predict_result_inverse.npy'.format(path_test))

    test_src_feature = pd.DataFrame(test_src_feature)
    test_src_feature = test_src_feature.fillna(0)
    test_src_feature = test_src_feature.astype('float64')

    test_tar_feature = pd.DataFrame(test_tar_feature)
    test_tar_feature = test_tar_feature.fillna(0)
    test_tar_feature = test_tar_feature.astype('float64')

    test_src_sample = random.sample(range(0, test_src_feature.shape[0]), sample_size)
    test_tar_sample = random.sample(range(0, test_tar_feature.shape[0]), sample_size)

    # oversample
    x_oversample_besttesttar = np.load('{}/oversample/x_oversample_{}.npy'.format(path, datamode_temp))
    y_oversample_besttesttar = np.load('{}/oversample/y_oversample_{}.npy'.format(path, datamode_temp))
    y_oversample_predict_besttesttar = np.load('{}/oversample/y_oversample_predict_{}.npy'.format(path, datamode_temp))
    oversample_feature = np.reshape(x_oversample_besttesttar, (x_oversample_besttesttar.shape[0], -1))

    oversample_feature = pd.DataFrame(oversample_feature)
    oversample_feature = oversample_feature.fillna(0)
    oversample_feature = oversample_feature.astype('float64')
    oversample_sample = random.sample(range(0, oversample_feature.shape[0]), sample_size)

    fit_feature = pd.concat((train_src_feature.iloc[train_src_sample],
                            train_tar_feature.iloc[train_tar_sample],
                            test_src_feature.iloc[test_src_sample],
                            test_tar_feature.iloc[test_tar_sample],
                            oversample_feature.iloc[oversample_sample]),
                            axis=0)
    print(fit_feature.shape)
    tsne_result = tsne(fit_feature)
    
    # train plot
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    plot_tsne_for_different_motor_testsrctrue(
        tsne_result[:sample_size],
        np.reshape(train_src_labels[train_src_sample], (sample_size)), axes[0],
        train_motor)
    plot_tsne_for_different_motor_testsrcpre(
        tsne_result[:sample_size],
        np.reshape(train_srcpredict_labels[train_src_sample], (sample_size)), axes[1],
        train_motor)
    plot_tsne_for_different_motor_testtartrue(
        tsne_result[sample_size:sample_size*2],
        np.reshape(train_tar_labels[train_tar_sample], (sample_size)), axes[2],
        test_motor)
    plot_tsne_for_different_motor_testtarpre(
        tsne_result[sample_size:sample_size*2],
        np.reshape(train_tarpredict_labels[train_tar_sample], (sample_size)), axes[3],
        test_motor)
    plt.savefig('{}/train_feature_tsne_{}.jpg'.format(path_pic, datamode))
    if ifPlot:
        plt.show()
    # test plot
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    plot_tsne_for_different_motor_testsrctrue(
        tsne_result[sample_size*2:sample_size*3],
        np.reshape(test_src_labels[test_src_sample], (sample_size)), axes[0],
        train_motor)
    plot_tsne_for_different_motor_testsrcpre(
        tsne_result[sample_size*2:sample_size*3],
        np.reshape(test_srcpredict_labels[test_src_sample], (sample_size)), axes[1],
        train_motor)
    plot_tsne_for_different_motor_testtartrue(
        tsne_result[sample_size*3:sample_size*4],
        np.reshape(test_tar_labels[test_tar_sample], (sample_size)), axes[2],
        test_motor)
    plot_tsne_for_different_motor_testtarpre(
        tsne_result[sample_size*3:sample_size*4],
        np.reshape(test_tarpredict_labels[test_tar_sample], (sample_size)), axes[3],
        test_motor)
    plt.savefig('{}/test_feature_tsne_{}.jpg'.format(path_pic, datamode))
    # oversample plot
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    plot_tsne_for_different_motor_testsrctrue(
        tsne_result[sample_size*4:sample_size*5],
        np.reshape(y_oversample_besttesttar[oversample_sample], (sample_size)), axes[0],
        train_motor)
    plot_tsne_for_different_motor_testsrcpre(
        tsne_result[sample_size*4:sample_size*5],
        np.reshape(y_oversample_predict_besttesttar[oversample_sample], (sample_size)), axes[1],
        train_motor)
    plt.savefig('{}/oversample_feature_tsne_{}.jpg'.format(path_pic, datamode))
    if ifPlot:
        plt.show()

