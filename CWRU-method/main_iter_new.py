# -*- coding: UTF-8 -*-

"""整个工程的入口函数，完成从命令行或shell脚本中读入argument参数的功能，并完成训练参数的配置，最后执行需执行训练过程的run函数。该文件必须通过直接执行生效。

示例：
python main.py --model CNN --src_motor 0 --tar_motor 1 --nepoch 1000 --batchsize 512

可进一步扩展：可以导入多个不同训练过程的run函数，使用if-else通过某种规则条件指定。
"""

import argparse
import logging
import time
import numpy as np

from myutils import mkdir
import train_iter_new
from collections import Counter

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # 对数据进行参数配置
    parser.add_argument('--src_motor', type=int, default=0)
    parser.add_argument('--tar_motor', type=int, default=3)
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--src_ratio', type=int, default=1)
    parser.add_argument('--tar_ratio', type=int, default=1)

    # 对训练过程进行参数配置
    parser.add_argument('--model', type=str, default='DANN')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--nepoch', type=int, default=200)
    parser.add_argument('--dcnepoch', type=int, default=200)
    parser.add_argument('--resultpath', type=str, default='result.csv')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--finetune', type=int, default=0)
    parser.add_argument('--finetunepath', type=str, default='')
    parser.add_argument('--earlystopping', action='store_true')
    parser.add_argument('--preDANN', type=int, default=0)

    # 对损失函数进行参数配置
    parser.add_argument('--w_gamma', type=float, default=0.5)
    parser.add_argument('--w_mmd', type=float, default=0)
    parser.add_argument('--w_cmmd', type=float, default=0.5)
    parser.add_argument('--w_ceclass', action='store_true')
    parser.add_argument('--reallabel', action='store_true')

    args = parser.parse_args()

    # 对训练数据的读入路径进行配置
    src_train_feature = "dataset/CWRUtraindata_{}HP_imba.npy".format(args.src_motor)
    src_train_label = "dataset/CWRUtrainlabel_{}HP_imba.npy".format(args.src_motor)
    tar_train_feature = "dataset/CWRUtraindata_{}HP_imba.npy".format(args.tar_motor)
    tar_train_label = "dataset/CWRUtrainlabel_{}HP_imba.npy".format(args.tar_motor)
    src_valid_feature = ""
    src_valid_label = ""
    tar_valid_feature = ""
    tar_valid_label = ""
    src_test_feature = "dataset/CWRUtestdata_{}HP_imba.npy".format(args.src_motor)
    src_test_label = "dataset/CWRUtestlabel_{}HP_imba.npy".format(args.src_motor)
    tar_test_feature = "dataset/CWRUtestdata_{}HP_imba.npy".format(args.tar_motor)
    tar_test_label = "dataset/CWRUtestlabel_{}HP_imba.npy".format(args.tar_motor)
    
    # 对保存训练结果和中间数据的文件夹路径进行配置
    dic_path = "saved_model/{}_DANN_sliding_{}_motor_train_{}_test_{}".format(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()), 1024, args.src_motor, args.tar_motor)
    mkdir(dic_path)

    # 对日志文件进行配置
    logging.basicConfig(filename= dic_path +'/runmain.log', level=logging.INFO, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    # 读取训练数据的类别数量
    srccounter = Counter(np.load(src_train_label).flatten())
    srcclassnum = len(srccounter)

    # 将上面设置好的配置参数读入字典
    config = {}

    config['src_train_feature'] = src_train_feature
    config['src_train_label'] = src_train_label
    config['tar_train_feature'] = tar_train_feature
    config['tar_train_label'] = tar_train_label
    config['src_valid_feature'] = src_valid_feature
    config['src_valid_label'] = src_valid_label
    config['tar_valid_feature'] = tar_valid_feature
    config['tar_valid_label'] = tar_valid_label
    config['src_test_feature'] = src_test_feature
    config['src_test_label'] = src_test_label
    config['tar_test_feature'] = tar_test_feature
    config['tar_test_label'] = tar_test_label

    config['dic_path'] = dic_path
    config['srcclassnum'] = srcclassnum

    config['resultpath'] = args.resultpath
    config['model'] = args.model
    config['cuda'] = args.cuda
    config['lr'] = args.lr
    config['nepoch'] = args.nepoch
    config['batchsize'] = args.batchsize
    config['patience'] = args.patience
    config['finetune'] = args.finetune
    config['finetunepath'] = args.finetunepath
    config['earlystopping'] = args.earlystopping
    config['w_gamma'] = args.w_gamma
    config['w_mmd'] = args.w_mmd
    config['w_cmmd'] = args.w_cmmd
    config['w_ceclass'] = args.w_ceclass
    config['reallabel'] = args.reallabel
    config['source'] = args.src_motor
    config['target'] = args.tar_motor
    config['preDANN'] = args.preDANN
    
    #开始执行训练过程
    train_iter_new.run(config)
