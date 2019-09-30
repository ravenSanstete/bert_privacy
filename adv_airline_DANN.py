import random
import os
import numpy as np
import torch
import torch.utils.data as data_utils
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.svm import SVC
from torch.nn import GRU, Embedding, Linear
from util import embedding
from tools import balance
from tqdm import tqdm
from DANN import DANN

ARCH = "xlm"
CLS_NUM = 10
IS_BALANCED = True
GROUND_TRUTH = False
USE_DANN = True
KEY = 'Hong Kong'

# DANN PARA
MAXITER = 4000
VERBOSE = False
BATCH_SIZE = 20
LAMDA = 0.8
HIDDEN = 50

DEVICE = torch.device('cuda:1')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# file path
PATH = "/DATACENTER/data/pxd/bert_privacy/data/skytrax-reviews-dataset"
Wiki_DS_PATH = '/DATACENTER/data/yyf/Py/bert_privacy/data/Airline/part/wiki/{}.{}'
# DS_PATH = '/DATACENTER/data/yyf/Py/bert_privacy/data/Airline/part/train.{}.{}'
DS_PATH = '/DATACENTER/data/yyf/Py/bert_privacy/data/Airline/EX_part/train.{}.{}'
DS_EMB_PATH = '/DATACENTER/data/yyf/Py/bert_privacy/data/Airline/EX_part/EMB/{}/train'.format(ARCH) + '.{}.{}'
TARGET_PATH = '/DATACENTER/data/yyf/Py/bert_privacy/data/Airline/Target/test.txt'
TARGET_EMB_PATH = '/DATACENTER/data/yyf/Py/bert_privacy/data/Airline/Target/test'
CPT_PATH = '/DATACENTER/data/yyf/Py/bert_privacy/data/Airline/EX_part/checkpoint/{}/'.format(ARCH)

EMB_DIM_TABLE = {
    "bert": 1024,
    "gpt": 768,
    "gpt2": 768,
    "xl": 1024,
    "xlnet": 1024,
    "xlm": 2048
}


cls_names = {
    'Hong Kong',
    'London',
    'Toronto',
    'Paris',
    'Rome',
    'Sydney',
    'Dubai',
    'Bangkok',
    'Singapore',
    'Frankfurt'
}


def use_DANN(key):
    # X_train, Y_train
    X_train, Y_train = [], []
    # get training dataset
    for i in [0, 1]:
        f = open(DS_PATH.format(key, i) + '.txt', 'r')
        sents = [x[:-1] for x in f if x[:-1] != '']
        embs = embedding(sents, DS_EMB_PATH.format(key, i), ARCH)
        embs = embs[np.random.choice(len(embs), 110, replace=False), :]
        X_train.append(embs)
        Y_train.extend([i] * embs.shape[0])
        f.close()
    X_train = np.concatenate(X_train, axis=0)
    Y_train = np.array(Y_train)

    # X_valid, Y_valid
    raw_valid, X_valid = list(open(TARGET_PATH, 'r')), np.load(TARGET_EMB_PATH + '.' + ARCH + '.npy')
    X_valid_b = X_valid
    if (IS_BALANCED):
        raw_valid, X_valid = balance(key, raw_valid, X_valid)
    Y_valid = np.array([(key in x) for x in raw_valid])

    clf = DANN(input_size=EMB_DIM_TABLE[ARCH], maxiter=4000, verbose=False, name=key, batch_size=20, lambda_adapt=1.0,
               hidden_layer_size=50)

    # How to chose X_adapt? X_valid(after/before balanced),
    acc = clf.fit(X_train, Y_train, X_adapt=X_valid, X_valid=X_valid, Y_valid=Y_valid)
    return acc



if __name__ == '__main__':
    # DS_prepare()
    # EX_DS_prepare()
    Source_Acc_sum = 0
    Target_Acc_sum = 0

    if USE_DANN:
        for key in cls_names:
            TA = use_DANN(key)
            Target_Acc_sum += TA

    print('Target_Acc_Top1_Average: {}'.format(Target_Acc_sum / len(cls_names)))
