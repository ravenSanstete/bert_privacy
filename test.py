import torch
import os
import random
import pytorch_transformers
from bert_serving.client import BertClient
from tqdm import tqdm
import numpy as np
from util import embedding
from tools import balance
from mpl_toolkits.mplot3d import Axes3D
from DANN import DANN
import paddlehub

import pandas
import matplotlib
import csv
import string
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
# file path
PATH = "/DATACENTER/data/pxd/bert_privacy/data/skytrax-reviews-dataset"
Wiki_DS_PATH= '/DATACENTER/data/yyf/Py/bert_privacy/data/Airline/part/wiki/{}.{}'
DS_PATH = '/DATACENTER/data/yyf/Py/bert_privacy/data/Airline/part/train.{}.{}'
TARGET_PATH = '/DATACENTER/data/yyf/Py/bert_privacy/data/Airline/Target/test.txt'
# TARGET_EMB_PATH = '/DATACENTER/data/yyf/Py/bert_privacy/data/Airline/Target/test'
ARCH = 'bert'
TARGET_EMB_PATH = DS_PATH + '.' + ARCH + '.npy'

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"



def main():
    # embs = np.load('/DATACENTER/data/yyf/Py/bert_privacy/data/Airline/EX_part/EMB/ernie/train.Hong Kong.0.ernie2.npy')
    # print(embs[0])
    # embss = np.load('/DATACENTER/data/yyf/Py/bert_privacy/data/Airline/EX_part/EMB/ernie2/train.Hong Kong.0.ernie2.npy')
    # print(embss[0])
    print('.')

if __name__ == "__main__":
