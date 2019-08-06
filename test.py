import torch
import random
import pytorch_transformers
from bert_serving.client import BertClient
import torch
from tqdm import tqdm
import numpy as np
from util import embedding
from tools import balance
from mpl_toolkits.mplot3d import Axes3D
from DANN import DANN
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


def main():
    f = open('/DATACENTER/data/yyf/Py/bert_privacy/data/Airline/EX_part/train.{}.{}'.format('Hong Kong','1') + '.txt', 'r')
    target_f = [x[:-1] for x in f if x[:-1] != '']
    print(len(target_f))


if __name__ == "__main__":
    main()

