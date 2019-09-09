## to explore the genome case
from util import embedding
import numpy as np
from sklearn.svm import SVC
from sklearn import linear_model
import torch
import torch.utils.data as data_utils
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRU, Embedding, Linear
import random
from tqdm import tqdm
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import argparse
from scipy.spatial.distance import pdist
from scipy.stats import describe

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

TABLE = {
    "A": 0,
    "G": 1,
    "C": 2,
    "T": 3
    }
REVERSE_TABLE  = ["A", "G", "C", "T"]
INTERVAL_LEN = 1
TOTAL_LEN = 20
ARCH = 'gpt'

def gen(target = 0):
    # @param target: which specifies the inverval to infer (i.e. [target, target + inverval_LEN))
    # key = [random.choice(REVERSE_TABLE) for i in range(target, target+INTERVAL_LEN)]
    part_A = [random.choice(REVERSE_TABLE) for i in range(0, target)]
    part_B = [random.choice(REVERSE_TABLE) for i in range(target+INTERVAL_LEN, TOTAL_LEN)]
    # to 
    return [("".join(part_A + [key] + part_B), seq2id("".join([key]))) for key in REVERSE_TABLE]

def get_batch(target = 0, batch_size = 10):
    batch = []
    for i in range(batch_size):
        batch.extend(gen(target))
    z = embedding([x for x, y in batch], "tmp", ARCH, cached = False)
    y = [int(y) for x, y in batch]
    z = torch.FloatTensor(z)
    y = torch.LongTensor(y)
    return z, y, [x for x, y in batch]


def seq2id(s):
    val = 0
    base = 4 ** (INTERVAL_LEN - 1)
    for i, c in enumerate(s):
        val += base * TABLE[c]
        base = base // 4
    return val

def id2seq(val):
    s = np.base_repr(val, base = 4).zfill(INTERVAL_LEN)  
    return "".join([REVERSE_TABLE[int(c)] for c in s])

def compute_diff(z):
    diff = []
    for i in range(z.shape[0]//4):
        dist = pdist(z[i*4:(i+1)*4, :], 'euclidean')
        diff.append(np.mean(dist))
    print(describe(diff))
    return np.mean(diff)
    
        
    

if __name__ == '__main__':
    diffs = []
    for i in range(TOTAL_LEN):
        TARGET = i
        sample_size = 1024
        z, y, x = get_batch(TARGET, sample_size)
        diff = compute_diff(z)
        diffs.append(diff)
    
    plt.plot(list(range(TOTAL_LEN)), diffs)
    plt.savefig('curve.png')
    
    # print(z.shape)
    # print(y)
    
    
