import random

import numpy as np
import torch
import torch.utils.data as data_utils
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRU, Embedding, Linear

from util import Embedder
from tqdm import tqdm
import time 


class Classifier(nn.Module):
    def __init__(self, embedding_size, hidden_size, cls_num = 12, device = torch.device('cuda:1')):
        super(Classifier, self).__init__()
        self.fc1 = Linear(embedding_size, hidden_size)
        self.fc2 = Linear(hidden_size, cls_num)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = F.softmax(self.fc2(x), dim = 0)
        return x
    
    def predict(self, x):
        outputs = self(x)
        _, preds = torch.max(outputs, 1)
        return preds.cpu().numpy()

    def predict_topk(self, x, k = 5):
        with torch.no_grad():
            probs = self(x)
            _, topk = torch.topk(probs, k)
        return topk.cpu().numpy()

    def loss(self, x, y):

        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)

        _loss = self.criterion(x, y)
        return _loss

    def evaluate(self, x, y):
        preds = self.predict(x)
        y = y.numpy()
        return np.mean(preds == y)

    def evaluate_topk(self, x, y, k = 5):
        y = y.numpy()
        with torch.no_grad():
            probs = self(x)
            _, topk = torch.topk(probs, k)
            topk = topk.cpu().numpy()
            acc = [int(y[i] in topk[i, :]) for i in range(len(y))]
        return np.mean(acc)

    
    
def main():
    cls_num = 36000
    emb_size = 1024
    year = 400
