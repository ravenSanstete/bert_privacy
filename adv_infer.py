import random

import numpy as np
import torch
import torch.utils.data as data_utils
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRU, Embedding, Linear
from util import embedding
from tqdm import tqdm

def load_state_code(path = 'state.txt'):
    f = open(path, 'r')
    code = []
    for line in f:
        code.append(line.split(" ")[1])
    return code

STATE_CODE = load_state_code()


NUM = [str(i) for i in range(10)]
INFER_PART = "date"

HIDDEN_DIM_TABLE = {
    "month": 25,
    "date" : 200,
    "year" : 400
}

ARCH = "bert"
EMB_DIM_TABLE = {
    "bert": 1024,
    "gpt": 768,
    "gpt2": 768,
    "xl": 1024
    }

CLS_NUM_TABLE = {
    "month": 12,
    "date" : 30,
    "year" : 100
    }


EMB_DIM = EMB_DIM_TABLE[ARCH]

def gen(part = "month"):
    part_year = random.choice(NUM) + random.choice(NUM)
    
    
    year = "19"+ part_year
    month = random.choice([str(i) for i in range(1, 13)])
    date = random.choice([str(i) for i in range(1, 31)])
    if(len(month) == 1):
        month = "0" + month
    if(len(date) == 1):
        date  = "0" + date
    state_id = random.choice(STATE_CODE)
    individual = "".join([random.choice(NUM) for i in range(4)])
    whole = state_id + year + month + date + individual

    if(part == 'month'):
        side_channel = month
    elif(part == 'date'):
        side_channel = date
    elif(part == 'year'):
        side_channel = int(part_year) + 1
    return whole, side_channel

def get_batch(batch_size = 10, part = INFER_PART):
    batch = [gen() for i in range(batch_size)]
    z = embedding([x for x, y in batch], "tmp", ARCH, cached = False)
    y = [int(y)-1 for x, y in batch]
    z = torch.FloatTensor(z)
    y = torch.LongTensor(y)
    return z, y

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

    def loss(self, x, y):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        _loss = self.criterion(x, y)
        return _loss

    def evaluate(self, x, y):
        preds = self.predict(x)
        y = y.numpy()
        return np.mean(preds == y)
    
    
    
if __name__ == '__main__':
    print("INFER {}".format(INFER_PART))
    MAX_ITER = 40000
    CACHED = False
    PRINT_FREQ = 100
    DEVICE = torch.device('cuda:1')
    TEST_SIZE = 1000
    HIDDEN_DIM = HIDDEN_DIM_TABLE[INFER_PART]
    CLS_NUM = CLS_NUM_TABLE[INFER_PART]
    BATCH_SIZE = 128
    PATH = "{}_cracker_tmp.cpt".format(INFER_PART)
    best_acc = 0.0
    
    classifier = Classifier(EMB_DIM, HIDDEN_DIM, CLS_NUM, DEVICE)
    if(CACHED):
        print("Loading Model...")
        classifier.load_state_dict(torch.load(PATH))
    classifier = classifier.to(DEVICE)
    test_x, test_y = get_batch(TEST_SIZE, INFER_PART)
    test_x = test_x.to(DEVICE)
    optimizer = optim.Adam(classifier.parameters(), lr = 0.001)
    running_loss = 0.0
    for i in tqdm(range(MAX_ITER)):
        x, y = get_batch(BATCH_SIZE, INFER_PART)
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = classifier.loss(x, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if((i + 1) % PRINT_FREQ == 0):
            acc = classifier.evaluate(test_x, test_y)
            print("Iteration {} Loss {:.4f} Acc.: {:.4f}".format(i+1, running_loss/PRINT_FREQ, acc))
            running_loss = 0.0
            if(acc >= best_acc):
                best_acc = acc
                torch.save(classifier.state_dict(), PATH)
                print("save model acc. {:.4f}".format(best_acc))
        
    

    
    
