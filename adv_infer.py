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

import argparse
parser = argparse.ArgumentParser(description='ID Attack')
parser.add_argument("-p", type=int, default= 5555, help = 'the comm port the client will use')
parser.add_argument("-a", type=str, default='bert', help = 'targeted architecture')
parser.add_argument("-t", action='store_true', help = "to switch between training or testing")
parser.add_argument("-m", type=str, default='date', help = 'targeted part to attack')
parser.add_argument("-c", action='store_true', help = 'whether to use cached model')
parser.add_argument("--pooling", type=str, default='last', help = 'the pooling option')
parser.add_argument("-f", type = str, default = 'atk', help = 'which function to do')


ARGS = parser.parse_args()

def explate(seq):
    out = ""
    for c in seq:
        out = out + c + ' '
    return out[:-1]


def load_state_code(path = 'state.txt'):
    f = open(path, 'r')
    code = []
    for line in f:
        code.append(line.split(" ")[1])
    return code

STATE_CODE = load_state_code()


NUM = [str(i) for i in range(10)]

INFER_PART = ARGS.m

HIDDEN_DIM_TABLE = {
    "month": 25,
    "date" : 200,
    "year" : 400
}


ARCH = ARGS.a

EMB_DIM_TABLE = {
    "bert": 768,
    'gpt' : 768,
    'gpt-2': 768,
    'transformer-xl': 1024,
    'xlnet': 768,
    'xlm': 1024,
    'roberta': 768,
    'ernie': 768
}

CLS_NUM_TABLE = {
    "month": 12,
    "date" : 30,
    "year" : 100
    }

        
OFFLINE_TABLE = {
    "year": 1,
    "month": 2,
    "date": 3
}
    

EMB_DIM = EMB_DIM_TABLE[ARCH]



embedder = Embedder(ARGS.p)
embedding = embedder.embedding # export the functional port

POOLING = ARGS.pooling



def gen(part = "year"):
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

def gen_offline():
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
    return whole, int(part_year) + 1, month, date
    
def get_batch_offline(batch_size):
    batch = [gen_offline() for i in range(batch_size)]
    ids = [b[0] for b in batch]
    if(ARCH == 'transformer-xl'):
        ids = [explate(x) for x in ids]
    z = embedding(ids, "tmp", ARCH, cached = False)

    year_y = [int(b[1])-1 for b in batch]
    month_y = [int(b[2])-1 for b in batch]
    date_y = [int(b[3])-1 for b in batch]
    year_y = np.array(year_y)
    month_y = np.array(month_y)
    date_y = np.array(date_y)
    return z, year_y, month_y, date_y, [b[0] for b in batch]
    
def get_batch(batch_size, part):
    batch = [gen(part) for i in range(batch_size)]
    ids = [x for x, y in batch]
    if(ARCH == 'transformer-xl'):
        ids = [explate(x) for x in ids]
    z = embedding(ids, "tmp", ARCH, cached = False)

    y = [int(y)-1 for x, y in batch]
    z = torch.FloatTensor(z)
    y = torch.LongTensor(y)
    return z, y, [x for x, y in batch]

def get_offline_batch_loader(batch_size, pooling):
    path_temp = 'citizen.{}.{}'.format(ARCH, pooling)+'.{}.npy'
    comps = ['k', 'year', 'month', 'date']
    z, year_y, month_y, date_y = list(map(lambda p: np.load(p), map(lambda x: path_temp.format(x), comps)))
    print(z.shape)
    z = torch.FloatTensor(z)
    year_y = torch.LongTensor(year_y)
    month_y = torch.LongTensor(month_y)
    date_y = torch.LongTensor(date_y)
    train_batches = data_utils.TensorDataset(z, year_y, month_y, date_y)
    train_batches = data_utils.DataLoader(train_batches, batch_size = batch_size, shuffle = True)
    return list(train_batches)

def get_batch_from_dataloader(dataloader, part):
    batch = random.choice(dataloader[1:])
    return batch[0], batch[OFFLINE_TABLE[part]]
 


def generate_offline_training_set(batch_num = 1000, pooling = 'mean'):
    batch_size = 128
    K, V1, V2, V3 = [], [], [], []
    for i in tqdm(range(batch_num)):
        z, year_y, month_y, date_y, _ = get_batch_offline(batch_size)
        K.append(z)
        V1.append(year_y)
        V2.append(month_y)
        V3.append(date_y)
    K, V1, V2, V3 = [np.concatenate(x, axis = 0) for x in [K, V1, V2, V3]]
    print(K.shape)
    print(V1.shape)
    print(V2.shape)
    print(V3.shape)
    np.save('citizen.{}.{}.k.npy'.format(ARCH, pooling), K)
    np.save( 'citizen.{}.{}.year.npy'.format(ARCH, pooling), V1)
    np.save('citizen.{}.{}.month.npy'.format(ARCH, pooling), V2)
    np.save( 'citizen.{}.{}.date.npy'.format(ARCH, pooling), V3)
    print(K[0:2, :])
    
    
    

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
    print("INFER {}".format(INFER_PART))
    MAX_ITER = 1000000
    CACHED = ARGS.c
    PRINT_FREQ = 1000
    OFFLINE = True

    DEVICE = torch.device('cuda:0')
    TEST_SIZE = 1000
    HIDDEN_DIM = HIDDEN_DIM_TABLE[INFER_PART]
    CLS_NUM = CLS_NUM_TABLE[INFER_PART]
    BATCH_SIZE = 128 # 64

    PATH = "{}_{}_{}_cracker.cpt".format(ARCH, INFER_PART, POOLING)

    best_acc = 0.0
    K = 5
    
    classifier = Classifier(EMB_DIM, HIDDEN_DIM, CLS_NUM, DEVICE)
    if(CACHED):
        print("Loading Model...")
        classifier.load_state_dict(torch.load(PATH))

    if(OFFLINE):
        offline_dataloader = get_offline_batch_loader(BATCH_SIZE, POOLING)

    
    classifier = classifier.to(DEVICE)
    # if(OFFLINE):
    #     test_x, test_y = offline_dataloader[0][0], offline_dataloader[0][OFFLINE_TABLE[INFER_PART]]
    # else:
    test_x, test_y, _ = get_batch(TEST_SIZE, INFER_PART)
    test_x = test_x.to(DEVICE)
    optimizer = optim.Adam(classifier.parameters(), lr = 0.001)
    running_loss = 0.0
    acc = classifier.evaluate(test_x, test_y)
    topk_acc = classifier.evaluate_topk(test_x, test_y, k = K)
    print("Iteration {} Loss {:.4f} Acc.: {:.4f} Top-{} Acc.: {:.4f}".format(0, running_loss/PRINT_FREQ, acc, K, topk_acc))
    for i in tqdm(range(MAX_ITER)):
        if(OFFLINE):
            x, y = get_batch_from_dataloader(offline_dataloader, INFER_PART)
        else:
            x, y, _ = get_batch(BATCH_SIZE, INFER_PART)
        
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        optimizer.zero_grad()
        loss = classifier.loss(x, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if((i + 1) % PRINT_FREQ == 0):
            acc = classifier.evaluate(test_x, test_y)
            topk_acc = classifier.evaluate_topk(test_x, test_y, k = K)
            print("Iteration {} Loss {:.4f} Acc.: {:.4f} Top-{} Acc.: {:.4f}".format(i+1, running_loss/PRINT_FREQ, acc, K, topk_acc))
            running_loss = 0.0
            if(acc >= best_acc):
                best_acc = acc
                torch.save(classifier.state_dict(), PATH)
                print("save model acc. {:.4f}".format(best_acc))
        
def padding(x):
    if(len(x) == 1):
        x = "0" + x
    return x

if __name__ == '__main__':

    if(ARGS.f == 'atk'):
        main()
    elif(ARGS.f == 'gen'):
        generate_offline_training_set(pooling = POOLING)
    else:
        parts = ["year", "month", "date"]
        PATH = "{}_{}_cracker.cpt"
        crackers = []
        DEVICE = torch.device('cuda:0')
        TEST_SIZE = 1000
        # DEMO_SIZE = 4
        
        K = 5
        for p in parts:
            path = PATH.format(ARCH, p)
            print("Loading {} Cracker...".format(path))
            classifier = Classifier(EMB_DIM, HIDDEN_DIM_TABLE[p], CLS_NUM_TABLE[p], DEVICE)
            
            classifier.load_state_dict(torch.load(path))
            classifier.to(DEVICE)
            test_x, test_y, _ = get_batch(TEST_SIZE, p)
            # print(test_y)
            test_x = test_x.to(DEVICE)
            acc = classifier.evaluate(test_x, test_y)
            topk_acc = classifier.evaluate_topk(test_x, test_y, k = K)
            print("Arch: {} Part: {} Acc.: {:.4f} Top-{} Acc.: {:.4f}".format(ARCH, p, acc, K, topk_acc))
            
            
            # crackers.append(classifier)
        
        
        # demo_x, _, demo_plain = get_batch(DEMO_SIZE, p)
        # demo_x = demo_x.to(DEVICE)
        # cracked = [cls.predict_topk(demo_x,k= K) for cls in crackers]
        # demo_x = demo_x.cpu().numpy()
        # for i, text in enumerate(demo_plain):
        #     print("============================ SAMPLE {} ===========================".format(i+1))
        #     print("Original ID: {}".format(text))
        #     print("Embedding: {}".format(demo_x[i, :]))
        #     year = [("19" + padding(str(x))) for x in cracked[0][i, :]]
        #     print("Top-{} Year: {}".format(K, year))
        #     print("Top-{} Month: {}".format(K, [padding(str(x)) for x in (cracked[1][i, :] + 1)]))
        #     print("Top-{} Date: {}".format(K,  [padding(str(x)) for x in (cracked[2][i, :] + 1)]))

    
        
        
    
