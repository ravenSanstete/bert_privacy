## the attack on the genome data
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

def explate(seq):
    out = ""
    for c in seq:
        out = out + c + ' '
    return out[:-1]

def extract_genomes(path):
    f = open(path, 'r')
    out = []
    for i in range(4): next(f)
    for line in f:
        line = line.split(' ')
        out.append(explate(line[-1][:-1]))
    return out

def prepare_raw_datasets():
    TRUE_PATH = "data/acceptor_hs3d/IE_true.seq"
    F_PATH_PAT = "data/acceptor_hs3d/IE_false.seq.00{}"
    true_akpt = extract_genomes(TRUE_PATH)
    false_akpt = []
    for i in range(1, 5):
        false_akpt.extend(extract_genomes(F_PATH_PAT.format(i)))
    # random select 1:10 false samples
    false_akpt = np.random.choice(false_akpt, size = 10 * len(true_akpt), replace = False).tolist()
    print("# of Positive Samples: {}".format(len(true_akpt)))
    print("# of Negative Samples: {}".format(len(false_akpt)))
    return true_akpt, false_akpt

def train_test_split(embs, ratio = 0.9):
    np.random.shuffle(embs)
    train = embs[:int(ratio * len(embs))]
    test = embs[int(ratio*len(embs)):]
    return train, test

def construct_datasets(arch = 'bert'):
    embedding_path = "data/acceptor_hs3d/IE.{}"
    true_akpt, false_akpt = prepare_raw_datasets()
    true_embeddings = embedding(true_akpt, embedding_path.format(1), arch, False)
    false_embeddings = embedding(false_akpt, embedding_path.format(0), arch, False)
    return

# let us just test svm
def predict(embedding_path = "data/acceptor_hs3d/IE.{}"):
    arch = 'bert'
    true_embeddings = embedding(None, embedding_path.format(1), arch)
    false_embeddings = embedding(None, embedding_path.format(0), arch)[:len(true_embeddings)]
    print(true_embeddings)
    print(false_embeddings)
    # do a train test split
    train_1, test_1 = train_test_split(true_embeddings)
    train_0, test_0 = train_test_split(false_embeddings)
    print("# of train_0: {}".format(len(train_0)))
    print("# of train_1: {}".format(len(train_1)))
    print("# of test_0: {}".format(len(test_0)))
    print("# of test_1: {}".format(len(test_1)))
    # clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-5, verbose = 1)
    clf = SVC(kernel = 'linear', gamma = 'scale', verbose = True)
    train_x = np.concatenate([train_0, train_1], axis = 0)
    test_x = np.concatenate([test_0, test_1], axis = 0)
    train_y = np.array([0] * len(train_0) + [1] * len(train_1))
    test_y = np.array([0] * len(test_0) + [1] * len(test_1))
    clf.fit(train_x, train_y)
    preds = clf.predict(test_x)
    print(np.sum(preds))
    true_p = np.mean(preds[test_y == 1])
    false_p = np.mean(1 - preds[test_y == 1])
    print('ACC: {:.4f} TP: {:.4f} FP: {:.4f}'.format(np.mean(preds == test_y), true_p, false_p))
    pass


# The attacker model, which is used to infer the genetic subsequence at a fixed interval (a 4)
TABLE = {
    "A": 0,
    "G": 1,
    "C": 2,
    "T": 3
    }
REVERSE_TABLE  = ["A", "G", "C", "T"]
EMB_DIM_TABLE = {
    "bert": 1024
    }
INTERVAL_LEN = 4

TOTAL_LEN = 140
ARCH = 'bert'

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




def gen(target = 0):
    # @param target: which specifies the inverval to infer (i.e. [target, target + inverval_LEN))
    key = [random.choice(REVERSE_TABLE) for i in range(target, target+INTERVAL_LEN)]
    part_A = [random.choice(REVERSE_TABLE) for i in range(0, target)]
    part_B = [random.choice(REVERSE_TABLE) for i in range(target+INTERVAL_LEN, TOTAL_LEN)]
    return " ".join(part_A + key + part_B), seq2id("".join(key)) 
def get_batch(target = 0, batch_size = 10):
    batch = [gen(target) for i in range(batch_size)]
    z = embedding([x for x, y in batch], "tmp", ARCH, cached = False)
    y = [int(y) for x, y in batch]
    z = torch.FloatTensor(z)
    y = torch.LongTensor(y)
    return z, y, [x for x, y in batch]
    
class Classifier(nn.Module):
    def __init__(self, embedding_size, hidden_size, cls_num = 12, device = torch.device('cuda:1')):
        super(Classifier, self).__init__()
        self.fc1 = Linear(embedding_size, hidden_size)
        self.fc2 = Linear(hidden_size, cls_num)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        print(cls_num)
        

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


def train_attacker():
    TARGET = 0
    CLS_NUM = 4 ** INTERVAL_LEN
    print("INFER GENE SUBSEQ [{}, {}) CLS NUMBER {}".format(TARGET, TARGET + INTERVAL_LEN, CLS_NUM))
    MAX_ITER = 20000
    CACHED = True
    PRINT_FREQ = 100
    DEVICE = torch.device('cuda:0')
    TEST_SIZE = 1000
    HIDDEN_DIM = 500
    BATCH_SIZE = 512 # 128 #64

    EMB_DIM = EMB_DIM_TABLE[ARCH]
    PATH = "{}-{}_cracker.cpt".format(TARGET, TARGET + INTERVAL_LEN)
    best_acc = 0.0
    K = 5
    classifier = Classifier(EMB_DIM, HIDDEN_DIM, CLS_NUM, DEVICE)
    if(CACHED):
        print("Loading Model...")
        classifier.load_state_dict(torch.load(PATH))
    classifier = classifier.to(DEVICE)
    test_x, test_y, _ = get_batch(TARGET, TEST_SIZE)
    test_x = test_x.to(DEVICE)

    optimizer = optim.Adam(classifier.parameters(), lr = 0.001)
    running_loss = 0.0
    
    acc = classifier.evaluate(test_x, test_y)
    topk_acc = classifier.evaluate_topk(test_x, test_y, k = K)
    print("Iteration {} Loss {:.4f} Acc.: {:.4f} Top-{} Acc.: {:.4f}".format(0, running_loss/PRINT_FREQ, acc, K, topk_acc))
    for i in tqdm(range(MAX_ITER)):
        x, y, _ = get_batch(TARGET, BATCH_SIZE)
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
    
if __name__ == '__main__':
    # prepare_raw_datasets()
    # construct_datasets("bert")
    # predict()
    train_attacker()


