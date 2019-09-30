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
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import argparse
from pytorch_revgrad import RevGrad
from sklearn.decomposition import PCA
from scipy.stats import describe
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree
from sklearn.manifold import MDS
from numpy import linalg as LA

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

TOTAL_LEN = 20


parser = argparse.ArgumentParser(description='Genome Attack')
parser.add_argument("-p", type=int, default= 0, help = 'the position to attack')
parser.add_argument("-c", action='store_true', help = 'whether to use cached model')
ARGS = parser.parse_args()

def do_kdtree(combined_x_y_arrays,points):
    mytree = cKDTree(combined_x_y_arrays)
    dist, indexes = mytree.query(points)
    return dist, indexes


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

        out.append(line[-1][:TOTAL_LEN])

    return out
## extraction 
def _extract_genomes(path):
    f = open(path, 'r')
    out = []
    for i in range(4): next(f)
    for line in f:
        line = line.split(' ')
        out.append(line[-1][:TOTAL_LEN])
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
    print(len(true_akpt[0]))
    print(len(false_akpt[0]))

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
    "bert": 1024,
    'gpt' : 768
    }
INTERVAL_LEN = 1

ARCH = 'gpt'


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
<<<<<<< HEAD
    # key = [random.choice(REVERSE_TABLE) for i in range(target, target+INTERVAL_LEN)]
    part_A = [random.choice(REVERSE_TABLE) for i in range(0, target)]
    part_B = [random.choice(REVERSE_TABLE) for i in range(target+INTERVAL_LEN, TOTAL_LEN)]
    # to 
    return [("".join(part_A + [key] + part_B), seq2id("".join([key]))) for key in REVERSE_TABLE]

CENTERS = []
PLOTTED = False


def get_batch(target = 0, batch_size = 10):
    global PLOTTED
    pca = PCA(n_components=2)
    batch = []
    for i in range(batch_size):
        batch.extend(gen(target))
    z = embedding([x for x, y in batch], "tmp", ARCH, cached = False)
    # to centralize the embeddings
    centers = []
    inner_cluster_dist = []
    # for i in range(z.shape[0]//4):
    #     c = np.mean(z[i*4:(i+1)*4, :], axis = 0)
    #     z[i*4:(i+1)*4] =  z[i*4:(i+1)*4] - c
    # A_vecs = []
    # for k in range(4):
    #     A_vecs.append(np.array([z[i, :] for i in range(z.shape[0]) if i % 4 == k]))
    # total = np.concatenate(A_vecs, axis = 0)
    # pca = MDS(n_components=2)
    # total = pca.fit_transform(total)
    # colors = sns.color_palette("hls", 4)
    # interval = len(total) // 4
    # if(not PLOTTED):
    #     for k in range(4):
    #         plt.scatter(total[k*interval:(k+1)*interval,0], total[k*interval:(k+1)*interval,1], c = [colors[k] for i in range(interval)])
    #     plt.savefig('delta_mds_center.png')
    #     PLOTTED = True
    
        # centers.append(c)
        # if(np.random.rand() < 0.1):
        #    CENTERS.append(c) # collect the centers
    # inner_cluster_dist = np.linalg.norm(z, axis = 0)
    # centers = np.array(centers)
    # dist = pdist(centers, 'euclidean')
    # print("OUTER: {}".format(describe(dist)))
    # print("INNER: {}".format(describe(inner_cluster_dist)))
    # for i in range(z.shape[0]):
    #     z[i, :] = z[i, :] - np.mean(z[i, :]) # what about the average
    y = [int(y) for x, y in batch]
    z = torch.FloatTensor(z)
    y = torch.LongTensor(y)
    return z, y, [x for x, y in batch]


def get_batch_ground_truth(target = 0, batch_size = 10):
    embedding_path = "data/acceptor_hs3d/IE.{}"
    TRUE_PATH = "data/acceptor_hs3d/IE_true.seq"
    z = embedding(None, embedding_path.format(1), ARCH)[:batch_size, :]
    y = _extract_genomes(TRUE_PATH)[:batch_size]
    y = [seq2id(x[target:target+INTERVAL_LEN]) for x in y]
    z = torch.FloatTensor(z)
    y = torch.LongTensor(y)
    return z, y, None
    
class Classifier(nn.Module):
    def __init__(self, embedding_size, hidden_size, cls_num = 12, device = torch.device('cuda:1')):
        super(Classifier, self).__init__()
        self.encoder = nn.Sequential(Linear(embedding_size, 400),
                                        nn.Sigmoid(),
                                        Linear(400, 200),
                                     nn.Sigmoid(),
                                     Linear(200, 100))
        
        self.decoder = nn.Sequential(Linear(20, 100),
                                     nn.ReLU(True),
                                     Linear(100, 200),
                                     nn.ReLU(True),
                                     Linear(200, 400),
                                     nn.ReLU(True),
                                     Linear(400, embedding_size),
                                     nn.ReLU(True))
        self.classifier = Linear(100, cls_num)

        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        print(cls_num)
        

    def forward(self, x):
        x = self.classifier(self.encoder(x))

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

    
    def pretrain_loss(self, x):
        z = self.encoder(x)
        z = self.decoder(z)
        _loss = F.mse_loss(z, x)
        return _loss
        
        
    def loss(self, x, y):
        x = self(x)

        _loss = self.criterion(x, y)
        return _loss

    def evaluate(self, x, y):
        # x = x.cpu().numpy()
        # dist, indexes = do_kdtree(CENTERS, x)
        # print("DIST:{}".format(dist))
        # print("INDICES:{}".format(indexes))
        # for i in range(x.shape[0]):
        #     x[i, :] = x[i, :] - CENTERS[indexes[i]]
        # x = torch.FloatTensor(x).cuda()
        with torch.no_grad():
            preds = self.predict(x)
            y = y.numpy()
            print(np.histogram(y))
            print(np.histogram(preds))
        return np.mean(preds == y)

    def evaluate_topk(self, x, y, k = 5):
        y = y.numpy()
        with torch.no_grad():
            probs = self(x)
            _, topk = torch.topk(probs, k)
            topk = topk.cpu().numpy()
            acc = [int(y[i] in topk[i, :]) for i in range(len(y))]
        return np.mean(acc)


class DANNClassifier(nn.Module):
    def __init__(self, embedding_size, hidden_size, cls_num = 12, device = torch.device('cuda:1')):
        super(DANNClassifier, self).__init__()
        self.encoderA = Linear(embedding_size, hidden_size)
        self.encoderB = Linear(embedding_size, hidden_size)
        self.classifier = Linear(hidden_size, cls_num)
        self.device = device
        self.rev_grad = RevGrad()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        z = torch.sigmoid(self.encoderA(x))
        x = self.classifier(z)
        return x
    
    def loss(self, x, y):
        z = torch.sigmoid(self.encoderA(x))
        a = torch.sigmoid(self.encoderB(x))
        delta = z + a
        z = self.classifier(z)
        a = self.classifier(self.rev_grad(a)) # put a gradient reversal layer
        _lossA = self.criterion(z, y)
        _lossB = self.criterion(a, y)
        reconstruction_loss = F.mse_loss(x, delta)
        coeff = 0.01
        return _lossA + _lossB + coeff * reconstruction_loss

    def predict(self, x):
        outputs = self(x)
        _, preds = torch.max(outputs, 1)
        return preds.cpu().numpy()

    def predict_topk(self, x, k = 5):
        with torch.no_grad():
            probs = self(x)
            _, topk = torch.topk(probs, k)
        return topk.cpu().numpy()

    def evaluate(self, x, y):
        with torch.no_grad():
            preds = self.predict(x)
            y = y.numpy()
            print(np.histogram(y))
            print(np.histogram(preds))

        return np.mean(preds == y)

    def evaluate_topk(self, x, y, k = 5):
        y = y.numpy()
        with torch.no_grad():
            probs = self(x)
            _, topk = torch.topk(probs, k)
            topk = topk.cpu().numpy()
            acc = [int(y[i] in topk[i, :]) for i in range(len(y))]
        return np.mean(acc)

def train_attacker(target = 0):
    TARGET = target
    CLS_NUM = 4 ** INTERVAL_LEN
    print("INFER GENE SUBSEQ [{}, {}) CLS NUMBER {}".format(TARGET, TARGET + INTERVAL_LEN, CLS_NUM))
    MAX_ITER = 10000
    CACHED = ARGS.c
    PRINT_FREQ = 100
    DEVICE = torch.device('cuda:1')
    TEST_SIZE = 1000
    HIDDEN_DIM = 200
    BATCH_SIZE = 64 # 128 #64
    TRUTH = True
    EMB_DIM = EMB_DIM_TABLE[ARCH]
    PATH = "checkpoints/{}-{}_cracker_tmp_len_20_hidden_400_100.cpt".format(TARGET, TARGET + INTERVAL_LEN)
    best_acc = 0.0
    K = 2
    classifier = Classifier(EMB_DIM, HIDDEN_DIM, CLS_NUM, DEVICE)
    if(CACHED and Path(PATH).exists()):
        print("Loading Model...")
        classifier.load_state_dict(torch.load(PATH))
    classifier = classifier.to(DEVICE)


    if(TRUTH):
        test_x, test_y, _ = get_batch_ground_truth(TARGET, TEST_SIZE)
    else:
        test_x, test_y, _ = get_batch(TARGET, TEST_SIZE)

    test_x = test_x.to(DEVICE)
    # optimizer = optim.SGD(classifier.parameters(), lr = 0.1)
    optimizer = optim.Adam(classifier.parameters(), lr = 0.001)
    running_loss = 0.0
    running_mean = np.zeros([EMB_DIM])

    # print("PRETRAIN")
    # for i in tqdm(range(MAX_ITER)):
    #     x, _, _ = get_batch(TARGET, BATCH_SIZE)
    #     x = x.to(DEVICE)
    #     optimizer.zero_grad()
    #     loss = classifier.pretrain_loss(x)
    #     loss.backward()
    #     optimizer.step()
    #     running_loss += loss.item()
        
    #     if((i + 1) % PRINT_FREQ == 0):
    #         print("Iteration {} Reconstruction Loss: {:.4f}".format(i+1, running_loss/PRINT_FREQ))
    #         running_loss = 0.0
        

    running_loss = 0.0    
    
    # acc = classifier.evaluate(test_x, test_y)
    # topk_acc = classifier.evaluate_topk(test_x, test_y, k = K)
    # print("Iteration {} Loss {:.4f} Acc.: {:.4f} Top-{} Acc.: {:.4f}".format(0, running_loss/PRINT_FREQ, acc, K, topk_acc))
    for i in tqdm(range(MAX_ITER)):
        x, y, raw = get_batch(TARGET, BATCH_SIZE)
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
            running_mean = np.zeros([EMB_DIM])
            # print(raw[:4])
            # print(y[:4])

            if(acc >= best_acc):
                best_acc = acc
                torch.save(classifier.state_dict(), PATH)
                print("save model acc. {:.4f}".format(best_acc))
                if(best_acc > 0.99):
                    break
    return best_acc


def train_random_forest(target = 0):
    train_sample_num = 10000
    test_sample_num = 1000
    test_x, test_y, _ = get_batch_ground_truth(target, test_sample_num)
    x, y, _ = get_batch(target, train_sample_num)
    x, y = x.numpy(), y.numpy()
    test_x, test_y = test_x.numpy(), test_y.numpy()
    clf = RandomForestClassifier(n_estimators = 100)
    # clf = SVC()
    clf.fit(x, y)
    preds = clf.predict(test_x)
    acc = np.mean(preds == test_y)
    print("Target {} -- Top-1 Acc. {:.4f}".format(target, acc))
    return acc

if __name__ == '__main__':
    # prepare_raw_datasets()
    construct_datasets("gpt")
    # predict()
    # import sys; sys.exit()
    # acc = 1.0
    for target in range(5, 20):
        # target =ARGS.p
        acc = train_attacker(target)
        # acc *= train_random_forest(target)
    print("Restore 20-length gene Acc.: {}".format(acc))

