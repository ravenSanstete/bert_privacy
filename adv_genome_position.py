## the attack on the genome data
from util import Embedder
import numpy as np
from sklearn.svm import SVC, LinearSVC
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
from defense import initialize_defense


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


parser = argparse.ArgumentParser(description='Genome Attack')
parser.add_argument("-p", type=int, default= 5555, help = 'the comm port the client will use')
parser.add_argument("-c", action='store_true', help = 'whether to use cached model')
parser.add_argument("-t", action='store_true', help = "to switch between training or testing")
parser.add_argument("--save_p", type=str, default="default", help = 'the place to store the model')
parser.add_argument("-a", type=str, default='bert', help = 'targeted architecture')
parser.add_argument("-d", type=str, default='none', help = 'the type of defense to do')
ARGS = parser.parse_args()




TOTAL_LEN = 20

# The attacker model, which is used to infer the genetic subsequence at a fixed interval (a 4)
TABLE = {
    "A": 0,
    "G": 1,
    "C": 2,
    "T": 3
    }
REVERSE_TABLE  = ["A", "G", "C", "T"]
EMB_DIM_TABLE = {
    "bert": 768,
    'bert-large': 1024,
    'gpt' : 768,
    'gpt-2': 768,
    'transformer-xl': 1024,
    'xlnet': 768,
    'xlm': 1024,
    'roberta': 768,
    'ernie': 768,
    "gpt-2-medium": 1024,
    "gpt-2-large": 1280
    }
INTERVAL_LEN = 1

ARCH = ARGS.a

POS_EMBED_DIM = EMB_DIM_TABLE[ARCH]

# 

TRUNCATE_RATIO = 0.1




embedder = Embedder(ARGS.p)
embedding = embedder.embedding # export the functional port

COUNTER = 0

# offline_archs = ['transformer-xl']
offline_archs = []

if(ARCH in offline_archs):
    # construct the transformer
    batch_size = 512
    z = torch.FloatTensor(np.load('{}.z.npy'.format(ARCH)))
    y = torch.LongTensor(np.load('{}.y.npy'.format(ARCH)))
    # to do some truncating
    batch_num = z.shape[0] // batch_size
    print(batch_num)
    ARG.save_p += ".{:.1f}".format(TRUNCATE_RATIO)
    current_batch_num = int(batch_num * TRUNCATE_RATIO)
    print("Batch Size {}/{}".format(current_batch_num, batch_num))
    #
    
    
    xl_dataset = data_utils.TensorDataset(z, y)
    xl_dataloader = data_utils.DataLoader(xl_dataset, batch_size = batch_size, shuffle = True)
    xl_dataloader = [(z, y) for z, y in xl_dataloader]
    print(len(xl_dataloader))



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

def dump(sents, path):
    f = open(path, 'w+')
    for sent in sents:
        f.write(sent + '\n')
    f.close()
    

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
    # dump(true_akpt, "data/acceptor_hs3d/genome.1.txt")
    # dump(false_akpt, "data/acceptor_hs3d/genome.0.txt")
    return true_akpt, false_akpt

def load_raw_datasets():
    true_akpt = [s[:-1] for s in open("data/acceptor_hs3d/genome.1.txt", 'r')]
    false_akpt = [s[:-1] for s in open("data/acceptor_hs3d/genome.0.txt", 'r')]
    return true_akpt, false_akpt

def train_test_split(embs, ratio = 0.9):
    np.random.shuffle(embs)
    train = embs[:int(ratio * len(embs))]
    test = embs[int(ratio*len(embs)):]
    return train, test

def construct_datasets(arch = 'bert'):
    embedding_path = "data/acceptor_hs3d/IE.{}"
    true_akpt, false_akpt = load_raw_datasets() # prepare_raw_datasets()
    
    true_embeddings = embedding(true_akpt, embedding_path.format(1), arch, False)
    false_embeddings = embedding(false_akpt, embedding_path.format(0), arch, False)
    return


class GenomeClassifier(nn.Module):
    def __init__(self, embedding_size):
        super(GenomeClassifier, self).__init__()
        hidden_size = 200
        self.mlp = nn.Sequential(Linear(embedding_size, hidden_size),
                                     nn.Sigmoid(),
                                     Linear(hidden_size, 2))
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.mlp(x)
    
    def predict(self, x):
        outputs = self(x)
        _, preds = torch.max(outputs, 1)
        return preds.cpu().numpy() 
        
    def train(self, X, Y, test_X, test_Y):
        X = torch.FloatTensor(X)
        Y = torch.LongTensor(Y)
        test_X = torch.FloatTensor(test_X).cuda()
        dataset = data_utils.TensorDataset(X, Y)
        dataloader = data_utils.DataLoader(dataset, batch_size = 128, shuffle = True)
        optimizer = optim.Adam(self.parameters(), lr = 0.001)
        self.cuda()
        running_loss = 0.0
        PRINT_FREQ = 100
        counter = 0
        max_epoch = 100
        best_acc = 0.5
        for i in range(max_epoch):
            for x, y in dataloader:
                x, y = x.cuda(), y.cuda()
                loss = self.criterion(self(x), y)
                # print(loss)
                running_loss += loss.data
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                counter += 1
                if(counter % PRINT_FREQ == 0):
                    running_loss /= PRINT_FREQ
                    preds = self.predict(test_X)
                    acc = np.mean(preds == test_Y)
                    print("Iteration {}: Loss {:.4f} Acc: {:.4f}".format(counter, running_loss, acc))
                    running_loss = 0.0
                    if(acc > best_acc):
                        best_acc = acc
                        torch.save(self.state_dict(), "functional.genome.{}.cpt".format(ARCH))
                        print("save best acc. {:.4f}".format(best_acc))
        
                    
                
        
        
        
    

# let us just test svm
def predict(embedding_path = "data/acceptor_hs3d/IE.{}"):
    true_akpt, false_akpt = load_raw_datasets()
    if(ARCH == 'transformer-xl'):
        true_akpt = [explate(x) for x in true_akpt]
        false_akpt = [explate(x) for x in false_akpt]
    true_embeddings = embedding(true_akpt, embedding_path.format(1), ARCH)
    false_embeddings = embedding(false_akpt, embedding_path.format(0), ARCH)[:len(true_embeddings),:]
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
    # clf = LinearSVC(verbose = 1, max_iter = 5000)
    clf = GenomeClassifier(true_embeddings.shape[1])
    train_x = np.concatenate([train_0, train_1], axis = 0)
    test_x = np.concatenate([test_0, test_1], axis = 0)
    train_y = np.array([0] * len(train_0) + [1] * len(train_1))
    test_y = np.array([0] * len(test_0) + [1] * len(test_1))
    clf.train(train_x, train_y, test_x, test_y)
    # clf.fit(train_x, train_y)
    # preds = clf.predict(test_x)
    # print(np.sum(preds))
    # true_p = np.mean(preds[test_y == 1])
    # false_p = np.mean(1 - preds[test_y == 1])
    # print('ACC: {:.4f} TP: {:.4f} FP: {:.4f}'.format(np.mean(preds == test_y), true_p, false_p))





def get_positional_embedding(d_pos_vec, n_position):
    position_enc = np.array([
    [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)]
    if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return position_enc

POS_EMBEDDING = get_positional_embedding(POS_EMBED_DIM, TOTAL_LEN)



def seq2id(s):
    return TABLE[s]

def id2seq(val):
    s = np.base_repr(val, base = 4).zfill(INTERVAL_LEN)  
    return "".join([REVERSE_TABLE[int(c)] for c in s])


def gen(target = 0):
    # @param target: which specifies the inverval to infer (i.e. [target, target + inverval_LEN))
    # key = [random.choice(REVERSE_TABLE) for i in range(target, target+INTERVAL_LEN)]
    seq = [random.choice(REVERSE_TABLE) for i in range(TOTAL_LEN)]
    return [("".join(seq), seq2id(seq[target])), target]


CENTERS = []
PLOTTED = False
CONCAT = True


def generate_offline_training_data(total_number, batch_size = 64, pos_embedding = POS_EMBEDDING):
    z = []
    y = []
   
    for idx in tqdm(range(total_number // batch_size)):
        batch = []
        TARGETS = list(range(TOTAL_LEN))
        for i in range(batch_size):
            target = random.choice(TARGETS)
            batch.append(gen(target))
        z_ = embedding([explate(b[0][0]) for b in batch], "tmp", ARCH, cached = False)
        y.extend([b[0][1] for b in batch])
        pos_embeddings = np.array([pos_embedding[b[1]] for b in batch])
        if(CONCAT):
            z_ = np.concatenate([z_, pos_embeddings], axis = 1)
        else:
            z_ = z_ + pos_embeddings
        z.append(z_)
    z = np.concatenate(z, axis = 0)
    y = np.array(y)
    # save the numpy file
    np.save(open("{}.z.npy".format(ARCH), 'w+b'), z)
    np.save(open("{}.y.npy".format(ARCH), 'w+b'), y)
    
    return z, y
    


"""
Now the batch consists of (seq, id, positional_embedding)
"""
def get_batch(batch_size = 10, is_offline = False, dataloader = None, pos_embedding = POS_EMBEDDING):
    global COUNTER
    if(is_offline):
        z, y = dataloader[COUNTER]
        COUNTER = (COUNTER+1) % len(dataloader)
        z, y = random.choice(dataloader)
        return z, y, None
    batch = []
    TARGETS = list(range(TOTAL_LEN))
    for i in range(batch_size):
        target = random.choice(TARGETS)
        batch.append(gen(target))
    # for i in range(batch_size):
    #     target = random.choice(TARGETS)
    #     batch.append([gen(target), target])
    z = embedding([b[0][0] for b in batch], "tmp", ARCH, cached = False)
    pos_embeddings = np.array([pos_embedding[b[1]] for b in batch])
    if(CONCAT):
        z = np.concatenate([z, pos_embeddings], axis = 1)
    else:
        z = z + pos_embeddings
    # z = z + pos_embeddings
    # to centralize the embeddings
    y = [b[0][1] for b in batch]
    z = torch.FloatTensor(z)
    y = torch.LongTensor(y)
    # print(z.shape)
    return z, y, [b[0][0] for b in batch]


def get_batch_ground_truth(target = 0, batch_size = 10, use_defense = False ,defense = None, arch = ARCH, pos_embedding = POS_EMBEDDING):
    embedding_path = "data/acceptor_hs3d/IE.{}"
    # TRUE_PATH = "data/acceptor_hs3d/IE_true.seq"
    y_1 = [s[:-1] for s in open("data/acceptor_hs3d/genome.1.txt", 'r')]
    y_0 = [s[:-1] for s in open("data/acceptor_hs3d/genome.0.txt", 'r')]
    y_1 = y_1[:batch_size]
    y_0 = y_0[:batch_size]
    y = y_1 + y_0

    if(arch == 'transformer-xl'):
        y_0 = [explate(x) for x in y_0]
        y_1 = [explate(x) for x in y_1]
    # print(len(y))
    z_1 = embedding(y_1, embedding_path.format(1), arch)[:batch_size, :]
    z_0 = embedding(y_0, embedding_path.format(0), arch)[:batch_size, :]
    # print(z_1.shape)
    # print(z_0.shape)
    z = np.concatenate([z_1, z_0], axis = 0)

    utility_y = np.array([1]*batch_size + [0]*batch_size)

    # the sensitive y
    y = [seq2id(x[target:target+INTERVAL_LEN]) for x in y]

    # seems a bug here
    if(use_defense):
        z = defense(z, y) # add the defense
    raw_z = z
    ## obtain the correposnding positional embedding
    pos_embeddings = np.array([pos_embedding[target] for i in range(2*batch_size)])
    if(CONCAT):
        z = np.concatenate([z, pos_embeddings], axis = 1)
    else:
        z = z + pos_embeddings
    # y = _extract_genomes(TRUE_PATH)[:batch_size]
   
    z = torch.FloatTensor(z)
    y = torch.LongTensor(y)
    return z, y, utility_y, raw_z
    
class Classifier(nn.Module):
    def __init__(self, embedding_size, hidden_size, cls_num = 12, device = torch.device('cuda:0')):
        super(Classifier, self).__init__()
        self.encoder = nn.Sequential(Linear(embedding_size, 400),
                                     nn.BatchNorm1d(400),
                                        nn.Sigmoid(),
                                     Linear(400, 100),
                                        nn.Sigmoid(),
                                     nn.BatchNorm1d(100))
        
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
        
        
    def loss(self, x, y):
        x = self(x)
        _loss = self.criterion(x, y)
        return _loss

    def evaluate(self, x, y):
        with torch.no_grad():
            preds = self.predict(x)
            y = y.numpy()
            # print(np.histogram(y))
            # print(np.histogram(preds))
        return np.mean(preds == y)

    def evaluate_topk(self, x, y, k = 5):
        y = y.numpy()
        with torch.no_grad():
            probs = self(x)
            _, topk = torch.topk(probs, k)
            topk = topk.cpu().numpy()
            acc = [int(y[i] in topk[i, :]) for i in range(len(y))]
        return np.mean(acc)

DEVICE = torch.device('cuda:0')

def train_attacker(target = 0, path = None):
    TARGET = target
    CLS_NUM = 4 ** INTERVAL_LEN
    print("INFER GENE SUBSEQ [{}, {}) CLS NUMBER {}".format(TARGET, TARGET + INTERVAL_LEN, CLS_NUM))
    MAX_ITER = 100000
    CACHED = False
    PRINT_FREQ = 100

    TEST_SIZE = 1000
    HIDDEN_DIM = 200
    BATCH_SIZE = 128 # 128 #64
    TRUTH = True
    EMB_DIM = EMB_DIM_TABLE[ARCH]
    PATH = path
    best_acc = 0.0
    K = 2
    if(CONCAT):
        emb_dim = EMB_DIM + POS_EMBED_DIM
    else:
        emb_dim = EMB_DIM
    classifier = Classifier(emb_dim, HIDDEN_DIM, CLS_NUM, DEVICE)
    if(CACHED and Path(PATH).exists()):
        print("Loading Model...")
        classifier.load_state_dict(torch.load(PATH, map_location = DEVICE))
    classifier = classifier.cuda()

    if(TRUTH):
        test_x, test_y, _, _ = get_batch_ground_truth(TARGET, TEST_SIZE)
    else:
        test_x, test_y, _ = get_batch(TEST_SIZE)

            

    test_x = test_x.cuda()
    # optimizer = optim.SGD(classifier.parameters(), lr = 0.01)
    optimizer = optim.Adam(classifier.parameters(), lr = 0.001)
    running_loss = 0.0

    
    acc = classifier.evaluate(test_x, test_y)
    topk_acc = classifier.evaluate_topk(test_x, test_y, k = K)
    print("Iteration {} Loss {:.4f} Acc.: {:.4f} Top-{} Acc.: {:.4f}".format(0, running_loss/PRINT_FREQ, acc, K, topk_acc))
    evaluate("", ARCH, None, given_clf = True, clf = classifier)
    for i in tqdm(range(MAX_ITER)):
        if(not ARCH in offline_archs):
            x, y, _ = get_batch(BATCH_SIZE)
        else:
            x, y, _ = get_batch(BATCH_SIZE, is_offline = True, dataloader = xl_dataloader)
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        loss = classifier.loss(x, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if((i + 1) % PRINT_FREQ == 0):
            acc = classifier.evaluate(test_x, test_y)
            topk_acc = classifier.evaluate_topk(test_x, test_y, k = K)
            print("Iteration {} Loss {:.4f} Acc.: {:.4f} Top-{} Acc.: {:.4f}".format(i+1, running_loss/PRINT_FREQ, acc, K, topk_acc))
            evaluate("", ARCH, None, given_clf = True, clf = classifier)
            running_loss = 0.0
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


def evaluate(path, arch, defense = None, given_clf = False, clf = None):
    pos_embed_dim = EMB_DIM_TABLE[arch]
    local_pos_embedding =  get_positional_embedding(pos_embed_dim, TOTAL_LEN)
    TEST_SIZE = 1000
    EMB_DIM = EMB_DIM_TABLE[arch]

    if(CONCAT):
        emb_dim = EMB_DIM + pos_embed_dim
    else:
        emb_dim = EMB_DIM
    # print(emb_dim)
    CLS_NUM = 4
    if(not given_clf):
        classifier = Classifier(emb_dim, 0, CLS_NUM, DEVICE)
        classifier.load_state_dict(torch.load(path, map_location = DEVICE))
        print("Loading Model from {} ...".format(path))
        classifier = classifier.cuda()
        classifier.eval() # this line is important, to deactivate the effect of the batch normalization
    else:
        classifier = clf
        clf.eval()
    
    average_acc = 0.0
    average_topk_acc =0.0
    avg_util  = 0.0
    protected_avg_acc = 0.0
    protected_avg_topk_acc = 0.0
    protected_avg_util = 0.0

    print("Loading the Utility Model ")
    genome_clf_path = "checkpoints/functional.genome.{}.cpt".format(arch)
    
    genome_clf = GenomeClassifier(EMB_DIM)
    # print(EMB_DIM)
    
    # genome_clf.load_state_dict(torch.load(genome_clf_path, map_location = DEVICE))
    genome_clf.cuda()

    # defense = initialize_defense('rounding', decimals = 1)
    # defense = initialize_defense('dp', delta = 12.0, eps = 20.0)
    # defense = initialize_defense('minmax', cls_num = 2, eps = 0.001)
    atk_acc_arr = []
    protected_acc_arr = []
    baseline_acc = []
    
    for target in range(0, TOTAL_LEN):
        test_x, test_y, test_util_y, raw_x = get_batch_ground_truth(target, TEST_SIZE, arch = arch, pos_embedding = local_pos_embedding)
        # impose the defense with raw_x and the test util_y
        histg = np.histogram(test_y, bins = 4)
        baseline_acc.append(max(histg[0] / np.sum(histg[0])))
        
        test_x = test_x.cuda()
        acc = classifier.evaluate(test_x, test_y)
        topk_acc = classifier.evaluate_topk(test_x, test_y, k = 2)
        average_acc += acc
        average_topk_acc += topk_acc
        raw_x = torch.FloatTensor(raw_x).cuda()
        preds = genome_clf.predict(raw_x)
        util_acc = np.mean(preds == test_util_y)
        avg_util += util_acc
        
        atk_acc_arr.append(acc)

        protected_util_acc, protected_acc, protected_topk_acc = 0.0, 0.0, 0.0
        if(defense):
            protected_test_x, _, _, protected_raw_x = get_batch_ground_truth(target, TEST_SIZE, True, defense, arch = arch, pos_embedding = local_pos_embedding)
            protected_test_x = torch.FloatTensor(protected_test_x).cuda()
            protected_acc = classifier.evaluate(protected_test_x, test_y)
            protected_topk_acc = classifier.evaluate_topk(protected_test_x, test_y, k = 2)
            protected_avg_acc += protected_acc
            protected_avg_topk_acc += protected_topk_acc
            protected_raw_x = torch.FloatTensor(protected_raw_x).cuda()
            preds = genome_clf.predict(protected_raw_x)
            protected_util_acc = np.mean(preds == test_util_y)
            protected_avg_util += protected_util_acc
            protected_acc_arr.append(protected_acc)
        
        
        # print("Util Acc: {:.4f} Protected Util Acc.: {:.4f} TARGET INDEX {} ACC: {:.4f} TOP-2: {:.4f} Protected: {:.4f} Protected Top-2: {:.4f}".format(util_acc, protected_util_acc, target, acc, topk_acc, protected_acc, protected_topk_acc))
    # print("Average Acc: {:.4f} Average Top-2 Acc.: {:.4f} Avergage Util: {:.4f} Protected: {:.4f} {:.4f} {:.4f}".format(average_acc/TOTAL_LEN, average_topk_acc/TOTAL_LEN, avg_util/TOTAL_LEN, protected_avg_acc/TOTAL_LEN, protected_avg_topk_acc/TOTAL_LEN, protected_avg_util/TOTAL_LEN))

    """
    # THE ADV. Utility
    # print(atk_acc_arr)
    # print(average_acc /  TOTAL_LEN)
    # print(average_topk_acc / TOTAL_LEN)
    """
    protected_avg_acc  /= TOTAL_LEN
    avg_util /= TOTAL_LEN
    protected_avg_util /= TOTAL_LEN

    print("{},{}".format(atk_acc_arr, np.array(atk_acc_arr).mean()))
    # print(avg_util)
    # print(protected_avg_util)
    # print(protected_acc_arr)
    # print(protected_avg_acc)
    # for 
    # print(baseline_acc)
    # print()
    
    classifier.train()
    return (protected_acc_arr, [avg_util, protected_avg_util]) # protected_acc_arr

    
    
        

        

if __name__ == '__main__':
    # generate_offline_training_data(102400)
    
    # predict()
    # import sys; sys.exit()
    TRAIN = (not ARGS.t)
    DEFENSE = ARGS.d
    TEST_ARCHS = ["bert", "gpt", "gpt-2", "xlm", "xlnet", "roberta", "transformer-xl", "ernie"]
    # TEST_ARCHS = TEST_ARCHS[:1]
    
    
    
    # TEST_ARCHS = ["transformer-xl"]
    # prepare_raw_datasets()
    DELTA_TABLE = {
    "bert": 81.82,
    'gpt' : 73.19,
    'gpt-2': 110.2,
    'transformer-xl': 17.09,
    'xlnet': 601.5,
    'xlm': 219.4,
    'roberta': 4.15,
    'ernie': 28.20        
    }
    

    # predict()
    # import sys; sys.exit()
    # acc = 1.0
    
    # prepare_raw_datasets()
    # predict()
    # import sys; sys.exit()
    TEMPLATE = "checkpoints/genome_{}_{}.cpt"
    PATH = "checkpoints/genome_{}_{}.cpt".format(ARGS.save_p, ARCH)

    
    
    if(TRAIN):
        # generate_offline_training_data(102400)
        acc = train_attacker(0, PATH)
    elif(DEFENSE != 'none'):
        # construct_datasets(ARCH)
        defenses = []
        if(DEFENSE == 'rounding'):
            for i in range(10):
                defenses.append((i, "rounding to {} decimals".format(i), initialize_defense('rounding', decimals = i)))
            RESULTS = []
            for param, descript, _def in defenses:
                RESULT = dict()
                for arch in TEST_ARCHS:
                    print("EVALUATE {} With Defense {}".format(arch, descript))
                    RESULT[arch] = evaluate(TEMPLATE.format(ARGS.save_p, arch), arch, _def)
                    # RESULT.append()
                RESULTS.append((param, RESULT))
            print(RESULTS)         
        elif(DEFENSE == 'dp'):
            eps_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
            RESULTS = [dict() for _ in eps_list]
            for i, arch in enumerate(TEST_ARCHS):
                defenses = []
                for eps in eps_list:
                    defenses.append((eps, "laplace with eps {}".format(eps), initialize_defense("dp", delta = DELTA_TABLE[arch], eps = eps)))
                for j, defense in enumerate(defenses):
                    param, descript, _def = defense
                    print("Evaluate {} with Defense {}".format(arch, descript))
                    RESULTS[j][arch] = evaluate(TEMPLATE.format(ARGS.save_p, arch), arch, _def)
            RESULTS = [(eps_list[i], RESULTS[i]) for i in range(len(RESULTS))]
            print(RESULTS)
        elif(DEFENSE == 'minmax'):
            eps_list = [0.001, 0.005, 0.01, 0.1, 0.5, 1.0]
            RESULTS = [dict() for _ in eps_list]
            # defenses = []
            for i, arch in enumerate(TEST_ARCHS):
                defenses = []
                for eps in eps_list:
                    defenses.append((eps, "minmax with eps {}".format(eps), initialize_defense("minmax", cls_num = 4, eps = eps)))
                for j, defense in enumerate(defenses):
                    param, descript, _def = defense
                    print("Evaluate {} with Defense {}".format(arch, descript))
                    RESULTS[j][arch] = evaluate(TEMPLATE.format(ARGS.save_p, arch), arch, _def)
            RESULTS = [(eps_list[i], RESULTS[i]) for i in range(len(RESULTS))]
            print(RESULTS)
    else:
        evaluate(TEMPLATE.format(ARGS.save_p, ARGS.a), ARGS.a)
        # evaluate(PATH)

    # z, y = generate_offline_training_data(1024 * 100)

    # print(z.shape)
    # print(y.shape)


    
        

