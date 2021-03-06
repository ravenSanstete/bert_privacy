# create an adversary to infer keywords from embeddings of medical sentences
from util import embedding
from tools import balance
from sklearn.svm import SVC
from sklearn import linear_model
import numpy as np
from tqdm import tqdm
from scipy.stats import describe
import argparse
import torch
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA, KernelPCA, FastICA
from main import MODEL_SAVE_PATH, get_dataloader, evaluate as evaluate_main, CLS_NUM, MODEL_MAP, NonLinearClassifier, INPUT_DIM, total_acc
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns; sns.set_style("white")
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
import torch.optim as optim
import torch.utils.data as data_utils
from sklearn.semi_supervised import label_propagation
from sklearn.neighbors import KNeighborsClassifier
from TSVM import TSVM
from DANN import DANN


parser = argparse.ArgumentParser(description='Privacy Testbed for Setence Embedding Services')
parser.add_argument('--arch', type = str, default = 'bert')
parser.add_argument('--nonlinear', action='store_true')
parser.add_argument('--truth', action = 'store_true', default = False)
parser.add_argument('--prefix', type=str, default = 'part_fake_2')
parser.add_argument('--pca', action = 'store_true')
args = parser.parse_args()

KEY = 'shoulder'
PREFIX = '/DATACENTER/data/pxd/bert_privacy/data/{}/'.format(args.prefix)

PATH = PREFIX + '{}.{}.txt'
EMB_PATH = PREFIX + '{}.{}'
ARCH = args.arch
MODEL = "linear"
# WORDS = ["leg", "hand", "spine", "chest", "ankle", "head", "hip", "arm", "face", "shoulder", "potato"]
WORDS = ["leg", "hand", "spine", "chest", "ankle", "head", "hip", "arm", "face", "shoulder"]

# WORDS = ["potato"]

# WORDS = WORDS[:-1]
# WORDS = ["potato"]
# WORDS = ["hand"]

IS_BALANCED = False
PRINT_SOURCE = True
VERBOSE = True
GROUND_TRUTH = args.truth
NONLINEAR = True # args.nonlinear
IS_SEMI = True

print("TESTING {}".format(ARCH))
DO_PCA = False # args.pca

if(GROUND_TRUTH):
    DO_PCA = False

DO_TRANSFER = True

scenario = "medical"

if scenario == "daily":
    TARGET_PATH = '/DATACENTER/data/pxd/bert_privacy/data/part/daily.train.txt'
    TARGET_EMB_PATH = '/DATACENTER/data/pxd/bert_privacy/data/part/daily.train.x.{}.npy'.format(ARCH)
    TRAIN_PATH = TARGET_PATH
    TRAIN_EMB_PATH = TARGET_EMB_PATH
elif scenario == 'medical':
    TARGET_PATH = '/DATACENTER/data/pxd/bert_privacy/data/medical.test.txt'
    TARGET_EMB_PATH = '/DATACENTER/data/pxd/bert_privacy/data/medical.test.x.{}.npy'.format(ARCH)
    TRAIN_PATH = '/DATACENTER/data/pxd/bert_privacy/data/medical.train.txt'
    TRAIN_EMB_PATH = '/DATACENTER/data/pxd/bert_privacy/data/medical.train.x.{}.npy'.format(ARCH)

    
EMB_DIM_TABLE = {
    "bert": 1024,
    "gpt": 768,
    "gpt2": 768,
    "xl": 1024
    }

EMB_DIM = EMB_DIM_TABLE[ARCH]
    



            
    

def visualize(key):
    X = []
    Y = []
    num = 0
    print("extract embedding inform\n")
    for i in [0, 1]:
        f = open(PATH.format(key, i), 'r')
        sents = [x[:-1] for x in f if x[:-1] != '']
        embs = embedding(sents, EMB_PATH.format(key, i), ARCH)
        X.append(embs)
        num = embs.shape[0]
        Y.extend([i]*embs.shape[0])
        

    # reformat the data
    X = np.concatenate(X, axis = 0)
    print(X.shape)
    Y = np.array(Y)
    pca = PCA(n_components=3)
    mds = MDS(n_components=3)
    X = mds.fit_transform(X)

    # plot
    print(X.shape)
    fig, ax = plt.subplots(1, 1)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X[:num, 0], X[:num, 1], X[:num, 2], c = 'b')
    ax.scatter(X[num:, 0], X[num:, 1], X[num:, 2], c = 'g')
    plt.savefig('visual/{}.{}.mds3.png'.format(key, ARCH))
    return


def train_atk_classifier(key, size = 1900):
    pca = None
    X_train, Y_train = [], []

    for i in [0, 1]:
        f = open(PATH.format(key, i), 'r')
        sents = [x[:-1] for x in f if x[:-1] != '']
        embs = embedding(sents, EMB_PATH.format(key, i), ARCH)
        if args.prefix != 'part':
            embs = embs[np.random.choice(len(embs), size, replace = False), :]
        X_train.append(embs)
        Y_train.extend([i]*embs.shape[0])
    X_train = np.concatenate(X_train, axis = 0)
    Y_train = np.array(Y_train)
    train_embs = np.load(TRAIN_EMB_PATH)

    # BottleNeck
    # X_train = np.load(TRAIN_EMB_PATH)
    # raw_train = list(open(TRAIN_PATH, 'r'))
    # if IS_BALANCED:
        # raw_train, X_train = balance(key, raw_train, X_train)
    # Y_train = np.array([(key in x) for x in raw_train])

    # load validation set

    raw_valid, X_valid = list(open(TARGET_PATH, 'r')), np.load(TARGET_EMB_PATH)
    if(key != 'potato' and IS_BALANCED):
        raw_valid, X_valid = balance(key, raw_valid, X_valid)
    print(len(raw_valid))
    Y_valid = np.array([(key in x) for x in raw_valid])
    acc = -1
    # learn a transfer

    # clf = linear_model.SGDClassifier(max_iter = 1000,  verbose = 0)
    # clf = SVC(kernel = 'rbf', gamma = 'scale', verbose = False)
    # clf = KNeighborsClassifier(n_neighbors=1, p = 1)
    if(NONLINEAR):
        # clf = DANN(input_size = EMB_DIM, maxiter = 2000, verbose = False, name = key, batch_size = 128)
        clf = DANN(input_size=EMB_DIM, maxiter=4000, verbose=True, name=key, batch_size=64, lambda_adapt=1.0,
               hidden_layer_size=25)
        acc = clf.fit(X_train, Y_train, X_adapt = train_embs, X_valid = X_valid, Y_valid = Y_valid)
        print("DANN Acc.: {:.4f}".format(acc))
    # train_embs = train_embs[np.random.choice(len(train_embs), 2000), :]

    # # apply pca first
    # if(DO_PCA):
        # train_embs = train_embs[np.random.choice(len(train_embs), size = 6 * int(len(X_train)), replace = False)]
        # package = np.concatenate([X_train, train_embs], axis = 0)
        # pca = PCA(n_components=INPUT_DIM)
        # pca.fit(package)
        # X_train, train_embs = pca.transform(X_train), pca.transform(train_embs)

    # if NONLINEAR:
        # clf = NonLinearClassifier(key, ARCH, cls_num = 2, pca = pca, use_pca = DO_PCA)


    # clf.fit(X_train, Y_train)


    if NONLINEAR:
        clf.to(torch.device('cpu'))
    # on current set
    # correct = 0
    if(VERBOSE):
        print("TRAIN INFERENCE MODEL FROM EXTERNAL SOURCES (# = {})".format(len(X_train)))
        correct = np.sum((clf.predict(X_train) == Y_train))
        print("Source Domain Acc.: {:.4f}".format(correct/len(Y_train)))
    return clf, pca, acc
    

# given 200 sentences 
def train_ground_truth_classifier(key, size = 2000):
    X_0, X_1, Y= list(),list(),list()
    train_embs = np.load(TRAIN_EMB_PATH)
    c = 0
    for i, s in enumerate(open(TRAIN_PATH, 'r')):
        if(len(X_1) > size and len(X_0) > size):
            break
        if(key in s):
            X_1.append(train_embs[i, :])
        else:
            X_0.append(train_embs[i, :])
    _size = size
    size = min(len(X_0), len(X_1))
    signal = False
    if(size == 0):
        size = _size
        signal = True
        X_1 = [np.zeros((EMB_DIM))]
    X_0, X_1 = [random.choice(X_0) for i in range(size)], X_1[:size]
    X = X_0 + X_1
    Y.extend([0]*size)
    if not signal: 
        Y.extend([1]*size)
    else: # deal with non-existant words
        Y.extend([1])
    X, Y = np.array(X), np.array(Y)
    clf = SVC(kernel = 'linear', gamma = 'auto')
    
    clf.fit(X, Y)

    
    # clf.to(torch.device('cpu')) 
    # on current set
    correct = 0
    if(VERBOSE):
        print("TRAIN INFERENCE MODEL FROM GROUND TRUTH (# = {})".format(len(X)))
        correct = np.sum((clf.predict(X) == Y))
        print("Source Domain Acc.: {:.4f}".format(correct/len(Y)))
    return clf, None, correct/len(Y)
    



def evaluate(clf, key, use_dp = False, dp_func = None, is_balanced = IS_BALANCED, pca = None, transfer_func = None):
    # load the target set
    target_f = list(open(TARGET_PATH, 'r'))
    target_embs = np.load(TARGET_EMB_PATH)
    # if the flag use_dp is true, then apply the given mechanism to the target embedding


    
    if(use_dp):
        target_embs = dp_func(target_embs)
        
    if(is_balanced):
        target_f, target_embs = balance(key, target_f, target_embs)

    if(DO_PCA):
        target_embs = pca.transform(target_embs)
    # if(VERBOSE):
    #    print("TARGET SAMPLE NUM OF {}".format(len(target_f)))
    #    print("TARGET EMBS SHAPE {}".format(target_embs.shape))
    results = np.zeros((2,2))
    count = 0
    for i, sent in enumerate(list(target_f)):
        pred_ = clf.predict([target_embs[i]])[0]
        truth_ = int(key in sent)
        results[pred_][truth_] += 1
        count += 1

    results /= (count * 1.0)
    acc = results[0][0] + results[1][1]
    # print("Inference Accuracy: {:.3f}".format(results[0][0] + results[1][1]))
    # print("Details:")
    # print(results)
    return acc

        
    

def main(key = KEY, use_dp = False, dp_func = None, is_balanced = IS_BALANCED):
    # clf = train_atk_classifier(KEY)

    clf, pca, acc = train_ground_truth_classifier(key) if GROUND_TRUTH else train_atk_classifier(key)
    return evaluate(clf, key, use_dp, dp_func, is_balanced, pca)




 # size experiments
def size_experiments():
    pts = [1, 2, 5, 10, 50]
    pts.extend(list(range(100, 1200, 100)))
    accs = []
    for num in tqdm(pts):
        acc = []
        for k in WORDS:
            clf = train_ground_truth_classifier(k, num)
            acc.append(evaluate(clf, k))
            if(VERBOSE):
                print("INFER {}\tAcc. {:.3f}".format(k, acc[-1]))
        acc = np.mean(acc)
        accs.append(acc)
        print("AVG ACC {} NUM {}".format(acc, num))
    print(accs)
    return
        
        



def attacker_utility(use_dp, dp_func, is_balanced = IS_BALANCED):
    print("Whether Balanced: {}".format(is_balanced))
    acc = []
    for i, k in enumerate(WORDS):
        if(i >= 10):
            is_balanced = False
        acc.append(main(k, use_dp, dp_func, is_balanced))
        if(VERBOSE):
            print("INFER {}\tAcc. {:.3f}".format(k, acc[-1]))
    avg_acc = np.mean(acc)
    print("Average Acc. {:.4f}".format(avg_acc))
    # print("Expected Bottleneck Acc. {:.4f}".format(total_acc))
    return avg_acc

def user_utility(use_dp, dp_func):
    user_model = MODEL_MAP[MODEL]() # linear or non-linear
    user_model.load_state_dict(torch.load(MODEL_SAVE_PATH.format(ARCH, MODEL)))
    user_model = user_model.cuda()
    user_model.eval()
    print(user_model)

    emb_path = "/DATACENTER/data/pxd/bert_privacy/data/medical.{}.{}.{}.npy"
    X = np.load(emb_path.format("test", 'x', ARCH))
    if(use_dp):
        X = dp_func(X)
    X = torch.FloatTensor(X)
    Y = torch.LongTensor(np.load(emb_path.format("test", 'y','univ')))
    ds = data_utils.TensorDataset(X, Y)
    test_loader = data_utils.DataLoader(ds, batch_size = 64, shuffle = True, pin_memory = True, num_workers = 4)
    acc = evaluate_main(test_loader, user_model) / 100.0
    print(acc)
    return acc


# an entry point for testing the attacker's utility 
def attack_test_without_dp():
    rep = 1
    acc = []
    for k in WORDS:
        avg_acc = 0.0
        # for i in tqdm(range(rep)):
        acc.append(main(KEY = k))
        avg_acc += (acc[-1]/rep)
        print("INFER {}\tAcc. {:.3f}".format(k, avg_acc))
        # print("Statistics:")
        # print(describe(acc))

# used to estimate the sensitivity of bert for appplying Laplace mechanism
def estimate_sensitivity_mean():
    delta = 0.0
    total_count = 0
    for k in WORDS:
        x = np.load(EMB_PATH.format(k, 0)+ '.' +ARCH +".npy")
        y = np.load(EMB_PATH.format(k, 1)+ '.' + ARCH +".npy")
        sense_per_sample = np.linalg.norm(x-y, ord = 1, axis = 0)
        delta += np.sum(sense_per_sample)
        total_count += sense_per_sample.shape[0]
    return delta / total_count

# used to estimate the sensitivity of bert for appplying Laplace mechanism
def estimate_sensitivity_max():
    delta = 0.0
    for k in WORDS:
        x = np.load(EMB_PATH.format(k, 0)+  '.' +ARCH + ".npy")
        y = np.load(EMB_PATH.format(k, 1)+  '.' +ARCH + ".npy")
        sense_per_sample = np.linalg.norm(x-y, ord = 1, axis = 0)
        delta = max(delta, np.max(sense_per_sample))
    return delta

# used to estimate the sensitivity of bert for appplying Laplace mechanism
def estimate_sensitivity_min():
    delta = 100.0
    for k in WORDS:
        x = np.load(EMB_PATH.format(k, 0)+  '.' +ARCH + ".npy")
        y = np.load(EMB_PATH.format(k, 1)+  '.' +ARCH + ".npy")
        sense_per_sample = np.linalg.norm(x-y, ord = 1, axis = 0)
        delta = min(delta, np.min(sense_per_sample))
    return delta

def init_laplace(delta, eps):
    b = delta / eps
    def func(x):
        perturb = np.random.laplace(loc = 0.0, scale = b, size = x.shape)
        return x + perturb
    return func
        

def attack_test_with_laplace_dp(start = 1.0, end = 100.0, num = 100):
    delta = estimate_sensitivity_mean()
    print("Estimated L1 Sensitivity:\t{}".format(delta))
    # x = embedding(None, EMB_PATH.format("head", 0))
    atk_utils_b, atk_utils_imb, usr_utils = [], [], []
    for eps in tqdm(np.linspace(start, end, num = num)):
        dp_func = init_laplace(delta, eps)
        atk_utils_b.append(attacker_utility(True, dp_func, True))
        atk_utils_imb.append(attacker_utility(True, dp_func, False))
        usr_utils.append(user_utility(True, dp_func))
    print("atk_utils_b")
    print(atk_utils_b)
    print("atk_utils_imb")
    print(atk_utils_imb)
    print("usr_utils")
    print(usr_utils)
    # print(usr_utils)
    # dp_func = init_laplace(delta, eps)
    # atk_util = attacker_utility(True, dp_func)
    # usr_util = user_utility(True, dp_func)
    # print(usr_util)
    
    


if __name__ == '__main__':
    # print(estimate_sensitivity_min())
    # attack_test()
    # delta = estimate_sensitivity_mean()
    # print(delta)
    # user_utility(False, None)

    attacker_utility(False, None, IS_BALANCED)

    # attacker_utility(False, None, True)
    # attack_test_with_laplace_dp(20.,60.,4)
    # print(attacker_utility(False, None))
    # print(attacker_utility(False, None))
    # print(user_utility(False, None))
    # attack_test_with_laplace_dp(1.0, 100.0, 20.0)
    # for k in WORDS:
        # visualize(k)
    # size_experiments()
 
    
    


        
    

