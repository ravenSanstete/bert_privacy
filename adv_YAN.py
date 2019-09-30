import random
import os
import numpy as np
import torch
import torch.utils.data as data_utils
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.svm import SVC
from torch.nn import GRU, Embedding, Linear
from util import Embedder
from tools import balance
from tqdm import tqdm
from DANN import DANN
import argparse



parser = argparse.ArgumentParser(description='Medical Attack')
parser.add_argument("-p", type=int, default= 5555, help = 'the comm port the client will use')
parser.add_argument("-c", action='store_true', help = 'whether to use cached model')
parser.add_argument("-t", action='store_true', help = "to switch between training or testing")
parser.add_argument("--save_p", type=str, default="default", help = 'the place to store the model')
parser.add_argument("-a", type=str, default='bert', help = 'targeted architecture')
parser.add_argument("-d", type=str, default='none', help = 'the type of defense to do')
parser.add_argument("--clf", type = str, default='SVM', help = 'the type of attack model to use')
parser.add_argument("-v", action='store_true', help = 'whether to be wordy')
ARGS = parser.parse_args()



ARCH = ARGS.a
CLS = ARGS.clf


CLS_NUM = 10
VERBOSE = ARGS.v

#SVM parameter
SVM_KERNEL = 'linear'

# DANN parameter
MAXITER = 1000
BATCH_SIZE = 64
LAMDA = 1.0
HIDDEN = 25
# DANN_CPT_PATH = '/DATACENTER/data/yyf/Py/bert_privacy/data/part_fake_5/DANN_CPT/'
DANN_CACHED = False

# MLP parameter
CACHED = False
EPOCH = 1000
HIDDEN_DIM = 80
BATCH_SIZE = 15
LEARNING_RATE = 0.01
PRINT_FREQ = 100
K = 5


# CPT_PATH = '/DATACENTER/data/yyf/Py/bert_privacy/data/part_fake_5/MLP_CPT/'
CPT_PATH = 'data/part_fake_5/MLP_CPT/'
DANN_CPT_PATH = 'data/part_fake_5/DANN_CPT/'




DEVICE = torch.device('cuda:0')
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# LOCAL = '/DATACENTER/data/yyf/Py/bert_privacy/data/part_fake_4/'
DS_LOCAL = '/DATACENTER/data/pxd/bert_privacy/data/part_fake_5/'

DS_PATH = DS_LOCAL + '{}.{}'
DS_EMB_PATH = DS_LOCAL + '{}.{}'

TARGET_PATH = '/DATACENTER/data/yyf/Py/bert_privacy/data/medical.test.txt'
TARGET_EMB_PATH = '/DATACENTER/data/yyf/Py/bert_privacy/data/medical.test.x'

TRAIN_PATH = '/DATACENTER/data/yyf/Py/bert_privacy/data/medical.train.txt'
TRAIN_EMB_PATH = '/DATACENTER/data/yyf/Py/bert_privacy/data/medical.train.x'


P_TABLE = {
    "bert": 5001,
    "gpt": 768,
    "gpt2": 768,
    "xl": 1024,
    "xlnet": 5002,
    "xlm": 5004,
    "roberta":5003,
    "ernie": 5005
}

p = P_TABLE[ARCH]


EMB_DIM_TABLE = {
    "bert": 1024,
    "gpt": 768,
    "gpt2": 768,
    "xl": 1024,
    "xlnet": 768,
    "xlm": 1024,
    "roberta": 768,
    "ernie":768
}


cls_names = ["leg", "hand", "spine", "chest", "ankle", "head", "hip", "arm", "face", "shoulder"]

embedder = Embedder(p)
embedding = embedder.embedding # export the functional port

def data_embedding():
    f = open(TARGET_PATH, 'r')
    sents = [x[:-1] for x in f if x[:-1] != '']
    embs = embedding(sents, TARGET_EMB_PATH, ARCH)

    f = open(TRAIN_PATH, 'r')
    sents = [x[:-1] for x in f if x[:-1] != '']
    embs = embedding(sents, TRAIN_EMB_PATH, ARCH)

    for key in cls_names:
        for i in [0, 1]:
            f = open(DS_PATH.format(key, i) + '.txt', 'r')
            sents = [x[:-1] for x in f if x[:-1] != '']
            embs = embedding(sents, DS_EMB_PATH.format(key, i), ARCH)

def DANNA(key, size=2000):
    X, Y = [], []
    for i in [0, 1]:  # while my training data is from gpt
        f = open(DS_PATH.format(key, i) + '.txt', 'r')
        sents = [x[:-1] for x in f if x[:-1] != '']
        embs = embedding(sents, DS_EMB_PATH.format(key, i), ARCH)
        embs = embs[np.random.choice(len(embs), min(size, len(embs)), replace=False), :]
        X.append(embs)
        Y.extend([i] * embs.shape[0])
    X = np.concatenate(X, axis=0)
    Y = np.array(Y)

    train_embs = np.load(TRAIN_EMB_PATH + '.' + ARCH + '.npy')

    # load validation set (let us load gpt2)
    raw_valid, X_valid = list(open(TARGET_PATH, 'r')), np.load(TARGET_EMB_PATH + '.' + ARCH + '.npy')
    if (key != 'potato'):
        raw_valid, X_valid = balance(key, raw_valid, X_valid)
    print(len(raw_valid))
    Y_valid = np.array([(key in x) for x in raw_valid])

    # learn a transfer
    clf = DANN(input_size=EMB_DIM_TABLE[ARCH], maxiter=4000, verbose=False, name=key, batch_size=BATCH_SIZE,
               lambda_adapt=LAMDA, hidden_layer_size=HIDDEN)
    acc = clf.fit(X, Y, X_adapt=train_embs, X_valid=X_valid, Y_valid=Y_valid)

    return acc

class NonLinearClassifier(nn.Module):
    def __init__(self, key, embedding_size, hidden_size, cls_num=2, device=DEVICE):
        super(NonLinearClassifier, self).__init__()
        self.fc1 = Linear(embedding_size, hidden_size)
        self.fc2 = Linear(hidden_size, cls_num)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.key = key

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = F.softmax(self.fc2(x))  # , dim=0)
        return x

    def predict(self, x):
        x = torch.FloatTensor(x)
        # print(x)
        outputs = self(x.cuda())
        # print(outputs)
        _, preds = torch.max(outputs, 1)

        # print(preds)
        return preds.cpu().numpy()

    def predict_topk(self, x, k=5):
        with torch.no_grad():
            probs = self(x)
            _, topk = torch.topk(probs, k)
        return topk.cpu().numpy()

    def loss(self, x, y):
        x = self(x)
        _loss = self.criterion(x, y)
        return _loss

    def _evaluate(self, x, y):
        preds = self.predict(x)
        # y = y.numpy()
        return np.mean(preds == y)

    def _evaluate_topk(self, x, y, k = K):
        # y = y.numpy()
        with torch.no_grad():
            probs = self(x)
            _, topk = torch.topk(probs, k)
            topk = topk.cpu().numpy()
            acc = [int(y[i] in topk[i, :]) for i in range(len(y))]
        return np.mean(acc)

    def fit(self, X, Y, epoch_num=EPOCH):  # 2000, 4000
        y_cpu = Y.copy()
        self.cuda()
        X = torch.FloatTensor(X)
        Y = torch.LongTensor(Y)

        if (CACHED and os.path.exists(CPT_PATH + "{}_cracker_{}.cpt".format(self.key, ARCH))):
            # print("Loading Model...")
            self.load_state_dict(torch.load(CPT_PATH + "{}_cracker_{}.cpt".format(self.key, ARCH)))
            preds = self.predict(X)
            correct = np.sum(preds == y_cpu)
            correct = correct / len(y_cpu)
            print("Source Domain batch Acc.: {:.4f}".format(correct))
            return

        ds = data_utils.TensorDataset(X, Y)
        train_loader = data_utils.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
        counter = 0
        best_acc = 0.0

        for epoch in tqdm(range(epoch_num)):
            running_loss = 0.0
            criterion = nn.CrossEntropyLoss()
            # optimizer = optim.SGD(self.parameters(), lr=0.1, momentum = 0.9)
            optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
            for i, data in enumerate(train_loader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                counter += 1

            if ((epoch + 1) % 100 == 0):
                print('Epoch %d loss: %.5f Count: %d' % (epoch + 1, running_loss, counter))
                running_loss = 0.0
                counter = 0
                preds = self.predict(X)
                correct = np.sum(preds == y_cpu)
                correct = correct / len(y_cpu)
                print("Source Domain batch Acc.: {:.4f}".format(correct))

                top1 = early_stopping_evaluate(self, self.key)
                print("Early stopping Acc.: {:4f}".format(top1))
                if (top1 >= best_acc):
                    best_acc = top1
                    torch.save(self.state_dict(),CPT_PATH + "{}_cracker_{}.cpt".format(self.key, ARCH))

        print("Early stopping set Infer {} Best acc top1. {:.4f}".format(self.key, best_acc))

def early_stopping_evaluate(clf, key):
    # load the early stopping set
    f = open(TARGET_PATH, 'r')
    target_f = [x[:-1] for x in f if x[:-1] != '']
    f.close()
    target_embs = embedding(target_f, TARGET_EMB_PATH, ARCH)
    target_f, target_embs = balance(key, target_f, target_embs)

    results = np.zeros((2, 2))
    count = 0

    for i, sent in enumerate(list(target_f)):
        pred_ = clf.predict([target_embs[i]])[0]
        truth_ = int(key in sent)
        results[pred_][truth_] += 1
        count += 1
    results /= (count * 1.0)
    acc = results[0][0] + results[1][1]
    # print("early_stopping set Inference {} Acc: {:.3f}".format(key, results[0][0] + results[1][1]))
    return acc

def ATTACK(key, use_dp=False, dp_func=None, verbose=VERBOSE, size = 2000):
    # (X, Y) is from external corpus.
    # X are sentence embeddings. Y are labels.
    # To prepare an external corpus, we substitute the food keywords in Yelp dataset to body keywords.
    X, Y = [], []
    for i in [0, 1]:  # while my training data is from gpt
        f = open(DS_PATH.format(key, i) + '.txt', 'r')
        sents = [x[:-1] for x in f if x[:-1] != '']
        embs = embedding(sents, DS_EMB_PATH.format(key, i), ARCH)
        embs = embs[np.random.choice(len(embs), min(size, len(embs)), replace=False), :]
        X.append(embs)
        Y.extend([i] * embs.shape[0])
    X = np.concatenate(X, axis=0)
    Y = np.array(Y)

    # (Target_sents, Target_X) is from target domain.
    # Target_X are sentence embeddings. Target_sents are original sentences.
    f = open(TRAIN_PATH, 'r')
    Target_sents = [x[:-1] for x in f if x[:-1] != '']
    f.close()
    Target_X = embedding(Target_sents, TRAIN_EMB_PATH, ARCH)
    Target_sents, Target_X = balance(key, Target_sents, Target_X)
    Target_Y = np.array([(key in x) for x in Target_sents])

    # (X_valid, Y_valid) is from valid set.
    # SVM: This is regarded as shadow corpus of Target domain.
    # DANN or MLP: This is used to early stop.
    # X_valid are sentence embeddings. Y_valid are labels.
    raw_valid, X_valid = list(open(TARGET_PATH, 'r')), np.load(TARGET_EMB_PATH + '.' + ARCH + '.npy')
    raw_valid, X_valid = balance(key, raw_valid, X_valid)
    Y_valid = np.array([(key in x) for x in raw_valid])

    # learn a transfer
    print("The current CLS: {}".format(CLS))
    if CLS == 'MLP':
        clf = NonLinearClassifier(key, EMB_DIM_TABLE[ARCH], HIDDEN_DIM)
        clf.fit(X, Y)
        print("here")
    elif CLS == 'SVM':
        clf = SVC(kernel='{}'.format(SVM_KERNEL), gamma='scale', verbose=False)
        clf.fit(X_valid, Y_valid)
    elif CLS == 'DANN':
        DANN_CPT_PATHs = DANN_CPT_PATH + "{}_cracker_{}.cpt".format(key, ARCH)
        clf = DANN(input_size=EMB_DIM_TABLE[ARCH], maxiter=MAXITER, verbose=False, name=key, batch_size=BATCH_SIZE, lambda_adapt=LAMDA, hidden_layer_size=HIDDEN, cached = DANN_CACHED, cpt_path = DANN_CPT_PATHs)
        clf.fit(X, Y, X_adapt=Target_X, X_valid=X_valid, Y_valid=Y_valid)
        Target_X = torch.FloatTensor(Target_X)
        acc = clf.validate(Target_X, Target_Y)
        print("Target Domain Inference {} Acc: {:.3f}".format(key, acc))
        return acc
    else:
        clf = None
        print('wrong cls\' name')


    # predict on Target_X
    results = np.zeros((2, 2))
    count = 0
    for i, sent in enumerate(list(Target_sents)):
        pred_ = int(clf.predict([Target_X[i]])[0])
        truth_ = int(key in sent)
        results[pred_][truth_] += 1
        count += 1
    results /= (count * 1.0)
    acc = results[0][0] + results[1][1]
    print("Target Domain Inference {} Acc: {:.3f}".format(key, results[0][0] + results[1][1]))
    return acc

if __name__ == '__main__':
    # DS_prepare()
    # EX_DS_prepare()

    Source_Acc_sum = 0
    Target_Acc_sum = 0
    Target_Acc_list = []
    data_embedding()

    for key in cls_names:
        TA = ATTACK(key)
        Target_Acc_sum += TA
        Target_Acc_list.append([key, TA])

    print('Keyword Attacker {} on {} Ebeddings'.format(CLS, ARCH))
    for KT in Target_Acc_list:
        print('INFER {} ACC: %.4f'.format(KT[0])%KT[1])
    print('Target_Acc_Top1_Average: %.3f'%(Target_Acc_sum / len(cls_names)))

