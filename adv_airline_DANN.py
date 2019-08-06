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
from util import embedding
from tools import balance
from tqdm import tqdm
from DANN import DANN


ARCH = "bert"
CLS_NUM = 10
VERBOSE = False
IS_BALANCED = True
GROUND_TRUTH = False
NONLINEAR = True
USE_DANN = True
KEY = 'Hong Kong'

# MLP
HIDDEN_DIM = 100
BATCH_SIZE = 10
CACHED = False
MAX_ITER = 20000
PRINT_FREQ = 100
DEVICE = torch.device('cuda:0')
TEST_SIZE = 1000
best_acc = 0.0
K = 5

# file path
PATH = "/DATACENTER/data/pxd/bert_privacy/data/skytrax-reviews-dataset"
Wiki_DS_PATH= '/DATACENTER/data/yyf/Py/bert_privacy/data/Airline/part/wiki/{}.{}'
# DS_PATH = '/DATACENTER/data/yyf/Py/bert_privacy/data/Airline/part/train.{}.{}'
DS_PATH = '/DATACENTER/data/yyf/Py/bert_privacy/data/Airline/EX_part/train.{}.{}'
TARGET_PATH = '/DATACENTER/data/yyf/Py/bert_privacy/data/Airline/Target/test.txt'
TARGET_EMB_PATH = DS_PATH + '.' + ARCH + '.npy'


EMB_DIM_TABLE = {
    "bert": 1024,
    "gpt": 768,
    "gpt2": 768,
    "xl": 1024
    }

trainDS_cls_names = {
    'Hong Kong',
    'London',
    'Toronto',
    'Paris',
    'New York',
    'Chicago',
    'Rome',
    'Sydney',
    'Dubai',
    'LA',
    'Beijing',
    'Shanghai',
    'Bangkok',
    'BKK',
    'Tokyo',
    'Moscow',
    'Dublin'
}

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

def DS_prepare():
    for key in cls_names:
        f = open(Wiki_DS_PATH.format(key, 1) + '.txt', 'r')
        w1 = open(DS_PATH.format(key, 1) + '.txt', 'w')
        w0 = open(DS_PATH.format(key, 0) + '.txt', 'w')
        for x in f:
            if key in x:
                w1.write(x)
                w1.write('\n')
                x = x.replace(key, random.choice(list(cls_names)))
                w0.write(x)
                w0.write('\n')
        f.close()
        w1.close()
        w0.close()

class LinearClassifier(nn.Module):
    def __init__(self):
        super(LinearClassifier, self).__init__()
        self.fc1 = nn.Linear(EMB_DIM_TABLE[ARCH], CLS_NUM)

    def forward(self, x):
        x = self.fc1(x)
        return x

    def predict(self, x):
        with torch.no_grad():
            outputs = self(x)
            _, predicted = torch.max(outputs.data, 1)
        return predicted


class NonLinearClassifier(nn.Module):
    def __init__(self, embedding_size, hidden_size, cls_num = 2, device=torch.device('cuda:0')):
        super(NonLinearClassifier, self).__init__()
        self.fc1 = Linear(embedding_size, hidden_size)
        self.fc2 = Linear(hidden_size, cls_num)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = F.softmax(self.fc2(x))#, dim=0)
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

    def fit(self, X, Y, epoch_num=4000):  # 2000, 4000
        y_cpu = Y.copy()
        self.cuda()
        X = torch.FloatTensor(X)
        Y = torch.LongTensor(Y)

        if (CACHED and os.path.exists("{}_cracker.cpt".format(KEY))):
            # print("Loading Model...")
            self.load_state_dict(torch.load("{}_cracker.cpt".format(KEY)))
            preds = self.predict(X)
            correct = np.sum(preds == y_cpu)
            correct = correct / len(y_cpu)
            # print("Source Domain batch Acc.: {:.4f}".format(correct))
            return

        ds = data_utils.TensorDataset(X, Y)
        train_loader = data_utils.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
        counter = 0
        best_acc = 0.0

        for epoch in tqdm(range(epoch_num)):
            running_loss = 0.0
            criterion = nn.CrossEntropyLoss()
            # optimizer = optim.SGD(self.parameters(), lr=0.1, momentum = 0.9)
            optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)
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

                top1 = evaluate(self, KEY)
                print("Target Domain Acc.: {:4f}".format(top1))
                if (top1 >= best_acc):
                    best_acc = top1
                    torch.save(self.state_dict(), "{}_cracker.cpt".format(KEY))
        print("Target Domain Infer {} Best acc top1. {:.4f}".format(KEY, best_acc))

# given Wiki dataset
def train_atk_classifier(key, size=110, verbose = VERBOSE):
    X_train, Y_train = [], []

    # get training dataset
    for i in [0, 1]:
        f = open(DS_PATH.format(key, i) + '.txt', 'r')
        sents = [x[:-1] for x in f if x[:-1] != '']
        embs = embedding(sents, DS_PATH.format(key, i), ARCH)
        embs = embs[np.random.choice(len(embs), size, replace=False), :]
        X_train.append(embs)
        Y_train.extend([i] * embs.shape[0])
        f.close()
    X_train = np.concatenate(X_train, axis=0)
    Y_train = np.array(Y_train)

    # define clf
    if NONLINEAR:
        clf = NonLinearClassifier(EMB_DIM_TABLE[ARCH], HIDDEN_DIM)
        clf.to(torch.device('cpu'))
    else:
        clf = SVC(kernel='linear', gamma='scale', verbose=False)
        # clf = LinearClassifier(EMB_DIM_TABLE[ARCH], HIDDEN_DIM, CLS_NUM)

    clf.fit(X_train, Y_train)

    Source_Acc = 0
    if (verbose):
        print("TRAIN INFERENCE MODEL FROM EXTERNAL(Wiki) SOURCES (# = {})".format(len(X_train)))
        correct = np.sum((clf.predict(X_train) == Y_train))
        Source_Acc = correct / len(Y_train)
        print("Source Domain(Wiki) infers #{}# Acc.: {:.4f}".format(key, Source_Acc))
    return clf, Source_Acc


def evaluate(clf, key, use_dp=False, dp_func=None,is_balanced = IS_BALANCED, verbose = VERBOSE):
    # load the target set
    f = open(TARGET_PATH, 'r')
    target_f = [x[:-1] for x in f if x[:-1] != '']
    f.close()
    # print("Waiting Embedding...")
    target_embs = embedding(target_f, DS_PATH, ARCH)
    # print("Embedding Finished.")

    if (use_dp):
        target_embs = dp_func(target_embs)
    if (is_balanced):
        target_f, target_embs = balance(key, target_f, target_embs)

    results = np.zeros((2, 2))
    count = 0
    f = open("/DATACENTER/data/yyf/Py/bert_privacy/data/Airline/MLP_PRE/pre.txt",'a')
    f.write(str(KEY))
    for i, sent in enumerate(list(target_f)):
        pred_ = clf.predict([target_embs[i]])[0]
        # f.write(str(pred_))
        truth_ = int(key in sent)
        results[pred_][truth_] += 1
        count += 1
    f.close()
    results /= (count * 1.0)
    acc = results[0][0] + results[1][1]
    print("Target Domain Inference {} Acc: {:.3f}".format(key, results[0][0] + results[1][1]))
    return acc

def use_DANN(key):
    # X_train, Y_train
    X_train, Y_train = [], []
    # get training dataset
    for i in [0, 1]:
        f = open(DS_PATH.format(key, i) + '.txt', 'r')
        sents = [x[:-1] for x in f if x[:-1] != '']
        embs = embedding(sents, DS_PATH.format(key, i), ARCH)
        embs = embs[np.random.choice(len(embs), 110, replace=False), :]
        X_train.append(embs)
        Y_train.extend([i] * embs.shape[0])
        f.close()
    X_train = np.concatenate(X_train, axis=0)
    Y_train = np.array(Y_train)

    # X_valid, Y_valid
    raw_valid, X_valid = list(open(TARGET_PATH, 'r')), np.load(TARGET_EMB_PATH)
    X_valid_b = X_valid
    if (IS_BALANCED):
        raw_valid, X_valid = balance(key, raw_valid, X_valid)
    Y_valid = np.array([(key in x) for x in raw_valid])

    clf = DANN(input_size=EMB_DIM_TABLE[ARCH], maxiter=4000, verbose=False, name=key, batch_size=32, lambda_adapt=1.0,
               hidden_layer_size=64)

    # How to chose X_adapt? X_valid(after/before balanced),
    acc = clf.fit(X_train, Y_train, X_adapt=X_valid, X_valid=X_valid, Y_valid=Y_valid)
    return acc



def main(key = KEY, use_dp = False, dp_func = None, is_balanced = IS_BALANCED):
    clf, Source_Acc = train_atk_classifier(key)
    Target_Acc = evaluate(clf, key, use_dp, dp_func, is_balanced)
    return Source_Acc, Target_Acc


if __name__ == '__main__':
    # DS_prepare()
    Source_Acc_sum = 0
    Target_Acc_sum = 0

    if USE_DANN:
        for key in cls_names:
            TA = use_DANN(key)
            Target_Acc_sum += TA
    else:
        for key in cls_names:
            KEY = key
            SA, TA = main(key = KEY, is_balanced = IS_BALANCED)
            Source_Acc_sum += SA
            Target_Acc_sum += TA
        # print('test.txt IS BALANCED = {}'.format(IS_BALANCED))
        # print('Source_Acc_Average: {}'.format(Source_Acc_sum / len(cls_names)))
    print('Target_Acc_Top1_Average: {}'.format(Target_Acc_sum / len(cls_names)))
