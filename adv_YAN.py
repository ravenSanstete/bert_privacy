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
from defense import initialize_defense





parser = argparse.ArgumentParser(description='Medical Attack')
parser.add_argument("-p", type=int, default= 5555, help = 'the comm port the client will use')
parser.add_argument("-c", action='store_true', help = 'whether to use cached model')
parser.add_argument("-t", action='store_true', help = "to switch between training or testing")
# parser.add_argument("--save_p", type=str, default="default", help = 'the place to store the model')
parser.add_argument("-a", type=str, default='bert', help = 'targeted architecture')
parser.add_argument("-d", type=str, default='none', help = 'the type of defense to do')
parser.add_argument("--clf", type = str, default='SVM', help = 'the type of attack model to use')
parser.add_argument("-v", action='store_true', help = 'whether to be wordy')
parser.add_argument("-f", type=str, default='atk', help = 'to specify the functional')

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

FUNCTION = 'atk'

# DANN_CPT_PATH = '/DATACENTER/data/yyf/Py/bert_privacy/data/part_fake_5/DANN_CPT/'
# DANN_CACHED = False

# MLP parameter

EPOCH = 50
HIDDEN_DIM = 80
BATCH_SIZE = 15
LEARNING_RATE = 0.001
PRINT_FREQ = 100
K = 5
DATASET = 'skytrax'

# CPT_PATH = '/DATACENTER/data/yyf/Py/bert_privacy/data/part_fake_5/MLP_CPT/'
# CPT_PATH = 'data/part_fake_5/MLP_CPT/'



if(not ARGS.t):
    DANN_CPT_PATH = 'data/part_fake_5/DANN_CPT/'
    DANN_CACHED = False
    CPT_PATH = 'data/part/MLP_CPT/'
    CACHED = False
else: # toggle it to use Yan's pretrained model
    DANN_CPT_PATH = '/DATACENTER/data/yyf/Py/bert_privacy/data/part_fake_5/DANN_CPT/'
    CPT_PATH = '/DATACENTER/data/yyf/Py/bert_privacy/data/part_fake_5/MLP_CPT/'
    DANN_CACHED = True
    CACHED = True



    
    
    




DEVICE = torch.device('cuda:0')
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


if(DATASET == 'medical'):
    # LOCAL = '/DATACENTER/data/yyf/Py/bert_privacy/data/part_fake_4/'
    DS_LOCAL = '/DATACENTER/data/pxd/bert_privacy/data/part_fake_5/'

    DS_PATH = DS_LOCAL + '{}.{}'
    DS_EMB_PATH = DS_LOCAL + '{}.{}'

    TARGET_PATH = '/DATACENTER/data/yyf/Py/bert_privacy/data/medical.test.txt'
    TARGET_EMB_PATH = '/DATACENTER/data/yyf/Py/bert_privacy/data/medical.test.x'
    # TARGET_EMB_PATH = 'data/medical.test.x'

    TRAIN_PATH = '/DATACENTER/data/yyf/Py/bert_privacy/data/medical.train.txt'
    TRAIN_EMB_PATH = '/DATACENTER/data/yyf/Py/bert_privacy/data/medical.train.x'
    # TRAIN_EMB_PATH = 'data/medical.train.x'

    cls_names = ["leg", "hand", "spine", "chest", "ankle", "head", "hip", "arm", "face", "shoulder"]
else:
    cls_names = ['Hong Kong','London','Toronto','Paris','Rome','Sydney','Dubai','Bangkok','Singapore','Frankfurt']
    TARGET_PATH = '/DATACENTER/data/yyf/Py/bert_privacy_Yan/data/Airline/Target/valid.txt'
    TARGET_EMB_PATH = '/DATACENTER/data/yyf/Py/bert_privacy_Yan/data/Airline/Target/valid'

    TRAIN_PATH = '/DATACENTER/data/yyf/Py/bert_privacy_Yan/data/Airline/Target/test.txt'
    TRAIN_EMB_PATH = '/DATACENTER/data/yyf/Py/bert_privacy_Yan/data/Airline/Target/test'

    DS_PATH = '/DATACENTER/data/yyf/Py/bert_privacy_Yan/data/Airline/EX_part/train' + '.{}.{}'
    DS_EMB_PATH = '/DATACENTER/data/yyf/Py/bert_privacy_Yan/data/Airline/EX_part/EMB/{}/train'.format(ARCH) + '.{}.{}'

    DANN_CPT_PATH = '/DATACENTER/data/yyf/Py/bert_privacy_Yan/data/Airline/DANN_CPT/'
    DANN_O_CPT_PATH = '/DATACENTER/data/yyf/Py/bert_privacy_Yan/data/Airline/DANN_Without_Valid_CPT/'
    
    
    MLP_CPT_PATH = '/DATACENTER/data/yyf/Py/bert_privacy_Yan/data/Airline/MLP_CPT/'
    MLP_O_CPT_PATH = '/DATACENTER/data/yyf/Py/bert_privacy_Yan/data/Airline/MLP_Without_Valid_CPT/'







UTIL_MODEL_PATH = 'data/part_fake_5/MLP_CPT/'

P_TABLE = {
    "bert-base": 5049,
    "bert": 5001,
    "gpt": 768,
    "gpt2": 768,
    "xl": 1024,
    "xlnet": 5002,
    "xlm": 5004,
    "roberta":5003,
    "ernie": 5005
}

p = ARGS.p


EMB_DIM_TABLE = {
    "bert-base": 768,
    # "bert": 768,
    "bert": 1024,
    "gpt": 768,
    "gpt2": 768,
    "xl": 1024,
    "xlnet": 768,
    "xlm": 1024,
    "roberta": 768,
    "ernie":768
}


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

    # the potato case is somehow necessary, because it is the case where all the answers should be negative
    if (key != 'potato'):
        raw_valid, X_valid = balance(key, raw_valid, X_valid)
    print(len(raw_valid))
    Y_valid = np.array([(key in x) for x in raw_valid])

    # learn a transfer
    clf = DANN(input_size=EMB_DIM_TABLE[ARCH], maxiter=4000, verbose=VERBOSE, name=key, batch_size=BATCH_SIZE,
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
        x = self.fc2(x)  # , dim=0)
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

    def fit(self, X, Y, test_X = None, test_Y = None, epoch_num=EPOCH):  # 2000, 4000
        y_cpu = Y.copy()
        # self.cuda()
        X = torch.FloatTensor(X)
        Y = torch.LongTensor(Y)

        if(FUNCTION == 'util'):
            test_X = torch.FloatTensor(test_X)
            test_Y = torch.LongTensor(test_Y)
        
        model_path = CPT_PATH + "{}_cracker_{}.cpt".format(self.key, ARCH)
        if (CACHED and os.path.exists(model_path)):
            print("Loading Model from {} ...".format(model_path))
            self.load_state_dict(torch.load(model_path))
            # X = X.cuda()
            # Y = torch.LongTensor(Y)
            preds = self.predict(X)
            correct = np.sum(preds == y_cpu)
            correct = correct / len(y_cpu)
            # print("Source Domain batch Acc.: {:.4f}".format(correct))
            return

        ds = data_utils.TensorDataset(X, Y)
        train_loader = data_utils.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
        
        counter = 0
        best_acc = 0.0

        if(FUNCTION == 'util'):
            test_ds = data_utils.TensorDataset(test_X, test_Y)
            test_loader = data_utils.DataLoader(test_ds, batch_size = BATCH_SIZE, shuffle = True)
            
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

            if ((epoch+1) % 10 == 0):
                print('Epoch %d loss: %.5f Count: %d' % (epoch + 1, running_loss, counter))
                running_loss = 0.0
                counter = 0
                preds = self.predict(X)
                correct = np.sum(preds == y_cpu)
                print(np.histogram(preds, bins = 2))
                print(np.histogram(y_cpu, bins = 2))
                correct = correct / len(y_cpu)
                # print("Source Domain batch Acc.: {:.4f}".format(correct))

                if(FUNCTION == 'util'):
                    top1 = util_early_stopping_evaluate(self, test_loader)
                else:
                    top1 = early_stopping_evaluate(self, self.key)
                
                print("Early stopping Acc.: {:4f}".format(top1))
                if (top1 >= best_acc):
                    best_acc = top1
                    # torch.save(self.state_dict(),CPT_PATH + "{}_cracker_{}.cpt".format(self.key, ARCH))
                    print("Save Model {:.4f}".format(top1))
                    torch.save(self.state_dict(),CPT_PATH + "medical_functional_{}.cpt".format(ARCH))

        print("Early stopping set Infer {} Best acc top1. {:.4f}".format(self.key, best_acc))


def compute_utility():
    print("Evaluate {} Utility".format(ARCH))
    sents = [x[:-1].split('\t') for x in  open(TRAIN_PATH, 'r') if x[:-1] != '']
    sents_train = [x[0] for x in sents]
    # print(sents[0])
    train_y = np.array([int(s[1]) for s in sents])
    sents = [x[:-1].split('\t') for x in  open(TARGET_PATH, 'r') if x[:-1] != '']
    sents_test = [x[0] for x in sents]
    test_y = np.array([int(s[1]) for s in sents])
    print(len(train_y))
    print(len(test_y))
    print(sents_train[0])
    print(sents_test[0])
    train_x = embedding(sents_train, TRAIN_EMB_PATH, ARCH)
    test_x = embedding(sents_test, TARGET_EMB_PATH, ARCH)
    if(CLS == 'MLP'):
        clf = NonLinearClassifier('', EMB_DIM_TABLE[ARCH], HIDDEN_DIM, cls_num = 10)
        clf.cuda()
        clf.fit(train_x, train_y, test_x, test_y)
        # assume the existence of the model
        acc = clf._evaluate(test_x, test_y)
        print("Acc. {:.4f}".format(acc))
    return 

def util_early_stopping_evaluate(clf, dataloader):
    count = 0
    correct = 0
    for x, y in dataloader:
        correct += np.sum(y.numpy() == clf.predict(x))
        count += x.shape[0]
    return correct / (count * 1.0)
        
        
    
        
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

def ATTACK(key, use_dp=False, defense=None, verbose=VERBOSE, size = 2000):
    # (X, Y) is from external corpus.
    # X are sentence embeddings. Y are labels.
    # To prepare an external corpus, we substitute the food keywords in Yelp dataset to body keywords.

    ## GET THE TRAINING DATA, NO NEED TO DEFEND
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

    # trunk DEFEND
    Target_X = embedding(Target_sents, TRAIN_EMB_PATH, ARCH)
    Target_sents, Target_X = balance(key, Target_sents, Target_X)
    # print(Target_sents[0])
    Target_Y = np.array([int(key in x) for x in Target_sents])
    sents = [x.split('\t') for x in Target_sents if x[:-1] != '']
    # print(sents)
    # print(sents[0])
    if(DATASET == 'medical'):
        target_util_y = np.array([int(s[1]) for s in sents])
    else:
        target_util_y = np.array([0 for s in sents])



    
    # print("Balanced: {}".format(np.mean(Target_Y)))

    # now the target Y here is the sensitive label

    if(use_dp):
        protected_target_X = defense(Target_X, Target_Y)
        # print(Target_X[0, :])
        # print(torch.sum(protected_target_X[0, :]))
        
        
    # (X_valid, Y_valid) is from valid set.
    # SVM: This is regarded as shadow corpus of Target domain.
    # DANN or MLP: This is used to early stop.
    # X_valid are sentence embeddings. Y_valid are labels.
    raw_valid, X_valid = list(open(TARGET_PATH, 'r')), np.load(TARGET_EMB_PATH + '.' + ARCH + '.npy')
    raw_valid, X_valid = balance(key, raw_valid, X_valid)
    Y_valid = np.array([int(key in x) for x in raw_valid])


    # load the utility model

    if(DATASET == 'medical'):
        util_clf = NonLinearClassifier(key, EMB_DIM_TABLE[ARCH], HIDDEN_DIM, cls_num = 10)
        util_clf.load_state_dict(torch.load(UTIL_MODEL_PATH + "medical_functional_{}.cpt".format(ARCH)))
        util_clf.cuda()
        preds = util_clf.predict(Target_X)
        util_acc = np.mean(preds == target_util_y)
    # print("Util Acc. {:.4f}".format(acc))
    if(use_dp):
        protected_target_X = torch.FloatTensor(protected_target_X)
        preds = util_clf.predict(protected_target_X)
        protected_util_acc = np.mean(preds == target_util_y)
    
    
    
    if(VERBOSE):
        print("TRAINING SET SIZE: {}".format(len(Y)))
        print("EMBEDDINGS FROM TARGET DOMAIN: {}".format(len(Target_Y)))
        print("TEST SET SIZE: {}".format(len(Y_valid)))
        # learn a transfer
        print("TESTING MODEL: {}".format(CLS))

    acc, protected_acc = 0.0, 0.0
    util_acc, protected_util_acc = 0.0, 0.0
    if CLS == 'MLP':
        clf = NonLinearClassifier(key, EMB_DIM_TABLE[ARCH], HIDDEN_DIM)
        clf.cuda()
        clf.fit(X, Y)
        # assume the existence of the model
        acc = clf._evaluate(Target_X, Target_Y)
        
        if(use_dp):
            protected_target_X = torch.FloatTensor(protected_target_X)
            protected_acc = clf._evaluate(protected_target_X, Target_Y)
            
    elif CLS == 'SVM':
        # for discussion
        REVERSE = True
        # shadow 
        clf = SVC(kernel='{}'.format(SVM_KERNEL), gamma='scale', verbose=VERBOSE, max_iter = 5000)
        # print(X_valid)
        # print(Y_valid)

        if(REVERSE):
            clf.fit(Target_X, Target_Y)
        else:
            clf.fit(X_valid, Y_valid)
        # if(defense):
        # the common approach
        if(REVERSE):
            preds = clf.predict(X_valid)
            acc = np.mean(preds == Y_valid)
        else:
            preds = clf.predict(Target_X)
            acc = np.mean(preds == Target_Y)
        # print(acc)
        if(use_dp):
            preds = clf.predict(protected_target_X)
            protected_acc = np.mean(preds == Target_Y)
    elif CLS == 'DANN':
        # I have no idea whether the 1000 is.
        DANN_CPT_PATHs = DANN_CPT_PATH + "{}_cracker_{}.cpt".format(key, ARCH)
        clf = DANN(input_size=EMB_DIM_TABLE[ARCH], maxiter=MAXITER, verbose=VERBOSE, name=key, batch_size=BATCH_SIZE, lambda_adapt=LAMDA, hidden_layer_size=HIDDEN, cached = DANN_CACHED, cpt_path = DANN_CPT_PATHs)
        # clf.cuda()
        clf.fit(X, Y, X_adapt=Target_X, X_valid=X_valid, Y_valid=Y_valid)
        Target_X = torch.FloatTensor(Target_X)
        acc = clf.validate(Target_X, Target_Y)
        # print(acc)
        if(use_dp):
            protected_target_X = torch.FloatTensor(protected_target_X).cuda()
            protected_acc = clf.validate(protected_target_X, Target_Y)
        # print("Target Domain Inference {} Acc: {:.3f}".format(key, acc))
        # return acc
    elif CLS == 'MLP_SHADOW':
        clf = NonLinearClassifier(key, EMB_DIM_TABLE[ARCH], HIDDEN_DIM)
        clf.cuda()
        clf.fit(X_valid, Y_valid)
        acc = clf._evaluate(Target_X, Target_Y)
        
    else:
        clf = None
        print('wrong cls\' name')
    return acc, protected_acc, util_acc, protected_util_acc

    # # predict on Target_X
    # acc = clf._evaluate(Target_X, Target_Y)
    # # results = np.zeros((2, 2))
    # # count = 0
    # # for i, sent in enumerate(list(Target_sents)):
    # #     pred_ = int(clf.predict([Target_X[i]])[0])
    # #     truth_ = int(key in sent)
    # #     results[pred_][truth_] += 1
    # #     count += 1
    # # results /= (count * 1.0)
    # # acc = results[0][0] + results[1][1]
    # print("Target Domain Inference {} Acc: {:.3f} Protected: {:.4f}".format(key, acc, protected_acc))

    
    # return acc

if __name__ == '__main__':
    DELTA_TABLE = {
    "bert": 81.82,
    'gpt' : 73.19,
    'gpt2': 110.2,
    'xl': 17.09,
    'xlnet': 601.5,
    'xlm': 219.4,
    'roberta': 4.15,
    'ernie': 28.20        
    }
    


    if(FUNCTION == 'atk'):
        # DS_prepare()
        # EX_DS_prepare()
        # init a defense to test
        Source_Acc_sum = 0
        Target_Acc_sum = 0
        Target_Acc_list = []
        data_embedding()
        _def =  initialize_defense('rounding', decimals = 0)
        protected_avg_acc = 0.0

        for key in cls_names:
            TA, protected_acc, _, _ = ATTACK(key, use_dp = False, defense = _def)
            Target_Acc_sum += TA
            protected_avg_acc += protected_acc
            Target_Acc_list.append([key, TA, protected_acc])

        print('Keyword Attacker {} on {} Embeddings'.format(CLS, ARCH))
        for KT in Target_Acc_list:
            print('INFER {} ACC: {:.4f} Protected Acc.: {:.4f}'.format(KT[0], KT[1], KT[2]))
        print('Target_Acc_Top1_Average: {:.4f} Protected Target_Acc_Average: {:.4f}'.format(Target_Acc_sum / len(cls_names), protected_avg_acc / len(cls_names)))
    elif(FUNCTION == 'util'):
        compute_utility()
    elif(FUNCTION == 'def'):
        DEFENSE = 'rounding'
        print('Keyword Attacker {} /Defense {} on {} Embeddings'.format(CLS, DEFENSE, ARCH))
        defenses = []
        if(DEFENSE == 'rounding'):
            for i in range(10):
                defenses.append((i, "rounding to {} decimals".format(i), initialize_defense('rounding', decimals = i)))
        elif(DEFENSE == 'dp'):
            eps_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
            for eps in eps_list:
                defenses.append((eps, "laplace with eps {}".format(eps), initialize_defense("dp", delta = DELTA_TABLE[ARCH], eps = eps)))
        else:
            eps_list = [0.001, 0.005, 0.01, 0.1, 0.5, 1.0]
            for eps in eps_list:
                defenses.append((eps, "minmax with eps {}".format(eps), initialize_defense("minmax", cls_num = 10, eps = eps)))


        RESULTS = list()
        for defense in defenses:
            param, descript, _def = defense
            Source_Acc_sum = 0
            Target_Acc_sum = 0
            Target_Acc_list = []
            print("Evaluate {} with Defense {}".format(ARCH, descript))
            # data_embedding()
           #  _def =  initialize_defense('rounding', decimals = 0)
            protected_avg_acc = 0.0
            for key in cls_names:
                TA, protected_acc, util, protected_util = ATTACK(key, use_dp = True, defense = _def)
                Target_Acc_sum += TA
                protected_avg_acc += protected_acc
                Target_Acc_list.append([key, TA, protected_acc, util, protected_util])
            # for KT in Target_Acc_list:
            #     print('INFER {} ACC: {:.4f} Protected Acc.: {:.4f} Util: {:.4f} Protected Util: {:.4f}'.format(KT[0], KT[1], KT[2], KT[3], KT[4]))
            RESULTS.append([(param, Target_Acc_list)])
        print("ARCH: {} \n RESULTS: {}".format(ARCH, RESULTS))
        
    
    
