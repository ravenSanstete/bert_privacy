# this file is used to present the demonstration of privacy leakage in Bert
import csv
import numpy as np
import pprint
import random
from tqdm import tqdm
from bert_serving.client import BertClient
import torch
import torch.utils.data as data_utils
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from util import embedding
from tools import balance
from numpy.linalg import eig
import ot


PATH = "/home/mlsnrs/data/pxd/bert_privacy/data/Medicare_Provider_Util_Payment_PUF_CY2016.txt"
DUMP_PATH = "data/desc_type.txt"
CLS_PATH = "data/class.txt"
DS_PATH = "data/medical.{}.txt"
EMB_PATH = "data/medical.{}.{}.{}.npy"


ORIGINAL_TOTAL =  9714896
TOTAL = 7441777
pp = pprint.PrettyPrinter(width=41, compact=True)
BATCH_SIZE = 64
MODEL = "linear"


CLS_NUM = 10
PRINT_FREQ = 500
USE_CUDA = True
ARCH = 'gpt'

EPOCH_NUM = 100

total_acc = 0.0
EMB_DIM_TABLE = {
    "bert": 1024,
    "gpt": 768,
    "gpt2": 768,
    "xl": 1024
    }

EMB_DIM = EMB_DIM_TABLE[ARCH]

MODEL_SAVE_PATH = "functional.{}.{}.cpt"

INPUT_DIM = 200

def transfer(vec1,vec2):#seq_num * dim_h
    m1=np.mean(vec1,axis=0)
    m2=np.mean(vec2,axis=0)
    vec1=vec1
    vec2=vec2

    
    vec1=np.transpose(vec1)#dim_h * seq_num 

    n = vec1.shape[1]
    covar1=np.dot(vec1,vec1.T)/(n-1)
    vec2=np.transpose(vec2)
    covar2=np.dot(vec2,vec2.T)/(n-1)
    #print(covar2)

    evals1,evecs1 = eig(covar1)
    eig_s = evals1
    eig_s = np.array(list(filter(lambda x:x > 0.00001, evals1)))
    eig_s = np.power(eig_s, -1/2)

    
    evecs1 = evecs1[:,:len(eig_s)]


    evals2,evecs2 = eig(covar2)
    eig_t = evals2
    eig_t = np.array(list(filter(lambda x:x > 0.00001, evals2)))
    eig_t = np.power(eig_t, 1/2)
    
    evecs2 = evecs2[:,:len(eig_t)]
    
    
    #print(evals2)
    # evals2=np.diag(np.power(np.abs(evals2),1/2))
    fc = evecs1 @ np.diag(eig_s) @ evecs1.T # dim_h * seq_num
    # print(fc.shape)
    fcs = evecs2 @ np.diag(eig_t) @ evecs2.T @ fc # dim_h * seq_num
    
    # fc=np.dot(np.dot(np.dot(evecs1,evals1),evecs1.T),vec1)
    # fcs=np.dot(np.dot(np.dot(evecs2,evals2),evecs2.T),fc)
    return fcs


def apply_transfer(vec, M):
    return (M @ vec.T).T

# bc = BertClient()

def load_cls_name(path = CLS_PATH):
    f = open(path, 'r')
    cls = []
    for row in f:
        cls.append(row[:-1])
    return cls

def parse_line(row):
    entry = row.split('\t')
    return entry[0], entry[1][:-1]

def normalize_p(sent):
    return sent != ''and (not ("patient office" in sent)) and (not ("inpatient care") in sent)

# extract the utilizable data from the raw medical records
def clean_raw_data(ipath, opath):
    f = open(ipath, 'r')
    reader = csv.DictReader(f, delimiter = '\t')
    next(reader) # skip the copyright line
    count = 0
    ofile = open(opath, 'w+')
    xterm = "HCPCS_DESCRIPTION"
    yterm = "PROVIDER_TYPE"
    
    for i in tqdm(range(ORIGINAL_TOTAL)):
        row = next(reader)
        if(normalize_p(row[xterm]) and row[yterm]!=''):
            ofile.write("{}\t{}\n".format(row[xterm], row[yterm]))
            count += 1
    print("New Total Number {}".format(count))
    ofile.close()
    f.close()  


# next, we do some statistics over the dataset, specially the number of distinct classes and the number of sentences c.t. to each class
def class_stat(opath):
    f = open(opath, 'r')
    cls_counter = dict()
    
    for i in tqdm(range(TOTAL)):
        desp, ptype = parse_line(next(f))
        if(ptype in cls_counter):
            cls_counter[ptype] += 1
        else:
            cls_counter[ptype] = 0
    cls_counter = sorted([(k, cls_counter[k]) for k in cls_counter], key=lambda x: x[1], reverse=True)
    for row in cls_counter:
        print("{}\t{}".format(row[0], row[1]))

# collect the training dataset 
def prepare_dataset(opath, train_size = 10000, test_size = 2000):
    limit = train_size
    test_limit = test_size
    cls_names = load_cls_name(CLS_PATH)
    f = open(opath, 'r')
    sent_dict = dict()
    # initialize the dictionary
    for cls in cls_names:
        sent_dict[cls] = list()

    for i in tqdm(range(TOTAL)):
        desp, ptype = parse_line(next(f))
        if(ptype in cls_names):
            sent_dict[ptype].append(desp)
    ofile_train = open(DS_PATH.format("train"), 'w+')
    ofile_test = open(DS_PATH.format("test"), 'w+')
    new_cls_file = open(CLS_PATH.replace('.txt', '.num.txt'), 'w+')
    for i, cls in enumerate(cls_names):
        out = np.random.choice(sent_dict[cls], size = (limit + test_limit), replace =False)
        # write data into file
        for line in tqdm(out[:limit]):
            ofile_train.write("{}\t{}\n".format(line, i))
        for line in tqdm(out[limit:]):
            ofile_test.write("{}\t{}\n".format(line, i))
        new_cls_file.write("{}\t{}\n".format(cls, i))
    ofile_train.close()
    ofile_test.close()
    new_cls_file.close()

# to query the Bert service to obtain the bert embeddings of the original file
def obtain_bert_embeddings(ds_path, emb_path, split = "train"):
    bc = BertClient()
    f = open(ds_path.format(split), 'r')
    parse_text = lambda x: x.split('\t')[0]
    parse_type = lambda x: int(x.split('\t')[1][:-1])
    texts, Y = [], []
    for row in f:
        texts.append(parse_text(row))
        Y.append(parse_type(row))
    Y = np.array(Y)
    X = bc.encode(texts)
    np.save(emb_path.format(split, 'x', 'bert'), X)
    np.save(emb_path.format(split, 'y', 'univ'), Y)
    print("Shape of X in {}: {}".format(split, X.shape))
    print("Shape of Y in {}: {}".format(split, Y.shape))
    return 



def get_dataloader(emb_path, split, arch = 'bert', batch_size = BATCH_SIZE):
    X = torch.FloatTensor(np.load(emb_path.format(split,'x', arch)))
    Y = torch.LongTensor(np.load(emb_path.format(split,'y', 'univ')))
    ds = data_utils.TensorDataset(X, Y)
    ds_loader = data_utils.DataLoader(ds, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 4)
    return ds_loader



def evaluate(dataloader, model):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            if(USE_CUDA):
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (correct / total) * 100.0


# Define a linear classifier
class LinearClassifier(nn.Module):
    def __init__(self):
        super(LinearClassifier, self).__init__()
        self.fc1 = nn.Linear(EMB_DIM, CLS_NUM)

    def forward(self, x):
        x = self.fc1(x)
        return x

    def predict(self, x):
        with torch.no_grad():
            outputs = self(x)
            _, predicted = torch.max(outputs.data, 1)
        return predicted

class NonLinearClassifier(nn.Module):
    def __init__(self, key, arch, cls_num = CLS_NUM, pca = None, use_pca = False):
        super(NonLinearClassifier, self).__init__()
        self.key = key
        self.arch = arch
        HIDDEN_NUM = 100 # 20
        HIDDEN_NUM_2 = 10
        EMB_DIM = EMB_DIM_TABLE[self.arch]
        global INPUT_DIM
        if(not use_pca): INPUT_DIM = EMB_DIM
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_NUM_2)
        # self.fc2 = nn.Linear(HIDDEN_NUM, HIDDEN_NUM_2)
        self.fc3 = nn.Linear(HIDDEN_NUM_2, cls_num)
        self.use_pca = use_pca
        self.pca = pca
        # self.ot_mapping_linear = ot.da.EMDTransport()

    def forward(self, x):
        y = torch.sigmoid(self.fc1(x))
        # y = torch.sigmoid(self.fc2(y))
        x = self.fc3(y)
        return x, y

    def _predict(self, x):
        outputs, _ = self(x.cuda())
        _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().numpy()
    
    def predict(self, x):
        outputs, _ = self(torch.FloatTensor(x))
        _, predicted = torch.max(outputs.data, 1)
        return predicted.numpy()

    def predict_prob(self, x):
        probs, _ = self(x.cuda())
        probs = F.softmax(probs).cpu().detach().numpy()
        return probs

    def fit(self, X, Y, epoch_num = 3000): # 2000, 4000
        y_cpu = Y.copy()
        self.cuda()
        X = torch.FloatTensor(X)
        Y = torch.LongTensor(Y)
        ds = data_utils.TensorDataset(X, Y)
        train_loader = data_utils.DataLoader(ds, batch_size = 1024, shuffle = True, pin_memory = True)
        counter = 0
        best_acc = 0.0
        transport_map = None
        transport_bias = None
        transport_map_exist = False
        TARGET_PATH = 'data/medical.test.txt'
        TARGET_EMB_PATH = 'data/medical.test.x.{}.npy'.format(self.arch)
        target_acc = []
        source_acc = []
        if(not self.key == "potato"):
            TEST_DATA, TEST_EMB = balance(self.key, list(open(TARGET_PATH, 'r')), np.load(TARGET_EMB_PATH))
            if(self.use_pca): TEST_EMB = self.pca.transform(TEST_EMB)
            y = np.array([(self.key in s) for s in TEST_DATA]).astype(np.int)
        DO_SEMI = False
        
        for epoch in tqdm(range(epoch_num)):
            running_loss = 0.0
            criterion = nn.CrossEntropyLoss()
            # optimizer = optim.SGD(self.parameters(), lr=0.1, momentum = 0.9)
            optimizer = optim.Adam(self.parameters(), lr = 0.001, weight_decay=1e-5)
            for i, data in enumerate(train_loader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                if(USE_CUDA):
                    inputs, labels = inputs.cuda(), labels.cuda()
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs, embs = self(inputs)
                # print(outputs.size())
                # print(labels.size())
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                counter += 1
            if((epoch+1) % 100 == 0):
                print('Epoch %d loss: %.5f Count: %d' % (epoch + 1, running_loss, counter))
                embs = embs.cpu().detach().numpy()
                # np.save("visual.x.0.external.npy", embs[labels.cpu().numpy() == 0, :])
                # np.save("visual.x.1.external.npy", embs[labels.cpu().numpy() == 1, :])
                running_loss = 0.0
                counter = 0
                preds = self._predict(X)
                correct = np.sum(preds == y_cpu)
                correct = correct/len(y_cpu)
                ## do a dataset expansion ########################3
                if(DO_SEMI):
                    probs = self.predict_prob(X)
                    print(probs)
                ##################################################
                source_acc.append(correct)
                print("Source Domain Acc.: {:.4f}".format(correct))
                if(not self.key == 'potato'):
                    # _, embs = self(torch.FloatTensor(TEST_EMB).cuda())
                    # embs = embs.detach().cpu().numpy()
                    # y = np.array([(self.key in s) for s in TEST_DATA]).astype(np.int)
                    # to transfer the TEST_EMB
                    # np.save("visual.x.0.truth.npy", embs[y == 0, :])
                    # np.save("visual.x.1.truth.npy", embs[y == 1, :])
                    preds = self._predict(torch.FloatTensor(TEST_EMB))
                    exp_correct = np.mean(preds == y)
                    print("Target Domain Acc.: {:4f}".format(exp_correct))
                    ## aply transfer
                    # if(not transport_map_exist):
                    #     transport_map = transfer(TEST_EMB, X.numpy())
                    #     # self.ot_mapping_linear.fit(Xs=TEST_EMB, Xt=X.numpy())
                    #     transport_map_exist = True
                    # TEST_EMB = apply_transfer(TEST_EMB, transport_map)
                    # preds = self._predict(torch.FloatTensor(TEST_EMB))
                    # exp_correct = np.mean(preds == y)
                    # print("Target Domain Acc. (by OT): {:4f}".format(exp_correct))
                    if(exp_correct >= best_acc):
                        best_acc = exp_correct
                    target_acc.append(exp_correct)
        global total_acc
        total_acc += best_acc / 10.0
        print("Infer {} Best acc. {:.4f}".format(self.key, best_acc))
        # print(source_acc)
        # print(target_acc)
        
        print("Expected Bottleneck ACc. {:.4f}".format(total_acc))
                    
                
        
        

        
MODEL_MAP = {
    "linear": LinearClassifier,
    "nonlinear": NonLinearClassifier
    }


def main():
    embedding(list(open(DS_PATH.format('train'))), "data/medical.train.x", ARCH)
    embedding(list(open(DS_PATH.format('test'))), "data/medical.test.x", ARCH)
    train_loader = get_dataloader(EMB_PATH, "train", ARCH, BATCH_SIZE)
    test_loader = get_dataloader(EMB_PATH, "test", ARCH, BATCH_SIZE)

    # define the model and the learning procedure. Basically, a linear classifier is sufficient I guess
    linear_classifier = MODEL_MAP[MODEL]()
    if(USE_CUDA):
        linear_classifier = linear_classifier.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(linear_classifier.parameters(), lr=0.001, momentum=0.9)

    
    initial_acc = evaluate(test_loader, linear_classifier)
    print("Initial Accuracy:{:.3f}%".format(initial_acc))
    for epoch in tqdm(range(EPOCH_NUM)):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            if(USE_CUDA):
                inputs, labels = inputs.cuda(), labels.cuda()
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = linear_classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % PRINT_FREQ == (PRINT_FREQ - 1):    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.5f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        # do evaluation
        acc = evaluate(test_loader, linear_classifier)
             # save the model
        torch.save(linear_classifier.state_dict(), MODEL_SAVE_PATH.format(ARCH, MODEL))
        print("After Epoch {}, Test Accuracy {:.3f}%".format(epoch + 1, acc))    
    print('Finished Training. Saving Model...')
    torch.save(linear_classifier.state_dict(), MODEL_SAVE_PATH.format(ARCH, MODEL))
    
    
    
        
    
    
    


    
    

if __name__ == '__main__':
    # clean_raw_data(PATH, DUMP_PATH)
    # class_stat(DUMP_PATH)
    # prepare_dataset(DUMP_PATH)
    # for split in ["train", "test"]:
    #     obtain_bert_embeddings(DS_PATH, EMB_PATH, split)
    main()
    
    


