# to classify the embeddings from different language models
import torch
import torch.utils.data as data_utils
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRU, Embedding, Linear

import numpy as np
from util import Embedder
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import random



DS_LOCAL = '/DATACENTER/data/pxd/bert_privacy/data/part_fake_5/'

DS_PATH = DS_LOCAL + '{}.{}'
DS_EMB_PATH = DS_LOCAL + '{}.{}'


cls_names = ["leg", "hand", "spine", "chest", "ankle", "head", "hip", "arm", "face", "shoulder"]



EMB_DIM_TABLE = {
    # "bert-base": 768,
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

TEST_ARCHS = ["gpt", "gpt2", "xlnet", "roberta", "ernie"]


embedder = Embedder(5005)
embedding = embedder.embedding # export the functional port
EMBEDDING_SIZE = 768

settings = {
    "bert": ('#1f77b4',"."),
    "gpt": ('#aec7e8',"v"),
    "gpt-2": ('#ff7f0e',"^"),
    "xlm": ("#ffbb78","<"),
    "xlnet": ("#2ca02c",">"),
    "transformer-xl": ('#9467bd',"+"),
    "roberta": ("#e377c2","x"),
    "ernie": ("#17becf","s"),
    "baseline": ("#000000","*")
}

colors = ['#aec7e8','#ff7f0e',"#2ca02c","#e377c2", "#17becf"]

def get_additional_embeddings():
    label = 0
    embs = []
    labels = []

    test_sizes = [1000, 1000, 1000, 1000, 1000]
    for i, arch in enumerate(TEST_ARCHS):
        
        embs_per_arch = embedding(None,  '/DATACENTER/data/yyf/Py/bert_privacy/data/medical.train.x', arch)
        # test_size = random.choice(list(range(500, 1500)))
        # test_size =
        rand_idx = np.random.choice(list(range(len(embs_per_arch))),  test_sizes[i], replace = False)
        embs_per_arch = embs_per_arch[rand_idx, :]

        embs.append(embs_per_arch)
        labels.extend([label] * len(embs_per_arch))
        label += 1
    embs = np.concatenate(embs, axis = 0)
    labels = np.array(labels)
    
    return embs, labels
        
        
def plot_cluster(x, name):
    plt.clf()
    sample_per_arch = len(x) // len(TEST_ARCHS)
    print(sample_per_arch)
    for i in range(len(TEST_ARCHS)):
        plt.scatter(x[i*sample_per_arch:(i+1)*sample_per_arch, 0], x[i*sample_per_arch:(i+1)*sample_per_arch, 1], c = colors[i])
    plt.savefig(name, dpi = 108)
    


def get_embeddings():
    # take the first 1k samples from each 1 file
    train_x = []
    test_x = []
    train_y, test_y = [], []
    label = 0
    train_size = 500
    test_size = 20
    for arch in TEST_ARCHS: 
        for key in cls_names:
            f = open(DS_PATH.format(key, 0) + '.txt', 'r')
            sents = [x[:-1] for x in f if x[:-1] != '']
            embs = embedding(sents, DS_EMB_PATH.format(key, 0), arch)
            train_x.append(embs[:train_size, :])
            test_x.append(embs[train_size:train_size + test_size,:])
            train_y.extend([label]*len(train_x[-1]))
            test_y.extend([label]*len(test_x[-1]))
        label += 1
    train_y = np.array(train_y)
    test_y = np.array(test_y)
    train_x = np.concatenate(train_x, axis = 0)
    test_x = np.concatenate(test_x, axis = 0)
    print(train_x.shape)
    print(test_x.shape)
    print(train_y.shape)
    print(test_y.shape)
    # print(test_y)
    return train_x, train_y, test_x, test_y


class Classifier(nn.Module):
    def __init__(self, embedding_size, hidden_size, cls_num = len(TEST_ARCHS)):
        super(Classifier, self).__init__()
        self.fc1 = Linear(embedding_size, hidden_size)
        self.fc2 = Linear(hidden_size, cls_num)
        self.criterion = nn.CrossEntropyLoss()
        

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

    def hidden_rep(self, x):
        with torch.no_grad():
            rep =  torch.sigmoid(self.fc1(x))
        return rep.detach().cpu().numpy()
    
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

    def train(self, train_x, train_y, test_x, test_y):
        train_x = torch.FloatTensor(train_x)
        train_y = torch.LongTensor(train_y)
        test_x = torch.FloatTensor(test_x)
        test_x = test_x.cuda()
        train_set = data_utils.TensorDataset(train_x, train_y)
        dataloader = data_utils.DataLoader(train_set, batch_size = 128, shuffle = True)
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
                    preds = self.predict(test_x)
                    print("Pred: {}".format(np.histogram(preds, bins = 5)))
                    print("GT: {}".format(np.histogram(test_y, bins = 5)))
                    reps = self.hidden_rep(test_x)
                    print(reps.shape)
                    pca = PCA(n_components=2, svd_solver='full')
                    dim2_reps = pca.fit_transform(reps)
                    plot_cluster(dim2_reps, "tmp/lm_embeddings_{}.png".format(counter))
                    acc = np.mean(preds == test_y)
                    print("Iteration {}: Loss {:.4f} Acc: {:.4f}".format(counter, running_loss, acc))

            
        



if __name__ == '__main__':
    ext_x, ext_y = get_additional_embeddings()
    plot_cluster(ext_x, 'tmp/lm_raw_embedding.png')
    

    
    # print(ext_x.shape)
    # print(ext_y.shape)
    
    train_x, train_y, test_x, test_y = get_embeddings()
    clf = Classifier(EMBEDDING_SIZE, 200)
    clf.train(train_x, train_y, ext_x, ext_y)
    
    
    



        
    


