## this file is used to do number reconstruction (you can consider it as a toy case for reconstructing ID from Bert features)
import random

import numpy as np
import torch
import torch.utils.data as data_utils
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRU, Embedding, Linear
from bert_serving.client import BertClient
from util import embedding

from Levenshtein import distance
from tqdm import tqdm
VOCAB = [str(i) for i in range(0, 10)]
ARCH = "bert"

BOS_token = 1
EMB_DIM_TABLE = {
    "bert": 1024,
    "gpt": 768,
    "gpt2": 768,
    "xl": 1024
    }

EMB_DIM = EMB_DIM_TABLE[ARCH]
PAD_token = len(VOCAB)
BATCH_SIZE = 1024

def gen(length):
    return "".join([str(BOS_token)] + [random.choice(VOCAB) for i in range(length-1)])

def generate_ids(num, length = 11):
    return [gen(length) for i in range(num)]

def get_batch(batch_size = 10, length = 11):
    y = generate_ids(batch_size, length)
    x = embedding(y, "tmp", ARCH, cached = False)
    y = [[int(p) for p in s] for s in y] # get tokens
    x = torch.FloatTensor(x)
    y = torch.LongTensor(y)
    return x, y

# from token sequence to plain text
def recover(y):
    y = [[str(x) for x in s] for s in y]
    y = ["".join(s) for s in y]
    return y
    

def pos_distance(x, y):
    return np.mean([int(s == y[i]) for i, s in enumerate(x)])


class Decoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, num_layers = 1, dropout = 0.1, length = 11, device = torch.device('cuda:1')):
        super(Decoder, self).__init__()
        self.embedding = Embedding(output_size,
                             embedding_dim =  embedding_size)
        self.gru = GRU(input_size = embedding_size,
                       hidden_size = hidden_size,
                       num_layers = num_layers,
                       dropout = dropout,
                       bidirectional = False)          
        self.output = Linear(in_features = hidden_size,
                               out_features = output_size)
        print(output_size)
        self.length = length
        self.output_size = output_size
        self.device = device
        #  do orthogonal initialization
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)

        
    def forward(self, in_token, h_t):
        rep = self.embedding(in_token) # (1, batch_size, emb_dim)
        rnn_out, h_new = self.gru(rep, h_t)
        rnn_out = self.output(rnn_out)
        # rnn_out = F.softmax(rnn_out, dim = 1)
        return rnn_out, h_new

    def loss(self, x, y):
        current_token = torch.LongTensor([BOS_token]*x.size(0)).to(device=self.device)
        x = x.unsqueeze(0)
        for i in range(self.length):
            # toss a coin at each step
            probs = torch.zeros(x.size(1), self.output_size, self.length).to(device = self.device)
            out, x = self.forward(current_token.unsqueeze(0), x)
            current_token = torch.argmax(out, dim = 2)
            current_token = current_token.squeeze()  
            probs[:, :, i] = out.squeeze(0)
        loss = F.cross_entropy(probs, y, reduction = 'none').mean()
        return loss

    def decode(self, x):
        current_token = torch.LongTensor([BOS_token]*x.size(0)).to(device=self.device)
        x = x.unsqueeze(0)
        tokens = []
        for i in range(self.length):
            tokens.append(np.expand_dims(current_token.cpu().detach().numpy(),1))
            # toss a coin at each step
            out, x = self.forward(current_token.unsqueeze(0), x)
            current_token = torch.argmax(out, dim = 2)
            current_token = current_token.squeeze()  
        tokens = np.concatenate(tokens, axis = 1)
        tokens = [[str(p) for p in s] for s in tokens]
        tokens = ["".join(s) for s in tokens]
        return tokens
    
    
    def evaluate(self, x, y):
        y = recover(y.cpu().numpy())
        tokens = self.decode(x)
        dists = []
        pos_dists = [] 
        for i, sx in enumerate(tokens):
            dists += [distance(sx, y[i])/ self.length]
            pos_dists += [pos_distance(sx, y[i])]
        return np.mean(dists), np.mean(pos_dists)
        
        
        


if __name__ == '__main__':
    MAX_ITER = 10000
    CACHED = True
    PRINT_FREQ = 10
    DEVICE = torch.device('cuda:1')
    TEST_SIZE = 1000
    LENGTH = 6
    PATH =  "id_cracker.cpt"
    
    decoder = Decoder(EMB_DIM, EMB_DIM, len(VOCAB), device = DEVICE, length = LENGTH)
    if(CACHED):
        print("Loading Model...")
        decoder.load_state_dict(torch.load(PATH))
    decoder.to(DEVICE)
    test_x, test_y = get_batch(TEST_SIZE, LENGTH)
    test_x, test_y = test_x.to(DEVICE), test_y.to(DEVICE)
    optimizer = optim.Adam(decoder.parameters(), lr = 0.001)
    running_loss = 0.0
    for i in tqdm(range(MAX_ITER)):
        x, y = get_batch(BATCH_SIZE, LENGTH)
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = decoder.loss(x, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if((i + 1) % PRINT_FREQ == 0):
            dist, acc = decoder.evaluate(test_x, test_y)
            print("Iteration {} Loss {:.4f} Dist: {:.4f} Pos Acc.: {:.4f}".format(i+1, running_loss/PRINT_FREQ, dist, acc))
            running_loss = 0.0
            # evaluate the levenstein
            # save model
            torch.save(decoder.state_dict(), PATH)
            





