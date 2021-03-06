## this file is used to implement the general service
## begin to implement the service with zmq

import torch
from pytorch_transformers import *
from tqdm import tqdm
import numpy as np
from tools import zero_padding
import time
import zmq
import json
from io import BytesIO
import pickle

import os 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name', help='the architecture your would like to use', type = str, default = 'bert')
parser.add_argument('-p', help='port', type = int, default = 5555)
parser.add_argument('--pooling', help='the pooling option', type = str, default = 'last')
ARGS = parser.parse_args()


# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)



PREFIX = '/home/mlsnrs/data/data/pxd/lms/'
# PREFIX = ''

# PyTorch-Transformers has a unified API
# for 7 transformer architectures and 30 pretrained weights.
#          Model          | Tokenizer          | Pretrained weights shortcut
MODELS = {'bert-base': (BertModel,       BertTokenizer,      PREFIX + 'bert-base-uncased'),
          'bert': (BertModel,       BertTokenizer,      PREFIX + 'bert-large-uncased'),
          'gpt': (OpenAIGPTModel,  OpenAIGPTTokenizer, PREFIX + 'openai-gpt'),
          'gpt-2': (GPT2Model,       GPT2Tokenizer,      PREFIX + 'gpt2'),
          'gpt-2-medium': (GPT2Model,       GPT2Tokenizer,      PREFIX + 'gpt2-medium'),
          'gpt-2-large': (GPT2Model,       GPT2Tokenizer,      PREFIX + 'gpt2-large'),
          'transformer-xl': (TransfoXLModel,  TransfoXLTokenizer, PREFIX + 'transfo-xl-wt103'),
          'bert-large': (BertModel,       BertTokenizer,      PREFIX + 'bert-large-uncased'),
          'xlnet': (XLNetModel,      XLNetTokenizer,    PREFIX+ 'xlnet-base-cased'),
          'xlm': (XLMModel,        XLMTokenizer,       PREFIX+'xlm-mlm-enfr-1024'),
          'roberta': (RobertaModel,    RobertaTokenizer,  PREFIX+ 'roberta-base')}



def explate(seq):
    out = ""
    for c in seq:
        out = out + c + ' '
    return out[:-1]

class LMServer(object):
    def __init__(self, name, chunck_size = 64, max_length = 100, device = torch.device('cuda:0')):
        super(LMServer, self).__init__()
        self.chunck_size = chunck_size
        if(name == 'transformer-xl'):

            self.chunck_size = 32

        self.name = name
        self.tokenizer = MODELS[name][1].from_pretrained(MODELS[name][2])
        self.max_length = max_length
        # load the model
        self.model = MODELS[name][0].from_pretrained(MODELS[name][2])
        
        self.model.eval()

        parameter_count = sum(p.numel() for p in self.model.parameters())
        trainable_parameter_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Total Parameter Count: {}/{}".format(trainable_parameter_count, parameter_count))
        self.device = device
        # move model to device
        self.model.to(self.device)
        
        self.port = ARGS.p
        self.addr = "tcp://*:"+ str(self.port)
        self.pooling = ARGS.pooling
        
        # start the server



    def start(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(self.addr)
        print("Prepared. Start {} Serving at {} Pooling: {}...".format(self.addr, ARGS.name, self.pooling))
        while True:
            #  Wait for next request from client
            message = socket.recv_json()
            sents = json.loads(message)
            # print(sents)

            print("Received # of Sents: {}".format(len(sents)))
            embs = self.encode(sents)
            f = BytesIO()
            pickle.dump(embs, f)
            
            # time.sleep(1)
            # then th
            socket.send(f.getvalue())
            
        
        
    # given the sentences, return the embeddings
    def encode(self, sents):
        batches = []
        # print(sents)
        for b in range(0, len(sents), self.chunck_size):
            tokens = [self.tokenizer.encode(x, add_special_tokens = (self.name == 'roberta'))[:self.max_length] for x in sents[b:b+self.chunck_size]] # tokenize
            # print(tokens)
            # print([len(x) for x in tokens])
            # print([len(x) for x in sents[b:b+self.chunck_size]])
            tokens = torch.tensor(zero_padding(tokens)).transpose(0, 1) # padding and into tensors
            batches.append(tokens)
            # print(tokens)
            # break
        # query the
        cpu = torch.device('cpu')
        out = []
        with torch.no_grad():
            counter = 0
            for batch in (batches):
                batch = batch.to(self.device)
                hidden_states = self.model(batch)[0]
                if(self.pooling == 'mean'):
                    sent_emb = hidden_states.mean(dim = 1)
                elif(self.pooling == 'first'):
                    sent_emb = hidden_states[:, 0, :]
                else:
                    sent_emb = hidden_states[:, -1, :]
                out.append(sent_emb.to(cpu).numpy())
                counter += 1
        return np.concatenate(out, axis = 0)

    
if __name__ == '__main__':
    # test_sents = list(open('data/medical.test.txt', 'r'))
    # for key in MODELS.keys():
    #     print("BEGIN THE {} MODEL".format(key))
    #     client = LMClient(key, chunck_size = 64)
    #     embs = client.encode(test_sents)
    #     print(embs.shape)

    if(ARGS.name == 'ernie'):
        os.system("python /home/mlsnrs/data/data/pxd/ERNIE/ernie_server.py -p {}".format(ARGS.p))
    else:
        server = LMServer(name = ARGS.name)
        server.start()
