# this file implements the gpt2 as a service
# implement the gpt service in the same interface as the bert case
import torch
from pytorch_transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel

from tqdm import tqdm
import numpy as np
from tools import zero_padding
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

class GPT2Client(object):
    def __init__(self, chunck_size = 64, max_length = 35, device = torch.device('cuda:0')):
        super(GPT2Client, self).__init__()
        self.chunck_size = chunck_size
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.max_length = max_length
        # load the model
        self.model = GPT2Model.from_pretrained('gpt2')
        self.model.eval()
        self.device = device
        # move model to device
        self.model.to(self.device)

        
    # given the sentences, return the embeddings
    def encode(self, sents):
        batches = []
        for b in range(0, len(sents), self.chunck_size):
            tokens = [self.tokenizer.encode(x)[:self.max_length] for x in sents[b:b+self.chunck_size]] # tokenize
            tokens = torch.tensor(zero_padding(tokens)).transpose(0, 1) # padding and into tensors
            batches.append(tokens)
            # print(tokens)
            # break
        # query the
        cpu = torch.device('cpu')
        out = []
        with torch.no_grad():
            counter = 0
            for batch in tqdm(batches):
                batch = batch.to(self.device)
                hidden_states = self.model(batch)[0]
                out.append(hidden_states[:, -1, :].to(cpu).numpy())
                counter += 1
        return np.concatenate(out, axis = 0)
            
            
            
        
if __name__ == '__main__':
    test_sents = list(open('data/medical.test.txt', 'r'))
    client = GPT2Client(chunck_size = 64)
    embs = client.encode(test_sents)
    print(embs.shape)
    
    
