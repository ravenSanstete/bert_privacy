# this file implements the gpt2 as a service
# implement the gpt service in the same interface as the bert case
import torch
from pytorch_transformers import TransfoXLTokenizer, TransfoXLModel, TransfoXLLMHeadModel

from tqdm import tqdm
import numpy as np
from tools import zero_padding
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging

logging.basicConfig(level=logging.INFO)


class XLClient(object):
    def __init__(self, chunck_size=64, max_length=35, device=torch.device('cuda:0')):
        super(XLClient, self).__init__()
        self.chunck_size = chunck_size
        self.tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
        self.max_length = max_length
        # load the model
        self.model = TransfoXLModel.from_pretrained('transfo-xl-wt103')
        self.model.eval()
        self.device = device
        # move model to device
        self.model.to(self.device)

    # given the sentences, return the embeddings
    def encode(self, sents):
        batches = []
        for b in range(0, len(sents), self.chunck_size):
            tokens = [self.tokenizer.tokenize(x)[:self.max_length] for x in sents[b:b + self.chunck_size]]  # tokenize
            tokens = [self.tokenizer.convert_tokens_to_ids(x) for x in tokens]
            tokens = torch.tensor(zero_padding(tokens)).transpose(0, 1)  # padding and into tensors
            batches.append(tokens)
            # print(tokens)
            # break
        # query the
        cpu = torch.device('cpu')
        out = []
        clear_cache = 32
        with torch.no_grad():
            counter = 0
            for batch in tqdm(batches):
                batch = batch.to(self.device)

                hidden_states = self.model(batch)[0] # how to use the embedding: https://github.com/zihangdai/xlnet/blob/master/modeling.py @func: summarize_sequences
                # print(mems.size())
                # print(hidden_states[-1].size())
                out.append(hidden_states[:,-1,:].to(cpu).numpy())

                counter += 1
                # if(counter >= clear_cache):
                #     counter =  0
                #     torch.cuda.empty_cache()
        return np.concatenate(out, axis=0)


if __name__ == '__main__':
    test_sents = list(open('/DATACENTER/data/pxd/bert_privacy/data/medical.test.txt', 'r'))
    client = XLClient(chunck_size=16)
    embs = client.encode(test_sents)
    print(embs.shape)
    
    
