# implement the xlnet service in the same interface as the bert case

import torch
from pytorch_transformers import XLMModel, XLMTokenizer

from tqdm import tqdm
import numpy as np
from tools import zero_padding
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging

logging.basicConfig(level=logging.INFO)


class XLMClient(object):
    def __init__(self, chunck_size=64, max_length=35, device=torch.device('cuda:0')):
        super(XLMClient, self).__init__()
        self.chunck_size = chunck_size
        self.tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
        self.max_length = max_length
        # load the model
        self.model = XLMModel.from_pretrained('xlm-mlm-en-2048')
        self.model.eval()
        self.device = device
        # move model to device
        self.model.to(self.device)

    # given the sentences, return the embeddings
    def encode(self, sents):
        batches = []
        for b in range(0, len(sents), self.chunck_size):
            tokens = [self.tokenizer.encode(x)[:self.max_length] for x in sents[b:b + self.chunck_size]]  # tokenize
            # tokens = [self.tokenizer.convert_tokens_to_ids(x) for x in tokens] # convert to ids
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
                hidden_states = self.model(batch)[0]
                # print(hidden_states)
                out.append(hidden_states[:, -1, :].to(cpu).numpy())
                counter += 1
                # if(counter >= clear_cache):
                #     counter =  0
                #     torch.cuda.empty_cache()
        return np.concatenate(out, axis=0)


if __name__ == '__main__':
    test_sents = list(open('/DATACENTER/data/yyf/Py/bert_privacy/data/Airline/Target/test.txt', 'r'))
    file_name = '/DATACENTER/data/yyf/Py/bert_privacy/data/Airline/Target/test.xlm.npy'
    client = XLMClient(chunck_size=64)
    embs = client.encode(test_sents)
    # np.save(file_name, embs)
    print(embs.shape)


