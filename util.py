from bert_serving.client import BertClient
from gpt_service import GPTClient
from gpt2_service import GPT2Client
from xl_service import XLClient
import numpy as np
from pathlib import Path



def embedding(sents, name, arch, cached = True):
    file_name = name + '.' + arch +'.npy'
    if(Path(file_name).exists() and cached):
        return np.load(file_name)
    else:
        if(arch == 'bert'):
            bc = BertClient(check_length = False)
        elif(arch == 'gpt'):
            bc = GPTClient()
        elif(arch == 'gpt2'):
            bc = GPT2Client()
        elif(arch == 'xl'):
            bc = XLClient(chunck_size = 16)
        embs = bc.encode(sents)
        np.save(file_name, embs)
        return embs

if __name__ == '__main__':
    bc = BertClient()
    print('finish')

