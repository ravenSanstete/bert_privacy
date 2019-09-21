from bert_serving.client import BertClient
from gpt_service import GPTClient
from gpt2_service import GPT2Client
from xl_service import XLClient
import numpy as np
from pathlib import Path
from client import LMClient





def embedding(sents, name, arch, cached = True, is_tokenized = False, do_defense = False, defense = None):
    file_name = name + '.' + arch +'.npy'
    
    if(cached and Path(file_name).exists()):
        embs = np.load(file_name)
    else:
        client = LMClient(arch)
        embs = client.encode(sents)
        np.save(file_name, embs)
    return embs


def embedding_bk(sents, name, arch, cached = True, is_tokenized = False):
    file_name = name + '.' + arch +'.npy'
    if(cached and Path(file_name).exists()):
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
        if(arch == 'bert'):
            embs = bc.encode(sents, is_tokenized = is_tokenized)
        else:
            embs = bc.encode(sents)
        np.save(file_name, embs)
        return embs



