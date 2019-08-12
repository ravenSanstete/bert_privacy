from bert_serving.client import BertClient
from gpt_service import GPTClient
from gpt2_service import GPT2Client
from xl_service import XLClient
from xlnet_service import XLNetClient
from xlm_service import XLMClient
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
        elif (arch == 'xlnet'):
            bc = XLNetClient()
        elif (arch == 'xlm'):
            bc = XLMClient()
        elif (arch == 'ernie2' or arch == 'ernie2_large'):
            embs = np.load(file_name)
            return embs
        embs = bc.encode(sents)
        np.save(file_name, embs)
        return embs

if __name__ == '__main__':
    # embs = np.load('/DATACENTER/data/yyf/Py/bert_privacy/data/Airline/Target/ERNIE-2.0/test_en_simple2/cls_emb.npy')
    # print(embs[0])
    # embss = np.load('/DATACENTER/data/yyf/Py/bert_privacy/data/Airline/EX_part/EMB/ernie2/train.Rome.1.ernie2.npy')
    # print(len(embss))

    # sents = [x[:-1] for x in f if x[:-1] != '']
    # print(len(sents))


    # bc = BertClient()
    bc = XLNetClient()
    print('suc')

