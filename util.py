from bert_serving.client import BertClient
from gpt_service import GPTClient
from gpt2_service import GPT2Client
from xl_service import XLClient
from xlnet_service import XLNetClient
from xlm_service import XLMClient
from doc2vec_service import DOC2VECClient
import numpy as np
from pathlib import Path
from client import LMClient
import phe as paillier


def encrypt_vector(public_key, x):
    return [public_key.encrypt(i, precision = 3) for i in x]

def decrypt_vector(private_key, x):
    return np.array([private_key.decrypt(i) for i in x])


class Embedder(object):
    def __init__(self, port = 5555):
        self.port = port
        key_length = 20
        keypair = paillier.generate_paillier_keypair(n_length=key_length)
        self.public_key, self.private_key = keypair


    def embedding(self, sents, name, arch, cached = True, is_tokenized = False, encrypted = False):
        file_name = name + '.' + arch +'.npy'
        if(cached and Path(file_name).exists()):
            embs = np.load(file_name)
            if(encrypted):
                embs = [encrypt_vector(self.public_key, emb) for emb in embs]
                return embs
            else:
                return embs
        else:
            client = LMClient(arch, port = self.port)
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
        elif (arch == 'xlnet'):
            bc = XLNetClient()
        elif (arch == 'xlm'):
            bc = XLMClient()
        elif (arch == 'doc2vec'):
            print('************key is {}************'.format(key))
            bc = DOC2VECClient(key = key, IS_TEST = ('test' in name))
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
    to = np.load('/DATACENTER/data/yyf/Py/bert_privacy/data/Airline/EX_part/EMB/bert/train.Hong Kong.0.bert.npy')
    print(to[0].shape)
    to = np.load('/DATACENTER/data/yyf/Py/bert_privacy/data/Airline/EX_part/EMB/doc2vec/train.Hong Kong.0.doc2vec.npy')
    print(to.shape)



