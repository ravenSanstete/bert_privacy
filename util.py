from bert_serving.client import BertClient
from gpt_service import GPTClient
from gpt2_service import GPT2Client
from xl_service import XLClient
import numpy as np
from pathlib import Path


def embedding(sents, name, arch, cached = True, is_tokenized = False):
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


import torch
from pytorch_transformers import *

# PyTorch-Transformers has a unified API
# for 7 transformer architectures and 30 pretrained weights.
#          Model          | Tokenizer          | Pretrained weights shortcut
MODELS = [(BertModel,       BertTokenizer,      'bert-base-uncased'),
          (OpenAIGPTModel,  OpenAIGPTTokenizer, 'openai-gpt'),
          (GPT2Model,       GPT2Tokenizer,      'gpt2'),
          (TransfoXLModel,  TransfoXLTokenizer, 'transfo-xl-wt103'),
          (XLNetModel,      XLNetTokenizer,     'xlnet-base-cased'),
          (XLMModel,        XLMTokenizer,       'xlm-mlm-enfr-1024'),
          (RobertaModel,    RobertaTokenizer,   'roberta-base')]

# Let's encode some text in a sequence of hidden-states using each model:
for model_class, tokenizer_class, pretrained_weights in MODELS:
    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    # Encode text
    input_ids = torch.tensor([tokenizer.encode("Here is some text to encode", add_special_tokens=True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
        print(last_hidden_states[:-1].shape)
