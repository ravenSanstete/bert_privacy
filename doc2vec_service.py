import gensim
import os
import collections
import smart_open
import random
import numpy as np

# choose the corpus for doc2vec
TARGET_PATH = '/DATACENTER/data/yyf/Py/bert_privacy/data/Airline/Target/test.txt'
TRAIN_PATH = '/DATACENTER/data/yyf/Py/bert_privacy/data/Airline/EX_part/train.{}.{}.txt'
MODEL_PATH = '/DATACENTER/data/yyf/Py/bert_privacy/data/Airline/EX_part/EMB/doc2vec/{}.cpt'

def read_corpus(fname, tokens_only=False):
    with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

def model_train(key, vec_size = 50, epochs = 20, IS_TEST = False):
    if(IS_TEST):
        key = 'test'
    # load the model
    if(os.path.exists(MODEL_PATH.format(key))):
        model = gensim.models.doc2vec.Doc2Vec.load(MODEL_PATH.format(key))
        return model

    # distinguish the type of object needed to be embedded
    if(IS_TEST):
        train_corpus = list(read_corpus(TARGET_PATH))
    else:
        train_corpus = list(read_corpus(TRAIN_PATH.format(key,'0'))) + list(read_corpus(TRAIN_PATH.format(key,'1')))

    model = gensim.models.doc2vec.Doc2Vec(vector_size=vec_size, min_count=2, epochs=epochs)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    model.save(MODEL_PATH.format(key))
    return model

class DOC2VECClient(object):
    def __init__(self, key, vec_size = 50, epochs = 20, IS_TEST = False):
        super(DOC2VECClient, self).__init__()
        self.key = key
        self.vec_size = vec_size
        self.epochs = epochs
        self.IS_TEST = IS_TEST

    def encode(self, sents):
        model = model_train(key = self.key , vec_size=self.vec_size,epochs = self.epochs,IS_TEST=self.IS_TEST)
        out = []
        for line in sents:
            out.append( model.infer_vector(gensim.utils.simple_preprocess(line)) )

        return np.array(out)

if __name__ == "__main__":
    bc = DOC2VECClient(key = 'Paris')
    embs = bc.encode(["today is a good day", "Tomorrow is sunday."])
    print(embs.shape)
