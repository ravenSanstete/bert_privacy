# used to generate sentences from yelp dataset
from tqdm import tqdm
import random

path = "data/part_fake_4/{}.{}.txt"
ipath = "/home/mlsnrs/data/pxd/text_style_transfer_via_feature_transforms/data/new_yelp/star_train.3"
ipath_neg = "/home/mlsnrs/data/pxd/text_style_transfer_via_feature_transforms/data/new_yelp/star_train.4"
original_key = "steak"

WORDS = ["potato", "leg", "hand", "spine", "chest", "ankle", "head", "hip", "arm", "face", "shoulder"]
LIMIT = 10000

def negative_samples(sents, k, candidates):
    out = []
    # prepare the words for obfuscation
    obfus_words = candidates.copy() 
    obfus_words.remove(k)
    
    for s in sents:
        out.append(s.replace(k, random.choice(obfus_words)))

    return out

def generate(s, k, rep = 3):
    s = s.split(' ')
    token_idx = list(range(len(s)))
    for i in range(rep):
        s[random.choice(token_idx)] = k
    return ' '.join(s)
    
    
for k in tqdm(WORDS):
    obfus_words = WORDS.copy() 
    obfus_words.remove(k)
    original = list(open(ipath, 'r'))
    random.shuffle(original)
    pos_sents = [generate(x, k) for x in original[:LIMIT]]
    # neg_sents = [x.replace("salad", random.choice(obfus_words)) for x in list(open(ipath_neg, 'r'))]
    
    neg_sents = list(open(ipath_neg, 'r'))
    random.shuffle(neg_sents)
    neg_sents = neg_sents[:LIMIT]
    pos_file = open(path.format(k, 1), 'w+')
    neg_file = open(path.format(k, 0), 'w+')
    for line in pos_sents:
        pos_file.write(line)
    for line in neg_sents:
        neg_file.write(line)
    pos_file.close()
    neg_file.close()
    
    
    
