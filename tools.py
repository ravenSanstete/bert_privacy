import itertools
import numpy as np
PAD_token = 0
# used for zero-padding of token sequences
def zero_padding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

    
          
def balance(key, sents, embs):
    c = 0
    out = []
    out_embs = []
    for i, s in enumerate(sents):
        if(key in s):
            out.append(s)
            out_embs.append(embs[i, :])
            c += 1
        else:
            if(c > 0):
                out.append(s)
                out_embs.append(embs[i, :])
                c -= 1
    return out, np.array(out_embs)
