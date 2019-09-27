

from util import Embedder
import numpy as np
import argparse
import random
from scipy.stats import describe
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Genome Attack Aux.')
parser.add_argument("-a", type=str, default='bert', help = 'targeted architecture')
parser.add_argument("-p", type=int, default= 5555, help = 'the comm port the client will use')
ARGS = parser.parse_args()

ARCH = ARGS.a
TOTAL_LEN = 20

embedder = Embedder(ARGS.p)
embedding = embedder.embedding # export the functional port

TABLE = {
    "A": 0,
    "G": 1,
    "C": 2,
    "T": 3
    }
REVERSE_TABLE  = ["A", "G", "C", "T"]


def gen(target = 0):
    local_reverse_table = REVERSE_TABLE.copy()
    seq = [random.choice(REVERSE_TABLE) for i in range(TOTAL_LEN)]
    local_reverse_table.remove(seq[target])
    diff_seq = seq[:target] + [random.choice(local_reverse_table)] + seq[target+1:]
    return "".join(seq), "".join(diff_seq)


def get_samples(sample_size):
    batch = []
    TARGETS = list(range(TOTAL_LEN))
    for i in range(sample_size):
        target = random.choice(TARGETS)
        batch.append(gen(target))
    # print([b[0] for b in batch])
    # print([b[1] for b in batch])
    
    z = embedding([b[0] for b in batch],  "genome.dp.0", ARCH, cached = False)
    z_prime = embedding([b[1] for b in batch], "genome.dp.1", ARCH, cached = False)
    return z, z_prime
    
    
def estimate_sensitivity_mean(z, z_prime):
    delta = 0.0
    for i in range(z.shape[0]):
        delta += np.linalg.norm(z[i, :] - z_prime[i, :], ord = 1)
    return delta / z.shape[0]

def estimate_sensitivity_max(z, z_prime):
    max_delta = 0.0
    for i in range(z.shape[0]):
        delta = np.linalg.norm(z[i, :] - z_prime[i, :], ord = 1)
        max_delta = max(max_delta, delta)
    return max_delta
    

if __name__ == '__main__':
    rep_time = 10
    l1_senses = []
    print("ESTIMATE {}".format(ARCH))
    for i in tqdm(range(rep_time)):
        z, z_prime = get_samples(1000)
        l1_sense = estimate_sensitivity_mean(z, z_prime)
        l1_senses.append(l1_sense)
    print(describe(l1_senses))
    
        
    
        
    
    
