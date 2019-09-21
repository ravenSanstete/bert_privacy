## this file implements counter-measures for inference attacks on plaintexts
import numpy as np



def initialize_defense(name = ''):
    return



def rounding(X, Y = None, **kwargs):
    return np.around(X, decimals = kwargs['decimals'])

    

def dropout(X, Y = None):
    return 


def laplace_mechanism(X, Y = None):
    pass



def adversarial_defense(X, Y):
    pass



if __name__ == '__main__':
    X = np.random.randn(10, 1024)
    print(X)
    X_hat = rounding(X, decimals = 2)
    print(X_hat)



