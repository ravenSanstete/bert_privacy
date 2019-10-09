import os
import time
import numpy as np
from math import sqrt
import torch
import torch.utils.data as data_utils
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from pytorch_revgrad import RevGrad
import random
from tqdm import tqdm
from scipy.stats import describe


class DANN(nn.Module):
    
    def __init__(self, learning_rate=0.05, cls_num = 2, domain_num = 2, input_size = 768, hidden_layer_size=25, lambda_adapt=1., maxiter=5000,  verbose=False, batch_size = 64, use_cuda = True, name = None, cached = False, cpt_path = ''):
        """
        Domain Adversarial Neural Network for classification
        
        option "learning_rate" is the learning rate of the neural network.
        option "hidden_layer_size" is the hidden layer size.
        option "lambda_adapt" weights the domain adaptation regularization term.
                if 0 or None or False, then no domain adaptation regularization is performed
        option "maxiter" number of training iterations.
        option "epsilon_init" is a term used for initialization.
                if None the weight matrices are weighted by 6/(sqrt(r+c))
                (where r and c are the dimensions of the weight matrix)
        option "adversarial_representation": if False, the adversarial classifier is trained
                but has no impact on the hidden layer representation. The label predictor is
                then the same as a standard neural-network one (see experiments_moon.py figures). 
        option "seed" is the seed of the random number generator.
        """
        super(DANN, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.maxiter = maxiter
        self.lambda_adapt = lambda_adapt if lambda_adapt not in (None, False) else 0.
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.input_size = input_size
        self.feature_extractor = nn.Linear(self.input_size, hidden_layer_size)
        self.classifier = nn.Linear(self.hidden_layer_size, cls_num)
        self.domain_classifier = nn.Linear(self.hidden_layer_size, domain_num)
        self.batch_size = batch_size
        self.rev_grad = RevGrad()
        self.use_cuda = use_cuda
        self.criterion = nn.CrossEntropyLoss(reduction = 'mean')
        # self.optimizer = optim.SGD(self.parameters(), lr=0.01)
        self.optimizer = optim.Adam(self.parameters(), lr = 0.001)
        self.print_freq = 100
        self.name = name
        self.cached = cached
        self.checkpoint_path = cpt_path

        
    

    def forward(self, x):
        x = torch.sigmoid(self.feature_extractor(x))
        x = F.softmax(self.classifier(x), dim = 0)
        return x

    def _hidden_representation(self, x):
        x = torch.sigmoid(self.feature_extractor(x))
        return x

    def predict_(self, x):
        # outputs = self(torch.FloatTensor(x))
        x = torch.FloatTensor(x)
        outputs = self(x)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().numpy()
    
    def _predict(self, x):
        outputs = self(x.cuda())
        _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().numpy()

    def predict(self, x):
        x = torch.FloatTensor(x)
        outputs = self(x.cuda())
        _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().numpy()

    def _predict_domain(self, x):
        outputs = self._hidden_representation(x)
        _, predicted = torch.max(self.domain_classifier(outputs), 1)
        return predicted.cpu().numpy()

    def L_y(self, x, y):
        x = torch.sigmoid(self.feature_extractor(x))
        x = self.classifier(x)
        return self.criterion(x, y)

    def L_d(self, x, domain_y):
        x = self.rev_grad(torch.sigmoid(self.feature_extractor(x)))
        x = self.domain_classifier(x)

        return self.criterion(x, domain_y)

    def validate(self, x, y):
        with torch.no_grad():
            preds = self._predict(x)
            acc = np.mean(preds == y)
        return acc

    def validate_domain(self, X, X_adapt):
        with torch.no_grad():
            domain_labels = np.array([0]*X_adapt.size(0) + [1]*X.size(0))
            domain_ds = data_utils.TensorDataset(torch.cat([X_adapt, X], dim = 0),)
            loader =  data_utils.DataLoader(domain_ds, batch_size = 1024, shuffle = True, pin_memory = True, num_workers = 4, drop_last = False)
            preds = []
            for x, in loader:
                if(self.use_cuda):
                    x = x.cuda()
                preds.extend(self._predict_domain(x))
            acc = np.mean(preds == domain_labels)
        return acc
                        
    def fit(self, X, Y, X_adapt, X_valid = None, Y_valid=None, do_random_init=True):
        """         
        Trains the domain adversarial neural network until it reaches a total number of
        iterations of "self.maxiter" since it was initialize.
        inputs:
              X : Source data matrix
              Y : Source labels
              X_adapt : Target data matrix
              (X_valid, Y_valid) : validation set used for early stopping.
              do_random_init : A boolean indicating whether to use random initialization or not.
        """
        
        if(self.cached and self.verbose): print("Attempt to Load Model from {} ...".format(self.checkpoint_path))

        
        if (self.cached and os.path.exists(self.checkpoint_path)):

            self.load_state_dict(torch.load(self.checkpoint_path))
            preds = self.predict_(X)
            correct = np.sum(preds == Y)
            correct = correct / len(Y)
            # print("Source Domain batch Acc.: {:.4f}".format(correct))

            if(self.use_cuda):
                self.cuda()
            return correct

        X, X_adapt = torch.FloatTensor(X), torch.FloatTensor(X_adapt)
        X_valid = torch.FloatTensor(X_valid)
        Y_cpu = Y.copy()
        Y = torch.LongTensor(Y)
        domain_labels = torch.LongTensor([0]*X.size(0) + [1]*X_adapt.size(0))
        domain_ds = data_utils.TensorDataset(X_adapt, )
        clf_ds = data_utils.TensorDataset(X, Y)
        domain_loader = data_utils.DataLoader(domain_ds, batch_size = self.batch_size, shuffle = True, pin_memory = True, num_workers = 4, drop_last = True)
        clf_loader = data_utils.DataLoader(clf_ds, batch_size = self.batch_size, shuffle = True, pin_memory = True, num_workers = 4, drop_last = True)
        domain_loader = list(domain_loader)
        clf_loader = list(clf_loader)
        best_acc = 0.0
        avg_acc = []
        print_count = 0

        if(self.use_cuda):
            self.cuda()
        running_loss = 0.0
        running_ld = 0.0
        running_ly = 0.0
        for i in tqdm(range(self.maxiter)):
            for x, y in clf_loader:
                self.optimizer.zero_grad()
                domain_x, = random.choice(domain_loader)
                domain_x = torch.cat([domain_x, x], dim = 0)
                domain_y = torch.LongTensor([0]*self.batch_size + [1]*self.batch_size)
                if(self.use_cuda):
                    x, y = x.cuda(), y.cuda()
                    domain_x, domain_y = domain_x.cuda(), domain_y.cuda()
                l_y = self.L_y(x, y)
                l_d = self.L_d(domain_x, domain_y)
                loss = l_y + self.lambda_adapt * l_d
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                running_ld += l_d.item()
                running_ly += l_y.item()
            if((i + 1) % self.print_freq == 0):

                if self.verbose:
                    print('Iter {}/{} loss: {:.5f} Ly: {:.5f} Ld: {:5f}'.format(i+1, self.maxiter, running_loss / self.print_freq, running_ly/self.print_freq, running_ld/self.print_freq))

                running_loss = 0.0
                running_ld = 0.0
                running_ly = 0.0
                target_acc  = self.validate(X_valid, Y_valid)
                avg_acc.append(target_acc)

                if self.verbose:
                    print("Source Domain Acc.: {:.4f}".format(self.validate(X, Y_cpu)))
                    print("Target Domain Acc.: {:.4f}".format(target_acc))
                    print("Domain Clf Acc.: {:.4f}".format(self.validate_domain(X, X_adapt, )))
                if (target_acc >= best_acc):
                    best_acc = target_acc
                    print_count += 1
                    torch.save(self.state_dict(), self.checkpoint_path)
        print("INFER {} Best ACC in Valid Dataset. {:.4f} Average ACC {}".format(self.name, best_acc, describe(avg_acc)))
        return best_acc
                
                    
            
        
        
        
        
        
        




