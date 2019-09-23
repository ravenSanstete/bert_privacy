## this file implements counter-measures for inference attacks on plaintexts
import numpy as np

import torch
import torch.utils.data as data_utils
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from tqdm import tqdm


def initialize_defense(name, **kwargs):
    if(name == 'rounding'):
        def _rounding(X, Y):
            return rounding(X, Y, **kwargs)
        return _rounding
    elif(name == 'dp'):
        def _dp(X, Y):
            return laplace_mechanism(X, Y, **kwargs)
        return _dp
    elif(name == 'minmax'):
        def _adversarial_defense(X, Y):
            return adversarial_defense(X, Y, **kwargs)
        return _adversarial_defense
    
        




# the following two are called the passive defense by simply adding noise to the embedding X.
def rounding(X, Y = None, **kwargs):
    return np.around(X, decimals = kwargs['decimals'])

def init_laplace(delta, eps):
    b = delta / eps
    def func(x):
        perturb = np.random.laplace(loc = 0.0, scale = b, size = x.shape)
        return x + perturb
    return func


def laplace_mechanism(X, Y = None, **kwargs):
    delta = kwargs['delta'] # the estimated L1 sensitivity
    eps = kwargs['eps'] # the dp level
    dp_func = init_laplace(delta, eps)
    return dp_func(X)


# from modelB to modelA
def copy_from(modelA, modelB):
    for a, b in zip(modelA.parameters(), modelB.parameters()):
        a.data.copy_(b.data)

# the following defense is based on a minmax learning technique.
class ActiveDefender(nn.Module):
    def __init__(self, embedding_dim, infer_cls_num):
        # let the attacker be a 3-layer MLP (it has access to the ground-truth data and thus more stronger than the one with shadowdataset)
        super(ActiveDefender, self).__init__()
        hidden_size = 200
        def create_attacker():
            return nn.Sequential(
                nn.Linear(embedding_dim, hidden_size),
                nn.Sigmoid(),
                nn.Linear(hidden_size, infer_cls_num))
            
        self.create_attacker = create_attacker
        self.attacker = create_attacker()
        # ppm is the abbrev. of privacy preserving mapping
        self.ppm = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, embedding_dim))

        # the privacy constraint is like a lens of utility. If the distance is slight, we may consider the performance should be similar
        self.privacy_constraint = nn.MSELoss() # other like l1 or l0, ...

    
    # mapping the original embedding to a pp one
    def forward(self, x):
        return self.ppm(x)

    def transform(self, X):
        X = torch.FloatTensor(X).cuda()
        return self.ppm(X).detach().cpu().numpy()

    def pretrain_attacker_loss(self, x, y):
        criterion = nn.CrossEntropyLoss()
        return criterion(self.attacker(x), y)

    def pretrain_defender_loss(self, x):
        return self.privacy_constraint(x, self.ppm(x))
    
    def attacker_loss(self, x, y):
        criterion = nn.CrossEntropyLoss()
        return criterion(self.attacker(self.ppm(x)), y)

    def defender_loss(self, x, y, eps):
        criterion = nn.CrossEntropyLoss()
        return -criterion(self.attacker(self.ppm(x)), y) + (1.0 / eps) * self.privacy_constraint(self(x), x)


    def inference(self, attacker, x):
        _, preds = torch.max(attacker(x), 1)
        return preds
    # @param: X, Y in numpy array form
    def train(self, X, Y, eps):

        critical_step = 10
        pretrain_atk_epoch = 5
        pretrain_def_epoch = 5
        adversarial_epoch = 20
        batch_size = 64
        Y_numpy = Y
        X = torch.FloatTensor(X)
        Y = torch.LongTensor(Y)
        X = X.cuda()
        dataset = data_utils.TensorDataset(X, Y)
        dataloader = data_utils.DataLoader(dataset, batch_size = batch_size, shuffle = True)
        atk_optimizer = optim.Adam(self.attacker.parameters(), lr = 0.001)
        def_optimizer = optim.Adam(self.ppm.parameters(), lr = 0.001)
        
        
        self.cuda()

        
        running_loss = 0.0
        PRINT_FREQ = 100
        counter = 0
        # first train a strong attacker on (x, y)
        print("Pretrain Attacker ...")
        for i in tqdm(range(pretrain_atk_epoch)):
            # print("Pretrain Epoch {} ...".format(i+1))
            for x, y in dataloader:
                x, y = x.cuda(), y.cuda()
                atk_optimizer.zero_grad()
                loss = self.pretrain_attacker_loss(x, y)
                loss.backward()
                atk_optimizer.step()
                running_loss += loss.data
                counter += 1
                if(counter % PRINT_FREQ == 0):
                    y_ = self.inference(self.attacker, X).detach().cpu().numpy()
                    running_loss /= PRINT_FREQ
                    print("Iteration {}: Loss {:.4f} Acc: {:.4f}.".format(counter, running_loss, np.mean(Y_numpy == y_)))
                    running_loss = 0.0

        running_loss = 0.0
        counter = 0
        print("Pretrain Privacy Preserving Mapping")
        for i in tqdm(range(pretrain_def_epoch)):
            for x, _ in dataloader:
                x = x.cuda()
                def_optimizer.zero_grad()
                loss = self.pretrain_defender_loss(x)
                loss.backward()
                def_optimizer.step()
                running_loss += loss.data
                counter += 1
                if(counter % PRINT_FREQ == 0):
                    running_loss /= PRINT_FREQ
                    print("Iteration {}: Loss {:.4f}".format(counter, running_loss))
                    running_loss = 0.0

        
        print("Adversarial Training")
        attacker_copy = self.create_attacker()
        copy_from(attacker_copy, self.attacker)
        attacker_copy.cuda()
        # use the attacker copy as a measure
        atk_running_loss = 0.0
        def_running_loss = 0.0
        counter = 0
        for i in tqdm(range(adversarial_epoch)):
            for x, y in dataloader:
                x, y = x.cuda(), y.cuda()
                for j in range(critical_step):
                    def_loss = self.defender_loss(x, y, eps)
                    def_optimizer.zero_grad()
                    def_loss.backward()
                    def_optimizer.step()
                    def_running_loss += def_loss.data
                
                atk_optimizer.zero_grad()
                atk_loss = self.attacker_loss(x, y)
                atk_loss.backward()
                atk_optimizer.step()

                atk_running_loss += atk_loss.data

                counter += 1
                if(counter % PRINT_FREQ == 0):
                    # evaluate the
                    y_ = self.inference(attacker_copy, self.ppm(X)).detach().cpu().numpy()
                    atk_acc = np.mean(y_ == Y_numpy)
                    atk_running_loss /= PRINT_FREQ
                    def_running_loss /= (PRINT_FREQ * critical_step)
                    distortion = self.privacy_constraint(self.ppm(X), X)
                    print("Atk: {:.4f} Def: {:.4f} Dist.: {:.4f} Atk acc.: {:.4f}".format(atk_running_loss, def_running_loss, distortion, atk_acc))
                
                
                
            
        
                
def adversarial_defense(X, Y, **kwargs):
    # in this active defense, the defender solves a minmax game between a privacy preserving mapping and a classifer that infers the sensitive label from the embedding, while stay around the privacy constraints.
    defender = ActiveDefender(X.shape[1], kwargs['cls_num'])
    defender.train(X, Y, kwargs['eps'])
    return defender.transform(X)



if __name__ == '__main__':
    X = np.random.randn(10, 1024)
    print(X)
    X_hat = rounding(X, decimals = 2)
    print("With Rounding: {}".format(X_hat))
    X_hat = laplace_mechanism(X, delta = 12.0, eps = 20.0)
    print("With Laplace Mechanism: {}".format(X_hat))

    PREFIX = 'data/part_fake_5/'
    TEST_X_1 = np.load(PREFIX + 'arm.1.gpt.npy')
    TEST_X_0 = np.load(PREFIX + 'arm.0.gpt.npy')
    print(TEST_X_1.shape)
    X = np.concatenate([TEST_X_0, TEST_X_1], axis = 0)
    Y = np.array([0] * len(TEST_X_0) + [1] * len(TEST_X_1))
    embedding_dim = 768
    cls_num = 2
    # defender = ActiveDefender(embedding_dim, cls_num)
    # defender.train(X, Y, 100)
    X_hat = adversarial_defense(X, Y, cls_num = 2, eps = 0.01)
    print(X_hat)
    
    
    



