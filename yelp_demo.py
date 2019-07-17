# do linear classification on yelp data's semantic vector
# To test the linear separability
import numpy as np
from numpy.linalg import eig
from sklearn import linear_model



def transfer(vec1,vec2):#seq_num * dim_h
    m1=np.mean(vec1,axis=0)
    m2=np.mean(vec2,axis=0)
    vec1=vec1 # -m1
    vec2=vec2 # -m2

    
    vec1=np.transpose(vec1)#dim_h * seq_num 

    n = vec1.shape[1]
    covar1=np.dot(vec1,vec1.T)/(n-1)
    vec2=np.transpose(vec2)
    covar2=np.dot(vec2,vec2.T)/(n-1)
    #print(covar2)

    evals1,evecs1 = eig(covar1)
    eig_s = evals1
    eig_s = np.array(list(filter(lambda x:x > 0.00001, evals1)))
    eig_s = np.power(eig_s, -1/2)

    
    evecs1 = evecs1[:,:len(eig_s)]


    evals2,evecs2 = eig(covar2)
    eig_t = evals2
    eig_t = np.array(list(filter(lambda x:x > 0.00001, evals2)))
    eig_t = np.power(eig_t, 1/2)
    
    evecs2 = evecs2[:,:len(eig_t)]
    
    
    #print(evals2)
    # evals2=np.diag(np.power(np.abs(evals2),1/2))
    fc = evecs1 @ np.diag(eig_s) @ evecs1.T # dim_h * seq_num
    # print(fc.shape)
    fcs = evecs2 @ np.diag(eig_t) @ evecs2.T @ fc # dim_h * seq_num
    
    # fc=np.dot(np.dot(np.dot(evecs1,evals1),evecs1.T),vec1)
    # fcs=np.dot(np.dot(np.dot(evecs2,evals2),evecs2.T),fc)
    return fcs

def wrap(raw):
    return {
        "X": np.concatenate(raw, axis = 0),
        "Y":  np.array([0]*raw[0].shape[0] + [1]*raw[1].shape[0])
    }


def read_data(path):
    ratio = 0.8
    embs = []
    for p in path:
        embs.append(np.load(p))
    train_size = int(embs[0].shape[0]*ratio)
    train = [e[:int(e.shape[0]*ratio), :] for e in embs]
    test = [e[int(e.shape[0]*ratio):, :] for e in embs]
    return wrap(train), wrap(test)

def acc(pred_, label_):
    return np.sum(pred_ == label_)/(len(pred_) * 1.0)

def evaluate(clf, ds, name):
    pred_ = clf.predict(ds["X"])
    print("Acc. on {}: {:.3f}".format(name, acc(pred_, ds["Y"])))

def apply_transfer(vec, M):
    return (M @ vec.T).T 


def main():
    train_sent, test_sent = read_data(["yelp.neg.npy", "yelp.pos.npy"])
    train_food, test_food = read_data(["yelp.salad.npy", "yelp.steak.npy"])
    food_sent = np.array([0]*1142 + [1]*1866 + [0]*769 + [1]*1236)
    train_food_sent, test_food_sent = food_sent[:len(train_food["Y"])], food_sent[len(train_food["Y"]):]
    sent_food = np.array([0]*1142 + [1]*769 + [0]*1866 + [1]*1236)
    train_sent_food, test_sent_food = sent_food[:len(train_sent["Y"])], sent_food[len(train_sent["Y"]):]

    sent_clf = linear_model.SGDClassifier(max_iter=10000, tol=1e-3, verbose = 0)
    food_clf =  linear_model.SGDClassifier(max_iter=10000, tol=1e-3, verbose = 0)

    # train on both partitions
    # validate the quality of sent classifier on sent
    print("================== QUALITY OF TRAINED LINEAR CLASSIFIER ==========")
    sent_clf.fit(train_sent["X"], train_sent["Y"])
    evaluate(sent_clf, train_sent, "TRAIN SENT")
    evaluate(sent_clf, test_sent, "TEST SENT")

    
    # validate the quality of food classifier on food
    food_clf.fit(train_food["X"], train_food["Y"])
    evaluate(food_clf, train_food, "TRAIN FOOD")
    evaluate(food_clf, test_food, "TEST FOOD")

    print("=========== PROTECT SALAD WITH STYLE TRANSFER =======")
    target = np.load("yelp.salad.npy") # use a neutral sentiment salad corpus to obfuscate the unprotected sentiment corpus
    salad_label = np.array([0]*len(target))
    salad_sent_label = np.array([0]*1142 + [1]*1866)
    # validate the quality of sent classifier on food (negative)
    evaluate(food_clf, {"X": target,
                        "Y": salad_label}, "(UNPROTECTED) INFER SALAD ON SENT")
    
    obfus_table = transfer(target, np.load("yelp.steak.npy"))
    salad_X_obfus = apply_transfer(target, obfus_table)
    
    evaluate(food_clf, {"X": salad_X_obfus,
                        "Y": salad_label}, "(PROTECTED) INFER SALAD ON SENT")
    evaluate(sent_clf, {"X": salad_X_obfus,
                        "Y": salad_sent_label}, "(REGRESSION TEST) SENT PREDICTION")  


    print("=========== PROTECT STEAK WITH STYLE TRANSFER =======")
    target = np.load("yelp.steak.npy") # use a neutral sentiment salad corpus to obfuscate the unprotected sentiment corpus
    steak_label = np.array([1]*len(target))
    steak_sent_label = np.array([0]*769 + [1]*1236) 
    # validate the quality of sent classifier on food (negative)
    evaluate(food_clf, {"X": target,
                        "Y": steak_label}, "(UNPROTECTED) INFER STEAK ON SENT")

    # get the postitive steak
    pos_steak = np.load("yelp.steak.npy")[769:, :]
    pos_all = np.load("yelp.pos.npy")
    print(pos_steak.shape)
    print(pos_all.shape)
    
    obfus_table = transfer(pos_steak, pos_all) # use mixture methods
    steak_X_obfus = apply_transfer(target, obfus_table)
    
    evaluate(food_clf, {"X": steak_X_obfus,
                        "Y": steak_label}, "(PROTECTED with MIXTURE OBFUSC.) INFER STEAK ON SENT")
    evaluate(sent_clf, {"X": steak_X_obfus,
                        "Y": steak_sent_label}, "(REGRESSION TEST) SENT PREDICTION")  
if __name__ == '__main__':
    main()
