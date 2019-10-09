##

import zmq
import json
import time
from io import BytesIO
import pickle



class LMClient(object):
    def __init__(self, name = None, port = 5432):
        self.name = name
        context = zmq.Context()
        #  Socket to talk to server
        # print("Connecting to LM server…")
        self.socket = context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:"+str(port))

    # @param: sents are in the form of array of strings
    def encode(self, sents):
        self.socket.send_json(json.dumps(sents))
        # print(sents)
        # time.sleep(1)
        message = self.socket.recv() # a binary file of pickled numpy string
        embs = pickle.load(BytesIO(message))
        # print(embs.shape)
        return embs
        
        

if __name__ == '__main__':
    PATH = "/home/mlsnrs/data/data/yyf/Py/bert_privacy_Yan/data/Airline/Target/test.txt"
    test_sents = list(open(PATH, 'r'))
    # test_sents = list(open('data/part_fake_5/ankle.0.txt', 'r'))
    test_sents = test_sents[:10]

    client = LMClient()
    for request in range(10):
        embs = client.encode(test_sents)
        # embs = client.encode(['[CLS]'])
        print(embs.shape)

