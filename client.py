##

import zmq
import json
import time
from io import BytesIO
import pickle



class LMClient(object):
    def __init__(self):
        context = zmq.Context()
        #  Socket to talk to server
        print("Connecting to LM server…")
        self.socket = context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:5555")

    # @param: sents are in the form of array of strings
    def encode(self, sents):
        self.socket.send_json(json.dumps(sents))
        time.sleep(1)
        message = self.socket.recv() # a binary file of pickled numpy string
        embs = pickle.load(BytesIO(message))
        print(embs.shape)
        return embs
        
        




if __name__ == '__main__':
    test_sents = list(open('data/medical.test.txt', 'r'))
    test_sents = test_sents[:256]

    client = LMClient()
    for request in range(10):
        client.encode(test_sents)
