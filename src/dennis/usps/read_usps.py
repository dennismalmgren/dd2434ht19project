import h5py
import gin
import os

class UspsDataset:
    @gin.configurable   
    def __init__(self, path="./datasets/usps/usps.h5"):
        self.path = path

    def load_data(self):
        with h5py.File(self.path, 'r') as hf:
            train = hf.get('train')
            X_tr = train.get('data')[:]
            y_tr = train.get('target')[:]
            test = hf.get('test')
            X_te = test.get('data')[:]
            y_te = test.get('target')[:]

if __name__=="__main__":
    dataset = UspsDataset()
    dataset.load_data()
