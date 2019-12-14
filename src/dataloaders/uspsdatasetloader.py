import data_utils
import gin
import h5py
import numpy

class UspsDatasetLoader:
    @gin.configurable   
    def __init__(self, path=data_utils.get_datasets_folder() + "/usps/usps.h5"):
        self.path = path

    def load_dataset(self):
        with h5py.File(self.path, 'r') as hf:
            train = hf.get('train')
            test = hf.get('test')
            self.train = train
            self.test = test

            X_te = test.get('data')[:]
            y_te = test.get('target')[:]

            X_tr = train.get('data')[:]
            y_tr = train.get('target')[:]

            print('loaded')


    """
    Returns all the data, in a tuple (input, output)
    """
    def get_full_dataset(self):
        return (self.train, self.test)

    def visualize(document):
        #Do nothing
        return
        
if __name__=="__main__":
    dataset = UspsDataset()
    dataset.load_data()
