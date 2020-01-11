import gin
import h5py
import numpy as np
import os
"""
Returns the path to the datasets folder based on the location of utils.py
"""


def get_datasets_folder():
    scriptpath = os.path.realpath(
        __file__
    )  # This refers to the utils file which needs to be in src for this to work. Should be enough for this simple project.
    cur_dir = os.path.dirname(scriptpath)
    base_dir = os.path.dirname(cur_dir)
    base_dir = os.path.dirname(base_dir)
    datasets_dir = os.path.join(base_dir, 'datasets')
    return datasets_dir


class UspsDatasetLoader:
    @gin.configurable
    def __init__(self, path=get_datasets_folder() + "/usps/usps.h5"):
        self.path = path

    def load_dataset(self):
        with h5py.File(self.path, 'r') as hf:
            train = hf.get('train')
            test = hf.get('test')
            self.train = train
            self.test = test

            self.X_te = test.get('data')[:]
            self.y_te = test.get('target')[:]
            self.y_te = self._to_binary_label(self.y_te)

            self.X_tr = train.get('data')[:]
            self.y_tr = train.get('target')[:]
            self.y_tr = self._to_binary_label(self.y_tr)

            print('loaded')

    """
    Returns all the data, in a tuple (input, output)
    """

    def get_full_dataset(self):
        a1 = np.concatenate((self.X_tr, self.X_te))
        a2 = np.concatenate((self.y_tr, self.y_te))
        return (a1, a2)

    def get_training_dataset(self):
        return (self.X_tr, self.y_tr)

    def get_test_dataset(self):
        return (self.X_te, self.y_te)

    def _to_binary_label(self, labels):
        """Transform the 10-class dataset to 2-class dataset:
            - original label = 0-4 -> new label = 0
            - original label = 5-9 -> new label = 1
        """
        labels[labels < 5] = 0
        labels[labels >= 5] = 1
        return labels


if __name__ == "__main__":
    dataset = UspsDatasetLoader()
    dataset.load_dataset()
    x, y = dataset.get_full_dataset()
    x, y = dataset.get_training_dataset()
    x, y = dataset.get_test_dataset()
