#Example code for clustering the dataset here: 
# https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#sphx-glr-auto-examples-text-plot-document-clustering-py
from sklearn.datasets import fetch_20newsgroups
from os import listdir
from os.path import isfile, join
import string

# Constants indicating news categories
CATEGORY_COMP_WINDOWS_X = 'comp.windows.x'
CATEGORY_COMP_SYS_MAC_HARDWARE = 'comp.sys.mac.hardware'
CATEGORY_COMP_OS_MS_WINDOWS_MISC = 'comp.os.ms-windows.misc'

class NewsGroupDataset:
    def __init__(self):
        self.dataset = fetch_20newsgroups(subset='all', categories=[CATEGORY_COMP_WINDOWS_X, CATEGORY_COMP_SYS_MAC_HARDWARE, CATEGORY_COMP_OS_MS_WINDOWS_MISC], 
                            download_if_missing=True)

    def get_dataset(self):
        return self.dataset

if __name__ =="__main__":
    setloader = NewsGroupDataset()
    dataset = setloader.get_dataset()
    
    print("%d documents" % len(dataset.data))
    #2936 documents
    print("%d categories" % len(dataset.target_names))

    #3 categories
    #dataset.data[dataset.target]
    print('what should we do?')
