#Example code for clustering the dataset here: 
# https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#sphx-glr-auto-examples-text-plot-document-clustering-py
from sklearn.datasets import fetch_20newsgroups
from os import listdir
from os.path import isfile, join
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import gin
from sklearn.feature_extraction.text import CountVectorizer

# Constants indicating news categories
# Intended for use with experiment configuration
CATEGORY_COMP_WINDOWS_X = 'comp.windows.x'
CATEGORY_COMP_SYS_MAC_HARDWARE = 'comp.sys.mac.hardware'
CATEGORY_COMP_OS_MS_WINDOWS_MISC = 'comp.os.ms-windows.misc'

class NewsGroupDataset:
    @gin.configurable
    def __init__(self, shuffle=True, random_state=42, categories=[CATEGORY_COMP_WINDOWS_X, CATEGORY_COMP_SYS_MAC_HARDWARE, CATEGORY_COMP_OS_MS_WINDOWS_MISC]):
        self.shuffle = shuffle
        self.random_state = random_state
        self.categories = categories

    """
    Loads the data from the web for the three required categories. 
    This needs revising to work with the counting pipeline.
    """
    def load_data(self):
        self.dataset = fetch_20newsgroups(subset='all', categories=self.categories, 
                            download_if_missing=True, shuffle = self.shuffle, random_state = self.random_state)
        self.data = np.asarray(self.dataset.data)
        #Depending on if 2 or 3 categories are sent the dimensions change
        #however, only between 3-5k, not 7k... trying with all categories results in 18k.
        #an alternative would be rare word-removal (not yet implemented)
        vectorizer = TfidfVectorizer(min_df=10)
        vectors = vectorizer.fit_transform(self.data)
        self.vectors = vectors
        
    """
    Returns the full data set
    """
    def get_dataset(self):
        return self.dataset

    """
    Returns the vectorized data in tf-idf form. 
    """
    def get_vectors(self):
        return self.vectors

    """
    Returns the actual text documents for a category selected 
    from this files category list.

    Args:
        - category: One of CATEGORY_COMP_WINDOWS_X, CATEGORY_COMP_SYS_MAC_HARDWARE
            or CATEGORY_COMP_OS_MS_WINDOWS_MISC
    """
    def get_data_for_category(self, category):
        targetid = self.dataset.target_names.index(category)
        return self.data[self.dataset.target == targetid]

    """
    Returns the documents in an individual category.

    Args:
        - category: One of CATEGORY_COMP_WINDOWS_X, CATEGORY_COMP_SYS_MAC_HARDWARE
            or CATEGORY_COMP_OS_MS_WINDOWS_MISC
    """
    def get_targets_for_category(self, category):
        targetid = self.dataset.target_names.index(category)
        return self.data[self.dataset.target == targetid]

    #Todo: return with labels. First round of clustering can be done directly
    #on the vector data (unsupervised).

if __name__ =="__main__":
    categories = [CATEGORY_COMP_WINDOWS_X, CATEGORY_COMP_SYS_MAC_HARDWARE]
    #, CATEGORY_COMP_OS_MS_WINDOWS_MISC]
    setloader = NewsGroupDataset(categories=categories)
    setloader.load_data()
    dataset = setloader.get_dataset()
    
    cat_data = setloader.get_data_for_category(CATEGORY_COMP_OS_MS_WINDOWS_MISC)

