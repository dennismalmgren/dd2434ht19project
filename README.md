# dd2434ht19project
Project repository for working with the Kernel assignment

Pawels paper instructions can be found here

https://kth.instructure.com/courses/12423/files/2381305/download

The scientific paper can be found here

http://papers.nips.cc/paper/2257-cluster-kernels-for-semi-supervised-learning.pdf


Setup python environment then run pip install -r requirements.txt


To manage experiment configurations we use
https://github.com/google/gin-config

##E-level data sets

### Dataset 1
- Dataset 1: Text classification task. The mac and windows subsets from the 20 newsgroups dataset, 
- From the paper: [Processed as 20news-18827](http://www.ai.mit.edu/˜jrennie/20Newsgroups/), removing rare words, duplicate documents, and performing tf-idf mapp

#### Notes from Dennis

Website no longer exists, but jrennie is at the bottom of [this page](http://qwone.com/~jason/20Newsgroups/) which Kaggle references as the 'original source'. Neither of the datasets are named '18827', however, there is an 18828. I haven't validated if they are the same, '18827' is mentioned in multiple books and papers.

scikit-learn has [helper functions for this dataset](https://scikit-learn.org/stable/datasets/index.html), functions fetch_20newsgroups and fetch_20newsgroups_vectorized,  but it is uncertain how well they correspond to our requirements. 

From [another source](https://www.hybrid-analysis.com/sample/5401f33287dca3f6f7af135a4eb40b5fb990f864c2bb6e520305f4200654ad7f?environmentId=100) I found: 

> "I initially published the original 20 Newsgroups data set as20news-19996.tar.gz. [Rainbow](http://www.cs.cmu.edu/~mccallum/bow) counts 19996documents using the default processing arguments. It skipssci.crypt/16017 because of the funky signature. I found 1169duplicates and subtracted to get 18827, even though there were 18828documents in the 20news-18827.tar.gz package. I've renamed thepackages to 20news-19997.tar.gz and 20news-18828.tar.gz to reflect theactual number of documents in each data set."

The original source link does have a 19997 so it's a next action to see if removal of duplicates leads us to the correct data set. 

**To summarize**:

Original data set was probably used (19997). Removing rare words (open question if this was done before or after 19997), removing duplicates (leads to 18827), and then tf-idf mapp refers to term frequency-inverse document frequency.
This is another description of the process: The dataset was 20news-18827, which consists of the 20newsgroups data with
headers and duplicates removed, and was preprocessed to remove all punctuation,
capitalization, and distinct numbers.
([Source](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.908.7148&rep=rep1&type=pdf). I'm fairly confident the note '20newsgroups data' refers to the original data set of 19997)


Two categories, mac and windows, with respectively 958 and 961 examples of dimension 7511. 

Next step is also to identify the format of the data when fed into the clustering algorithm.

### Dataset 2
The task of classifying the handwritten digits 0 to 4 against 5 to 9 of the USPS database. There is no actual source explained, but Dataset is available [here](https://www.kaggle.com/bistaumanga/usps-dataset).

So there are multiple flaws in the report reducing the ability to reproduce the experimental results.

