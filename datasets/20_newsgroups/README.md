Dataset analysis
=================
The use of the data set is explained in the paper "Partially labeled classification with Markov random walks" .

1. The data set is named the "20 newsgroups dataset"
2. The data set is supposed to be processed as "20news-18827" which by the dataset standard indicates 18827 documents. Rare words and duplicate documents were removed from an original (unnamed) data set.
3. A mac and windows category were used. 
4. There were 958 and 961 examples in the two classes, with 7511 dimensions.

Unfortunately no data set we managed to produce thus far matches these numbers. Here is a brief breakdown of the issues.
1. The original site for the data set, referenced in the paper and in several other papers, does no longer exist. The original author does have a new website set up but no 18827 version of the dataset exists. An 18848 does exist, where duplicate documents have been removed. An 18828 does also exist, where additional headers have been removed.
2. There are no mac and windows categories, there are however three candidate categories. 'comp.windows.x', 'comp.sys.mac.hardware' and 'comp.os.ms-windows.misc'.
3. In the 18828 dataset, the categories do not hold the number of samples expected.
    -'comp.os.ms-windows.misc': 985
    -'comp.windows.x': 980 
    -'comp.sys.mac.hardware': 961
4. There is no indication of which threshold of rarity was used in the original paper. 

Recommended assumptions and strategy
====================================
Given the constraints, I recommend we
-Use the scikit-learn api for working with the 18848 dataset.
  Duplicates have been removed and most irrelevant headers have been removed. From a data science perspective the data set has good qualities, the correct categories with data that should be distributed similarly to the experiment in the paper, enough to argue the validity of any results obtained, especially if also measured against the random walk implementation in the original dataset paper.
-Use the 'comp.sys.mac.hardware' but test against both 'comp.os.ms-windows.misc' and 'comp.windows.x' data sets for comparison. 
-Set the threshold for rarity to 10 and remove rare words.

Dataset processing analysis
===========================

Additional notes
================
The website "A topic model for the Twenty Newsgroups data" working with the 18848 data set indicates that removal of rare words and stop words in the training set of the documents resulted in the removal of one additional document. A hypothesis would be that this would have been how the original authors ended up with 18827.

References
==========
Paper link:
Partially labeled classification with Markov random walks
https://papers.nips.cc/paper/1967-partially-labeled-classification-with-markov-random-walks.pdf

20 Newsgroups
http://qwone.com/~jason/20Newsgroups/

A topic model for the Twenty Newsgroups data
https://ldavis.cpsievert.me/newsgroup/newsgroup.html