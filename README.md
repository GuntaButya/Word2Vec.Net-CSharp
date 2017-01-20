# Word2Vec.NET C Sharp Port



Word2Vec.Net C# from Tomas Mikolov's Word2Vec Toolkit - Please Note: The code in this repository is partly based on work from eabdullin/Word2Vec.Net



Tomas Mikolov changed the world of ML when releasing the paper and the code for Word2Vec! 

It is, however, quite unclear as to what the structures are of the classes/applications and what they do. It is also hard to follow due to lack of Code Documentation. The paper, below, does a very good job at describing the basic model, but not how it all works. Difficult to grasp for beginners!!!



I am still grasping much of this, but some fundamentals below.




# Word2Vec:


Specifically trains a Model/Vocabulary on a Corpus of Text (Corpus.txt).





# Distance:


Specifically a Vector Operation on the Vocabulary (Vectors.bin). 

Finds the closest Word.



Minimum Input: One Word





# WordAnalogy:


Specifically a Vector Operation on the Vocabulary (Vectors.bin). 

Looks for Patterns. 



EG:



vec(“Madrid”) - vec(“Spain”) + vec(“France”) is closer to vec(“Paris”) than to any other word vector




Another common analogy:




vec(“man”) - vec(“king”) + vec(“woman”) is closer to vec(“woman”) than to any other word vector.



Also:



vec(“one”) - vec(“two”) + vec(“three”) - can produce some very interesting results! A Machine that can Count!


    Word: 1  Position in vocabulary: 520
    Word: 2  Position in vocabulary: 561
    Word: 3  Position in vocabulary: 738
                                                  Word       Cosine distance
    ------------------------------------------------------------------------
                                                     4           0.9748517
                                                     5           0.9639122
                                                     6           0.9504161
                                                     7           0.9439411
                                                     8           0.9344452
                                                     9           0.9217442
                                                    10           0.8888409
                                                    11           0.8875012
                                                    23           0.8747311
                                                    15           0.8657212


Examples do vary depending on the training parameters.



Minimum Input: Three Words






# Word2Phrase:



A work still in progress...





# ComputeAccuracy:



A work still in progress...





# Data Structure:



The way the output file is written to (Vectors.txt/Vectors.bin), vary depending on the input parameters to the Word2Vec Constructor. See the Word2Vec Class.



Generally, the structure is for Vectors:


    Words.Count VectorSpace.Count

    </s>

    word 0.025 0.012 0.112...



Where Words.Count is the total Words in the Vocabulary, VectorSpace.Count is the Vectors specified by the 'size' variable input to the Word2Vec Class. '/s' is the start to the Vocabulary. Also where. 'word' is the first word in the Vocabulary, and the following Vectors are the Dimensions specified by the 'size' variable input to the Word2Vec Class.

This repeats, single word followed by Vectors.Count.

It is worth noting, the Vectors are of type: float


# Improvements:


The classes in this repository can Load the Google News Corpus (5Gb) and other very large Datasets/Corpus Texts! These classes can be H/W Intensive, requiring CPU and RAM in copious amounts!!!



Please Note: This code has not been optimised! I have kept the format and the Code structure as close to the original as possible in an effort to stay inline with Tomas Mikolov's work and also so it was readible between the C# code and also the C Code. Particular effort has been made not to change anything.



Machine Learning really does need Word2Vec and its Classes/Applications surrounding the Word Vectors. Credit to all who work on Source Code and share - This makes the world a better place!




# See Word2Vec - By Tomas Mikolov



"This tool provides an efficient implementation of the continuous bag-of-words and skip-gram architectures for computing vector representations of words. These representations can be subsequently used in many natural language processing applications and for further research."


Project: https://code.google.com/archive/p/word2vec/#


Citations: https://scholar.google.com/citations?user=oBu8kMMAAAAJ


Questions: https://groups.google.com/forum/#!forum/word2vec-toolkit


Source Code: https://code.google.com/archive/p/word2vec/source/default/source


Paper: https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
