'''
Assignment 2
CSE490UA
Jason You, ID: 1427878
'''
# Files used in the A2.py
#
# 1. enwik9, URL: https://code.google.com/archive/p/word2vec/
# 2. glove.6B.Xd.txt (X can be 50, 100, 200 or 300), URL: http://nlp.stanford.edu/projects/glove/

Class WordEmbed: 
	implementing the Word2Vec and GloVe word embeding to build word analogy models

word2vecmodel(self, test_data, modelname):
	return a predicted word based on the list of three words provided based on the Word2Vec word embeding

glovemodel(self, test_data):
	return a predicted word based on the list of three words provided based on the GloVe word embedding

getvectors(self, filename):
	return a dictionary that map word to n-dimensional vectors represent this word based on the given file

validatewords(self, words, keys):
	convert all unknown word to '<unk>'

gettestdata(self, filename, choose):
	return a dictionary maps chosen topics to list of four words sequences

evaluation(self, testdata):
	return a dictionary of accuracies for each topic chosen