'''
    CSE 490UA, Assignment 2
    Author: Jason You (ID: 1427878)
    Last Modified Date: Feb 2nd, 2017
'''

from operator import itemgetter
import numpy as np
from gensim.models import word2vec
import logging  # used to obtain the run time information of gensim (word2vec)
from datetime import datetime
from scipy.spatial.distance import cosine  # defined as 1 - cos(theta) for two vectors
import pickle

# Initiating the word2vec model
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class WordEmbed:

    def __init__(self, vectorfile=''):
        self.w2v_model = ''       # define the word2vec model for later use
        if vectorfile:
            self.vectors = self.getvectors(vectorfile)

    # Pre : test data should be a list/tuple of four words, with the fourth word to be predicted.
    # Post: returns a predicted word based on the analogy word pairs. (by Word2Vec word embedding)
    def word2vecmodel(self, test_data, modelname):
        if not bool(self.w2v_model):        # check if the word2vec model is initiated
            self.w2v_model = word2vec.Word2Vec.load(modelname)
        if len(test_data) not in [3, 4] or \
                any([type(word) is not str for word in test_data]):
            raise ValueError('Input must be a sequence of 3 or 4 words.')

        if all([word in self.w2v_model.wv.vocab for word in test_data]):
            return self.w2v_model.most_similar(positive=[test_data[1], test_data[2]],
                                               negative=[test_data[0]], topn=1)[0][0]
        else:
            return '<not found>'

    # Pre : test data should be a list/tuple of four words, with the fourth word to be predicted.
    # Post: returns a predicted word based on the analogy word pairs (by GloVe word embedding)
    def glovemodel(self, test_data):
        if len(test_data) not in [3, 4] or \
                any([type(word) is not str for word in test_data]):
            raise ValueError('Input must be a sequence of 3 or 4 words.')

        # get the vectors for the first three vocabularies
        v1, v2, v3 = self.validatewords(test_data[0:3], self.vectors.keys())
        vector1 = self.vectors[v1]
        vector2 = self.vectors[v2]
        vector3 = self.vectors[v3]
        a = vector3 + vector2 - vector1    # vector 'a' is used for finding the most similar vector

        most_similar = ('', 100)           # store the most similar word
        for word in self.vectors.keys():
            if word not in (v1, v2, v3):
                b = self.vectors[word]               # vector 'b' as a test vector
                similarity = cosine(a, b)            # from the scipy package
                if similarity < most_similar[1]:     # update the similarity
                    most_similar = (word, similarity)
        return most_similar[0]

    # Pre : take a file path where the word embedding vectors are stored
    # Post: returns a dict with key to be the words, and values to be the vectors
    def getvectors(self, filename):
        with open(filename) as file:
            data = file.readlines()
        vectors = {}
        for vector in data:
            elem = vector.strip().split(' ')
            vectors[elem[0]] = np.array(elem[1:], dtype=float)  # store as numpy array
        return vectors

    # Pre : get a sequence of words and a sequence of keys (pretrained words) from the client
    # Post: returns a sequence that replace the unknown word by '<unk>'
    def validatewords(self, words, keys):
        result = []
        for word in words:
            result += (['<unk>'] if word not in keys else [word])
        return result

    # Pre : get the file name of the test data, and give a dictionary of chosen topics
    # Post: returns a list of lists, with four analogy words in each sub-list
    def gettestdata(self, filename, choose):
        with open(filename) as file:
            data_str = file.read().lower()  # convert to lower case
        parts = data_str.split(':')[1:]
        result = {}
        for part in parts:
            part = part.strip().split('\n')  # trim whitespaces in both ends and split
            topic = part[0].strip()
            if topic in choose:
                result[topic] = [words.strip().split() for words in part[1:]]
        return result

    # Pre : get a file of testdata with different topics
    # Post: return a dictionary of accuracies for each topic chosen
    def evaluation(self, testdata):
        accuracies = {}  # a dictionary to store the accuracies for each topic
        for topic in testdata.keys():
            total = 0
            correct = 0
            for line in testdata[topic]:
                total += 1
                prediction = self.glovemodel(line)
                if line[3] == prediction:
                    correct += 1
            accuracies[topic] = correct / total
            print("%s: %d/%d" % (topic, correct, total))  # print the accuracy for this topic
        print('Total accuracies: ', accuracies)
        return accuracies

def main():
    word_embed = WordEmbed('vectors.txt')    # use the proper embedded-words vector file name
    choose = ['capital-world', 'currency', 'city-in-state', 'family',
              'gram1-adjective-to-adverb', 'gram2-opposite', 'gram3-comparative',
              'gram6-nationality-adjective']
    my_choose = ['my-test-one-sport', 'my-test-two-food']

    # word2vec model
    w2v_model = word2vec.Word2Vec.load('enwik9.model')
    w2v_result = w2v_model.accuracy('questions-words.txt')
    '''
    capital-world: 31.9% (53/166)
    currency: 11.1% (2/18)
    city-in-state: 6.2% (39/626)
    family: 74.6% (179/240)
    gram1-adjective-to-adverb: 26.3% (145/552)
    gram2-opposite: 55.3% (73/132)
    gram3-comparative: 80.3% (747/930)
    gram6-nationality-adjective: 56.7% (697/1229)
    '''

    # glove model
    starttime = datetime.now()      # Timing the whole model
    testdata = word_embed.gettestdata('word-test.v1.txt', choose)
    accuracies = word_embed.evaluation(testdata)
    print('Total time:', datetime.now() - starttime)  # print the timing information
    '''
    gram3-comparative: 291/1332
    city-in-state: 664/2467
    family: 182/506
    gram2-opposite: 26/812
    currency: 22/866
    capital-world: 992/4524
    '''

    # my tests
    my_test_glove = word_embed.gettestdata('my-test.txt', my_choose)
    my_test_w2v = w2v_model.accuracy('my-test.txt')
    '''
    0 accuracy for both tests
    '''


if __name__ == '__main__':
    main()
