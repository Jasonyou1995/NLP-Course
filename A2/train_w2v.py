'''
Author: Jason You
This script is used to train the model of word2vec
'''

import os, gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class ReadSentence(object):
    def __init__(self, filename):
        self.fname = filename

    def __iter__(self):
        for line in open(self.fname):
            yield line.split()

def main(corpusfile):
    sentences = ReadSentence(corpusfile)
    model = gensim.models.Word2Vec(sentences, size=200, workers=4)  # train the word2vec model
    model.save('enwik9.model')
    # model.accuracy('word_test_v1.txt')  # used for testing the model

if __name__ == '__main__':
    main('enwik9')