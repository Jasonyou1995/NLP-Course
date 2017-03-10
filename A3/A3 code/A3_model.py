'''
    CSE 490UA, Assignment 3
    Author: Jason You (ID: 1427878)
    Last Modified Date: Feb 20th, 2017

    This script train bigram an trigram models with the Viterbi algorithm, and
    can be used for predicting labels given sequences of words.
'''

import os
import pickle
import numpy as np
from operator import itemgetter
from ast import literal_eval  # evaluating the data as lists


class ParseModel:
    def __init__(self):
        self.emission_count = {}
        self.trans_count = {}
        self.trans_count_trigram = {}

    # these steps need to be run before using any other method
    # Use the training data from the train_file to build a bigram model, and save it in the
    # same directory
    def count_and_save(self, train_file='A3_data/twt.train.json',
                       emission_file='bigram_emission.pickle',
                       bigram_file='bigram_trans.pickle',
                       trigram_file='trigram_trans.pickle'):
        train = load_file(train_file)

        # count the emission relation for each tag
        self.emission_count = dict()
        for sent in train:
            for elem in sent:
                word = elem[0]
                label = elem[1]
                if label not in self.emission_count:
                    self.emission_count[label] = {word: 0, 'sum': 0}
                if word not in self.emission_count[label]:
                    self.emission_count[label][word] = 0
                self.emission_count[label][word] += 1
                self.emission_count[label]['sum'] += 1

        self.smoothing(threshold=100)  # smooth the emission probability

        # count the transition relation for each pair (Bigram)
        self.trans_count = dict()
        for sent in train:
            labels = [elem[1] for elem in sent]
            labels = ['<*>'] + labels + ['<%>']  # add start and stop symbols
            for i in range(len(labels) - 1):
                y1 = labels[i]
                y2 = labels[i + 1]
                if y1 not in self.trans_count:
                    self.trans_count[y1] = {y2: 0, 'sum': 0}
                if y2 not in self.trans_count[y1]:
                    self.trans_count[y1][y2] = 0
                self.trans_count[y1][y2] += 1
                self.trans_count[y1]['sum'] += 1

        # count the transition relation for each pair (Trigram)
        self.trans_count_trigram = dict()
        for sent in train:
            labels = [elem[1] for elem in sent]
            labels = ['<*>', '<*>'] + labels + ['<%>']
            for i in range(len(labels) - 2):
                y1 = labels[i:(i + 2)]
                y1 = '(' + y1[0] + ', ' + y1[
                    1] + ')'  # (label1, label2) is an easy to read format
                y2 = labels[i + 2]
                if y1 not in self.trans_count_trigram:
                    self.trans_count_trigram[y1] = {y2: 0, 'sum': 0}
                if y2 not in self.trans_count_trigram[y1]:
                    self.trans_count_trigram[y1][y2] = 0
                self.trans_count_trigram[y1][y2] += 1
                self.trans_count_trigram[y1]['sum'] += 1

        # save files
        with open(emission_file, 'wb') as f:
            pickle.dump(self.emission_count, f)
        with open(bigram_file, 'wb') as f:
            pickle.dump(self.trans_count, f)
        with open(trigram_file, 'wb') as f:
            pickle.dump(self.trans_count_trigram, f)

    # smooth the emission probability by the given threshold, and count the occurrences
    # that are less than the threshold as unknown words.
    def smoothing(self, threshold=1):
        for label, words in self.emission_count.items():
            counts = np.array(list(words.values()))
            count_one = np.sum(counts == threshold)
            self.emission_count[label]['<UNK>'] = count_one
            self.emission_count[label]['sum'] += count_one

    # load the model by the given file names
    def load_model(self, emission_file='bigram_emission.pickle',
                   trans_file_bigram='bigram_trans.pickle',
                   trans_file_trigram='trigram_trans.pickle'):
        if not os.path.isfile(emission_file) or not os.path.isfile(trans_file_bigram) \
                or not os.path.isfile(trans_file_trigram):
            raise ValueError('Please give the correct file name or '
                             'build the model by running "count_and_save()" first.')
        with open(emission_file, 'rb') as f:
            self.emission_count = pickle.load(f)
        with open(trans_file_bigram, 'rb') as f:
            self.trans_count = pickle.load(f)
        with open(trans_file_trigram, 'rb') as f:
            self.trans_count_trigram = pickle.load(f)

    # Return a list of labels based on the given sentence (tokens) by using the Viterbi Algorithm
    def make_prediction(self, sent, my_model='bigram'):
        if self.emission_count == {} or self.trans_count == {} or self.trans_count_trigram == {}:
            raise ValueError('Please load the model by running "load_model()" first.')
        if my_model not in ('bigram', 'trigram'):
            raise ValueError('Please give the correct model name: "bigram" or "trigram".')

        result = [{}]  # [{label1:[back-pointer, score], label2:[...], ...}, {...}, ...]
        sent_len = len(sent)
        # all possible labels at the begining place
        start_labels = [x for x in self.trans_count['<*>'].keys() if (x != 'sum' and x != '<%>')]

        # Determine the proper start symbol
        if my_model == 'bigram':
            start_symbol = '<*>'
            trans_count = self.trans_count
        else:  # my_model == 'trigram'
            start_symbol = '(<*>, <*>)'
            trans_count = self.trans_count_trigram

        # Assign the transitional probablity to the first column (word)
        for label in start_labels:
                # update the transitional probability
                trans_prob = trans_count[start_symbol][label] / \
                             trans_count[start_symbol]['sum']
                # start symbol can only be one '<*>'
                result[0][label] = [trans_prob, '<*>']  # [score, back-pointer]
        
        # Find all the possible arg-max pathes
        for i in range(sent_len):
            result.append({})  # store all the possible pathes to the next word
            possible_labels = result[i].keys()
            # update the emission probabilities to the current column(word)
            for label in possible_labels:
                word = sent[i]
                if word not in self.emission_count[label]:
                    word = '<UNK>'
                emission_prob = self.emission_count[label][word] / \
                                self.emission_count[label]['sum']
                result[i][label][0] *= emission_prob  # update the score for this cell
            
            all_labels = [x for x in self.trans_count.keys() if x != '<*>']  # labels for next word
            if i == (sent_len - 1):
                # the stop case
                all_labels = ['<%>']
            for next_label in all_labels:
                # fill the next column with arg_max paths
                arg_max = [0.0, '<?>']  # [score, back-pointer]
                for this_label in possible_labels:
                    history = this_label
                    if my_model == 'trigram':   # use previous two words, instead of one
                        back_pointer = result[i][this_label][1]  # extract the previous label
                        history = '(' + back_pointer + ', ' + this_label + ')'
                    if history in trans_count and next_label in trans_count[history]:
                        pre_score = result[i][this_label][0]
                        score = trans_count[history][next_label] / \
                                trans_count[history]['sum'] * pre_score
                        if score >   arg_max[0]:
                            arg_max = [score, this_label]
                result[i + 1][next_label] = arg_max

        # Backtrack the labels, start from the stop symbol '<%>'
        best_result = result[sent_len]['<%>']  # [score, back-pointer]
        predict_labels = []
        for i in range(sent_len - 1, -1, -1):
            back_pointer = best_result[1]
            predict_labels = [back_pointer] + predict_labels
            best_result = result[i][back_pointer]
        return predict_labels


    # Store and return sequence of predicted labels that are based on te given test data.
    # The size the of the result is the same as the test dataset.
    def get_result(self, save=None, test_file='A3_data/twt.test.json', model='bigram'):
        test = load_file(test_file)
        predict_labels = []
        for sent in test:
            test_sentence = [x[0] for x in sent]  # extract the tokens of sentence
            predict_labels.append(self.make_prediction(test_sentence, my_model=model))

        # save and return the predictions
        if save == None:
            if model == 'bigram':
                save = 'bigram-prediction.pickle'
            else:  # trigram model
                save = 'trigram-prediction.pickle'
        with open(save, 'wb') as f:
            pickle.dump(predict_labels, f)
        return predict_labels


# Pre : get the proper file path
# Post: returns a list of tagged data sequences
def load_file(filename):
    with open(filename) as f:
        data = f.read().split('\n')
    del data[-1]  # remove the last empty element
    data = [literal_eval(d) for d in data]
    return data


def main():
    model = ParseModel()
    model.count_and_save()  # only run in the first time to build the models
    model.load_model()
    # results will be stored in the same directory, and evaluated by 'A3_eval.py'
    # predict_bigram = model.get_result(model='bigram')
    predict_trigram = model.get_result(model='trigram')

    # # The code is run on the dev-set, and is not part of the model.
    # # use dev-set to tune the threshold of unknown words in the 'count_and_save()' method
    # model.count_and_save()
    # dev = load_file('A3_data/twt.dev.json')
    # with open('trigram-dev.pickle', 'rb') as f:
    #     prediction = pickle.load(f)
    # pred_token = np.array([item for sublist in prediction for item in sublist])
    # dev_token = [x[1] for pair in dev for x in pair]
    # dev_token = np.array([item for sublist in dev_token for item in sublist])
    # print(sum(pred_token == dev_token) / len(dev_token) * 100., '\%', sep='')


if __name__ == '__main__':
    main()
