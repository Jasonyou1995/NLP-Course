'''
    CSE 490UA, Assignment 3
    Author: Jason You (ID: 1427878)
    Last Modified Date: Feb 20th, 2017

    Evaluating the predictions made my the bigram and trigram HMM
'''

import pickle
import numpy as np
from ast import literal_eval  # evaluating the data as lists


# Pre : get the proper file path
# Post: returns a list of tagged data sequences
def load_file(filename):
    with open(filename) as f:
        data = f.read().split('\n')
    del data[-1]    # remove the last empty element
    data = [literal_eval(d) for d in data]
    return data

def main():
    # load test file and predictions
    test = load_file('A3_data/twt.test.json')
    with open('bigram-prediction.pickle', 'rb') as f:
        predict_bigram = pickle.load(f)
    with open('trigram-prediction.pickle', 'rb') as f:
        predict_trigram = pickle.load(f)

    # clear the test data and predictions
    pred_token_bigram = np.array([item for sublist in predict_bigram for item in sublist])
    pred_token_trigram = np.array([item for sublist in predict_trigram for item in sublist])
    test_token = [x[1] for pair in test for x in pair ]
    test_token = np.array([item for sublist in test_token for item in sublist])

    # print result
    print('The accuracy of using bigram model to parse:')
    print('%.2f%%' % (sum(pred_token_bigram == test_token) / len(test_token) * 100))
    print('The accuracy of using trigram model to parse:')
    print('%.2f%%' % (sum(pred_token_trigram == test_token) / len(test_token) * 100))

if __name__ == '__main__':
    main()
