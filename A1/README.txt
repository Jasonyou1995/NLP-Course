'''
    CSE 490UA, Assignment 1
    Author: Jason You (ID: 1427878)
    Last Modified Date: Jan 18, 2017
'''

###
This two model is based on the corpus provided by the 'nltk' package. Please
remember to install 'nltk' by run the following command in terminal:

$ pip install nltk

If using the 'plot_result' function, 'numpy' and 'matplotlib' package will
also be needed
###

# class LangModel: 
Building language model based on Brown, Gutenberg, and Reuters corpora. Two model was trained and tested: Back-off and Interpolation model start from trigrams.

# __init__():
Obtaining data from 'nltk' package in Python, and split into train/test/dev corpora
by the given percentage.

# split_corpus(corpus_token, train_percent=0.6):
Return well splited train, test, and development dataset as a Python dictionary
(default is 60% to train, 20% to test, and 20% to dev).

# train_freq_dist(dataset):
Return the freqency distribution for each of the tri-, bi-, and uni-gram(s) as
a Python dictionary.

# define_unknown(train):
Assign '<UNK>' to all the words that count less than or eqauls to 1, returns a set.

# get_BO_hyperparam(freqdist, dataset):
Returns the Back-off hyper-parameter based on the frequency distribution and
the development corpus.

# back_off_model(params, sentence, freqdist):
Return a probability for each sentence given based on the given parameters and
freqency distribution, with the back-off tri-grams approach.

# interpolation_params(dataset, freqdist):
Return the parameters that optimize the performance of the interpolation model
(minimize the perplexity of the model)

# interpolation_model(params, sentence, freqdist):
Return a probability for each sentence give based on the given parameters and
the frequency distribution, with the interpolation tri-grams approach

# get_perplexity(dataset, params, freqdist, model='backoff'):
Return the calculated perplexity based on either back-off approach (default), or interpolation approach ('inter')