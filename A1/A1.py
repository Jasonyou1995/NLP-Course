'''
    CSE 490UA, Assignment 1
    Author: Jason You (ID: 1427878)
    Last Modified Date: Jan 18, 2017
'''


import nltk                                         # the NLP toolkit
from math import log2                               # used for finding perplexity
from nltk.corpus import brown, gutenberg, reuters   # build in corpora

class LangModel:

    # read in the corpora as word tokens from the 'nltk' package, and then build the
    # train, test, development data sets (train: 60%, test: 20%, dev: 20%)
    def __init__(self):
        self.brown_data = self.split_corpus(brown.sents(), train_percent=0.6)
        self.gutenberg_data = self.split_corpus(gutenberg.sents(), train_percent=0.6)
        self.reuters_data = self.split_corpus(reuters.sents(), train_percent=0.6)

    # Pre : corpus should be tokens, and train percentage is between 0 and 1 exclusive.
    # Post: return train, test and development corpus sets from the given corpus as a dictionary.
    #       Test and development percentage is the same: half of the remaining part after
    #       setting the train set.
    def split_corpus(self, corpus_token, train_percent=0.6):
        if train_percent >= 1:
            return 'train percentage must between 0 and 1 exclusive.'

        corpus_token = [sent + ['<STOP>'] for sent in corpus_token]  # add <STOP> sign to each
        cop_length = len(corpus_token)                               # store the corpus length
        test_percent = (1 - train_percent) / 2                       # percentage of the test set

        # selecting train, test, and dev from the corpus
        train = corpus_token[:int(cop_length * train_percent)]
        test = corpus_token[int(cop_length * train_percent):
                            int(cop_length * (train_percent + test_percent))]
        dev = corpus_token[int(cop_length * (train_percent + test_percent)):]
        return {'train': train, 'test': test, 'dev': dev}

    # Pre : get a corpus data set, and the 'train' must include
    # Post: count the frequency distributions of tri-grams, bi-grams, and uni-gram,
    #       and return these frequency distributions with the count of total words as a dict.
    def train_freq_dist(self, dataset):
        if 'train' not in dataset.keys():
            return '\'train\' key must included in the dictionary.'

        train = dataset['train']
        unk_words = self.define_unknown(train)     # get a set of unknown words
        unigram, bigrams, trigrams = [], [], []

        # assign trigrams, bigrams, and unigram
        for sent in train:
            # convert all words in unk_words list to '<UNK>'
            sent = ['<UNK>' if word in unk_words else word for word in sent]

            trigrams += nltk.trigrams(sent)
            bigrams += nltk.bigrams(sent)
            unigram += sent

        # assign freqency distributions to each gram
        unigram_fd = nltk.FreqDist(unigram)
        bigrams_fd = nltk.FreqDist(bigrams)
        trigrams_fd = nltk.FreqDist(trigrams)
        return {'unigram_fd': unigram_fd, 'bigrams_fd': bigrams_fd, 'trigrams_fd': trigrams_fd,
                'count_uni': len(unigram)}

    # Pre : get the train corpus
    # Post: return a set of unknown words (frequency <= 1) found in this train corpus
    def define_unknown(self, train):
        vocabulary = []
        for sent in train:
            vocabulary += sent
        vocab_fd = nltk.FreqDist(vocabulary)

        unk_words = set()
        for key, value in vocab_fd.items():
            if value <= 1:
                unk_words.add(key)
        return unk_words

    # Pre : get the frequency distribution and a corpus data set ('dev' must included)
    # Post: return the hyper-parameters (B1 and B2) for the Back-off model
    def get_BO_hyperparam(self, freqdist, dataset):
        if 'dev' not in dataset.keys():
            return '\'dev\' key must included in the dictionary.'

        dev = dataset['dev']
        dev_trigrams = []
        for sent in dev:
            dev_trigrams += nltk.trigrams(sent)
        absense_rate = 1 - sum([gram not in freqdist['trigrams_fd'].keys()
                                for gram in dev_trigrams]) / len(dev_trigrams)

        param_list1 = []  # store the hyper-parameter of trigrams for each sentence in the dev
        for sent in dev:
            sent_trigrams = nltk.trigrams(sent)
            # temporary parameter
            beta1 = 1 - sum([freqdist['trigrams_fd'][gram] / freqdist['bigrams_fd'][gram[0:2]]
                             for gram in sent_trigrams
                             if gram in freqdist['trigrams_fd'].keys()])
            param_list1 += [beta1]

        # parameter for the percentage of back-off bi-grams case
        alpha1 = absense_rate * abs(sum(param_list1) / len(param_list1))

        # parameter for the percentage of back-off uni-gram case
        alpha2 = abs(absense_rate - alpha1)

        # set the upper bound to be 0.3
        if alpha1 > 0.5:
            alpha1 = 0.3
        if alpha2 > 0.5:
            alpha2 = 0.3

        return {'alpha1': alpha1, 'alpha2': alpha2}

    # Pre : get dataset (with test corpus in it), parameters, and frequency distribution
    # Post: return the probability for each sentence
    def back_off_model(self, params, sentence, freqdist):
        sentence = ['<UNK>' if word not in freqdist['unigram_fd'] else word
                    for word in sentence]
        trigrams = list(nltk.trigrams(sentence))     # generate trigrams for the test sentence
        result = 1
        for gram in trigrams:
            if gram in freqdist['trigrams_fd']:
                result *= freqdist['trigrams_fd'][gram] / freqdist['bigrams_fd'][gram[0:2]]
            elif gram[0:2] in freqdist['bigrams_fd']:
                result *= params['alpha1'] * freqdist['bigrams_fd'][gram[0:2]] / \
                       freqdist['unigram_fd'][gram[0]]
            elif gram[0] in freqdist['unigram_fd']:
                result *= freqdist['unigram_fd'][gram[0]] / freqdist['count_uni']
        return result

    # return three parameters that give the lowest perplexity for the interpolation model
    def interpolation_params(self, dataset, freqdist):
        params = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        perp_list = []
        dev = dataset['dev']
        for p1 in params:
            p2 = (1 - p1) / 2   # define the second parameter to be half of the remaining of p1
            p3 = p2
            sent_prob = []
            p = [p1, p2, p3]
            perp_list += [self.get_perplexity(dataset, [p1, p2, p3], freqdist, 'interp')]

        min_index = perp_list.index(min(perp_list))  # find the parameter that minimize perplexity
        best_param = params[min_index]
        return [best_param, (1 - best_param) / 2, (1 - best_param) / 2]

    # using the interpolation model to return a probability for the given sentence
    def interpolation_model(self, params, sentence, freqdist):
        sentence = ['<UNK>' if word not in freqdist['unigram_fd'] else word
                    for word in sentence]   # convert unseen words in this sentence to '<UNK>'
        trigrams = nltk.trigrams(sentence)

        result = 1
        for gram in trigrams:
            part1, part2, part3 = 0, 0, 0
            if gram in freqdist['trigrams_fd']:
                part1 = params[0] * freqdist['trigrams_fd'][gram] / \
                        freqdist['bigrams_fd'][gram[0:2]]
            if gram[0:2] in freqdist['bigrams_fd']:
                part2 = params[1] * freqdist['bigrams_fd'][gram[0:2]] / \
                        freqdist['unigram_fd'][gram[0]]
            if gram[0] in freqdist['unigram_fd']:
                part3 = params[2] * freqdist['unigram_fd'][gram[0]] / freqdist['count_uni']
            if part1 + part2 + part3 > 0:
                result *= (part1 + part2 + part3)
        return result

    # Pre : get a data set with test corpus in it, choose 'backoff' or 'interp' model
    # Post: return the perplexity of this model
    def get_perplexity(self, dataset, params, freqdist, model='backoff'):
        if model not in ['backoff', 'interp']:
            return 'Model must be either \'backoff\' or \'interp\''
        sent_prob = []
        total_words = 0
        for sent in dataset['test']:
            if model == 'backoff':
                sent_prob += [self.back_off_model(params, sent, freqdist)]
            else:  # use interpolation model
                sent_prob += [self.interpolation_model(params, sent, freqdist)]
            total_words += len(sent)

        log_sum = sum([log2(x) for x in sent_prob if 0 < x < 1])
        return 2 ** ((-1 / total_words) * log_sum)

    # Pre : two result must have same length and passed in as tuples
    # Post: plot the bar graph for perplexity comparision of two models
    def plot_result(self, backoff_result, interp_result):
        if len(backoff_result) != len(interp_result):
            return 'Two given results must be in the same length.'

        import numpy as np
        import matplotlib.pyplot as plt

        indent = np.arange(len(backoff_result))  # start points for each group of bars
        width = 0.36                             # length of each bar
        fig, ax = plt.subplots()
        rects1 = ax.bar(indent, backoff_result, width, color='r')
        rects2 = ax.bar(indent+width, interp_result, width, color='y')

        ax.set_ylabel('Perplexity')
        ax.set_title('Perplexity by Back-off and Interpolation')
        ax.set_xticks(indent+width)
        ax.set_xticklabels(('Brown', 'Gutenberg', 'Reuters'))
        ax.legend((rects1[0], rects2[0]), ('Back-off', 'Interpolation'))

        plt.show()

# main function: displays the estimating and performance of two language models
if __name__ == '__main__':
    LM = LangModel()    # class of the language model

    # gaining data sets
    brown_data = LM.brown_data
    gutenberg_data = LM.gutenberg_data
    reuters_data = LM.reuters_data

    # get the frequency distribution of tri-, bi-, and uni-gram(s) for each data set
    brown_fd = LM.train_freq_dist(brown_data)
    gutenberg_fd = LM.train_freq_dist(gutenberg_data)
    reuters_fd = LM.train_freq_dist(reuters_data)

    # ----------------Back-off Model----------------
    # Back-off parameters
    brown_BO_params = LM.get_BO_hyperparam(brown_fd, brown_data)
    gutenberg_BO_params = LM.get_BO_hyperparam(gutenberg_fd, gutenberg_data)
    reuters_BO_params = LM.get_BO_hyperparam(reuters_fd, reuters_data)
    print('----------------BO parameters----------------')
    # {'alpha2': 0.1281454755848253, 'alpha1': 0.1111153341612912}
    print('Brown parameters:', brown_BO_params)
    # {'alpha2': 0.14344361208963569, 'alpha1': 0.05632301048112203}
    print('Gutenberg parameters:', gutenberg_BO_params)
    # {'alpha2': 0.3, 'alpha1': 0.3}
    print('Reuters parameters:', reuters_BO_params)

    # test the back-off model by perplexity
    brown_BO_perp = LM.get_perplexity(brown_data, brown_BO_params, brown_fd)
    gutenberg_BO_perp = LM.get_perplexity(gutenberg_data, gutenberg_BO_params, gutenberg_fd)
    reuters_BO_perp = LM.get_perplexity(reuters_data, reuters_BO_params, reuters_fd)
    print('----------------Back-off perplexities----------------')
    print('Brown (BO):', brown_BO_perp)           # 205.55104725449993
    print('Gutenberg (BO)', gutenberg_BO_perp)    # 175.15678436577946
    print('Reuters (BO)', reuters_BO_perp)        # 51.2139812791161

    # ----------------Interpolation Model----------------
    brown_interp_params = LM.interpolation_params(brown_data, brown_fd)
    gutenberg_interp_params = LM.interpolation_params(gutenberg_data, gutenberg_fd)
    reuters_interp_params = LM.interpolation_params(reuters_data, reuters_fd)
    print('----------------Interpolation Parameters----------------')
    print('Brown parameters (Interp):', brown_interp_params)          # [0.15, 0.425, 0.425]
    print('Gutenberg parameters (Interp):', gutenberg_interp_params)  # [0.2, 0.4, 0.4]
    print('Reuters parameters (Interp):', reuters_interp_params)      # [0.3, 0.35, 0.35]

    # test the interpolation model by perplexity
    brown_interp_perp = LM.get_perplexity(brown_data, brown_interp_params, brown_fd, 'interp')
    gutenberg_interp_perp = LM.get_perplexity(gutenberg_data, gutenberg_interp_params,
                                             gutenberg_fd, 'interp')
    reuters_interp_perp = LM.get_perplexity(reuters_data, reuters_interp_params,
                                           reuters_fd, 'interp')
    print('----------------Interpolation perplexity----------------')
    print('Brown (Interp):', brown_interp_perp)                      # 104.3504096389727
    print('Gutenberg (Interp):', gutenberg_interp_perp)              # 76.27220475194363
    print('Reuters (Interp):', reuters_interp_perp)                  # 47.80923008035443

    # # plot the perplexity comparision result within a bar graph
    # backoff_result = (brown_BO_perp, gutenberg_BO_perp, reuters_BO_perp)
    # interp_result = (brown_interp_perp, gutenberg_interp_perp, reuters_interp_perp)
    # LM.plot_result(backoff_result, interp_result)
