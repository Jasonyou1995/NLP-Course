Assignment 3
CSE490UA
Jason You

# A3_model.py
	class BigramModel:
		def __init__: 		initiating the variables that stores the emission and transitional counts for both
							bigram and trigram models.

		def count_and_save: This method only need to be run in the first time to build models.
							store the counting of the occurence of words and arrangements for finding emission and 
							transitional probability.
							First add the number of occurance of different words given the labels.
							Second, add the number of occurance given the labels.
							For trigram model, the history labels are two word pairs.
							Store the emission and transitional counts as pickle files.

		def smoothing: 		count the number of unknow words based on the given threshold. Add the count of words
		 					below the threshold into <UNK> and sum. (retain the original words)

		def load_model: 	this method should be run after the "count_and_save", and it'll load the pickle models
							in the same directory of the script

		def make_prediction: get a list of tokens for one sentence, and returns a sequence of label with same size.
						     First initating the start case by finding the transitional probability for the first
						     word. Then loop through the rest of the words until the end.
						     In each step, first multiply the emission probability given the word, and find the
						     argmax for the next word, store the label that maximized the score as a backpointer.
						     After constructed the "table" (a list of dictionaries), backtrack all the backpointer
						     by start from the stop symbol.

        def get_result: 	Returns a list of sequences of labels with the same format as the given test dataset.
        					This method get a file name for the test dataset, and make prediction with the
        					"make_prediction" method by looping through all the sentences in the test. Store
        					the predicted labels in the same directory for later evaluation. 

		def load_file: 		return a list of word-tag paired data.

        main:				Predict the labels in the test dataset with both bigram and trigram models, and then
        					the method will automatically store the result in the same directory.
        					(Using dev-set to choose the best threshold for selecting the unknown words)


# A3_eval.py
	def load_file: 			return a list of word-tag paired data.

	main:					Evaluating the accuracy of prediction by numpy array. Sum up the correct predictions
							and then divided by total size of the test dataset. Convert the result to percentage.