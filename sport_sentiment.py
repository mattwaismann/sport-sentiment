"""
Versions:
Python 3.6
Tensorflow 2.0.0

A module to train a text classifier
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import tensorflow as tf

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout

class eda():

	def median_num_words_in_sample( texts):
	    return np.median([len(text.split()) for text in texts])

	def plot_comment_length(texts):
	    """
	    text: list/series, a list of texts 
	    
	    Returns a histogram of the the lenghts (in words) of each comment 
	   
	    """
	    lengths = [len(text.split()) for text in texts]
	    plt.figure(figsize = (20,8))
	    plt.hist(lengths, bins = 100, histtype = 'bar', ec = 'black')
	    plt.title("Distribution of comment lengths")
	    plt.xlabel("Number of words")
	    plt.ylabel("Frequency")
	    plt.xlim([1,40])

	def get_num_classes( labels):
	    """Gets the total number of classes.
	    # Arguments
	        labels: list, label values.
	            There should be at lease one sample for values in the
	            range (0, num_classes -1)
	    # Returns
	        int, total number of classes.
	    # Raises
	        ValueError: if any label value in the range(0, num_classes - 1)
	            is missing or if number of classes is <= 1.
	    """
	    labels = labels.astype('int')
	    num_classes = max(labels) + 1
	    missing_classes = [i for i in range(num_classes) if i not in labels]
	    if len(missing_classes):
	        raise ValueError('Missing samples with label value(s) '
	                         '{missing_classes}. Please make sure you have '
	                         'at least one sample for every label value '
	                         'in the range(0, {max_class})'.format(
	                            missing_classes=missing_classes,
	                            max_class=num_classes - 1))

	    if num_classes <= 1:
	        raise ValueError('Invalid number of labels: {num_classes}.'
	                         'Please make sure there are at least two classes '
	                         'of samples'.format(num_classes=num_classes))
	    return num_classes

	def plot_class_distribution( labels):
		plt.figure(figsize = (20,6))
		plt.hist(labels)
		plt.show()


	def plot_frequency_distribution_of_ngrams(self, texts,
	                                         ngram_range,
	                                         num_ngrams):
	    """
	    comments: list/series, a list of comments
	    ngram_range: tuple (min_n,max_n), the range of n-gram values to consider
	        min_n and max_n are the lower and upper bound vlaues for the range.
	    num_ngrams: int, number of n-grams to plot
	        Top 'num_ngrams' will be plotted
	    
	    Returns a histogram of the frequency of ngrams
	    """
	    #instaniate an CountVectorizer object
	    kwargs = {
	        'ngram_range': ngram_range,
	        'dtype': 'int32',
	        'strip_accents': 'unicode',
	        'decode_error':'replace',
	        'analyzer':'word'
	    }
	    vectorizer = CountVectorizer(**kwargs)
	    
	    # This creates a vocabulary dict (keys: n-grams, values: indicies). This also converts every text to an array the length
	    # of the vocabulary, where every element indicates  the count of the n-gram corresponding at that index in vocabulary
	    vectorized_texts = vectorizer.fit_transform(texts)
	    
	    #this is the list of all n-grams in the index order from the vocabulary (index is ordered alphabetically)
	    all_ngrams = list(vectorizer.get_feature_names())
	    
	    #Add up the counts per n-gram ie. column-wise (over the rows hence axis of moment is 0)
	    all_counts = vectorized_texts.sum(axis = 0).tolist()[0] #each column belongs to a specific ngram
	    
	    num_ngrams = min(num_ngrams, len(all_ngrams)) #in case we have less ngrams than specificed in function call
	    
	    #Sort and subset the ngrams and their counts
	    #Sort n-gram and counts by frequency and get top 'num_ngrams' ngrams. (* unpacks an iterable)
	    all_counts,all_ngrams = zip(*[(c,n) for c,n in sorted(zip(all_counts, all_ngrams), reverse = True)]) #sorted sorts by the first list in zip 
	    ngrams = list(all_ngrams)[:num_ngrams]
	    counts = list(all_counts)[:num_ngrams]
	    
	    #plot
	    idx = np.arange(num_ngrams)
	    plt.figure(figsize = (20,6))
	    plt.bar(idx,counts, width = 0.8, color = 'b')
	    plt.xlabel('N-grams')
	    plt.ylabel('Frequencies')
	    plt.title('Frequency distribution of {}-grams'.format(ngram_range))
	    plt.xticks(idx, ngrams, rotation = 45)
	    plt.show()

class preprocess:

	def ngram_vectorize( train_texts, train_labels, val_texts, ngram_range = (1,2)): 
	    
		    """Vectorizes texts as n-gram vectors.

		    1 text = 1 tf-idf vector the length of vocabulary of unigrams + bigrams.

		    # Arguments
		        train_texts: list, training text strings.
		        train_labels: np.ndarray, training labels.
		        val_texts: list, validation text strings.

		    # Returns
		        x_train, x_val: vectorized training and validation texts
		    """
		    
		    # Vectorization parameters
		    # Range (inclusive) of n-gram sizes for tokenizing text.
		    NGRAM_RANGE = ngram_range

		    # Limit on the number of features. We use the top 20K features.
		    TOP_K = 2000

		    # Whether text should be split into word or character n-grams.
		    # One of 'word', 'char'.
		    TOKEN_MODE = 'word'

		    # Minimum document/corpus frequency below which a token will be discarded.
		    MIN_DOCUMENT_FREQUENCY = 2

		    # Create keyword arguments to pass to the 'tf-idf' vectorizer.
		    kwargs = {
		            'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.
		            'strip_accents': 'unicode',
		            'decode_error': 'replace',
		            'analyzer': TOKEN_MODE,  # Split text into word tokens.
		            'min_df': MIN_DOCUMENT_FREQUENCY,
		    }
		    vectorizer = TfidfVectorizer(**kwargs)

		    # Learn vocabulary from training texts and vectorize training texts.
		    x_train = vectorizer.fit_transform(train_texts)
		    x_train = x_train.todense()


		    # Vectorize validation texts.
		    x_val = vectorizer.transform(val_texts)
		    x_val = x_val.todense()

		    # Select top 'k' of the vectorized features.
		    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
		    selector.fit(x_train, train_labels)
		    x_train = selector.transform(x_train).astype('float64')
		    x_val = selector.transform(x_val).astype('float32')
		    return x_train, x_val


class my_model:

	def _get_last_layer_units_and_activation(self,num_classes):
	    """Gets the # units and activation function for the last network layer.

	    # Arguments
	        num_classes: int, number of classes.

	    # Returns
	        units, activation values.
	    """
	    if num_classes == 2:
	        activation = 'sigmoid'
	        units = 1
	    else:
	        activation = 'softmax'
	        units = num_classes
	    print(units)
	    return units, activation


	def mlp_model(self,layers, units, dropout_rate, input_shape, num_classes):
	    """Creates an instance of a multi-layer perceptron model.

	    # Arguments
	        layers: int, number of `Dense` layers in the model.
	        units: int, output dimension of the layers.
	        dropout_rate: float, percentage of input to drop at Dropout layers.
	        input_shape: tuple, shape of input to the model.
	        num_classes: int, number of output classes.

	    # Returns
	        An MLP model instance.
	    """
	    op_units, op_activation =self._get_last_layer_units_and_activation(num_classes)
	    model = models.Sequential()
	    model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

	    for _ in range(layers-1):
	        model.add(Dense(units=units, activation='relu'))
	        model.add(Dropout(rate=dropout_rate))

	    model.add(Dense(units=op_units, activation=op_activation))
	    return model

	def train_ngram_model(self,data,
	                      learning_rate=1e-3,
	                      epochs=1000,
	                      batch_size=128,
	                      layers=2,
	                      units=128,
	                      dropout_rate=0.2):
	    """Trains n-gram model on the given dataset.

	    # Arguments
	        data: tuples of training and test texts and labels.
	        learning_rate: float, learning rate for training model.
	        epochs: int, number of epochs.
	        batch_size: int, number of samples per batch.
	        layers: int, number of `Dense` layers in the model.
	        units: int, output dimension of Dense layers in the model.
	        dropout_rate: float: percentage of input to drop at Dropout layers.

	    # Raises
	        ValueError: If validation data has label values which were not seen
	            in the training data.
	    """
	    # Get the data.
	    (train_texts, train_labels), (val_texts, val_labels) = data

	    # Verify that validation labels are in the same range as training labels.
	    num_classes = eda.get_num_classes(train_labels)
	    unexpected_labels = [v for v in val_labels if v not in range(num_classes)]
	    if len(unexpected_labels):
	        raise ValueError('Unexpected label values found in the validation set:'
	                         ' {unexpected_labels}. Please make sure that the '
	                         'labels in the validation set are in the same range '
	                         'as training labels.'.format(
	                             unexpected_labels=unexpected_labels))

	    # Vectorize texts.
	    x_train, x_val = preprocess.ngram_vectorize(
	        train_texts, train_labels, val_texts)

	    # Create model instance.
	    model = self.mlp_model(layers=layers,
	                                  units=units,
	                                  dropout_rate=dropout_rate,
	                                  input_shape=x_train.shape[1:],
	                                  num_classes=num_classes)

	    # Compile model with learning parameters.
	    if num_classes == 2:
	        loss = 'binary_crossentropy'
	    else:
	        loss = 'sparse_categorical_crossentropy'
	    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
	    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

	    # Create callback for early stopping on validation loss. If the loss does
	    # not decrease in two consecutive tries, stop training.
	    callbacks = [tf.keras.callbacks.EarlyStopping(
	        monitor='val_loss', patience=2)]

	 
	
	    # Train and validate model.
	    history = model.fit(
	            x_train,
	            train_labels,
	            epochs=epochs,
	            callbacks=callbacks,
	            validation_data=(x_val, val_labels),
	            verbose=2,  # Logs once per epoch.
	            batch_size=batch_size)

	    # Print results.
	    history = history.history
	    print('Validation accuracy: {acc}, loss: {loss}'.format(
	            acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

	    # Save model.
	    model.save('sport_sentiment_model.h5')
	    return model, history['val_acc'][-1], history['val_loss'][-1]

