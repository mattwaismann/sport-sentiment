import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.feature_extraction.text import CountVectorizer

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


def plot_frequency_distribution_of_ngrams(texts,
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
