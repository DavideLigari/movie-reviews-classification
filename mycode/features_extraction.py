import numpy as np
import os
import build_vocabulary as bv
import porter


def words_to_dict(words: list):
    """Load the vocabulary and returns it.
    The return value is a dictionary mapping words to numerical
indices.

    """
    n = 0
    voc = {}
    for w in words:
        voc[w] = n
        n += 1
    return voc


def stem_words(words: list):
    """Apply stemming to all words in the list"""
    stemmed_words = []
    for word in words:
        word = porter.stem(word)
        if (word not in stemmed_words):
            stemmed_words.append(word)
    return stemmed_words


def read_document(filename, voc):
    """Read a document and return its BoW representation."""
    f = open(filename, encoding="utf8")
    text = f.read()
    f.close()
    text = bv.remove_punctuation(text.lower())
    bow = np.zeros(len(voc))
    for w in text.split():
        if w in voc:
            index = voc[w]
            bow[index] += 1
    return bow


def read_document_stemming(filename, voc):
    """Read a document,apply stemming to all words and return its BoW representation."""
    f = open(filename, encoding="utf8")
    text = f.read()
    f.close()
    text = bv.remove_punctuation(text.lower())
    bow = np.zeros(len(voc))
    for w in text.split():
        w = porter.stem(w)
        if w in voc:
            index = voc[w]
            bow[index] += 1
    return bow


def get_bow_representation(words, path='../data/smalltrain/', save=False, stemming=False):
    """Read all documents and return the BoW representation, as well as the kind of each document (positive or negative)."""
    voc = words_to_dict(words)
    documents = []
    labels = []
    if stemming:
        for f in os.listdir(path+"/pos"):
            documents.append(read_document_stemming(path+"/pos/" + f, voc))
            labels.append(1)
        for f in os.listdir(path+"/neg"):
            documents.append(read_document_stemming(path+"/neg/" + f, voc))
            labels.append(0)
    else:
        for f in os.listdir(path+"/pos"):
            documents.append(read_document(path+"/pos/" + f, voc))
            labels.append(1)
        for f in os.listdir(path+"/neg"):
            documents.append(read_document(path+"/neg/" + f, voc))
            labels.append(0)
    # np.stack transforms the list of vectors into a 2D array.
    X = np.stack(documents)
    Y = np.array(labels)
    data = np.concatenate([X, Y[:, None]], 1)
    if save:
        np.savetxt("train.txt.gz", data)
    return data
