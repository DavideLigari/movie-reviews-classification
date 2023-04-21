import numpy as np
import os
import bagOfWords


def load_vocabulary(vocab):
    """Load the vocabulary and returns it.
The return value is a dictionary mapping words to numerical
indices. """
    n = 0
    voc = {}
    for w in vocab:
        voc[w] = n
        n += 1
    return voc


def read_document(filename, voc):
    """Read a document and return its BoW representation."""
    f = open(filename, encoding="utf8")
    text = f.read()
    f.close()
    text = bagOfWords.remove_punctuation(text.lower())
    # Start with all zeros
    bow = np.zeros(len(voc))
    for w in text.split():
        # If the word is the vocabulary...
        if w in voc:
            # ...increment the proper counter.
            index = voc[w]
            bow[index] += 1
    return bow


# The script compute the BoW representation of all the training
# documents.  This need to be extended to compute similar
# representations for the validation and the test set.
def get_train(path, vocabulary, save=False):
    documents = []
    labels = []
    voc = load_vocabulary(vocabulary)
    for f in os.listdir(path+"/pos"):
        documents.append(read_document(path+"/pos/" + f, voc))
        labels.append(1)
    for f in os.listdir(path+"/neg"):
        documents.append(read_document(path+"/neg/" + f, voc))
        labels.append(0)
    # np.stack transforms the list of vectors into a 2D array.
    X = np.stack(documents)
    Y = np.array(labels)
    # The following line append the labels Y as additional column of the
    # array of features so that it can be passed to np.savetxt.
    data = np.concatenate([X, Y[:, None]], 1)

    if save:
        np.savetxt("train.txt", data)
    return data


voc = bagOfWords.get_vocabulary('data/smalltrain', save=True)
train_data = get_train('data/smalltrain', voc, save=True)
