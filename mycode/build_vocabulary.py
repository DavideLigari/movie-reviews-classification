import collections
import os


def remove_punctuation(text):
    """Remove punctuation from the text."""
    punct = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    for p in punct:
        text = text.replace(p, " ")
    return text


def read_document(filename):
    """Read the file and returns a list of words."""
    f = open(filename, encoding="utf8")
    text = f.read()
    f.close()
    words = []
    text = remove_punctuation(text.lower())
    for w in text.split():
        if len(w) > 2:
            words.append(w)
    return words


def write_vocabulary(voc, filename, n):
    """Write the n most frequent words to a file."""
    f = open(filename, "w")
    for word, count in sorted(voc.most_common(n)):
        print(word, file=f)
    f.close()


def get_words(voc, n):
    words = []
    for word, count in sorted(voc.most_common(n)):
        words.append(word)
    return words


def get_vocabulary(path='data/smalltrain', numWords=1000, save=False):
    # The script reads all the documents in the smalltrain directory, uses
    # the to form a vocabulary, writes it to the 'vocabulary.txt' file.
    voc = collections.Counter()
    for f in os.listdir(path+"/pos"):
        voc.update(read_document(path+"/pos/" + f))
    for f in os.listdir(path+"/neg"):
        voc.update(read_document(path+"/neg/" + f))
    if save:
        write_vocabulary(voc, "vocabulary.txt", numWords)
    return get_words(voc, numWords)
