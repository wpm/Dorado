#!/usr/bin/env python

"""
Train a neural network n-gram model.
"""

import logging
import numpy as np
from ngramnn import NgramNeuralNetwork, IndexedVocabulary

def train(training_corpus, validation_corpus,
            batch_size = 100, epochs = 1000, learning_rate = 0.13):
    vocabulary = IndexedVocabulary(training_corpus + validation_corpus)
    model = NgramNeuralNetwork(vocabulary)
    logging.info(model)
    y, X = model.embed_tokens(training_corpus)
    training_batches = X.shape[0]/batch_size
    items = 0 # Number of datapoints trained on
    best_validation_perplexity = np.inf
    for epoch in xrange(epochs):
        for batch in xrange(training_batches):
            i = batch * batch_size
            j = i + batch_size
            y_batch = y[i:j]
            X_batch = X[i:j]
            p_train = model.training_update(X_batch, y_batch, learning_rate)
            items += y_batch.size
        p_valid = model.perplexity(validation_corpus)
        logging.info("%d. %0.4f\t%0.4f" % (items, p_train, p_valid))
        if p_valid < best_validation_perplexity:
            best_validation_perplexity = p_valid
        else:
            break
    return model


if __name__ == '__main__':    
    import argparse
    import cPickle
    from nltk.corpus import brown

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('model', type = argparse.FileType('w'), help = 'model file to create')
    parser.add_argument('--batch', type = int, default = 100, help = 'batch size')
    parser.add_argument('--epochs', type = int, default = 1000, help = 'maximum training epochs')
    parser.add_argument('--rate', type = float, default = 0.13, help = 'learning rate')
    parser.add_argument('--log', default = 'INFO', help = 'logging level')
    args = parser.parse_args()

    batch_size = args.batch
    epochs = args.epochs
    learning_rate = args.rate

    training_corpus = brown.words()[:10000]
    validation_corpus = brown.words()[1000000:1001000]

    logging.basicConfig(format='%(asctime)s %(message)s',
        level = getattr(logging, args.log.upper()))
    model = train(training_corpus, validation_corpus, batch_size, epochs, learning_rate)
    cPickle.dump(model, args.model)
