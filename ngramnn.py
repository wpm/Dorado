#!/usr/bin/env python

from nltk import word_tokenize as tokenize
import numpy as np
import theano
import theano.tensor as T
from theano import function

"""
An n-gram language model that uses a neural network with distributed word representations.
"""

class NgramNeuralNetwork(object):
    """
    Neural Network language model

    This uses single hidden player to predict a word embedding given
    preceding word embeddings.
    """
    def __init__(self, vocabulary, n = 3, m = 10, h = 20):
        """
        @param V: vocabulary
        @type V: L{IndexedVocabulary}
        @param n: n-gram order
        @type n: C{int}
        @param m: word embedding size
        @type m: C{int}
        @param h: hidden layer size
        @type h: C{int}
        """
        self.vocabulary = vocabulary
        self.V = len(self.vocabulary)
        self.n = n
        self.m = m
        self.h = h
        # Word embeddings
        self.C = theano.shared(random_matrix(self.V, m), name = 'C')
        # Hidden layer
        self.H = theano.shared(random_matrix(h, (n - 1) * m), name = 'H')
        # Hidden layer bias
        self.d = theano.shared(np.zeros((h,)), name = 'd')
        # Projection layer
        self.U = theano.shared(random_matrix(self.V, h), name = 'U')
        # Projection layer bias     
        self.b = theano.shared(np.zeros((self.V,)), name = 'b')        
        # Set of n-gram context indexes
        X = T.lmatrix('X')
        # Set of indexes of words following the contexts
        y = T.lvector()
        # Learning rate
        e = T.scalar('e')
        # Symbolic functions
        embeddings = self.C[X].reshape((X.shape[0], -1))
        self._p_y_given_X = T.nnet.softmax(T.dot(T.dot(embeddings, self.H) + self.d, self.U.T) + self.b)
        self._negative_log_likelihood = -T.mean(T.log(self._p_y_given_X)[T.arange(y.shape[0]), y])
        self._L2_sqr = (self.H.flatten() ** 2).sum() + (self.C.flatten() ** 2).sum()
        self._objective = self._negative_log_likelihood + self._L2_sqr
        # Derivatives of model parameters
        self.g_C = T.grad(self._objective, self.C)
        self.g_H = T.grad(self._objective, self.H)
        self.g_d = T.grad(self._objective, self.d)
        self.g_U = T.grad(self._objective, self.U)
        self.g_b = T.grad(self._objective, self.b)
        # Training updates
        updates = [
            (self.C, self.C - e * self.g_C),
            (self.H, self.H - e * self.g_H),
            (self.d, self.d - e * self.g_d),
            (self.U, self.U - e * self.g_U),
            (self.b, self.b - e * self.g_b)
        ]
        # Compiled functions.
        self.p_y_given_X = function([X], self._p_y_given_X)
        self.negative_log_likelihood = function([X, y], self._negative_log_likelihood)
        self.training_update = function(
            inputs = [X, y, e],
            outputs = self._negative_log_likelihood,
            updates = updates)

    def __repr__(self):
        return "%s(V = %d, n = %d, m = %d, h = %d)" % \
            (self.__class__.__name__, self.V, self.n, self.m, self.h)

    def train(self, tokens, training_rate = 0.1):
        """
        Train the model

        @param tokens: training data
        @type tokens: sequence of C{str}
        @param training_rate: training rate
        @type training_rate: C{float}
        """
        y, X = self._embed_tokens(tokens)
        for i in xrange(10):
            p = self.training_update(X, y, training_rate)
            print("%d. %0.4f" % (i + 1, p))

    def perplexity(self, tokens):
        y, X = self._embed_tokens(tokens)
        return pow(2, self.negative_log_likelihood(X, y)/y.shape[0])

    def _embed_tokens(self, tokens):
        pad = [None] * (self.n - 1)
        tokens = pad + tokens + pad
        y = np.asarray([self.vocabulary[token] for token in tokens[self.n - 1:]])
        X = np.asarray([self.vocabulary(tokens[i:i + self.n - 1]) for i in xrange(len(tokens) - self.n + 1)])
        return y, X


def random_matrix(r, c):
    """
    Matrix with random elements selected uniformly from [0,1].
    @param r: rows
    @type n: C{int}
    @param c: columns
    @type n: C{int}
    @return: randomly generated matrix
    @rtype: C{np.array}
    """
    return np.random.uniform(size = (r, c))


class IndexedVocabulary(dict):
    """Vocabulary of words mapped to integers

    The 0-th element is an out-of-vocabulary token.
    """
    def __init__(self, tokens):
        """
        @param tokens: tokens in a corpus
        @type tokens: sequence of strings
        """
        super(IndexedVocabulary, self).__init__()
        types = sorted(list(set(tokens)))
        for i, word_type in enumerate(types):
            self[word_type] = i + 1

    def __call__(self, tokens):
        """
        Vocabulary indexes corresponding to a sequence of tokens

        @param tokens: tokens to get
        @type tokens: sequence of strings
        @return: token indexes
        @rtype: C{np.array} of C{int}
        """
        return np.asarray([self.get(t, 0) for t in tokens])

    def __getitem__(self, token):
        """
        Index of a token

        @param token: a token
        @type token: C{str}
        @return: token index or 0 if the token is out of vocabulary
        @rtype: C{int}
        """
        return self.get(token, 0)

    def __len__(self):
        """
        The size of this vocabulary includes an out-of-vocabulary token
        """
        return len(self.keys()) + 1


def main(corpus = "to be or not to be that is the question"):
    vocabulary = IndexedVocabulary(tokenize(corpus))
    n = NgramNeuralNetwork(vocabulary)
    print(n)
    n.train(tokenize(corpus))
    return corpus, n


if __name__ == '__main__':
    main()
