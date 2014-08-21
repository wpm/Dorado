import argparse
import cPickle
from dorado.classifier import LogisticRegression, NeuralNetwork
import gzip
import numpy as np


def model_training_arguments(description):
    parser = argparse.ArgumentParser(description)
    parser.add_argument('train', type = load_compressed, help = 'training data')
    parser.add_argument('validation', type = load_compressed,
        help = 'validation data')
    parser.add_argument('type', choices = ['lg', 'nn'], 
        help = 'Classifier type: logistic regression or neural network')
    parser.add_argument('model', help = 'Trained model file')
    parser.add_argument('--classes', type = int, 
        help = 'number of classes, default is the number of unique labels in training')
    parser.add_argument('--l1', type = float, default = 0.0, help = 'L1 regularization constant')
    parser.add_argument('--l2', type = float, default = 0.0, help = 'L2 regularization constant')
    parser.add_argument('--hidden', type = int, default = 1000,
        help = 'Number of neural network hidden nodes')
    parser.add_argument('--batch', type = int, default = 100, help = 'batch size')
    parser.add_argument('--max-epochs', dest = 'max_epochs', type = int, default = np.inf,
        help = 'maximum training epochs')
    parser.add_argument('--min-epochs', dest = 'min_epochs', type = int, default = 1,
        help = 'minimum training epochs')
    parser.add_argument('--patience', type = int, default = 1,
        help = 'number of epochs to see before an early stop, default is one')
    parser.add_argument('--frequency', type = int, default = 1,
        help = 'how often to check the validation set, default every epoch')
    parser.add_argument('--rate', type = float, default = 0.13, help = 'learning rate')
    parser.add_argument('--log', default = 'CRITICAL', help = 'logging level')

    args = parser.parse_args()
    if args.classes == None:
        args.classes = args.train.classes()
    args.classifier = {
        'lg':LogisticRegression(args.train.dim(), args.classes, args.l1, args.l2),
        'nn':NeuralNetwork(args.train.dim(), args.classes, args.hidden, args.l1, args.l2)
    }[args.type]

    return args

def load_compressed(filename):
    with gzip.open(filename) as file:
        return cPickle.load(file)

def dump_compressed(obj, filename):
    with gzip.open(filename, 'w') as file:
        return cPickle.dump(obj, file)


class LabeledData(object):
    """
    A set of training example vectors with associated labels
    """
    def __init__(self, y, x):
        assert y.shape[0] == x.shape[0], \
            "Unmatched number of labels (%d) and training points (%d)" % (y.shape[0], x.shape[0])
        self.y = y
        self.x = x

    def __repr__(self):
        return "<%s %d examples, dimension %d, %d classes>" % \
            (self.__class__.__name__, len(self), self.dim(), self.classes())

    def __len__(self):
        """
        Numbered of labeled examples
        """
        return len(self.y)

    def dim(self):
        """
        The dimensionality of the example vectors
        """
        return self.x.shape[1]

    def classes(self):
        """The number of unique classes"""
        return np.unique(self.y).size

    def partition(self, batch_size):
        batches = []
        n = len(self)/batch_size
        for batch in xrange(n):
            i = batch * batch_size
            y_batch = self.y[i : i + batch_size]
            x_batch = self.x[i : i + batch_size]
            batches.append(LabeledData(y_batch, x_batch))
        return batches

    def shuffle(self):
        # TODO Randomly shuffle the data.
        return LabeledData(self.y, self.x)
