#!/usr/bin/env python

import argparse
from classifier import LabeledData, sgd_train
from classifier import LogisticRegression, NeuralNetwork
import cPickle
import gzip
import logging

"""
Logistic Regression of Digit Recognition

The digits data set is here http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
"""

parser = argparse.ArgumentParser()
parser.add_argument('data', help = 'MNIST digits data set')
parser.add_argument('model', help = 'Trained model file')
parser.add_argument('--l1', type = float, default = 0.0, help = 'L1 regularization constant')
parser.add_argument('--l2', type = float, default = 0.0, help = 'L2 regularization constant')
parser.add_argument('--batch', type = int, default = 100, help = 'batch size')
parser.add_argument('--epochs', type = int, default = 1000, help = 'maximum training epochs')
parser.add_argument('--patience', type = int,
    help = 'number of training examples to see before an early stop, default is the entire set')
parser.add_argument('--frequency', type = int,
    help = 'how often to check the validation set, default is once per epoch')
parser.add_argument('--rate', type = float, default = 0.13, help = 'learning rate')
parser.add_argument('--log', default = 'CRITICAL', help = 'logging level')
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s %(message)s', 
    level = getattr(logging, args.log.upper()))

# Read in digit data.
with gzip.open(args.data) as f:
    train_set, valid_set, test_set = cPickle.load(f)

training_set = LabeledData(train_set[1], train_set[0])
validation_set = LabeledData(valid_set[1], valid_set[0])

# Train the model.
classifier = LogisticRegression(training_set.dim(), 10, args.l1, args.l2)
# classifier = NeuralNetwork(training_set.dim(), 10, 1000, args.l1, args.l2)
model, validation_error = sgd_train(
        classifier, 
        training_set, validation_set,
        args.batch, args.patience, args.epochs, args.rate, args.frequency
    )

# Write the model to a zipped file.
print("model %s" % args.model)
with gzip.open(args.model, 'w') as f:
    cPickle.dump(model, f)
