#!/usr/bin/env python

from classifier import LogisticRegressionClassifier, LabeledData
import sgd_train
import cPickle
import gzip
import argparse
import logging

"""
Logistic Regression of Digit Recognition

The digits data set is here http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
"""

parser = argparse.ArgumentParser(__doc__)
parser.add_argument('data', help = 'MNIST digits data set')
parser.add_argument('--batch', type = int, default = 100, help = 'batch size')
parser.add_argument('--epochs', type = int, default = 1000, help = 'maximum training epochs')
parser.add_argument('--patience', type = int, help = 'number of training examples to see before an early stop')
parser.add_argument('--frequency', type = int, help = 'how often to check the validation set')
parser.add_argument('--rate', type = float, default = 0.13, help = 'learning rate')
parser.add_argument('--log', default = 'CRITICAL', help = 'logging level')
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s %(message)s', 
    level = getattr(logging, args.log.upper()))

# Read in digit data.
f = gzip.open(args.data)
train_set, valid_set, test_set = cPickle.load(f)
f.close()

training_set = LabeledData(train_set[1], train_set[0])
validation_set = LabeledData(valid_set[1], valid_set[0])

# Train the model.
if args.patience == None:
    patience = len(training_set)
else:
    patience = args.patience
if args.frequency == None:
    frequency = len(training_set)/args.batch
else:
    frequency = args.frequency
training_parameters = {
    'batch_size':args.batch,
    'epochs':args.epochs,
    'patience':patience,
    'validation_frequency':frequency,
    'rate':args.rate
    }

model, validation_error = sgd_train.sgd_train(
        LogisticRegressionClassifier(training_set.dim(), 10), 
        training_parameters, training_set, validation_set
    )
logging.info("Best validation error rate %04f" % validation_error)
