import argparse
import logging
import urllib

import numpy as np

from dorado import load_compressed
from dorado.classifier import LogisticRegression, NeuralNetwork
import dorado.train


def model_training_arguments(description):
    parser = argparse.ArgumentParser(description)
    parser.add_argument('train', type=load_compressed, help='training data')
    parser.add_argument('validation', type=load_compressed,
                        help='validation data')
    parser.add_argument('type', choices=['lg', 'nn'],
                        help='Classifier type: logistic regression or neural network')
    parser.add_argument('model', help='Trained model file')
    parser.add_argument('--classes', type=int,
                        help='number of classes, default is the number of unique labels in training')
    parser.add_argument('--l1', type=float, default=0.0,
                        help='L1 regularization constant')
    parser.add_argument('--l2', type=float, default=0.0,
                        help='L2 regularization constant')
    parser.add_argument('--hidden', type=int, default=1000,
                        help='Number of neural network hidden nodes')
    parser.add_argument('--batch', type=int, default=100, help='batch size')
    parser.add_argument('--max-epochs', dest='max_epochs', type=int, default=np.inf,
                        help='maximum training epochs')
    parser.add_argument('--min-epochs', dest='min_epochs', type=int, default=1,
                        help='minimum training epochs')
    parser.add_argument('--patience', type=int, default=1,
                        help='number of epochs to see before an early stop, default is one')
    parser.add_argument('--frequency', type=int, default=1,
                        help='how often to check the validation set, default every epoch')
    parser.add_argument('--rate', type=float, default=0.13,
                        help='learning rate')
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='randomly shuffle data')
    parser.add_argument('--log', default='CRITICAL', help='logging level')

    args = parser.parse_args()
    if args.classes == None:
        args.classes = args.train.classes()
    args.classifier = {
        'lg': LogisticRegression(args.train.dim(), args.classes, args.l1, args.l2),
        'nn': NeuralNetwork(args.train.dim(), args.classes, args.hidden, args.l1, args.l2)
    }[args.type]

    return args


def train():
    """Train a classifier using stochastic gradient descent."""
    args = model_training_arguments(train.__doc__)

    logging.basicConfig(
        format='%(levelname)s %(asctime)s %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S',
        level=getattr(logging, args.log.upper()))

    iterations = dorado.train.sequential_iterations(
        args.classifier, args.train, args.batch, args.rate, args.shuffle)
    model = dorado.train.train(
        iterations, args.validation, args.min_epochs, args.max_epochs,
        args.frequency, args.patience, args.shuffle)[1]

    dorado.dump_compressed(model, args.model)
    logging.info("Created model %s" % args.model)
    return 0


def distributed_train():
    """Train a classifier using stochastic gradient descent on Spark."""
    args = model_training_arguments(distributed_train.__doc__)

    try:
        from pyspark import SparkContext
    except ImportError:
        raise Exception("Run this with spark-submit")

    logging.basicConfig(format='%(asctime)s %(message)s',
                        level=getattr(logging, args.log.upper()))

    sc = SparkContext("local", "Distributed Training")

    iterations = dorado.train.parallel_iterations(
        sc, args.classifier, args.train, args.batch, args.rate)
    model = dorado.train.train(
        iterations, args.validation, args.min_epochs, args.max_epochs,
        args.frequency, args.patience)[1]

    dorado.dump_compressed(model, args.model)
    logging.info("Created model %s" % args.model)
    return 0


def test():
    """Run a model on labeled data"""
    parser = argparse.ArgumentParser(test.__doc__)
    parser.add_argument('model', help='zipped model file')
    parser.add_argument('data', help='labeled data')
    args = parser.parse_args()

    model = dorado.load_compressed(args.model)
    data = dorado.load_compressed(args.data)

    print("Error rate %04f" % model.error_rate(data.y, data.x))
    return 0


def download_mnist_digits():
    """
    Download the MNIST digit recognition data.

    Write it to mnist.train.pkl.gz, mnist.test.pkl.gz, and mnist.valid.pkl.gz
    files in the current directory.
    """
    parser = argparse.ArgumentParser(download_mnist_digits.__doc__)
    parser.parse_args()

    print('Download data')
    origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    filename = urllib.urlretrieve(origin)[0]

    train_set, valid_set, test_set = dorado.load_compressed(filename)

    for data, name in zip([train_set, valid_set, test_set], ['train', 'valid', 'test']):
        n = "mnist.%s.pkl.gz" % name
        print("Write %s" % n)
        x, y = data
        dorado.dump_compressed(dorado.LabeledData(y, x), n)

    return 0
