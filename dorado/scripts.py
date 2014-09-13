import argparse
import logging
import urllib
import os.path

import numpy as np

from dorado import load_compressed
from dorado.classifier import LogisticRegression, NeuralNetwork
import dorado.train


def command_line():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest = 'command')
    # Shared
    parser.add_argument("--log", default='INFO', help="logging level")
    # Train
    train_parser = subparsers.add_parser('train', help = 'Train a model')
    train_parser.add_argument('train', type=load_compressed, help='training data')
    train_parser.add_argument('validation', type=load_compressed,
                              help='validation data')
    train_parser.add_argument('type', choices=['lg', 'nn'],
                              help='Classifier type: logistic regression or neural network')
    train_parser.add_argument('model', help='Trained model file')
    train_parser.add_argument('--classes', type=int,
                              help='number of classes, default is the number of unique labels in training')
    train_parser.add_argument('--l1', type=float, default=0.0,
                              help='L1 regularization constant')
    train_parser.add_argument('--l2', type=float, default=0.0,
                              help='L2 regularization constant')
    train_parser.add_argument('--hidden', type=int, default=1000,
                              help='Number of neural network hidden nodes')
    train_parser.add_argument('--batch', type=int, default=100, help='batch size')
    train_parser.add_argument('--max-epochs', dest='max_epochs', type=int, default=np.inf,
                              help='maximum training epochs')
    train_parser.add_argument('--min-epochs', dest='min_epochs', type=int, default=1,
                              help='minimum training epochs')
    train_parser.add_argument('--patience', type=int, default=1,
                              help='number of epochs to see before an early stop, default is one')
    train_parser.add_argument('--frequency', type=int, default=1,
                              help='how often to check the validation set, default every epoch')
    train_parser.add_argument('--rate', type=float, default=0.13,
                              help='learning rate')
    train_parser.add_argument('--shuffle', type=bool, default=True,
                              help='randomly shuffle data')
    # Test
    test_parser = subparsers.add_parser('test', help = 'Apply a model')
    test_parser.add_argument('model', help='zipped model file')
    test_parser.add_argument('data', help='labeled data')
    # Fetch
    fetch_parser = subparsers.add_parser('fetch', help = 'Fetch data')
    fetch_parser.add_argument('--destination', default='.',
                              help='Download destination, default current directory')
    args = parser.parse_args()

    logging.basicConfig(
        format='%(levelname)s %(asctime)s %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S',
        level=getattr(logging, args.log.upper()))

    if args.command == 'train':
        return train(args)
    elif args.command == 'test':
        return test(args)
    elif args.command == 'fetch':
        return download_mnist_digits(args)
    else:
        raise Exception("Invalid parsed arguments %s" % args)


def train(args):
    """Train a classifier using stochastic gradient descent."""
    if args.classes == None:
        args.classes = args.train.classes()
    args.classifier = {
        'lg': LogisticRegression(args.train.dim(), args.classes, args.l1, args.l2),
        'nn': NeuralNetwork(args.train.dim(), args.classes, args.hidden, args.l1, args.l2)
    }[args.type]

    iterations = dorado.train.sequential_iterations(
        args.classifier, args.train, args.batch, args.rate, args.shuffle)
    model = dorado.train.train(
        iterations, args.validation, args.min_epochs, args.max_epochs,
        args.frequency, args.patience, args.shuffle)[1]

    dorado.dump_compressed(model, args.model)
    logging.info("Created model %s" % args.model)
    return 0


# TODO Add distributed training as a command line option
def distributed_train(args):
    """Train a classifier using stochastic gradient descent on Spark."""
    try:
        from pyspark import SparkContext
    except ImportError:
        raise Exception("Run this with spark-submit")

    sc = SparkContext("local", "Distributed Training")

    iterations = dorado.train.parallel_iterations(
        sc, args.classifier, args.train, args.batch, args.rate)
    model = dorado.train.train(
        iterations, args.validation, args.min_epochs, args.max_epochs,
        args.frequency, args.patience)[1]

    dorado.dump_compressed(model, args.model)
    logging.info("Created model %s" % args.model)
    return 0


def test(args):
    """Run a model on labeled data"""
    model = dorado.load_compressed(args.model)
    data = dorado.load_compressed(args.data)
    print("Error rate %04f" % model.error_rate(data.y, data.x))
    return 0


def download_mnist_digits(args):
    """
    Download the MNIST digit recognition data.

    Write it to mnist.train.pkl.gz, mnist.test.pkl.gz, and mnist.valid.pkl.gz
    files in the specified directory.
    """
    try:
        os.makedirs(args.destination)
    except OSError:
        if not os.path.isdir(args.destination):
            raise
    origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    logging.info("Download %s" % origin)
    filename = urllib.urlretrieve(origin)[0]

    train_set, valid_set, test_set = dorado.load_compressed(filename)

    for data, name in zip([train_set, valid_set, test_set], ['train', 'valid', 'test']):
        n = os.path.join(args.destination, "mnist.%s.pkl.gz" % name)
        logging.info("Write %s" % n)
        x, y = data
        dorado.dump_compressed(dorado.LabeledData(y, x), n)

    return 0
