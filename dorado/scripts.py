import argparse
import dorado.train
import logging
import urllib


def train():
    """Train a classifier using stochastic gradient descent."""
    args = dorado.model_training_arguments(train.__doc__)

    logging.basicConfig(
        format='%(levelname)s %(asctime)s %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S',
        level = getattr(logging, args.log.upper()))

    iterations = dorado.train.sequential_iterations(
        args.classifier, args.train, args.batch, args.rate)
    model = dorado.train.train(
        iterations, args.validation, args.min_epochs, args.max_epochs,
        args.frequency, args.patience)[1]

    dorado.dump_compressed(model, args.model)
    logging.info("Created model %s" % args.model)
    return 0


def distributed_train():
    """Train a classifier using stochastic gradient descent onx Spark."""
    args = dorado.model_training_arguments(distributed_train.__doc__)

    try:
        from pyspark import SparkContext
    except ImportError:
        raise Exception("Run this with spark-submit")


    logging.basicConfig(format='%(asctime)s %(message)s', 
        level = getattr(logging, args.log.upper()))

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
    parser.add_argument('model', help = 'zipped model file')
    parser.add_argument('data', help = 'labeled data')
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

    for data, name  in zip([train_set, valid_set, test_set], ['train', 'valid', 'test']):
        n = "mnist.%s.pkl.gz" % name
        print("Write %s" % n)
        x, y = data
        dorado.dump_compressed(dorado.LabeledData(y, x), n)

    return 0
