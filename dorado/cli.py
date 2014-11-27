import argparse
import cPickle
import gzip
import logging
import os
import sys
import subprocess
import urllib

import numpy

from dorado import train
from dorado.data import LabeledData
from dorado.model.logistic_regression import LogisticRegression
from dorado.model.neural_network import NeuralNetwork
from dorado.model.static_linear import StaticLinearModel
from dorado.epochs import SparkParallelAveragedEpochs, SequentialEpochs, ParallelAveragedEpochs


def run(spark_context=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default='INFO', help='logging level')
    train_test_parser = parser.add_subparsers(dest='command')
    train_parser = train_test_parser.add_parser('train', help='train a model')
    common_train = argparse.ArgumentParser(add_help=False)
    common_train.add_argument('training', help='training data')
    common_train.add_argument('validation', help='validation data')
    common_train.add_argument('model', help='compressed model file')
    common_train.add_argument('--learning-rate', type=float, default=0.13, help='learning rate, default 0.13')
    common_train.add_argument('--patience', type=int, default=5, help='epochs of patience, default 5')
    common_train.add_argument('--batches', type=int, default=100, help='number of batches, default 100')
    common_train.add_argument('--seed', type=int, help='random number seed')
    train_exclude = common_train.add_mutually_exclusive_group()
    train_exclude.add_argument('--serial', action='store_true', help='handle batches sequentially instead of averaging')
    train_exclude.add_argument('--spark', action='store_true', help='submit to spark')
    common_train.add_argument('--spark-submit', default='spark-submit', help='path to spark-submit')
    regularization_parser = argparse.ArgumentParser(add_help=False)
    regularization_parser.add_argument('--l1', type=float, default=0.0, help='l1 regularization, default 0.0')
    regularization_parser.add_argument('--l2', type=float, default=0.0, help='l2 regularization, default 0.0')
    model_type_parser = train_parser.add_subparsers(dest='model_type')
    model_type_parser.add_parser('random', parents=[common_train], help='random model')
    model_type_parser.add_parser('logreg', parents=[common_train, regularization_parser],
                                 help='logistic regression model')
    neural_parser = model_type_parser.add_parser('neural', parents=[common_train, regularization_parser],
                                                 help='neural network model')
    neural_parser.add_argument('--hidden', type=int, default=20, help='number of hidden layer nodes, default 20')
    test_parser = train_test_parser.add_parser('test', help='apply a model')
    test_parser.add_argument('model', type=load_compressed, help='compressed model file')
    test_parser.add_argument('test', type=load_compressed, help='testing data')
    mnist_parser = train_test_parser.add_parser('mnist', help='download MNIST data')
    mnist_parser.add_argument('--destination', default='.', help='download destination, default current directory')
    args, extra_args = parser.parse_known_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%y/%m/%d %H:%M:%S',
        level=getattr(logging, args.log.upper()))

    logging.info("Begin")
    if not (args.command == 'train' and (args.spark or spark_context)) and extra_args:
        parser.error("Unrecognized argument %s" % ' '.join(extra_args))

    if args.command == 'train':
        if not args.seed is None:
            numpy.random.seed(args.seed)
        for filename in [args.training, args.validation]:
            if not os.path.isfile(filename):
                parser.error("%s is not a file" % filename)
        if os.path.exists(args.model):
            parser.error("%s already exists" % args.model)
        validation_data = load_compressed(args.validation)
        model_factory, initial_parameters = \
            select_model_type(args, validation_data.dimension(), validation_data.classes())
        if not spark_context and args.spark:
            spark_cmd = "%s %s %s" % (args.spark_submit,
                                      os.path.join(os.path.dirname(__file__), 'spark.py'),
                                      ' '.join(sys.argv[1:]))
            logging.info(spark_cmd)
            subprocess.call(spark_cmd, shell=True)
        else:
            training_data = load_compressed(args.training)
            if spark_context:
                epochs = SparkParallelAveragedEpochs(spark_context, args.batches, args.learning_rate, args.patience)
            else:
                if args.serial:
                    epochs = SequentialEpochs(args.batches, args.learning_rate, args.patience)
                else:
                    epochs = ParallelAveragedEpochs(args.batches, args.learning_rate, args.patience)
            model, validation_error = train(model_factory, initial_parameters, training_data, validation_data, epochs)
            logging.info("Best validation error %0.4f" % validation_error)
            write_compressed(model, args.model)
    elif args.command == 'test':
        logging.info("Model %s, Data %s" % (args.model, args.test))
        print(args.model.error_rate(args.test))
    elif args.command == "mnist":
        try:
            os.makedirs(args.destination)
        except OSError:
            if not os.path.isdir(args.destination):
                raise
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        logging.info("Download %s" % origin)
        filename = urllib.urlretrieve(origin)[0]
        train_set, valid_set, test_set = load_compressed(filename)

        for data, name in zip([train_set, valid_set, test_set], ['train', 'valid', 'test']):
            n = os.path.join(args.destination, "mnist.%s.pkl.gz" % name)
            logging.info("Write %s" % n)
            vectors, labels = data
            data = LabeledData(vectors, labels)
            write_compressed(data, n)
    else:
        raise Exception("Invalid command %s" % args.command)
    logging.info("Done")


def select_model_type(args, dimension, classes):
    if args.model_type == 'random':
        return StaticLinearModel.factory(), StaticLinearModel.initial_parameters(dimension, classes)
    elif args.model_type == 'logreg':
        return LogisticRegression.factory(args.l1, args.l2), LogisticRegression.initial_parameters(dimension, classes)
    elif args.model_type == 'neural':
        return NeuralNetwork.factory(args.l1, args.l2), \
               NeuralNetwork.initial_parameters(dimension, classes, args.hidden)
    else:
        raise Exception("Invalid model type %s" % args.model_type)


def load_compressed(filename):
    logging.info("Read %s" % filename)
    with gzip.open(filename) as f:
        return cPickle.load(f)


def write_compressed(o, filename):
    logging.info("Write %s" % filename)
    with gzip.open(filename, 'w') as f:
        cPickle.dump(o, f)


if __name__ == "__main__":
    run()
