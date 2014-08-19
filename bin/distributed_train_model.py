import argparse
from copy import deepcopy
import dorado
import logging
import operator

try:
    from pyspark import SparkContext
except ImportError:
    raise Exception("Run this with spark-submit")


def train(sc, model, training_data, batch_size, rate):
    def map_train(batch):
        model.sgd_training_iteration(batch.y, batch.x, rate)
        return model

    zero = sc.broadcast(deepcopy(model).zero())
    batches = sc.broadcast(training_data.partition(batch_size))
    k = len(batches.value)
    while True:
        ensemble = sc.parallelize(batches.value)
        model = ensemble.map(map_train).fold(zero.value, operator.add) / k
        yield model


parser = argparse.ArgumentParser(
    description = "Train a classifier using stochastic gradient descent.")
parser.add_argument('train', type = dorado.LabeledData.from_file,
    help = 'training data')
parser.add_argument('validation', type = dorado.LabeledData.from_file,
    help = 'validation data')
parser.add_argument('--batch', type = int, default = 100, help = 'batch size')
args = parser.parse_args()

sc = SparkContext("local", "Distributed Training")

model = dorado.LogisticRegression(args.train.dim(), args.train.classes())
logging.info("Initial validation error %044f" % 
    model.error_rate(args.validation.y, args.validation.x))
for epoch, model in enumerate(train(sc, model, args.train, args.batch, 0.13)):
    epoch += 1
    print("%d. %s: %04f" % 
        (epoch, model,
            model.error_rate(args.validation.y, args.validation.x)))
    if epoch == 3:
        break
