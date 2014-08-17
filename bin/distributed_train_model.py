import argparse
import dorado
import logging

try:
    from pyspark import SparkContext
except ImportError:
    raise Exception("Run this with spark-submit")


def gizpped_dataset(filename):
    return dorado.LabeledData.from_file(filename)

def batch_train(batch, rate = 0.001):
    model = dorado.LogisticRegression(batch.dim(), batch.classes())
    cost = model.sgd_training_iteration(batch.y, batch.x, rate)
    logging.info("Training cost %04f" % cost)
    return model


parser = argparse.ArgumentParser(
    description = "Train a classifier using stochastic gradient descent.")
parser.add_argument('train', type = gizpped_dataset, help = 'training data')
parser.add_argument('validation', type = gizpped_dataset,
    help = 'validation data')
parser.add_argument('--batch', type = int, default = 100, help = 'batch size')
args = parser.parse_args()

sc = SparkContext("local", "Distributed Training")

batches = sc.parallelize(args.train.partition(args.batch))
models = batches.map(batch_train).collect()
averaged_model = dorado.average_models(models)
print(averaged_model)
