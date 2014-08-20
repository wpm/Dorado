import dorado.train
import logging

try:
    from pyspark import SparkContext
except ImportError:
    raise Exception("Run this with spark-submit")


args = dorado.model_training_arguments(
    "Train a classifier using stochastic gradient descent.")

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
