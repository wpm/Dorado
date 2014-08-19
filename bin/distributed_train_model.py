import dorado
import logging

try:
    from pyspark import SparkContext
except ImportError:
    raise Exception("Run this with spark-submit")


args = dorado.model_training_arguments(
    "Train a classifier using stochastic gradient descent.")

sc = SparkContext("local", "Distributed Training")
model = dorado.LogisticRegression(args.train.dim(), args.train.classes())
logging.info("Initial validation error %044f" % 
    model.error_rate(args.validation.y, args.validation.x))

for epoch, model in \
    enumerate(dorado.parallel_train(sc, model, args.train, args.batch, args.rate), 1):
    print("%d. %s: %04f" % 
        (epoch, model,
            model.error_rate(args.validation.y, args.validation.x)))
    if epoch == 3:
        break
