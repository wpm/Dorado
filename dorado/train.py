import copy
import logging
import operator

import numpy


def train_model(epochs, model, train_data, validate_data, n, patience):
    logging.info("Train %s: training %s, validation %s, %d batches, %d epochs patience" % \
                 (model, train_data, validate_data, n, patience))
    train_batches = list(train_data.shuffle().partition(n))
    best_error = numpy.Inf
    best_model = model
    wait = patience
    logging.info("Initial validation error %0.4f" % model.error_rate(validate_data))
    for i, model in enumerate(epochs(model, train_batches), 1):
        validation_error = model.error_rate(validate_data)
        logging.info("Epoch %d (%d), validation error %0.4f" % (i, wait, validation_error))
        if validation_error >= best_error:
            wait -= 1
            if not wait:
                break
        else:
            best_model = model
            best_error = validation_error
            wait = patience
    return best_model, best_error


def spark_averaged_epochs(spark_context, model, train_batches, **parameters):
    logging.info("Optimization parameters: %s" % dictionary_as_string(parameters))
    n = len(train_batches)
    zero = spark_context.broadcast(model.zero())
    while True:
        model = spark_context.parallelize(train_batches). \
            map(lambda batch: model.train(batch)).fold(zero.value, operator.add)
        model /= n
        yield model


def averaged_epochs(model, train_batches, **parameters):
    logging.info("Optimization parameters: %s" % dictionary_as_string(parameters))
    n = len(train_batches)
    while True:
        model = reduce(operator.add, (model.train(batch) for batch in train_batches))
        model /= n
        yield model


def serial_epochs(model, train_batches, **parameters):
    logging.info("Optimization parameters: %s" % dictionary_as_string(parameters))
    while True:
        for batch in train_batches:
            model = model.train(batch)
        yield model


def dictionary_as_string(d):
    return ','.join(["%s=%s" % (k, v) for k, v in d.items()])