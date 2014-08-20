from copy import deepcopy
import logging
import numpy as np
import operator


def train(iterations, validation_data, min_epochs = 1, max_epochs = np.inf, freq = 1, patience = 1):
    best_error = np.inf
    best_model = None
    wait = patience
    for epoch, model in iterations:
        if epoch > max_epochs:
            break
        if epoch > min_epochs and epoch % freq == 0:
            e = model.error_rate(validation_data.y, validation_data.x)
            logging.info("epoch = %d error = %04f" % (epoch, e))
            if e < best_error:
                best_error = e
                best_model = deepcopy(model)
                wait = patience
            elif wait == 0:
                break
        wait -= 1
    logging.info("Best error %04f" % best_error)
    return best_error, best_model


def sequential_iterations(model, training_data, batch_size, rate):
    epoch = 1
    batches = training_data.partition(batch_size)
    while True:
        for batch in batches:
            model.sgd_training_iteration(batch.y, batch.x, rate)
        yield epoch, model
        epoch += 1


def parallel_iterations(sc, model, training_data, batch_size, rate):
    def map_train(batch):
        model.sgd_training_iteration(batch.y, batch.x, rate)
        return model

    epoch = 1
    zero = sc.broadcast(deepcopy(model).zero())
    batches = sc.broadcast(training_data.partition(batch_size))
    k = len(batches.value)
    while True:
        ensemble = sc.parallelize(batches.value)
        model = ensemble.map(map_train).fold(zero.value, operator.add) / k
        yield epoch, model
        epoch += 1
