from copy import deepcopy
import logging
import numpy as np


def sgd_train(model, training_parameters, training_set, validation_set):
    logging.info(training_parameters)
    training_set = training_set.shuffle()
    batches = training_set.partition(training_parameters['batch_size'])
    training_points = 0
    iterations = 0
    patience = training_parameters['patience']
    best_error_rate = np.inf
    early_stop = False
    epoch = 0
    best_model = model
    while epoch < training_parameters['epochs'] and not early_stop:
        epoch += 1
        logging.info("Epoch %d" % epoch)
        for b, batch in enumerate(batches):
            logging.debug("Batch %d" % (b + 1))
            iterations += 1
            p = model.training_iteration(batch.y, batch.x, training_parameters['rate'])
            logging.debug("Training NLL %04f" % p)
            training_points += len(batch)
            if iterations % training_parameters['validation_frequency'] == 0:
                error_rate = model.error_rate(validation_set.y, validation_set.x)
                logging.info("%d. Validation error rate %04f" % (iterations, error_rate))
                if error_rate < best_error_rate:
                    best_error_rate = error_rate
                    best_model = deepcopy(model)
                    patience = training_points + training_parameters['patience']
                elif training_points > patience:
                    early_stop = True
                    break
    return best_model, best_error_rate
