import operator
import itertools


class Search(object):
    def __init__(self, n, learning_rate, patience):
        self.n = n
        self.learning_rate = learning_rate
        self.patience = patience

    def epochs(self, model_factory, initial_parameters, training_data):
        raise NotImplementedError()

    def _partitions(self, training_data):
        return training_data.shuffle().partition(self.n)


class SequentialSearch(Search):
    def epochs(self, model_factory, initial_parameters, training_data):
        training_batches = self._partitions(training_data)
        model = model_factory(initial_parameters)
        while True:
            for batch in training_batches:
                model.train(batch, self.learning_rate)
                yield model.get_parameters()


class ParallelAveragedSearch(Search):
    def epochs(self, model_factory, initial_parameters, training_data):
        zero = self._zero(initial_parameters)
        ensemble = self._create_ensemble(model_factory, initial_parameters, training_data)
        while True:
            self._train_models(ensemble)
            parameters = self._average_parameters(ensemble, zero)
            yield parameters
            self._update_parameters(ensemble, parameters)

    def _zero(self, initial_parameters):
        return initial_parameters.zero()

    def _create_ensemble(self, model_factory, initial_parameters, training_data):
        batches = self._partitions(training_data)
        return zip(batches, itertools.repeat(model_factory(initial_parameters), self.n))

    def _train_models(self, ensemble):
        reduce(operator.add, [self._train_model(model, batch) for batch, model in ensemble])

    def _average_parameters(self, ensemble, zero):
        s = reduce(operator.add, [model.get_parameters() for _, model in ensemble], zero)
        return s / self.n

    def _update_parameters(self, ensemble, parameters):
        for _, model in ensemble:
            model.set_parameters(parameters)

    def _train_model(self, model, data):
        model.train(data, self.learning_rate)
        return model


class SparkParallelAveragedSearch(ParallelAveragedSearch):
    def __init__(self, spark_context, n, learning_rate, patience):
        self.spark_context = spark_context
        super(SparkParallelAveragedSearch, self).__init__(n, learning_rate, patience)

    def _zero(self, initial_parameters):
        zero = super(SparkParallelAveragedSearch, self)._zero(initial_parameters)
        return self.spark_context.broadcast(zero)

    def _create_ensemble(self, model_factory, initial_parameters, training_data):
        ensemble = super(SparkParallelAveragedSearch, self). \
            _create_ensemble(model_factory, initial_parameters, training_data)
        return self.spark_context.parallelize(ensemble)

    def _train_models(self, ensemble):
        ensemble.map(lambda batch, model: self._train_model(model, batch))

    def _average_parameters(self, ensemble, zero):
        s = ensemble.map(lambda _, model: model.get_parameters()).fold(zero.value, operator.add)
        return s / self.n

    def _update_parameters(self, ensemble, parameters):
        ensemble.map(lambda batch, model: (batch, model.set_parameters(parameters)))
