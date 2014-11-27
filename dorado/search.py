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
                parameters = model.train(batch, self.learning_rate)
                yield parameters


class ParallelAveragedSearch(Search):
    def epochs(self, model_factory, initial_parameters, training_data):
        ensemble = self._create_ensemble(model_factory, initial_parameters, training_data)
        while True:
            parameters = reduce(operator.add, [self._train_model(model, batch) for batch, model in ensemble])
            parameters /= self.n
            yield parameters
            ensemble = [(batch, self._set_parameters(model, parameters)) for batch, model in ensemble]

    def _create_ensemble(self, model_factory, initial_parameters, training_data):
        batches = self._partitions(training_data)
        return zip(batches, itertools.repeat(model_factory(initial_parameters), self.n))

    def _train_model(self, model, data):
        model.train(data, self.learning_rate)
        return model.get_parameters()

    def _set_parameters(self, model, parameters):
        model.set_parameters(parameters)
        return model


class SparkParallelAveragedSearch(ParallelAveragedSearch):
    def __init__(self, spark_context, n, learning_rate, patience):
        self.spark_context = spark_context
        super(SparkParallelAveragedSearch, self).__init__(n, learning_rate, patience)

    def epochs(self, model_factory, initial_parameters, training_data):
        zero = self.spark_context.broadcast(initial_parameters.zero())
        ensemble = self.spark_context.parallelize(
            self._create_ensemble(model_factory, initial_parameters, training_data))
        while True:
            parameters = ensemble.map(lambda batch, model:
                                      self._train_model(model, batch)).fold(zero.value, operator.add)
            parameters /= self.n
            yield parameters
            ensemble = ensemble.map(lambda batch, model: (batch, self._set_parameters(model, parameters)))
