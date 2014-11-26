import operator
import itertools


class Search(object):
    def __init__(self, n, learning_rate, patience):
        self.n = n
        self.learning_rate = learning_rate
        self.patience = patience

    def epochs(self, model_factory, initial_parameters, training_batches):
        raise NotImplementedError()

    def _partitions(self, training_data):
        return list(training_data.shuffle().partition(self.n))


class SequentialSearch(Search):
    def epochs(self, model_factory, initial_parameters, training_data):
        training_batches = self._partitions(training_data)
        model = model_factory(initial_parameters)
        while True:
            for batch in training_batches:
                parameters = model.train(batch, self.learning_rate)
                yield parameters


class ParallelAveragedSearch(Search):
    def epochs(self, model_factory, parameters, training_data):
        training_batches = self._partitions(training_data)
        models = self._create_models(n, model_factory, parameters)
        while True:
            models = [self._train_model(model, parameters, batch)
                      for model, batch in self._enumerate_models(models, training_batches)]
            parameters = reduce(operator.add, [model.parameters for model in models])
            parameters /= self.n
            yield parameters

    def _create_models(self, n, model_type, initial_parameters):
        return itertools.repeat(model_type(initial_parameters), n)

    def _enumerate_models(self, models, training_batches):
        for model, batch in itertools.izip(models, training_batches):
            yield model, batch

    def _train_model(self, model, parameters, batch):
        model.set_parameters(parameters)
        model.train(batch, self.learning_rate)
        return model


class SparkParallelAveragedSearch(ParallelAveragedSearch):
    def __init__(self, spark_context, n, learning_rate, patience):
        self.spark_context = spark_context
        super(SparkParallelAveragedSearch, self).__init__(n, learning_rate, patience)

    def epochs(self, model_factory, parameters, training_data):
        training_batches = self._partitions(training_data)
        zero = self.spark_context.broadcast(parameters.zero())
        models = self._create_models(self.n, model_factory, parameters)
        while True:
            models = self.spark_context.parallelize(self._enumerate_models(models, training_batches)). \
                map(lambda model, batch: self._train_model(model, parameters, batch))
            parameters = models.map(lambda model: model.parameters).fold(zero.value, operator.add)
            parameters /= self.n
            yield parameters
