from copy import deepcopy
import logging

from dorado import load_compressed
from dorado.classifier import LogisticRegression


def main(training_data, e=0.13):
    def map_train(batch):
        model.sgd_training_iteration(batch.y, batch.x, e)
        return model

    model = LogisticRegression(training_data.dim(), training_data.classes())
    print(model)
    zero = deepcopy(model).zero()
    # print("Zero:")
    # print(zero.W.get_value())
    # print(zero.b.get_value())
    batches = training_data.partition(100)[:5]
    k = len(batches)
    print("k = %d" % k)
    # batch = batches[0]
    # m = deepcopy(model)
    # m.sgd_training_iteration(batch.y, batch.x, e)
    new_models = []
    for i, batch in enumerate(batches):
        m = deepcopy(model)
        print(batch)
        m.sgd_training_iteration(batch.y, batch.x, e)
        print("%d. %s" % (i, m))
        new_models.append(m)
    print(new_models)
    new_model = reduce(lambda a,b: a + b, new_models) / k
    print(new_model)


if __name__ == '__main__':
    import sys

    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(module)s: %(message)s',
        datefmt='%d/%m/%y %I:%M:%S',
        level=getattr(logging, 'INFO'))

    train = load_compressed(sys.argv[1])
    main(train)