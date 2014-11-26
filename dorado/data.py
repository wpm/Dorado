import itertools

import numpy

class LabeledData(object):
    def __init__(self, vectors, labels):
        assert len(vectors) == len(labels)
        self.vectors = vectors
        self.labels = labels

    def __repr__(self):
        return "<%s, %d vectors, %d features, %d classes>" % \
               (self.__class__.__name__, len(self), self.dimension(), self.classes())

    def __len__(self):
        return len(self.labels)

    def dimension(self):
        return self.vectors.shape[1]

    def classes(self):
        return len(numpy.unique(self.labels))

    def take(self, n):
        return LabeledData(numpy.copy(self.vectors[:n]), numpy.copy(self.labels[:n]))

    def partition(self, n):
        return (LabeledData(numpy.copy(vectors), numpy.copy(labels))
                for vectors, labels in itertools.izip(numpy.array_split(self.vectors, n),
                                                      numpy.array_split(self.labels, n)))

    def shuffle(self):
        """Randomly shuffle this data set"""
        i = numpy.arange(len(self))
        numpy.random.shuffle(i)
        return LabeledData(self.vectors[i], self.labels[i])
