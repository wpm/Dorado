======
Dorado
======

Deep Learning algorithms using `Theano <http://deeplearning.net/software/theano/>`_


Basic Usage
-----------

To obtain MNIST digit data for testing, run

    dorado fetch

This will download the data and create training, validation, and test sets in the current directory: mnist.train.pkl.gz, mnist.valid.pkl.gz, and mnist.test.pkl.gz.

To train a model run

    dorado --train mnist.train.pkl.gz mnist.valid.pkl.gz lg model.gz

This creates a model.gz file.

To apply this model to a different data set run

   dorado --test model.gz mnist.test.pkl.gz

Run dorado --help for more details.


Spark Support
-------------
Dorado supports parallel training using the `Spark <https://spark.apache.org/>`_ distributed computing framework.

If Spark is installed and spark-submit is on the path, train a model by running

    dorado --spark-url local --train mnist.train.pkl.gz mnist.valid.pkl.gz lg model.gz

This divides the training data into batches, trains a model for each batch, then averages the models.
The --spark-url parameter can be used to send the job to a cluster of machines.

Spark support requires a version of Theano greater than 0.6.0.
This may be obtained directly from github.
