#!/usr/bin/env python

import argparse
import cPickle
import gzip


parser = argparse.ArgumentParser()
parser.add_argument('model', help = 'Zipped model file')
parser.add_argument('data', help = 'Pickled labeled data of the form (vectors, labels)')
args = parser.parse_args()

with gzip.open(args.model) as f:
    model = cPickle.load(f)

with gzip.open(args.data) as f:
    x, y = cPickle.load(f)

e = model.error_rate(y, x)
print("Error rate %04f" % e)
