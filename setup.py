import codecs
from setuptools import setup

with codecs.open('README.rst', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='Dorado',
    version='0.1.0',
    author='W.P. McNeill',
    author_email='billmcn@gmail.com',
    packages=['dorado'],
    scripts=[
        'bin/dorado_distributed_train.py', 
        'bin/dorado_test', 'bin/dorado_train',
        'bin/download_mnist_digits_dataset'],
    url='https://github.com/wpm/Dorado',
    description='Machine learning with Theano',
    long_description=long_description,
    install_requires=[
        "argparse",
        "theano",
        "nltk",
        "numpy"
    ],
)