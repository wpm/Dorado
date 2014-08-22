import codecs
from setuptools import setup, find_packages

with codecs.open('README.rst', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name = "Dorado",
    version = "0.1.0",
    packages = find_packages(),

    entry_points = {
        'console_scripts': [
            'dorado_train = dorado.scripts:train',
            'dorado_test = dorado.scripts:test',
            'dorado_download_mnist_digits = dorado.scripts:download_mnist_digits'
        ],
    },

    install_requires = [
        'docutils>=0.3'
        'theano',
        'nltk'
        ],

    package_data = {
        '': ['*.rst'],
    },

    author = "W.P. McNeill",
    author_email = "billmcn@gmail.com",
    description = "Distributed Machine learning with Theano",
    keywords = "theano machine learning Spark",
    url = "https://github.com/wpm/Dorado",
    long_description=long_description,
)
