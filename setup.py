import codecs

from setuptools import setup, find_packages


with codecs.open('README.rst', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="Dorado",
    version="0.1.0",
    packages=find_packages(),

    entry_points={
        'console_scripts': ['dorado = dorado.scripts:run']
    },

    install_requires=[
        'docutils>=0.3'
        'theano'
    ],

    package_data={
        '': ['*.rst'],
    },

    author="W.P. McNeill",
    author_email="billmcn@gmail.com",
    description="Distributed Machine learning with Theano",
    keywords="theano machine learning Spark",
    url="https://github.com/wpm/Dorado",
    long_description=long_description,
)
