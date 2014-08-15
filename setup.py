from distutils.core import setup

setup(
    name='Dorado',
    version='0.1.0',
    author='W.P. McNeill',
    author_email='billmcn@gmail.com',
    packages=['dorado'],
    scripts=['bin/train_digits','bin/test'],
    url='https://github.com/wpm/Dorado',
    description='Machine learning with Theano',
    long_description=open('README').read(),
    install_requires=[
        "argparse",
        "theano",
        "nltk",
        "numpy"
    ],
)