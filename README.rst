========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis| |appveyor| |requires|
        | |codecov|
    * - package
      - | |version| |doi| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/deeprace/badge/?style=flat
    :target: https://readthedocs.org/projects/deeprace
    :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/psteinb/deeprace.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/psteinb/deeprace

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/psteinb/deeprace?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/psteinb/deeprace

.. |requires| image:: https://requires.io/github/psteinb/deeprace/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/psteinb/deeprace/requirements/?branch=master

.. |codecov| image:: https://codecov.io/github/psteinb/deeprace/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/psteinb/deeprace

.. |version| image:: https://img.shields.io/pypi/v/deeprace.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/deeprace

.. |doi| image:: https://zenodo.org/badge/123592478.svg
    :alt: Package DOI
    :target: https://zenodo.org/badge/latestdoi/123592478

.. |commits-since| image:: https://img.shields.io/github/commits-since/psteinb/deeprace/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/psteinb/deeprace/compare/v0.0.0...master

.. |wheel| image:: https://img.shields.io/pypi/wheel/deeprace.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/deeprace

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/deeprace.svg
    :alt: Supported versions
    :target: https://pypi.org/project/deeprace

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/deeprace.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/deeprace


.. end-badges

A small python3 based benchmark tool to benchmark and hence compare Deep Learning Models. This project is currently heavy development. Please expect the usage to change without prior notice.

* Free software: BSD 3-Clause License

Installation
============

(currently not available)
::

    pip install deeprace

Usage
=====

::

   $ deeprace --help
   usage: deeprace [--version] [--help] [--verbose] [--loglevel <level>] <command> [<args>...]
   
   options:
      -h, --help                           Show this help message
      -v, --version                        Print the version of deeprace
      -V, --verbose                        Run in verbose mode
      -L <level> --loglevel=<level>        logging level to use [default: info]
   
   The most commonly used git commands are:
      list      list available models
      train     run training on a given model   
      describe  show available parameters for given model
   
   See 'deeprace help <command>' for more information on a specific command.

To run training, do:
::

   $ python3 ./deeprace.py train resnet50v1

   usage: deeprace train [options] [--] <models>

   options:
       -h, --help                                 print this help message
       -O <mopts> --meta-options=<mopts>          hyper-parameters for training, e.g. batch_size
       -D <dpath> --datapath=<dpath>              path used for temporary storage, e.g. for the input data, checkpoints etc [default: datasets]
       -R <rpath> --resultspath=<rpath>           path to store results or checkpoints [default: deeprace-results]
       -b <backe> --backend=<backe>               which backend to use [default: keras]
       -e <neps> --nepochs=<neps>                 number of epochs to train [default: 0]
       -d <ds> --dataset=<ds>                     the dataset to use (depends on the model of choice) [default: model_default]
       -f <dfrac> --datafraction=<dfrac>          fraction of the dataset to use, helpful for debugging/testing [default: 1.]
       -t <output> --timings=<output>             file to store the individual timings in [default: timings.tsv]
       -s <sep> --separator=<sep>                 seperator for the output data [default: 	]
       -c <cmt> --comment=<cmt>                   comment to add to the measurement
   

Documentation
=============

(currently not available)

https://deeprace.readthedocs.io/


Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
