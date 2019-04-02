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
      - | |version| |wheel| |supported-versions| |supported-implementations|
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

simple benchmark suite for ensemble based deep learning benchmarking

* Free software: BSD 3-Clause License

Installation
============

::

    pip install deeprace

Documentation
=============


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
