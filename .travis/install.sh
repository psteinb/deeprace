#!/bin/bash

if [[ $TRAVIS_OS_NAME == 'osx' ]]; then

    # Install some custom requirements on OS X
    # e.g. brew install pyenv-virtualenv
    alias python=python3
    alias pip=pip3
    alias pytest=pytest-3

    which python
    python --version
    which pip
    pip --version
    which pytest

    # case "${TOXENV}" in
    #     py32)
    #         # Install some custom Python 3.2 requirements on OS X
    #         ;;
    #     py33)
    #         # Install some custom Python 3.3 requirements on OS X
    #         ;;
    # esac
else
    # Install some custom requirements on Linux
    which python
    python --version
    which python3
    python3 --version
fi
