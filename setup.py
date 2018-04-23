# -*- coding: utf-8 -*-
import re
import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand


REQUIRES = [
    'docopt',
]

class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ''

    def run_tests(self):
        import shlex
        #import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)


def find_version(fname):
    '''Attempts to find the version number in the file names fname.
    Raises RuntimeError if not found.
    '''
    version = ''
    with open(fname, 'r') as fp:
        reg = re.compile(r'__version__ = [\'"]([^\'"]*)[\'"]')
        for line in fp:
            m = reg.match(line)
            if m:
                version = m.group(1)
                break
    if not version:
        raise RuntimeError('Cannot find version information')
    return version

__version__ = find_version("deeprace.py")


def read(fname):
    with open(fname) as fp:
        content = fp.read()
    return content

setup(
    name='deeprace',
    version="0.2.0",
    description='benchmark suite to time deep learning training in a framework agnostic fashion',
    long_description=read("README.rst"),
    author='Peter Steinbach',
    author_email='steinbach@scionics.de',
    url='https://github.com/psteinb/deeprace',
    install_requires=REQUIRES,
    license=read("LICENSE"),
    zip_safe=False,
    keywords='deeprace',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        # "Programming Language :: Python :: 2",
        # 'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    py_modules=["deeprace", "models", "datasets", "verbs"],
    entry_points={
        'console_scripts': [
            "deeprace = deeprace:main"
        ]
    },
    tests_require=['pytest'],
    cmdclass={'test': PyTest}
)
