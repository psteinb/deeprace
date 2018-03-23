"""
usage: deeprace list [--help]

Options:
    -h, --help  print this help message
"""

from docopt import docopt
import os
import sys
import glob
import importlib
import logging

if importlib.find_loader:
    finder = importlib.find_loader
elif importlib.utils.find_spec:
    finder = importlib.utils.find_spec

def available_models():

    value = {}

    basepath = '.'

    if not os.path.exists(os.path.join(".","models")):
        if not os.path.exists(os.path.join("..","models")):
            return value
        else:
            basepath='..'

    found_model_files = glob.glob(os.path.join(basepath,"models","*.py"))
    for it in found_model_files:
        stem = os.path.split(it)[-1]
        fname = os.path.splitext(stem)[0]
        name = "models.%s" % (fname)
        ld = finder(name)
        if ld != None:
            logging.warning("found %s but could not find a loader for it" % stem)
        else:
            current = importlib.import_module(name)
            value[fname] = current.provides()

    return value

def print_models():
    models = available_models()
    if not models:
        logging.error("no models found")
        return 1

    logging.info("available models:")
    for k,v in models.items():
        logging.info("[%s] %s" % (k," ".join(v)))

    return 0


def main():

    args = docopt(__doc__)
    sys.exit(print_models())


if __name__ == '__main__':
    main()
