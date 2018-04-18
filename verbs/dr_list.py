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

if importlib:
    import importlib.util
    finder = importlib.util.find_spec
else:
    raise Exception("unable to find importlib.util.find_spec, are you using python 3.4+ ?")

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
        fname = os.path.split(it)[-1]
        modelstem = os.path.splitext(fname)[0]

        if "base" in modelstem:
            continue

        name = "models.%s" % (modelstem)
        ld = finder(modelstem, package="models")
        if ld != None:
            logging.warning("found %s but could not find a loader for it" % fname)
        else:
            current = importlib.import_module(name)
            m = current.model()
            value[modelstem] = m.provides()

    return value

def print_models():
    models = available_models()
    if not models:
        logging.error("no models found")
        return 1

    logging.info("available models and backends:")
    for k,v in models.items():
        if len(v) < 2:
            continue
        logging.info("%s (backends: %s)" % (" ".join(v[0]), v[-1]))

    return 0


def main():

    args = docopt(__doc__)
    sys.exit(print_models())


if __name__ == '__main__':
    main()
