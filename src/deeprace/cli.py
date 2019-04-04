#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function
from deeprace import __version__ as myversion

__doc__ = """
usage: deeprace [options] <command> [<args>...]

options:
   -h, --help                           print the help message
   -v, --version                        print the version of deeprace ({version})
   -V, --verbose                        be verbose when running [default: False]
   -L <level> --loglevel=<level>        logging level to use [default: info]

The most commonly used git commands are:
   list      list available models
   train     run training for a given model
   describe  show available parameters for given model
   infer     run inference for a given model

See 'deeprace help <command>' for more information on a specific command.

For more information and bug reports, please see
https://github.com/psteinb/deeprace
""".format(version=myversion)

# import argparse

# parser = argparse.ArgumentParser(description='Command description.')
# parser.add_argument('names', metavar='NAME', nargs=argparse.ZERO_OR_MORE,
#                     help="A name of something.")

from docopt import docopt, DocoptExit
import os
import importlib
import logging
import socket
import datetime

def main(argv=None):
    try:
        parsed = docopt(__doc__, argv,
                        options_first=True)
    except Exception as ex:
        print("parsing deeprace command line args caused exception")
        print(ex)
        pass
    except DocoptExit as dex:
        print(dex)
        return 1

    numeric_level = getattr(logging, parsed['--loglevel'].upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % parsed['--loglevel'])

    hname = socket.gethostname().split(".")[0]

    logging.basicConfig(level=numeric_level, format=('[%(asctime)-15s ' + hname + '] :: %(message)s'), datefmt="%y%m%d %H:%M:%S")

    logging.debug(argv)
    argv = [parsed['<command>']] + parsed['<args>']
    logging.debug(argv)
    rvalue = 0

    if parsed['--version'] or 'version' in parsed['<command>']:
        logging.info('deeprace {}'.format(myversion))
        return 1

    if '--help' in parsed and parsed['--help'] or ('help' in parsed['<command>']):
        print(parsed)
        print(__doc__)
        return 1

    if parsed['<command>'] == 'list':
        from deeprace.verbs import dr_list
        list_args = docopt(dr_list.__doc__, argv=argv)
        return dr_list.print_models()

    elif parsed['<command>'] in ['help', None]:

        if len(parsed['<args>']) and os.path.exists(os.path.join('verbs', 'dr_' + parsed['<args>'][0] + '.py')):
            # print the help message of a verb
            verb = importlib.import_module('deeprace.verbs.dr_' + parsed['<args>'][0])
            print(verb.__doc__)
        else:
            logging.info("deeprace (%s)", myversion)
            logging.info("for more information, see https://github.com/psteinb/deeprace\n")
            print(__doc__)
            # return call([sys.executable, __file__, '--help'])
        rvalue = 1

    elif parsed['<command>'] == 'train':
        from deeprace.verbs import dr_train
        run_args = docopt(dr_train.__doc__, argv=argv)
        logging.debug("[train] handing over to run_model:")
        logging.debug(run_args)
        rvalue = dr_train.run_model(run_args)

    elif parsed['<command>'] == 'infer':
        from deeprace.verbs import dr_infer
        run_args = docopt(dr_infer.__doc__, argv=argv)
        logging.debug("[infer] handing over to run_model:")
        logging.debug(run_args)
        rvalue = dr_infer.infer_model(run_args)

    elif parsed['<command>'] == 'describe':
        from deeprace.verbs import dr_train
        dr_train.describe(parsed['<args>'])

    else:
        logging.warning("%r is not a deeprace command. See 'deeprace help'." % parsed['<command>'])
        rvalue = 1

    return rvalue
