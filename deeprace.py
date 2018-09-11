#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function

import versioneer

__doc__ = """
usage: deeprace [--version] [--help] [--verbose] [--loglevel <level>] <command> [<args>...]

options:
   -h, --help                           Show this help message
   -v, --version                        Print the version of deeprace ({version})
   -V, --verbose                        Run in verbose mode
   -L <level> --loglevel=<level>        logging level to use [default: info]

The most commonly used git commands are:
   list      list available models
   train     run training for a given model
   describe  show available parameters for given model
   infer     run inference for a given model

See 'deeprace help <command>' for more information on a specific command.

For more information and bug reports, please see
https://github.com/psteinb/deeprace
""".format(version=versioneer.get_version().split("+")[0])

from docopt import docopt
import os
import sys
import argparse
import glob
import importlib
import re
import datetime
import logging
import socket
import datetime
from subprocess import call

__version__ = versioneer.get_version()
__author__ = "Peter Steinbach"
__license__ = "BSD"

def main():

    args = docopt(__doc__, version=__version__, options_first=True)

    numeric_level = getattr(logging, args['--loglevel'].upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args['--loglevel'])
    hname = socket.gethostname().split(".")[0]
    if args['--verbose']:
        numeric_level = getattr(logging, 'DEBUG', None)

    logging.basicConfig(level=numeric_level,format=('[%(asctime)-15s '+hname+'] :: %(message)s'),datefmt="%y%m%d %H:%M:%S")

    logging.debug(args)
    argv = [args['<command>']] + args['<args>']
    logging.debug(argv)
    rvalue = 0

    if args['<command>'] == 'list':
        import verbs.dr_list
        list_args = docopt(verbs.dr_list.__doc__, argv=argv)
        sys.exit(verbs.dr_list.print_models())

    elif args['<command>'] in ['help', None]:

        if len(args['<args>']) and os.path.exists(os.path.join('verbs','dr_'+args['<args>'][0]+'.py')):
            verb = importlib.import_module('verbs.dr_'+args['<args>'][0])
            print(verb.__doc__)
        else:
            logging.info("deeprace (%s)", versioneer.get_version())
            logging.info("for more information, see https://github.com/psteinb/deeprace\n")

            sys.exit(call([sys.executable, __file__, '--help']))

    elif args['<command>'] == 'train':
        import verbs.dr_train
        run_args = docopt(verbs.dr_train.__doc__, argv=argv)
        logging.debug("[train] handing over to run_model:")
        logging.debug(run_args)
        rvalue = verbs.dr_train.run_model(run_args)

    elif args['<command>'] == 'infer':
        import verbs.dr_infer
        run_args = docopt(verbs.dr_infer.__doc__, argv=argv)
        logging.debug("[infer] handing over to run_model:")
        logging.debug(run_args)
        rvalue = verbs.dr_infer.infer_model(run_args)

    elif args['<command>'] == 'describe':
        import verbs.dr_train
        verbs.dr_train.describe(args['<args>'])

    else:
        exit("%r is not a deeprace command. See 'deeprace help'." % args['<command>'])

    sys.exit(rvalue)



if __name__ == '__main__':
    main()
