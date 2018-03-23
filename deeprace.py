#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
usage: deeprace [--version] [--help] [--verbose] [--loglevel <level>] <command> [<args>...]

options:
   -h, --help                           Show this help message
   -v, --version                        Print the version of deeprace
   -V, --verbose                        Run in verbose mode
   -L <level> --loglevel=<level>        logging level to use [default: info]

The most commonly used git commands are:
   list      list available models
   run       run training on a given model

See 'deeprace help <command>' for more information on a specific command.
"""

from __future__ import unicode_literals, print_function
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


__version__ = "0.1.0"
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

    if args['<command>'] == 'list':
        import verbs.dr_list
        list_args = docopt(verbs.dr_list.__doc__, argv=argv)
        sys.exit(verbs.dr_list.print_models())

    elif args['<command>'] in ['help', None]:

        if len(args['<args>']) and os.path.exists(os.path.join('verbs','dr_'+args['<args>'][0]+'.py')):
            verb = importlib.import_module('verbs.dr_'+args['<args>'][0])
            print(verb.__doc__)
        else:
            exit(call([sys.executable, __file__, '--help']))

    elif args['<command>'] == 'run':
        import verbs.dr_run
        run_args = docopt(verbs.dr_run.__doc__, argv=argv)
        logging.info(run_args)

        hist, timings = verbs.dr_run.run_model(args)

        with open(args.timings,'w') as csvout:

            runid = "{hostname}{sep}{model}{sep}{dataset}{sep}{load_dur_sec}{sep}{ntrain}{sep}{ntest}{sep}{df}{sep}{train_start}{sep}{train_end}".format(hostname=hname,
                                                                                                                                                         model=args.model[0],
                  dataset=args.dataset,
                  load_dur_sec=(end-start).total_seconds(),
                  ntrain=ntrain,
                  ntest=ntest,
                  df=args.datafraction,
                  train_start=timings.train_begin.strftime("%Y%m%d:%H%M%S"),
                  train_end=timings.train_end.strftime("%Y%m%d:%H%M%S"),
                  sep=args.seperator

        )

            csvout.write("host{sep}model{sep}dataset{sep}load_dur_sec{sep}ntrain{sep}ntest{sep}datafraction{sep}train_start{sep}train_end{sep}epoch{sep}rel_epoch_start_sec{sep}epoch_dur_sec{sep}loss{sep}acc{sep}val_loss{sep}val_acc{sep}opts{sep}comment\n".format(sep=args.seperator))
            for i in range(len(timings.epoch_durations)):
                line = "{constant}{sep}{num}{sep}{rel_epoch_start_sec}{sep}{epoch_dur_sec}{sep}{loss}{sep}{acc}{sep}{val_loss}{sep}{val_acc}{sep}{detail}{sep}{comment}\n".format(
                    constant=runid,
                    num=i,
                    rel_epoch_start_sec=timings.epoch_start[i],
                    epoch_dur_sec=timings.epoch_durations[i],
                    loss=hist.history['loss'][i],
                    acc=hist.history['acc'][i],
                    val_loss=hist.history['val_loss'][i],
                    val_acc= hist.history['val_acc'][i],
                    detail=opts,
                    sep=args.seperator,
                    comment=args.comment
                )
                csvout.write(line)

            csvout.close()
            logging.info('wrote %s',args.timings)

        logging.info('Done.')

    sys.exit(0)


    else:
        exit("%r is not a deeprace command. See 'deeprace help'." % args['<command>'])



if __name__ == '__main__':
    main()
