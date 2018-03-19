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


if importlib.find_loader:
    finder = importlib.find_loader
elif importlib.utils.find_spec:
    finder = importlib.utils.find_spec


def import_model(name):
    """ import a model and return the imported symbol """

    if not os.path.exists(os.path.join(".","models")):
        return None

    expected_location =os.path.join(".","models",name+".py")
    if not os.path.exists(expected_location):
        return None

    full_name = "models.%s" % name
    ld = finder(full_name)
    if ld == None:
        print("found %s at %s but could find a loader for it" % (name, expected_location))
        return None

    return importlib.import_module(full_name)

def load_model(descriptor):
    """ given a string, return a tuple of the respective module loader and a dictionary of constant parameters """

    model_id_candidates = re.findall('^[a-zA-Z]+',descriptor)
    if not model_id_candidates:
        return None

    model_id = model_id_candidates[0]
    logging.info("importing "+model_id)
    loaded = import_model(model_id)

    param_dict = loaded.params_from_name(descriptor)
    return (loaded,param_dict)


def main():

    args = docopt(__doc__, version=__version__, options_first=True)

    numeric_level = getattr(logging, args['--loglevel'].upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args['--loglevel'])
    hname = socket.gethostname().split(".")[0]
    if args['--verbose']:
        numeric_level = getattr(logging, 'DEBUG', None)

    logging.basicConfig(level=numeric_level,format=('[%(asctime)-15s '+hname+'] :: %(message)s'),datefmt="%Y-%m-%d %H:%M:%S")

    argv = [args['<command>']] + args['<args>']

    if args['<command>'] == 'list':
        import verbs.dr_list
        list_args = docopt(verbs.dr_list.__doc__, argv=argv)
        sys.exit(verbs.dr_list.print_models())

    elif args['<command>'] in ['help', None]:

        if len(args['<args>']) and os.path.exists(os.path.join('verbs','dr_'+args['<args>'][0]+'.py')):
            verb = importlib.import_module('verbs.dr_'+args['<args>'][0])
            verb.main()
        else:
            exit(call([sys.executable, __file__, '--help']))
    else:
        exit("%r is not a deeprace command. See 'deeprace help'." % args['<command>'])


def old_main():

    desc = """
    benchmarking tool to run predefined models and print the time per epoch either to the screen or save it to a file
"""

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('model', type=str, nargs='*',
                        help='a model descriptor to run (defaul resnet30)')

    parser.add_argument('-O','--meta-options', type=str, default="",
                        help='meta-options for the training')

    parser.add_argument('-D','--datapath', type=str, default="datasets",
                        help='path to store the input data in')

    parser.add_argument('-L','--loglevel', type=str, default="info",
                        help='logging level to use')

    parser.add_argument('-e','--nepochs', type=int, default=0,
                        help='number of epochs to run')


    parser.add_argument('-d','--dataset', type=str, default='cifar10',
                        help='specify the dataset to use')

    parser.add_argument('-f','--datafraction', type=float, default=1.,
                        help='fraction of the dataset to use')

    parser.add_argument('-t','--timings', type=str, default="timings.tsv",
                        help='file to store the individual timings in')

    parser.add_argument('-s','--seperator', type=str, default="\t",
                        help='seperator for the output data')

    parser.add_argument('-c','--comment', type=str, default="",
                        help='comment to add to dataset (mind the defined seperator)')

    args = parser.parse_args()

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)
    hname = socket.gethostname().split(".")[0]
    logging.basicConfig(level=numeric_level,format=('[%(asctime)-15s] '+hname+' :: %(message)s'))

    models = available_models()
    if not args.model:
        print("no model specified\navailable:")
        for k,v in models.items():
            logging.info("\t[%s] %s" % (k," ".join(v)))
            #TODO list available parameters
        sys.exit(1)

    (loaded,opts_from_name) = load_model(args.model[0])

    deciphered = loaded.options()
    deciphered.update(opts_from_name)
    meta_opts = {}

    if args.meta_options:
        meta_opts = dict((k.strip(), v.strip()) for k,v in
                         (item.split('=') for item in args.meta_options.split(',')))
        deciphered.update(meta_opts)

    if args.nepochs != 0:
        deciphered["epochs"] = args.nepochs

    opts = ",".join(["{k}={v}".format(k=item[0],v=item[1]) for item in deciphered.items() ])

    start = datetime.datetime.now()
    train, test, ntrain, ntest = loaded.data_loader(args.datapath)
    end = datetime.datetime.now()
    logging.info("loading the data took %f seconds", ((end-start).total_seconds()))
    logging.info("running %s", args.model[0])
    logging.info("hyper-parameters used %s", opts.replace(","," "))

    hist, timings = loaded.train(train,test,datafraction=args.datafraction,**deciphered)

    with open(args.timings,'w') as csvout:

        runid = "{hostname}-{model}-{dataset}{sep}{load_dur_sec}{sep}{ntrain}{sep}{ntest}{sep}{df}{sep}{train_start}{sep}{train_end}".format(hostname=hname,
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


        csvout.write("runid{sep}load_dur_sec{sep}ntrain{sep}ntest{sep}datafraction{sep}train_start{sep}train_end{sep}epoch{sep}rel_epoch_start_sec{sep}epoch_dur_sec{sep}loss{sep}acc{sep}val_loss{sep}val_acc{sep}opts{sep}comment\n".format(sep=args.seperator))
        for i in range(len(timings.epoch_durations)):
            line = "{runid}{sep}{num}{sep}{rel_epoch_start_sec}{sep}{epoch_dur_sec}{sep}{loss}{sep}{acc}{sep}{val_loss}{sep}{val_acc}{sep}{detail}{sep}{comment}\n".format(
                runid=runid,
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

if __name__ == '__main__':
    main()
