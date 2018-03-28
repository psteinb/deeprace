"""
usage: deeprace train [options] [--] <models>

options:
    -h, --help                                 print this help message
    -O <mopts> --meta-options=<mopts>          hyper-parameters for training, e.g. batch_size
    -D <dpath> --datapath=<dpath>              path used for temporary storage, e.g. for the input data, checkpoints etc [default: datasets]
    -R <rpath> --resultspath=<rpath>           path to store results or checkpoints [default: deeprace-results]
    -e <neps> --nepochs=<neps>                 number of epochs to train [default: 0]
    -d <ds> --dataset=<ds>                     the dataset to use [default: cifar10]
    -f <dfrac> --datafraction=<dfrac>          fraction of the dataset to use, helpful for debugging/testing [default: 1.]
    -t <output> --timings=<output>             file to store the individual timings in [default: timings.tsv]
    -s <sep> --separator=<sep>                 seperator for the output data [default: \t]
    -c <cmt> --comment=<cmt>                   comment to add to the measurement
"""

from docopt import docopt
import os
import re
import sys
import glob
import importlib
import logging
import datetime
import socket

try:
    from importlib import util as ilib_util
except:
    raise
else:
    finder = ilib_util.find_spec

def import_model(name):
    """ import a model and return the imported symbol """

    if not os.path.exists(os.path.join(".","models")):
        return None

    expected_location =os.path.join(".","models",name+".py")
    if not os.path.exists(expected_location):
        return None

    try_this = [ "", ".", ".."]
    full_name = "models.%s" % (name)

    ld = None
    for it in try_this:
        ld = finder(full_name, package="%smodels" % it)

        if ld:
            break
        else:
            logging.info("no finder with %s at %8s (cwd: %s)",full_name,it, os.path.abspath(os.curdir))

    if ld == None:
        ld = finder(name, path="models")

    if ld == None:
        return None

    return importlib.import_module(full_name)

def load_model(descriptor):
    """ given a string, return a tuple of the respective module loader and a dictionary of constant parameters """

    model_id_candidates = re.findall('^[a-zA-Z]+',descriptor)
    if not model_id_candidates:
        return None

    model_id = model_id_candidates[0]
    logging.info("importing %s (from %s)", model_id, descriptor)
    loaded = import_model(model_id)
    if not loaded:
        logging.error("unable to load %s inferred from %s",model_id,descriptor)
        return None

    param_dict = loaded.params_from_name(descriptor)
    return (loaded,param_dict)

def describe(modelname):
    
    (loaded,opts_from_name) = load_model(modelname[0])

    logging.info("available options for {}".format(modelname[0]))
    for (k,v) in loaded.model().options().items():
        print("  {name:20} = {default}".format(name=k,default=v))

def run_model(args):

    logging.debug("received %s as args",args)

    if not "<models>" in args.keys():
        logging.error("no model recieved")
        return 1

    modelname = args["<models>"]
    #logging.info("loading %s",modelname)
    (loaded,opts_from_name) = load_model(modelname)

    model = loaded.model()
    deciphered = model.options()

    logging.info("successfully imported %s",modelname)

    deciphered.update(opts_from_name)
    meta_opts = {}

    if "--meta-options" in args.keys() and args["--meta-options"]:
        meta_opts = dict((k.strip(), v.strip()) for k,v in
                         (item.split('=') for item in args["--meta-options"].split(',')))
        deciphered.update(meta_opts)

    if ("--nepochs") in args.keys():
        deciphered["epochs"] = int(args["--nepochs"])

    opts = ",".join(["{k}={v}".format(k=item[0],v=item[1]) for item in deciphered.items() ])

    start = datetime.datetime.now()
    train, test, ntrain, ntest = model.data_loader(args["--datapath"])
    end = datetime.datetime.now()
    logging.info("loading the data took %f seconds", ((end-start).total_seconds()))
    logging.info("running %s", modelname)

    #update dictionary here
    d2 = model.options()
    d3 = {key:d2[key] for key in deciphered if key in d2}
    if d3.keys() == deciphered.keys():
        model.__dict__ = deciphered
    else:
        logging.error("options received (%s) do not match supported options (%s)",deciphered.keys(),d2.keys())

    if not os.path.exists(args["--resultspath"]):
        os.makedirs(args["--resultspath"])

    model.scratchspace = args["--resultspath"]

    hname = socket.getfqdn().split(".")[0]
    hist, timings, details = model.train(train,test,datafraction=args["--datafraction"])
    with open(args["--timings"],'w') as csvout:

        runid = "{hostname}{sep}{model}{sep}{dataset}{sep}{load_dur_sec}{sep}{ntrain}{sep}{ntest}{sep}{df}{sep}{train_start}{sep}{train_end}".format(hostname=hname,
                                                                                                                                                         model=modelname,
                  dataset=args["--dataset"],
                  load_dur_sec=(end-start).total_seconds(),
                  ntrain=ntrain,
                  ntest=ntest,
                  df=args["--datafraction"],
                  train_start=timings.train_begin.strftime("%Y%m%d:%H%M%S"),
                  train_end=timings.train_end.strftime("%Y%m%d:%H%M%S"),
                  sep=args["--separator"]

        )
        csvout.write("host{sep}model{sep}dataset{sep}load_dur_sec{sep}ntrain{sep}ntest{sep}datafraction{sep}train_start{sep}train_end{sep}epoch{sep}rel_epoch_start_sec{sep}epoch_dur_sec{sep}loss{sep}acc{sep}val_loss{sep}val_acc{sep}top_k_catacc{sep}val_top_k_catacc{sep}opts{sep}n_model_params{sep}versions{sep}comment\n".format(sep=args["--separator"]))
        for i in range(len(timings.epoch_durations)):
            line = "{constant}{sep}{num}{sep}{rel_epoch_start_sec}{sep}{epoch_dur_sec}{sep}{loss}{sep}{acc}{sep}{val_loss}{sep}{val_acc}{sep}{top_k_catacc}{sep}{val_top_k_catacc}{sep}{detail}{sep}{n_model_params}{sep}{versions}{sep}{comment}\n".format(
                constant=runid,
                num=int(i),
                rel_epoch_start_sec=timings.epoch_start[i],
                epoch_dur_sec=timings.epoch_durations[i],
                loss=hist.history['loss'][i],
                acc=hist.history['acc'][i],
                val_loss=hist.history['val_loss'][i],
                val_acc= hist.history['val_acc'][i],
                top_k_catacc=hist.history['top_k_categorical_accuracy'][i],
                val_top_k_catacc=hist.history['val_top_k_categorical_accuracy'][i],
                detail=opts,
                sep=args["--separator"],
                n_model_params=details['num_weights'],
                versions=model.versions(),
                comment=args["--comment"]
            )
            csvout.write(line)

        csvout.close()
        logging.info('wrote %s',args["--timings"])

    logging.info('Done.')

    return 0

def main():

    args = docopt(__doc__)
    if args['--help']:
        print(__doc__)
        sys.exit(1)
    else:
        sys.exit(run_model(args))


if __name__ == '__main__':
    main()
