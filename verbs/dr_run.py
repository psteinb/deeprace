"""
usage: deeprace run [options] [--] <models>

options:
    -h, --help                                 print this help message
    -O <mopts> --meta-options=<mopts>          hyper-parameters for training, e.g. batch_size
    -D <dpath> --datapath=<dpath>              path used for temporary storage, e.g. for the input data, checkpoints etc [default: datasets ]
    -e <neps> --nepochs=<neps>                 number of epochs to train [default: 0]
    -d <ds> --dataset=<ds>                     the dataset to use [default: cifar10]
    -f <dfrac> --datafraction=<dfrac>          fraction of the dataset to use, helpful for debugging/testing [default: 1.]
    -t <output> --timings=<output>             file to store the individual timings in [default: timings.tsv]
    -s <sep> --separator=<sep>                 seperator for the output data [default: \t]
    -c <cmt> --comment=<cmt>                   comment to add to the measurement
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


def run_model(args):

    logging.info("dr_run received %s",args)
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

def main():

    args = docopt(__doc__)
    if args['--help']:
        print(__doc__)
        sys.exit(1)
    else:
        sys.exit(run_model(args))


if __name__ == '__main__':
    main()