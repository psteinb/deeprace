#!/usr/bin/env python3
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

if importlib.find_loader:
    finder = importlib.find_loader
elif importlib.utils.find_spec:
    finder = importlib.utils.find_spec

def available_models():

    value = {}

    if not os.path.exists(os.path.join(".","models")):
        return value

    for it in glob.glob(os.path.join(".","models","*.py")):
        stem = os.path.split(it)[-1]
        fname = os.path.splitext(stem)[0]
        name = "models.%s" % fname
        ld = finder(name)
        if ld != None:
            print("strange, found %s but could find a loader for it" % stem)
        else:
            current = importlib.import_module(name)
            value[fname] = current.provides()

    return value

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

    desc = """
    benchmarking tool to run predefined models and print the time per epoch either to the screen or save it to a file
"""

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('model', type=str, nargs='*',
                        help='a model descriptor to run (defaul resnet30)')

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

    parser.add_argument('-t','--timings', type=str, default="timings.csv",
                        help='file to store the individual timings in')

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
        sys.exit(1)

    (loaded,deciphered) = load_model(args.model[0])
    if args.nepochs != 0:
        deciphered["epochs"] = args.nepochs

    start = datetime.datetime.now()
    train, test = loaded.data_loader(args.datapath)
    end = datetime.datetime.now()
    logging.info("loading the data took %f seconds" % ((end-start).total_seconds()))
    logging.info("handing over %s to %s" % (deciphered,args.model[0]))

    hist, timings = loaded.train(train,test,datafraction=args.datafraction,**deciphered)

    with open(args.timings,'w') as csvout:

        runid = "{hostname}-{model}-{dataset},{load_end},{load_end},{train_end},{train_end}".format(hostname=hname,
                                                                                model=args.model[0],
                                                                                dataset=args.dataset,
                                                                                load_start=start.strftime("%Y%m%d:%H%M%S"),
                                                                                load_end=end.strftime("%Y%m%d:%H%M%S"),
                                                                                train_start=timings.train_begin.strftime("%Y%m%d:%H%M%S"),
                                                                                train_end=timings.train_end.strftime("%Y%m%d:%H%M%S")

        )

        csvout.write("runid,load_begin,load_end,train_start,train_end,epoch,rel_epoch_start_sec,epoch_dur_sec,loss,acc,val_loss,val_acc,details\n")
        for i in range(len(timings.epoch_durations)):
            line = "{runid},{num},{rel_epoch_start_sec},{epoch_dur_sec},{loss},{acc},{val_loss},{val_acc},{detail}\n".format(
                runid=constant,
                num=i,
                rel_epoch_start_sec=timings.epoch_start[i],
                epoch_dur_sec=timings.epoch_durations[i],
                loss=hist.history['loss'][i],
                acc=hist.history['acc'][i],
                val_loss=hist.history['val_loss'][i],
                val_acc= hist.history['val_acc'][i],
                detail="-"
            )
            csvout.write(line)
        csvout.close()
        logging.info('wrote %s',args.timings)


    sys.exit(0)

if __name__ == '__main__':
    main()
