"""
usage: deeprace train [options] [--] <models>

options:
    -h, --help                                 print this help message
    -O <mopts> --meta-options=<mopts>          hyper-parameters for training, e.g. batch_size
    -D <dpath> --datapath=<dpath>              path used for temporary storage, e.g. for the input data, checkpoints etc [default: datasets]
    -R <rpath> --resultspath=<rpath>           path to store results or checkpoints [default: deeprace-results]
    -b <backe> --backend=<backe>               which backend to use [default: keras]
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
import versioneer

try:
    from importlib import util as ilib_util
except:
    raise
else:
    finder = ilib_util.find_spec

def import_model(name):
    """ import a model and return the imported symbol """

    expected_models_dir = os.path.dirname(os.path.abspath(__file__))
    expected_models_dir = os.path.join(os.path.dirname(expected_models_dir),"models")

    if not os.path.exists(expected_models_dir):
        logging.warning("%s was not found in %s",expected_models_dir,os.curdir)
        return None

    expected_location =os.path.join(expected_models_dir,name+".py")
    if not os.path.exists(expected_location):
        logging.warning("%s was not found in %s",expected_location,os.curdir)
        return None
    else:
        logging.info("found %s implementation",name)

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
    (loaded,opts_from_name) = load_model(modelname)

    model = loaded.model()
    model.backend=args["--backend"]
    deciphered = model.options()

    logging.info("successfully imported %s",modelname)

    deciphered.update(opts_from_name)
    meta_opts = {}

    if "--meta-options" in args.keys() and args["--meta-options"]:
        meta_opts = dict((k.strip(), v.strip()) for k,v in
                         (item.split('=') for item in args["--meta-options"].split(',')))
        deciphered.update(meta_opts)

    if ("--nepochs") in args.keys() and int(args["--nepochs"]) > 0:
        deciphered["epochs"] = int(args["--nepochs"])
    
    opts = ",".join(["{k}={v}".format(k=item[0],v=item[1]) for item in deciphered.items() ])
    deciphered['datapath'] = args["--datapath"]
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

        front_tags = "host,model,dataset,load_dur_sec,ntrain,ntest,datafraction,train_start,train_end".split(",")
        hist_tags = "epoch,rel_epoch_start_sec,epoch_dur_sec".split(",")
        for k in sorted(hist.keys()):
            hist_tags.append(k)
        rear_tags = "opts,n_model_params,versions,deerace_version,comment".split(",")
        tags = front_tags + hist_tags + rear_tags


        header_str = args["--separator"].join(tags)
        csvout.write(header_str+"\n")
        logging.info("wrote %s",header_str)

        front_constant = ("{sep}".join([ str("{%s}" % item) for item in front_tags ])).format(
            host=hname,
            model=modelname,
            dataset=args["--dataset"],
            load_dur_sec=(end-start).total_seconds(),
            ntrain=ntrain,
            ntest=ntest,
            datafraction=args["--datafraction"],
            train_start=timings.train_begin.strftime("%Y%m%d:%H%M%S"),
            train_end=timings.train_end.strftime("%Y%m%d:%H%M%S"),
            sep=args["--separator"]
        )
        rear_constant = (args["--separator"].join([ str("{%s}" % item) for item in rear_tags ])).format(
            opts=opts,
            n_model_params=details['num_weights'],
            versions=model.versions(),
            deeprace_version=versioneer.get_version(),
            comment=args["--comment"]
        )

        for i in range(len(timings.epoch_durations)):

            fields = [front_constant, str(i), str(timings.epoch_start[i]), str(timings.epoch_durations[i])]

            for k in sorted(hist.keys()):
                fields.append(str(hist[k][i]))

            fields.append(rear_constant)

            #([opts, details['num_weights'], model.versions(), comment=args["--comment"]))

            # line = "{constant}{sep}{num}{sep}{rel_epoch_start_sec}{sep}{epoch_dur_sec}{sep}{loss}{sep}{acc}{sep}{val_loss}{sep}{val_acc}{sep}{top_k_catacc}{sep}{val_top_k_catacc}{sep}{detail}{sep}{n_model_params}{sep}{versions}{sep}{comment}\n".format(
            #     constant=front_constant,
            #     num=int(i),
            #     rel_epoch_start_sec=timings.epoch_start[i],
            #     epoch_dur_sec=timings.epoch_durations[i],
            #     loss=hist['loss'][i],
            #     acc=hist['acc'][i],
            #     val_loss=hist['val_loss'][i],
            #     val_acc= hist['val_acc'][i],
            #     top_k_catacc=hist['top_k_categorical_accuracy'][i],
            #     val_top_k_catacc=hist['val_top_k_categorical_accuracy'][i],
            #     detail=opts,
            #     sep=args["--separator"],
            #     n_model_params=details['num_weights'],
            #     versions=model.versions(),
            #     comment=args["--comment"]
            # )

            line = args["--separator"].join(fields)
            csvout.write(line+"\n")

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
