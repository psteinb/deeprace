"""
usage: deeprace train [options] [--] <model>

options:
    -h, --help                                 print this help message
    -D <dpath> --datapath=<dpath>              path used for temporary storage, e.g. for the input data, checkpoints etc [default: datasets]
    -r <repeats> --nrepeats=<repeats>          how many times to repeat the test phase (unsupported yet) [default: 1]
    -d <ds> --dataset=<ds>                     the dataset to use (depends on the model of choice) [default: model_default]
    -n <ninfer> --num_inferences=<dfrac>       mnumber of inferences to perform [default: 1.]
    -t <output> --timings=<output>             file to store the individual timings in [default: timings.tsv]
    -s <sep> --separator=<sep>                 seperator for the output data [default: \t]
    -c <cmt> --comment=<cmt>                   comment to add to the measurement
    -m <mpath> --modelpath=<mpath>             path where the model is stored [default: deeprace-results]
"""

from docopt import docopt
import os
import sys
import logging
import datetime
import socket
import versioneer
import yaml
from collections import OrderedDict

from verbs.utils import uuid_from_this, yaml_this, yaml_ordered

try:
    from importlib import util as ilib_util
except:
    raise
else:
    finder = ilib_util.find_spec

def infer_model(args):

    logging.debug("received %s as args",args)

    if not "<models>" in args.keys():
        logging.error("no model recieved")
        return 1
    else:
        logging.warning("not implemented yet")
        return 0

    location = args["<modeltoload>"]
    (loaded,opts_from_name) = load_model(modelname)

    model = loaded.model()
    logging.info("successfully imported %s",modelname)

    model.backend=args["--backend"]
    deciphered = model.options()
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
    if 'model_default' != args["--dataset"].lower():
        model.dataset = args["--dataset"].lower()

    try:
        start = datetime.datetime.now()
        train, test, ntrain, ntest = model.data_loader(args["--datapath"],dataset_name=model.dataset)
        end = datetime.datetime.now()
    except Exception as ex:
        logging.error("unable to load indicated dataset %s from/into %s (%s operates with %s)",model.dataset,args["--datapath"], modelname, model.available_datasets())
        logging.error(ex)
        sys.exit(1)

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
    uuid= str(uuid_from_this(modelname, model.dataset, opts,float(args["--datafraction"]),versioneer.get_version()))

    yaml_file = os.path.splitext(args["--timings"])[0]+".yaml"
    with open(yaml_file,'w') as yamlf:

        odict = OrderedDict()
        odict["host"]             =   hname
        odict["model"]            =   modelname
        odict["backend"]          =   args["--backend"]
        odict["dataset"]          =   model.dataset
        odict["load_dur_sec"]     =   (end-start).total_seconds()
        odict["ntrain"]           =   ntrain
        odict["ntest"]            =   ntest
        odict["datafraction"]     =   float(args["--datafraction"])
        odict["train_start"]      =   timings.train_begin.strftime("%Y%m%d:%H%M%S")
        odict["train_end"]        =   timings.train_end.strftime("%Y%m%d:%H%M%S")
        odict["opts"]             =   opts
        odict["n_model_params"]   =   int(details['num_weights'])
        odict["versions"]         =   model.versions()
        odict["deeprace_version"] =   versioneer.get_version()
        odict["uuid"]             =   uuid
        odict["comment"]          =   args["--comment"]

        to_write = yaml_ordered(odict)
        #yaml.dump(to_write, yamlf, default_flow_style=False)
        yamlf.write(to_write)
        logging.info("wrote %s",yaml_file)

    with open(args["--timings"],'w') as csvout:

        tags = "uuid,epoch,rel_epoch_start_sec,epoch_dur_sec".split(",")
        for k in sorted(hist.keys()):
            tags.append(k)

        header_str = args["--separator"].join(tags)
        csvout.write(header_str+"\n")


        for i in range(len(timings.epoch_durations)):

            fields = [uuid,str(i), str(timings.epoch_start[i]), str(timings.epoch_durations[i])]

            for k in sorted(hist.keys()):
                fields.append(str(hist[k][i]))

            line = args["--separator"].join(fields)
            csvout.write(line+"\n")
            logging.debug("+ %s",line)

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