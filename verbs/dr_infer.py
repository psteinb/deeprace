"""
usage: deeprace infer [options] [--] <model>

options:
    -h, --help                                 print this help message
    -O <mopts> --meta-options=<mopts>          hyper-parameters for training, e.g. batch_size
    -D <dpath> --datapath=<dpath>              path used for temporary storage, e.g. for the input data, checkpoints etc [default: datasets]
    -r <repeats> --nrepeats=<repeats>          how many times to repeat the test phase (unsupported yet) [default: 1]
    -d <ds> --dataset=<ds>                     the dataset to use (depends on the model of choice) [default: model_default]
    -n <ninfer> --num_inferences=<dfrac>       mnumber of inferences to perform [default: 1]
    -t <output> --timings=<output>             file to store the individual timings in [default: inference.tsv]
    -s <sep> --separator=<sep>                 seperator for the output data [default: \t]
    -c <cmt> --comment=<cmt>                   comment to add to the measurement
    -m <mpath> --modelpath=<mpath>             path where the model is stored [default: deeprace-results/test.json]
    -y <yaml> --yaml_results=<ypath>           path where yaml file from training run are stored, optional [default: ]
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
from verbs.loaders import import_model, load_model


try:
    from importlib import util as ilib_util
except:
    raise
else:
    finder = ilib_util.find_spec

def infer_model(args):

    logging.debug("received %s as args",args)

    if not "<model>" in args.keys():
        logging.error("no model recieved")
        return 1


    model_descriptor = args["<model>"]
    config_options = {}
    if os.path.exists(model_descriptor):
        with open(model_descriptor, 'r') as yf:
            config_options = yaml.load(yf)
            yf.close()
        modelname = config_options["model"]
    else:
        modelname = model_descriptor

    (loaded,opts_from_name) = load_model(modelname)

    model = loaded.model()
    model.weights_file = args["--modelpath"]
    logging.info("successfully imported %s",modelname)

    if "backend" in config_options:
        model.backend=config_options["backend"]
    else:
        model.backend=args["--backend"]

    if model.backend == "tensorflow" or model.backend == "tf":
        logging.error("inference with tensorflow as backend is not implemented yet, waiting for TF2 to mature")
        return 1

    #remove unneeded keys here
    deciphered = model.options()
    deciphered.update(opts_from_name)
    meta_opts = {}

    if "--meta-options" in args.keys() and args["--meta-options"]:
        meta_opts = dict((k.strip(), v.strip()) for k,v in
                         (item.split('=') for item in args["--meta-options"].split(',')))
        deciphered.update(meta_opts)

    opts = ",".join(["{k}={v}".format(k=item[0],v=item[1]) for item in deciphered.items() if item[0] not in ["epochs","subtract_pixel_mean","data_augmentation", "checkpoint_epochs"] ])
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
    logging.info("running %s inference", modelname)

    #update dictionary of model object here
    d2 = model.options()
    d3 = {key:d2[key] for key in deciphered if key in d2}
    if d3.keys() == deciphered.keys():
        model.__dict__ = deciphered
    else:
        logging.error("options received (%s) do not match supported options (%s)",deciphered.keys(),d2.keys())

    hname = socket.getfqdn().split(".")[0]

    hist, timings, details = model.infer(test,args["--num_inferences"])

    uuid= str(uuid_from_this(modelname, model.dataset, opts,float(args["--num_inferences"]),versioneer.get_version()))

    yaml_file = os.path.splitext(args["--timings"])[0]+".yaml"
    with open(yaml_file,'w') as yamlf:

        odict = OrderedDict()
        odict["host"]             =   hname
        odict["model"]            =   modelname
        odict["mode"]             =   "inference"
        odict["backend"]          =   model.backend
        odict["dataset"]          =   model.dataset
        odict["load_dur_sec"]     =   (end-start).total_seconds()
        odict["num_inferences"]   =   float(args["--num_inferences"])
        odict["opts"]             =   opts
        odict["versions"]         =   model.versions()
        odict["deeprace_version"] =   versioneer.get_version()
        odict["uuid"]             =   uuid
        odict["comment"]          =   args["--comment"]

        to_write = yaml_ordered(odict)
        #yaml.dump(to_write, yamlf, default_flow_style=False)
        yamlf.write(to_write)
        logging.info("wrote %s",yaml_file)

    with open(args["--timings"],'w') as csvout:

        tags = "uuid,batch_id,infer_dur_sec,predictions".split(",")

        header_str = args["--separator"].join(tags)
        csvout.write(header_str+"\n")


        for i in range(len(timings)):

            fields = [uuid,str(i),str(timings[i]),str(hist[i])]

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
