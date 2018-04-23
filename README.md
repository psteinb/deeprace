# DeepRace [![DOI](https://zenodo.org/badge/123592478.svg)](https://zenodo.org/badge/latestdoi/123592478)


A small python3 based benchmark tool to compare ([Keras](keras.io)) Deep Learning Models. This project is currently under development. Please expect the usage to change without prior notice.

## Usage 

Current features are:

``` bash
$ python3 ./deeprace.py --help
usage: deeprace [--version] [--help] [--verbose] [--loglevel <level>] <command> [<args>...]

options:
   -h, --help                           Show this help message
   -v, --version                        Print the version of deeprace
   -V, --verbose                        Run in verbose mode
   -L <level> --loglevel=<level>        logging level to use [default: info]

The most commonly used git commands are:
   list      list available models
   train     run training on a given model   
   describe  show available parameters for given model

See 'deeprace help <command>' for more information on a specific command.
```

### `run`

``` bash
$ python3 ./deeprace.py help train

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
    -s <sep> --separator=<sep>                 seperator for the output data [default: 	]
    -c <cmt> --comment=<cmt>                   comment to add to the measurement
```

To run a benchmark, make sure that tensorflow and keras are available. Then call:

``` bash
$ python3 ./deeprace.py train -t test.tsv resnet56v1
```

Example outout:

``` bash
runid,load_dur_sec,ntrain,ntest,datafraction,train_start,train_end,epoch,rel_epoch_start_sec,epoch_dur_sec,loss,acc,val_loss,val_acc,details
talisker-resnet32v1-cifar10,1.394287,50000,10000,0.1,20180302:165429,20180302:165452,0,4.502005,9.612364,2.0117400371551515,0.3044,1.9701387672424315,0.335,-
talisker-resnet32v1-cifar10,1.394287,50000,10000,0.1,20180302:165429,20180302:165452,1,14.116095,8.322776,1.7452791357040405,0.4022,2.158809609413147,0.33,-
```


### `list`

``` bash
$ python3 ./deeprace.py list
[180327 09:12:39 tauruslogin5] :: available models:
[180327 09:12:39 tauruslogin5] :: [resnet] resnet20v1 resnet32v1 resnet44v1 resnet56v1 resnet110v1 resnet164v1 resnet29v2 resnet47v2 resnet65v2 resnet83v2 resnet164v2 resnet245v2 resnet1001v2
```

### `describe`

``` bash
$ python3 ./deeprace.py describe resnet32v1
[180329 13:15:14 r02n12] :: importing resnet (from resnet32v1)
[180329 13:15:14 r02n12] :: found resnet implementation
[180329 13:15:14 r02n12] :: available options for resnet32v1
  num_classes          = 10
  batch_size           = 32
  scratchspace         = /home/steinbac
  checkpoint_epochs    = False
  subtract_pixel_mean  = True
  n_gpus               = 1
  epochs               = 200
  n                    = 5
  version              = 1
  data_augmentation    = True
```

## Dependencies

`deeprace` is written with flexibility in mind. THe core idea is that it can/will be distributed via PYPI (not available yet). During installation through pip (not available yet), only dependencies for the command line tooling will be added. It is up to the user to have a compliant environment to actually run benchmarks. 

### Required

- python 3.5+

### Optional

Currently, the benchmark relies on keras and tensorflow as a backend. This is subject to change in the near future.

- keras 2.1.3+ 
- tensorflow 1.3+


## Known Issues

- the parallel model in keras does not appear to work with tensorflow 1.3, only works with tensorflow 1.5 and above

