# DeepRace [![DOI](https://zenodo.org/badge/123592478.svg)](https://zenodo.org/badge/latestdoi/123592478)


A small python3 based benchmark tool to compare ([Keras](keras.io)) Deep Learning Models. This project is currently heavy development. Please expect the usage to change without prior notice.

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
    -b <backe> --backend=<backe>               which backend to use [default: keras]
    -e <neps> --nepochs=<neps>                 number of epochs to train [default: 0]
    -d <ds> --dataset=<ds>                     the dataset to use (depends on the model of choice) [default: model_default]
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
$ python3 ./deeprace.py train -e 5 -f 0.1 care_denoise2d
#...
[180509 12:27:09 talisker] :: wrote timings.tsv
[180509 12:27:09 talisker] :: Done.
$ cat timings.tsv
host	model	dataset	load_dur_sec	ntrain	ntest	datafraction	train_start	train_end	epoch	rel_epoch_start_sec	epoch_dur_sec	acc	loss	val_acc	val_loss	opts	n_model_params	versions	deeprace_version	comment
talisker-10g	care_denoise2d	care_2d	9.195996	768	0	0.1	20180509:122702	20180509:122709	0	0.000134	5.577799	0.00503091251148897	0.09069557444137685	0.0052471160888671875	0.0868075042963028	validation_split=0.1,dataset=care_2d,batch_size=32,n_dims=2,n_row=3,backend=keras,depth=2,version=1,n_gpus=1,checkpoint_epochs=False,n_depth=2,filter_base=16,epochs=5,n_conv_per_depth=2,scratchspace=/deeprace,n_col=3	83377	keras:2.1.5,backend:tensorflow:1.7.0	0.2.0+49.g9c8fa46.dirty	None
talisker-10g	care_denoise2d	care_2d	9.195996	768	0	0.1	20180509:122702	20180509:122709	1	5.578052	0.28374	0.005308263442095588	0.06664747700971715	0.0056743621826171875	0.04846179112792015	validation_split=0.1,dataset=care_2d,batch_size=32,n_dims=2,n_row=3,backend=keras,depth=2,version=1,n_gpus=1,checkpoint_epochs=False,n_depth=2,filter_base=16,epochs=5,n_conv_per_depth=2,scratchspace=/deeprace,n_col=3	83377	keras:2.1.5,backend:tensorflow:1.7.0	0.2.0+49.g9c8fa46.dirty	None
talisker-10g	care_denoise2d	care_2d	9.195996	768	0	0.1	20180509:122702	20180509:122709	2	5.861963	0.279709	0.005573272705078125	0.03972847233800327	0.005794525146484375	0.047284774482250214	validation_split=0.1,dataset=care_2d,batch_size=32,n_dims=2,n_row=3,backend=keras,depth=2,version=1,n_gpus=1,checkpoint_epochs=False,n_depth=2,filter_base=16,epochs=5,n_conv_per_depth=2,scratchspace=/deeprace,n_col=3	83377	keras:2.1.5,backend:tensorflow:1.7.0	0.2.0+49.g9c8fa46.dirty	None
talisker-10g	care_denoise2d	care_2d	9.195996	768	0	0.1	20180509:122702	20180509:122709	3	6.141824	0.282113	0.005601097555721507	0.044960127595592946	0.0057621002197265625	0.039718300104141235	validation_split=0.1,dataset=care_2d,batch_size=32,n_dims=2,n_row=3,backend=keras,depth=2,version=1,n_gpus=1,checkpoint_epochs=False,n_depth=2,filter_base=16,epochs=5,n_conv_per_depth=2,scratchspace=/deeprace,n_col=3	83377	keras:2.1.5,backend:tensorflow:1.7.0	0.2.0+49.g9c8fa46.dirty	None
talisker-10g	care_denoise2d	care_2d	9.195996	768	0	0.1	20180509:122702	20180509:122709	4	6.424077	0.279615	0.005590551039751838	0.03724859808297718	0.005687713623046875	0.04680463671684265	validation_split=0.1,dataset=care_2d,batch_size=32,n_dims=2,n_row=3,backend=keras,depth=2,version=1,n_gpus=1,checkpoint_epochs=False,n_depth=2,filter_base=16,epochs=5,n_conv_per_depth=2,scratchspace=/deeprace,n_col=3	83377	keras:2.1.5,backend:tensorflow:1.7.0	0.2.0+49.g9c8fa46.dirty	None
```


### `list`

``` bash
$ python3 ./deeprace.py list
Using TensorFlow backend.
[180509 12:26:02 talisker] :: found __init__.py but could not find a loader for it
[180509 12:26:02 talisker] :: available models and backends:
	[0] resnet20v1 resnet32v1 resnet44v1 resnet56v1 resnet110v1 resnet164v1 resnet29v2 resnet47v2 resnet65v2 resnet83v2 resnet164v2 resnet245v2 resnet1001v2
	     backend(s): keras,tensorflow.keras,tensorflow
	     datasets(s): cifar10
	[1] care_denoise_2Ddepth2
	     backend(s): keras,tensorflow.keras
	     datasets(s): care_2d
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
- docopt
- versioneer

### Optional

Currently, the benchmark relies on keras and tensorflow as a backend. This is subject to change in the near future.

- keras 2.1.3+ 
- tensorflow 1.3+
- tifffile
- snakemake

For an up to date impression on the dependencies, see the [singularity](http://singularity.lbl.gov/) container recipes shipped with this repo, e.g. for [tensorflow 1.7](share/singularity/tf1.7.Singularity)

## Known Issues

- the parallel model in keras does not appear to work with tensorflow 1.3, only works with tensorflow 1.5 and above

