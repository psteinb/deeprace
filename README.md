# DeepRace

A small python3 based benchmark tool to compare keras Deep Learning Models.

## Usage (WIP)

``` bash
$ deeprace -g 1 -m resnet30 -o timings.csv -e -1
```

The above starts to train the `resnet30` network on one GPU using the currently configured keras backend, loops through all epochs `-e -1` and writes the measurements to `timings.csv`.

Current features are:

``` bash
python ./deeprace.py -h
usage: deeprace.py [-h] [-D DATAPATH] [-L LOGLEVEL] [-e NEPOCHS] [-d DATASET]
                   [-f DATAFRACTION] [-t TIMINGS]
                   [model [model ...]]

benchmarking tool to run predefined models and print the time per epoch either
to the screen or save it to a file

positional arguments:
  model                 a model descriptor to run (defaul resnet30)

optional arguments:
  -h, --help            show this help message and exit
  -D DATAPATH, --datapath DATAPATH
                        path to store the input data in
  -L LOGLEVEL, --loglevel LOGLEVEL
                        logging level to use
  -e NEPOCHS, --nepochs NEPOCHS
                        number of epochs to run
  -d DATASET, --dataset DATASET
                        specify the dataset to use
  -f DATAFRACTION, --datafraction DATAFRACTION
                        fraction of the dataset to use
  -t TIMINGS, --timings TIMINGS
                        file to store the individual timings in
```

Example outout:

``` bash
runid,load_dur_sec,ntrain,ntest,datafraction,train_start,train_end,epoch,rel_epoch_start_sec,epoch_dur_sec,loss,acc,val_loss,val_acc,details
talisker-resnet32v1-cifar10,1.394287,50000,10000,0.1,20180302:165429,20180302:165452,0,4.502005,9.612364,2.0117400371551515,0.3044,1.9701387672424315,0.335,-
talisker-resnet32v1-cifar10,1.394287,50000,10000,0.1,20180302:165429,20180302:165452,1,14.116095,8.322776,1.7452791357040405,0.4022,2.158809609413147,0.33,-
```

## Dependencies

- python 3
- keras 2+
- tensorflow 1.3+



