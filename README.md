# DeepRace

A small python3 based benchmark tool to compare keras Deep Learning Models.

## Usage (WIP)

``` bash
$ deeprace -g 1 -m resnet30 -o timings.csv -e -1
```

The above starts to train the `resnet30` network on one GPU using the currently configured keras backend, loops through all epochs `-e -1` and writes the measurements to `timings.csv`.

## Dependencies

- python 3
- keras 2+
- tensorflow 1.3+



