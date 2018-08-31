BootStrap: docker
From: tensorflow/tensorflow:1.7.0-gpu-py3

%post
    apt-get -y update
    apt-get -y install python3-pip git numactl
    pip3 install keras docopt pytest versioneer snakemake tifffile

%runscript
    exec python "$@"