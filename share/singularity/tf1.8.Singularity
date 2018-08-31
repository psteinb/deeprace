BootStrap: docker
From: tensorflow/tensorflow:1.8.0-gpu-py3

%post
    apt-get -y update
    apt-get -y install python3-pip git
    pip3 install keras docopt pytest versioneer snakemake

%runscript
    exec python "$@"