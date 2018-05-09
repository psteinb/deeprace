BootStrap: docker
From: tensorflow/tensorflow:1.5.0-gpu-py3

%post
    apt-get -y update
    apt-get -y install python3-pip git 
    pip3 install keras docopt pytest snakemake versioneer

%runscript
    exec python "$@"