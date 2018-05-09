BootStrap: docker
From: tensorflow/tensorflow:1.6.0-gpu-py3

%post
    apt-get -y update
    apt-get -y install python3-pip
    pip3 install keras docopt pytest

%runscript
    exec python "$@"