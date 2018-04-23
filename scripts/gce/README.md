# Google Com[piute 

## Node details

- 10 GB SSD storage
- 1 K80 in europe-west1
- RHEL7 image
- any vCPU

# Setup

``` bash
$ sudo yum update && sudo yum groupinstall 'Development Tools'
$ git clone https://github.com/singularityware/singularity.git
$ cd singularity
$ git checkout 2.4.2
$ ./autogen.sh
$ ./configure --prefix=/usr/local --sysconfdir=/etc
$ make
$ sudo make install
```

# Benchmarks

``` bash

```

