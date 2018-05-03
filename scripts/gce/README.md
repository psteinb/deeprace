# Google Com[piute 

## Node details

- 10 GB SSD storage
- 1 K80 in europe-west1
- RHEL7 image
- any vCPU

# Setup

### Prepping the environment

``` bash
$ sudo yum update && sudo yum groupinstall 'Development Tools' && sudo yum install kernel-devel
#the driver has to be downloaded first
$ sudo ./NVIDIA-Linux-x86_64-390.42.run 
#cross check that driver is installed
instance-1 ~]$ nvidia-smi 
Mon Apr 23 10:10:55 2018       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 390.42                 Driver Version: 390.42                    |
|-------------------------------|----------------------|----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla K80           Off  | 00000000:00:04.0 Off |                    0 |
| N/A   32C    P0    59W / 149W |      0MiB / 11441MiB |    100%      Default |
+-------------------------------|----------------------|----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+


```

### prepping singularity

``` bash
$ git clone https://github.com/singularityware/singularity.git
$ cd singularity
$ git checkout 2.4.2
$ ./autogen.sh
$ ./configure --prefix=/usr/local --sysconfdir=/etc
$ make
$ sudo make install
#cross check that the container sees a GPU
$ singularity exec -B /home/steinbac/deeprace:/deeprace --nv /home/steinbac/tf1.3-plus.simg nvidia-smi
Mon Apr 23 10:11:55 2018       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 390.42                 Driver Version: 390.42                    |
|-------------------------------|----------------------|----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla K80           Off  | 00000000:00:04.0 Off |                    0 |
| N/A   33C    P0    68W / 149W |      0MiB / 11441MiB |     94%      Default |
+-------------------------------|----------------------|----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

# Benchmarks

``` bash
$ cd development/deeprace/scripts/gce
$ ./resnet30.sh > resnet30.log 2>&1 
$ ./resnet50.sh > resnet50.log 2>&1 

```

