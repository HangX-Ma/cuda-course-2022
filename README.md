# :rocket:`ZJUER`: cuda-course-2022

This repository contains homework based on _CUDA_By_Example_ reference book. It will be updated continuously until the course final.

## Environment
If you want to launch the demo in this repository, you need to install the following software and equips with a NVIDIA GPU.
- Ubuntu 20.04
- OpenGL 4.6, GLUT
- NVIDIA Driver (GeForce MX150 or **Newer**, CUDA 11.4)
- CMake 3.16.3

## CUDA Platform
```shell
./deviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "NVIDIA GeForce MX150"
  CUDA Driver Version / Runtime Version          11.4 / 11.4
  CUDA Capability Major/Minor version number:    6.1
  Total amount of global memory:                 4042 MBytes (4238737408 bytes)
  (003) Multiprocessors, (128) CUDA Cores/MP:    384 CUDA Cores
  GPU Max Clock rate:                            1532 MHz (1.53 GHz)
  Memory Clock rate:                             3004 Mhz
  Memory Bus Width:                              64-bit
  L2 Cache Size:                                 524288 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        98304 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 11.4, CUDA Runtime Version = 11.4, NumDevs = 1
Result = PASS
```

## Demo Lists
- [x] **A1** - [Julia Set](A1/README.md): Change the julia-set fractal function and decorate the julia-set graphic design with gradient color.
- [x] **A2** - [Dynamic Ray Tracing](A2/README.md): Realize the dynamic circumstance ray tracing, using noconst memory and const memory for comparison.

## _X11 Forwarding Service_ on GPU server
Both on server and client.
1. Install essential software and `xclock` testing app.
    ```shell
    sudo apt-get install xorg
    sudo apt-get install xauth
    sudo apt-get install openbox
    sudo apt-get install xserver-xorg-legacy
    sudo apt install x11-apps # xclock
    ```
2. Please install `Remote X11` extension in VScode if you want to use `X11 Forwarding` in your vscode terminal.
3. Configure `X11 Forwarding`.
    ```shell
    sudo cp /etc/ssh/sshd_config /etc/ssh/sshd_config.backup
    sudo vim /etc/ssh/sshd_config
    ```
    **sshd_config**
    ```vim
    # sshd_config
    ...
    X11Forwarding yes
    ForwardX11 yes
    ForwardX11Trusted yes
    ForwardAgent yes # or AllowAgentForwarding
    ...
    ```

    ```shell
    # restart ssh
    systemctl restart sshd
    # test X11 forwarding
    xclock
    ```

## Problem Record
I want to use X11 server forwarding function to display result returned from the rented GPU server, but it will occupy the GPU resource. Terminal on GPU server will return `Segmentation fault (core dumped)`. On your personal computer, if you run demos in `cuda_by_example/chapter08` provided by NVIDIA, terminal may return:
```shell
all CUDA-capable devices are busy or unavailable in ../common/gpu_anim.h at line 77
``` 

X11 server cannot provide service simultaneously when running GPU program on my own computer. If you want to run program, please do as follows:
```shell
nvidia-settings # type in terminal
PRIME Profiles->NVIDIA(Performance Mode) # Find in NVIDIA X Server Settings GUI
```


## LICENSE
Apache License Version 2.0