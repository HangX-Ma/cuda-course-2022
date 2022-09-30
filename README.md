# :rocket:`ZJUER`: cuda-course-2022

This repository contains homework based on _CUDA_By_Example_ reference book. It will be updated continuously until the course final.

## Environment
If you want to launch the demo in this repository, you need to install the following software and equips with a NVIDIA GPU.
- Ubuntu 20.04
- OpenGL 4.6, GLUT
- NVIDIA Driver (GeForce MX150 or **Newer**, CUDA 11.4)
- CMake 3.16.3


## Demo Lists
- [x] **A1** - [Julia Set](A1/README.md): Change the julia-set fractal function and decorate the julia-set graphic design with gradient color.
- [ ] **A2** - [Dynamic Ray Tracing](A2/README.md):

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