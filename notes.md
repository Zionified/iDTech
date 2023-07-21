# Debugging-101 Notes
## Install git on Windows 10
https://git-scm.com/download/win and restart vscode

## Install PyTorch on Jetson Nano
https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

```{bash}
wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl

sudo apt-get install python3-pip libopenblas-base libopenmpi-dev libomp-dev

pip3 install Cython

pip3 install numpy torch-1.8.0-cp36-cp36m-linux_aarch64.whl
```

## Install torchvision:
- Using `pip`
    ```
    pip install torchvision
    ```
- Building from source
    ```
    sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev

    git clone --branch 0.9 https://github.com/pytorch/vision torchvision

    cd torchvision

    export BUILD_VERSION=0.9.0

    python3 setup.py install --user

    cd ../  # attempting to load torchvision from build dir will result in import error

    pip install 'pillow<7'
    ```
## SSH from Command Prompt
```
ssh-keygen -t rsa

ssh-copy-id nvidia@ip-address

ssh nvidia@ip-address
```
## Troubleshooting
Plug and Unplug Nano to solve the vscode loading issue

## Legacy from the Bird
sudo apt-get -y install python3-pi/

pip3 install numpy==1.19.4 torch-1.8.0-cp36-cp36m-linux_aarch64.whl



