# Debugging-101 Notes
This repository contains all the codes or notes I made during my time working as an iD Tech instructor for the course ML/AI with NVIDIA.

- notes below to install all necessary dependencies and libraries on Windows and Jetson Nano
- `calculate_acc`: calculating the error rate, or equivalently the accuracy of students' trained model
- `detectnet`: combining a trained SSD-mobilenet model for fox detection with the pre-trained SSD-mobilenet v2 model to detect foxes, bears and people.
- `gesture`: adapting the Jupyter Notebook `tab_control.ipynb` to `image-test.py` (work on images) or `camera-test.py` (work on video streams) to perform tab controls on Chrome or keyboard controls in Minecraft. `train.py` modified the `Training and Evaluation` section in `tab_control.ipynb` to allow students train their ResNet18 models in vscode and save all models that have a good performance on both training and validation data (within 3.5% accuracy difference).
- `OA`: my traumatized Optiver Online Assessment practice

## Install git on Windows 10
https://git-scm.com/download/win and restart vscode

## Install PyTorch on Jetson Nano
https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

```{bash}
pip3 install --upgrade pip

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



