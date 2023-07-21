from jetcam.usb_camera import USBCamera
from jetcam.csi_camera import CSICamera
import pyautogui

def left():
    pyautogui.hotkey('ctrl', 'shift', 'tab')
    print("Tab switched Left.")

#switch right
def right():
    pyautogui.hotkey('ctrl', 'tab')
    print("Switch tab Right")

#open tab
def open():
    pyautogui.hotkey('ctrl', 't')
    print("New tab opened.")

def close():
    pyautogui.hotkey('ctrl', 'w')
    print("Tab closed.")

def none():
    print("N/A")


# for USB Camera (Logitech C270 webcam), uncomment the following line
camera = USBCamera(width=640, height=480, capture_device=0) # confirm the capture_device number

# for CSI Camera (Raspberry Pi Camera Module V2), uncomment the following line
# camera = CSICamera(width=224, height=224, capture_device=0) # confirm the capture_device number

camera.running = True
print("camera created")

import torchvision.transforms as transforms
from dataset import ImageClassificationDataset

# TASK = 'thumbs'
# TASK = 'emotions'
# TASK = 'fingers'
# TASK = 'diy'
TASK = 'tabs'

# CATEGORIES = ['thumbs_up', 'thumbs_down']
# CATEGORIES = ['none', 'happy', 'sad', 'angry']
# CATEGORIES = ['1', '2', '3', '4', '5']
# CATEGORIES = [ 'diy_1', 'diy_2', 'diy_3']
CATEGORIES = ['close', 'open', 'left', 'right']

# DATASETS = ['A', 'B']
# DATASETS = ['A', 'B', 'C']
DATASETS = ['A', 'B', 'C', 'D']

TRANSFORMS = transforms.Compose([
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# datasets = {}
# for name in DATASETS:
#     datasets[name] = ImageClassificationDataset('../data/classification/' + TASK + '_' + name, CATEGORIES, TRANSFORMS)
    
print("{} task with {} categories defined".format(TASK, CATEGORIES))

# ================ Load Models ============================
import torch
import torchvision

CATEGORY_LEN = 4
MODEL_PATH = 'models/gesture-resnet18-29-1.0000-0.9812.pth'
print("loading model ", MODEL_PATH)

device = torch.device('cuda')    #.device(0)

# RESNET 18
model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(512, CATEGORY_LEN)
#print(type(model))


# model = torch.load(MODEL_PATH, map_location=device)
# model = torchvision.models.resnet18()
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(device)
# print(type(model))
#model = model.to(device)
#or: model = torch.load(MODEL_PATH, map_location=device) as given by chatgpt

# display(model_widget)
print("model configured and model_widget created")

# ================ Live Execution ==========================
import threading
import datetime
import time
from utils import preprocess
import torch.nn.functional as F
import cv2
import numpy as np

CATEGORIES = ['close', 'open', 'left', 'right']
score_widgets = []

def live(model, camera):
    print("enter here")
    image = camera.value
    print("camera read: ", image.shape)
    preprocessed = preprocess(image)
    print("preprocessed image: ", preprocessed.shape)
    output = model(preprocessed)
    print("output shape:", output.shape)
    output = F.softmax(output, dim=1)
    print("a")
    output = output.detach()
    print("a")
    output = output.cpu()
    print("a")
    output = output.numpy()
    print("a")
    output = output.flatten()
    print("a")
    print("output: ", output)
    category_index = output.argmax()
    print("category index:", category_index)
    prediction_value = CATEGORIES[category_index]
    print("detected:", prediction_value)

    if output[category_index] > 0.40:
        if prediction_value == "open":
            open()
        elif prediction_value == "close":
            close()
        elif prediction_value == "left":
            left()
        elif prediction_value == "right":
            right()
        else:
            none()
    time.sleep(1)

# execute_thread = threading.Thread(target=live, args=(model, camera))
# execute_thread.start()
t_end = datetime.datetime.now() + datetime.timedelta(seconds=300)
print("end time: ", t_end)
while datetime.datetime.now() < t_end:
    live(model, camera)


# display(live_execution_widget)
print("live_execution_widget created")
# ================ Close Camera ============================
import os
# import IPython

# if type(camera) is CSICamera:
#     print("Ignore 'Exception in thread' tracebacks\n")
#     camera.cap.release()

os._exit(00)