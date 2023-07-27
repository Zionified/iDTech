# -*- coding: utf-8 -*-
import os
import datetime
import time
import random
import cv2
import torch
import torchvision
import PIL

# import pyautogui
import numpy as np
import torchvision.transforms as transforms
from utils import preprocess
import torch.nn.functional as F


def left():
    # pyautogui.hotkey('ctrl', 'shift', 'tab')
    print("Tab switched Left.")


# switch right
def right():
    # pyautogui.hotkey('ctrl', 'tab')
    print("Switch tab Right")


# open tab
def open():
    # pyautogui.hotkey('ctrl', 't')
    print("New tab opened.")


def close():
    # pyautogui.hotkey('ctrl', 'w')
    print("Tab closed.")


def none():
    print("N/A")


TASK = "tabs"

CATEGORIES = ["close", "open", "left", "right"]

DATASETS = ["A", "B", "C", "D"]

TRANSFORMS = transforms.Compose(
    [
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

print("{} task with {} categories defined".format(TASK, CATEGORIES))

# ================ Load Models ============================
CATEGORY_LEN = 4
MODEL_PATH = "models/gesture-resnet18-29-1.0000-0.9812.pth"

device = torch.device("cuda")  # .device(0)

# RESNET 18
model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(512, CATEGORY_LEN)
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(device)
model.eval()

print("model configured, using ", MODEL_PATH)

# ================ Live Execution ==========================
CATEGORIES = ["close", "open", "left", "right"]
score_widgets = []


def live(model, camera):
    # print("enter here")
    image = camera.value
    # print("camera read: ", image.shape)
    preprocessed = preprocess(image)
    # print("preprocessed image: ", preprocessed.shape)
    output = model(preprocessed)
    # print("output shape:", output.shape)
    output = F.softmax(output, dim=1)
    output = output.detach()
    output = output.cpu()
    output = output.numpy()
    output = output.flatten()
    # print("output: ", output)
    category_index = output.argmax()
    # print("category index:", category_index)
    prediction_value = CATEGORIES[category_index]
    # print("detected:", prediction_value, ", output:", [(CATEGORIES[i], prediction) for i, prediction in enumerate(output)])

    if output[category_index] > 0.75:
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
    # time.sleep(10)
    return prediction_value


t_end = datetime.datetime.now() + datetime.timedelta(seconds=100)
print("end time: ", t_end)


class FakeCamera:
    pass


fake_camera = FakeCamera()

# DATA_DIR = "nvdli-data/classification/tabs_B/"
DATA_DIR = "images/"

images = [
    "close-1.jpg",
    "close-2.jpg",
    "left-1.jpg",
    "left-2.jpg",
    "open-1.jpg",
    "open-2.jpg",
]
# for category in CATEGORIES:
#     files = [DATA_DIR + category + "/" + i for i in os.listdir(DATA_DIR + category) if i.endswith(".jpg")]
#     random.shuffle(files)
#     images += files

error_count = 0
for image in images:
    # print("image:", image)
    image_path = os.path.join(DATA_DIR, image)
    fake_camera.value = np.asarray(PIL.Image.open(image_path))
    prediction_value = live(model, fake_camera)
    # if prediction_value not in image:
    #     error_count += 1
    print("=" * 20)
# print(1.0 - error_count / len(images))

print("detection ends")

# ================ Close Session ============================
# import os
# import IPython

# os._exit(00)
