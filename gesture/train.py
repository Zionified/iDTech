# -*- coding: utf-8 -*-
import tqdm
from utils import preprocess
import torch.nn.functional as F
import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
from dataset import ImageClassificationDataset


class GestureModel:
    # TASK = 'thumbs'
    # TASK = 'emotions'
    # TASK = 'fingers'
    # TASK = 'diy'
    TASK = "tabs"

    # CATEGORIES = ['thumbs_up', 'thumbs_down']
    # CATEGORIES = ['none', 'happy', 'sad', 'angry']
    # CATEGORIES = ['1', '2', '3', '4', '5']
    # CATEGORIES = [ 'diy_1', 'diy_2', 'diy_3']
    CATEGORIES = ["close", "open", "left", "right"]

    # DATASETS = ['A', 'B']
    # DATASETS = ['A', 'B', 'C']
    DATASETS = ["A", "B", "C", "D"]

    TRANSFORMS = transforms.Compose(
        [
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
            transforms.Resize((400, 400)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    DATA_DIR = "./nvdli-data/classification/"

    device = torch.device("cuda")

    def __init__(self, model_name: str = "resnet18"):
        datasets = {}
        for name in GestureModel.DATASETS:
            datasets[name] = ImageClassificationDataset(
                GestureModel.DATA_DIR + GestureModel.TASK + "_" + name,
                GestureModel.CATEGORIES,
                GestureModel.TRANSFORMS,
            )

        print(
            "{} task with {} categories defined".format(
                GestureModel.TASK, GestureModel.CATEGORIES
            )
        )
        self.trainset = datasets[GestureModel.DATASETS[0]]
        self.evalset = datasets[GestureModel.DATASETS[1]]

        # ALEXNET
        # model = torchvision.models.alexnet(pretrained=True)
        # model.classifier[-1] = torch.nn.Linear(4096, len(self.trainset.categories))

        # SQUEEZENET
        # model = torchvision.models.squeezenet1_1(pretrained=True)
        # model.classifier[1] = torch.nn.Conv2d(512, len(self.trainset.categories), kernel_size=1)
        # model.num_classes = len(self.trainset.categories)

        if model_name == "resnet34":
            # RESNET 34
            model = torchvision.models.resnet34(pretrained=True)
            model.fc = torch.nn.Linear(512, len(self.trainset.categories))
        elif model_name == "resnet18":
            # RESNET 18
            model = torchvision.models.resnet18(pretrained=True)
            model.fc = torch.nn.Linear(512, len(self.trainset.categories))
        else:
            raise Exception("model {} not supported".format(model_name))
        self.model_name = model_name

        model = model.to(GestureModel.device)
        self.model = model

    def load(self, model_path: str):
        self.model.load_state_dict(torch.load(model_path))
        print("model loaded from {}".format(model_path))

    def predict(self, image_path: str):
        import PIL
        import numpy as np
        from utils import preprocess
        import torch.nn.functional as F

        input_image = preprocess(np.asarray(PIL.Image.open(image_path)), device=GestureModel.device)
        predictions = [
            (GestureModel.CATEGORIES[i], prediction)
            for i, prediction in enumerate(
                F.softmax(self.model(input_image), dim=1)
                .detach()
                .cpu()
                .numpy()
                .flatten()
            )
        ]
        return list(sorted(predictions, key=lambda i: i[1], reverse=True))

    def train(self, epoches=50, batch_size=64):
        device = GestureModel.device
        model = self.model
        optimizer = torch.optim.Adam(model.parameters())

        try:
            train_loader = torch.utils.data.DataLoader(
                self.trainset, batch_size=batch_size, shuffle=True
            )

            for epoch in tqdm.tqdm(range(epoches), desc="EPOCH", ascii=True):
                # for epoch in range(epoches):
                count = 0
                sum_loss = 0.0
                error_count = 0.0

                model.train()
                # pbar = tqdm.tqdm(iter(train_loader), ascii=True)
                # for images, labels in iter(train_loader):
                # for images, labels in pbar:
                for images, labels, image_paths in iter(train_loader):
                    # send data to device
                    images = images.to(device)
                    labels = labels.to(device)

                    # zero gradients of parameters
                    optimizer.zero_grad()

                    # execute model to get outputs
                    with torch.set_grad_enabled(True):
                        outputs = model(images)

                        # compute loss
                        loss = F.cross_entropy(outputs, labels)

                        # run backpropogation to accumulate gradients
                        loss.backward()

                        # step optimizer to adjust parameters
                        optimizer.step()

                    # increment progress
                    error_count += len(
                        torch.nonzero(outputs.argmax(1) - labels).flatten()
                    )
                    count += len(labels.flatten())
                    sum_loss += float(loss)
                loss = sum_loss / count
                train_accuracy = 1.0 - error_count / count

                model.eval()
                with torch.set_grad_enabled(False):
                    evaluation_loader = torch.utils.data.DataLoader(
                        self.evalset, batch_size=batch_size, shuffle=True
                    )
                    count = 0
                    sum_loss = 0.0
                    error_count = 0.0

                    # pbar = tqdm.tqdm(iter(evaluation_loader), ascii=True)
                    # for images, labels in pbar:
                    for images, labels, image_paths in iter(evaluation_loader):
                        # send data to device
                        images = images.to(device)
                        labels = labels.to(device)

                        # execute model to get outputs
                        outputs = model(images)

                        # calculate accuracy
                        error_count += len(
                            torch.nonzero(outputs.argmax(1) - labels).flatten()
                        )
                        count += len(labels.flatten())

                    eval_accuracy = 1.0 - error_count / count

                if abs(train_accuracy - eval_accuracy) <= 0.035:
                    torch.save(
                        model.state_dict(),
                        "models/gesture-{}-{:02d}-{:.4f}-{:.4f}.pth".format(
                            self.model_name, epoch + 1, train_accuracy, eval_accuracy
                        ),
                    )
        except KeyboardInterrupt:
            raise
        except:
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    gesture_model = GestureModel(model_name="resnet18")
    gesture_model.train()
    # gesture_model.load("models01/gesture-resnet18-29-1.0000-0.9812.pth")
    # for image in [
    #     "open-1.jpg",
    #     "open-2.jpg",
    #     "left-1.jpg",
    #     "left-2.jpg",
    #     "close-1.jpg",
    #     "close-2.jpg",
    # ]:
    #     print(image, gesture_model.predict("images/" + image))