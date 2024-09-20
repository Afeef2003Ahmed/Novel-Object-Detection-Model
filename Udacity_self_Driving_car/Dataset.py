import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import time
from torchvision import datasets,transforms
import os
from PIL import Image
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
import torch.cuda.amp as amp
import math
import xml.etree.ElementTree as ET
from PIL import Image
import warnings
import json

class custom_dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.class_names = ["car","pedestrian","trafficLight-Red","trafficLight-Green","truck","trafficLight","biker","trafficLight-RedLeft","trafficLight-GreenLeft","trafficLight-Yellow","trafficLight-YellowLeft"]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        self.annotations = []
        for xml_file in os.listdir(root_dir):
            if xml_file.endswith(".xml"):
                tree = ET.parse(os.path.join(root_dir, xml_file))
                root = tree.getroot()

                image_info = {
                    "filename": root.find("filename").text,
                    "size": {
                        "width": int(root.find("size/width").text),
                        "height": int(root.find("size/height").text)
                    },
                    "objects": []
                }

                for obj in root.findall("object"):
                    bbox = obj.find("bndbox")
                    bbox_info = {
                        "name": obj.find("name").text,
                        "pose": obj.find("pose").text,
                        "truncated": int(obj.find("truncated").text),
                        "difficult": int(obj.find("difficult").text),
                        "bbox": [
                            int(bbox.find("xmin").text),
                            int(bbox.find("ymin").text),
                            int(bbox.find("xmax").text),
                            int(bbox.find("ymax").text)
                        ]
                    }
                    image_info["objects"].append(bbox_info)

                self.annotations.append(image_info)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_info = self.annotations[idx]
        img_path = os.path.join(self.root_dir, img_info["filename"])
        image = Image.open(img_path).convert("RGB")

        width, height = img_info["size"]["width"], img_info["size"]["height"]

        boxes = []
        labels = []
        for obj in img_info["objects"]:
            xmin = obj["bbox"][0]
            ymin = obj["bbox"][1]
            xmax = obj["bbox"][2]
            ymax = obj["bbox"][3]

            x_center_norm = (xmin+xmax)/(2*width)
            y_center_norm = (ymin+ymax)/(2*height)
            width_norm = (xmax-xmin)/(width)
            height_norm = (ymax-ymin)/(height)

            boxes.append([x_center_norm,y_center_norm,width_norm,height_norm])
            try:
                labels.append(self.class_to_idx[obj["name"]])
            except KeyError:
                print(f"Class '{obj['name']}' not found in class_to_idx dictionary.")
                continue

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = torch.zeros((len(labels),6))
        annotation = torch.cat((labels.unsqueeze(1),boxes),dim=1)
        #print(boxes.shape)
        #print(labels.shape)
        target[:,1:] = annotation

        if self.transform:
            image = self.transform(image)

        return image, target



transform = transforms.Compose([
    transforms.Resize((416,416)),
    transforms.RandomGrayscale(0.3),
    transforms.RandomHorizontalFlip(0.3),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
])

transform_val = transforms.Compose([
    transforms.Resize((416,416)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
])




def collate_fn(batch):
  img,label = zip(*batch)
  for i, l in enumerate(label):
    l[:, 0] = i
  return torch.stack(img, 0), torch.cat(label, 0)


train_root_dir = "/raid/cs21resch15003/Afeef_Intern/CustomModel_Self_DrivingCars/Self_Driving_Car/export"
train_dataset = custom_dataset(train_root_dir, transform=transform)

#test_root_dir = "/raid/cs21resch15003/Afeef_Intern/VOCtest/VOCdevkit/VOC2007"
#val_dataset = Dataset(test_root_dir, transform=transform)





#val_loader = DataLoader(dataset_val,batch_size=64,collate_fn=collate_fn,shuffle=True,num_workers=2,pin_memory=True if torch.cuda.is_available() else False)
train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn, shuffle=True, num_workers=2, pin_memory=True if torch.cuda.is_available() else False)




checkpoint_path = "/raid/cs21resch15003/Afeef_Intern/CustomModel_Self_DrivingCars/checkpoint"

# Save the checkpoint
def save_checkpoint(epoch, loss,epoch_losses,model_state_dict, optimizer_state_dict):
    checkpoint = {
        "epoch": epoch,
        "epoch_losses": epoch_losses,
        "loss": loss,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
    }
    torch.save(checkpoint, checkpoint_path)
    print("Checkpoint saved successfully.")
    


# Load the checkpoint
def load_checkpoint():
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        print("Checkpoint loaded successfully.")
        return checkpoint
    except FileNotFoundError:
        print("No checkpoint found.")
        return None
