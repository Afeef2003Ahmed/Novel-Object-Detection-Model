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
from torchvision.datasets import VOCDetection
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
import torch.cuda.amp as amp
import math
import xml.etree.ElementTree as ET
from PIL import Image
import warnings
import json
from torch.utils.data.distributed import DistributedSampler

class custom_dataset(Dataset):
  def __init__(self,json_file,img_dir,transform=None):

    #images = os.listdir(self.img_dir)
    with open(json_file,'r') as f:
      self.data = json.load(f)
      
    self.img_dir = img_dir
    self.transform = transform

    self.class_to_idx = {
            "pedestrian": 1,
            "rider": 2,
            "car": 3,
            "truck": 4,
            "bus": 5,
            "train": 6,
            "motorcycle": 7,
            "bicycle": 8,
            "traffic light": 9,
            "traffic sign": 10
        }
  def __len__(self):
    return len(self.data)

  def __getitem__(self,idx):

    #img_name = self.images_path[idx]
    annotation = self.data[idx]
    img_name = annotation['name']
    img_path = os.path.join(self.img_dir,img_name)

    image = Image.open(img_path).convert("RGB")
    '''try:

    except FileNotFoundError:
      print(f"File not found: {img_path}. Skipping.")
      return None '''
    #image = Image.open(img_path).convert("RGB")

    boxes = []
    labels = []

    for label in annotation['labels']:
      category = label['category']

      if category in self.class_to_idx:
        class_idx = self.class_to_idx[category]
        box = label['box2d']
        x, y,w,h = box['x'], box['y'], box['w'], box['h']
        boxes.append([x,y,w,h])
        labels.append(class_idx)


    boxes = torch.tensor(boxes)
    #print(f"Box at dataset class is {boxes}")
    labels = torch.tensor(labels)
    target = torch.zeros((len(labels),6))
    annotation = torch.cat((labels.unsqueeze(1),boxes),dim=1)
    #print(boxes.shape)
    #print(labels.shape)
    target[:,1:] = annotation
    #print(f"Initial dataset target is {target}")
    if self.transform:
      image = self.transform(image)

    return image,target


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


json_file = "/raid/cs21resch15003/Afeef_Intern/CustomModel_BDD100K/BDD100K_Dataset/bdd100k_labels_train(xywh).json"
img_dir = "/raid/cs21resch15003/Afeef_Intern/CustomModel_BDD100K/BDD100K_Dataset/bdd100k/bdd100k/images/100k/train"
dataset = custom_dataset(json_file,img_dir,transform=transform)
json_val = "/raid/cs21resch15003/Afeef_Intern/CustomModel_BDD100K/BDD100K_Dataset/bdd100k_labels_val(xywh).json"
val_img = "/raid/cs21resch15003/Afeef_Intern/CustomModel_BDD100K/BDD100K_Dataset/bdd100k/bdd100k/images/100k/val"
dataset_val = custom_dataset(json_val,val_img,transform=transform_val)




#test_loader = DataLoader(test_dataset,batch_size=32,collate_fn=collate_fn,shuffle=True,num_workers=2,pin_memory=True if torch.cuda.is_available() else False)
val_loader = DataLoader(dataset_val,batch_size=32,collate_fn=collate_fn,shuffle=True,num_workers=2,pin_memory=True if torch.cuda.is_available() else False)
train_loader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn, shuffle=True, num_workers=2, pin_memory=True if torch.cuda.is_available() else False)
#val_loader = DataLoader(dataset_val,batch_size=128,collate_fn=collate_fn,shuffle=True,num_workers=2,pin_memory=True if torch.cuda.is_available() else False)
#train_loader = DataLoader(dataset, batch_size=128, collate_fn=collate_fn, shuffle=False,sampler = DistributedSampler(dataset,shuffle=True),num_workers=2, pin_memory=True if torch.cuda.is_available() else False)

# *** DISTRIBUTED TRAINING ***



def prepare(rank,world_size,batch_size=32,pin_memory=False,num_workers=0):
  dataset = custom_dataset(json_file,img_dir,transform=transform)
  sampler = DistributedSampler(dataset,num_replicas=world_size,rank=rank,shuffle=False,drop_last=False)
  dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=num_workers,pin_memory=pin_memory,sampler=sampler,drop_last=False)
  
  return dataloader
  



checkpoint_path = "/raid/cs21resch15003/Afeef_Intern/CustomModel_BDD100K/checkpoint_2"
checkpoint_path_ddp = "/raid/cs21resch15003/Afeef_Intern/CustomModel_BDD100K/checkpoint_ddp.pth"
#checkpoint_path_load = "/raid/cs21resch15003/Afeef_Intern/CustomModel_BDD100K/best_model.pt"
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
