import torch
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from tqdm import tqdm
from Model import Model,backbone
from Utilities import ComputeLoss
from Dataset import load_checkpoint,train_loader,val_loader,save_checkpoint




device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
model = Model(num_classes=10,backbone=backbone(),channels_list = [256,512,1024])
'''for param in model.backbone.parameters():
    param.requires_grad = False
print("Backbone Weights Freezed")'''

model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
#optimizer = optim.Adam(model.parameters(),lr=0.001)
scaler = GradScaler()
loss_fn = ComputeLoss()

best_acc = 0.0
best_epoch = 0
best_val_loss = float('inf')
patience = 5
val_loss_values=[]

epoch_num = step_num = 0
num_epochs = 50
batch_height = 416
batch_width = 416

#best_model = torch.load("/raid/cs21resch15003/Afeef_Intern/CustomModel_BDD100K/best_model.pt")
loaded_checkpoint = load_checkpoint()
if loaded_checkpoint is not None:
    Epoch = loaded_checkpoint["epoch"]
    model.load_state_dict(loaded_checkpoint["model_state_dict"])
    #model.load_state_dict(best_model)
    optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
    train_loss_values = loaded_checkpoint["epoch_losses"]
else:
    train_loss_values = []


for epoch in range(num_epochs):

    # Training
    model.train()
    running_loss = 0.0
    start_time = time.time()

    for batch_idx, batch in enumerate(tqdm(train_loader)):
        inputs = batch[0].to(device)
        #labels = [{k: v.to(device) for k, v in t.items()} for t in batch[1]]
        labels = batch[1].to(device)
        #print (f"The Labels are {labels}")

        optimizer.zero_grad()

        with autocast():
            #outputs = model.forward(inputs)
            outputs = model.forward(inputs)
            #print(f"Output of inference is {outputs}")
            loss, loss_components = loss_fn(
            outputs, labels, epoch_num, step_num, batch_height, batch_width
            )
        print(f"Batch{batch_idx+1} Loss: {loss.item()}")

        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_loss_values.append(epoch_loss)
    
    print('Epoch {}/{} Training Loss: {:.4f} Time: {:.4f}'.format(
        epoch, num_epochs - 1, epoch_loss, time.time() - start_time))
    
    save_checkpoint(epoch, epoch_loss,train_loss_values,model.state_dict(), optimizer.state_dict())
    

    
    # Validation
    
    if (epoch+1) % 2 == 0:
        print("VALIDATION")
        running_val_loss = 0.0
       
        for batch_idx, batch in enumerate(tqdm(val_loader)):
            inputs = batch[0].to(device)
            #labels = [{k: v.to(device) for k, v in t.items()} for t in batch[1]]
            labels = batch[1].to(device)
            #print (f"The Labels are {labels}")
            with autocast():
                outputs = model.forward(inputs)
                #print(f"Output of inference is {outputs}")
                loss, loss_components = loss_fn(
                outputs, labels, epoch_num, step_num, batch_height, batch_width
                )
            print(f"Batch{batch_idx+1} Loss: {loss.item()}")

            running_val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_loss_values.append(epoch_val_loss)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_without_improvement = 0
            # Save the best model (optional)
            torch.save(model.state_dict(), '/raid/cs21resch15003/Afeef_Intern/CustomModel_BDD100K/best_model.pt')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f'Early stopping after {epoch + 1} epochs')
                break
    
        
            

    
