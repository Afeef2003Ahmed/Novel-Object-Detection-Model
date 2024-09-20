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

# Distributed Training
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from Dataset import prepare

def setup(rank,world_size):
    '''
    Args: 
        rank: Unique Identifier of each process
        world_size: Total Number of Processes
    '''
    
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl",rank=rank,world_size=world_size)



'''local_rank = int(os.environ['LOCAL_RANK'])
print(f'Local Rank: {local_rank}')
global_rank = int(os.environ['RANK'])
print(f"Global Rank: {global_rank}")

assert local_rank != -1
assert global_rank !=-1

init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)'''
def main(rank,world_size):
    setup(rank,world_size)
    dataloader = prepare(rank,world_size)
    
    #device = torch.device('cuda')

    model = Model(num_classes=10,backbone=backbone(),channels_list = [256,512,1024])
    '''for param in model.backbone.parameters():
        param.requires_grad = False
    print("Backbone Weights Freezed")'''

    model = model.to(rank)
    model = DDP(model,device_ids=[rank],output_device=rank,find_unused_parameters=True)
    
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)

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

    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')


    for epoch in range(num_epochs):
        loaded_checkpoint = load_checkpoint()
        if loaded_checkpoint is not None:
            Epoch = loaded_checkpoint["epoch"]
            model.module.load_state_dict(loaded_checkpoint["model_state_dict"])
            optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
            train_loss_values = loaded_checkpoint["epoch_losses"]
        else:
            train_loss_values = []
            
        dataloader.sampler.set_epoch(epoch)

        # Training
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            inputs = batch[0].to(rank)
            #labels = [{k: v.to(device) for k, v in t.items()} for t in batch[1]]
            labels = batch[1].to(rank)
            #print (f"The Labels are {labels}")

            optimizer.zero_grad()

            with autocast():
                #outputs = model.forward(inputs)
                outputs = model.module.forward(inputs)
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
        if rank==0:
            
            save_checkpoint(epoch, epoch_loss,train_loss_values,model.module.state_dict(), optimizer.state_dict())
        
    destroy_process_group()
    
    if __name__ == '__main__':
        world_size = 2
        mp.spawn(main,
                 args=(world_size),
                 nprocs = world_size)
        
   
    
    
    
