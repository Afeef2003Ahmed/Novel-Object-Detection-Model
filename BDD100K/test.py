import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
from Model import Model,backbone

from Utilities import ComputeLoss,ConfusionMatrix,non_max_suppression,xywh2xyxy,process_batch,ap_per_class,LOGGER
from Dataset import load_checkpoint,train_loader,val_loader


device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
model_test = Model(num_classes=10,backbone=backbone(),channels_list = [256,512,1024])
model_test.to(device).half()
model_test.eval()
loss_fn = ComputeLoss()
'''loaded_checkpoint = load_checkpoint()
if loaded_checkpoint is not None:
    model_test.load_state_dict(loaded_checkpoint["model_state_dict"])'''
best_model = torch.load("/raid/cs21resch15003/Afeef_Intern/CustomModel_BDD100K/best_model.pt")
model_test.load_state_dict(best_model)
stats, ap = [], []
seen = 0
iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
niou = iouv.numel()

confusion_matrix = ConfusionMatrix(nc=10)

running_loss = 0.0
all_preds = []
all_labels = []
epoch_num = step_num = 0
num_epochs = 4
batch_height = 416
batch_width = 416
with torch.no_grad():
  for inputs, labels in tqdm(val_loader):
    imgs = inputs.to(device).half()
    targets = labels.to(device)
    #print(f"Input shape = {inputs.shape}")
    #print(f"Labels shape = {labels.shape}")
    outputs = model_test.forward(imgs)
    conf_thres = 0.25
    iou_thres = 0.5
    #print('nms')
    outputs = non_max_suppression(outputs, conf_thres, iou_thres, multi_label=False)
    #print('NMS')
    #print(f'Outputs: {outputs.Shape}')

    eval_outputs = [x.detach().cpu() for x in outputs]

    for si, pred in enumerate(eval_outputs):
      labels = targets[targets[:, 0] == si, 1:]
      #print(labels)
      nl = len(labels)
      tcls = labels[:, 0].tolist() if nl else []
      #print(f'Actual Classes in Image is {tcls}')
      seen += 1

      if len(pred) == 0:
        if nl:
          stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
          continue

      predn = pred.clone()
      correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
      if nl:
        tbox = xywh2xyxy(labels[:, 1:5])
        tbox[:, [0, 2]] *= imgs[si].shape[1]
        tbox[:, [1, 3]] *= imgs[si].shape[2]
        labelsn = torch.cat((labels[:, 0:1], tbox), 1)

        correct = process_batch(predn, labelsn, iouv)
        confusion_matrix.process_batch(predn, labelsn)

      stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

save_dir = '/raid/cs21resch15003/Afeef_Intern/CustomModel_BDD100K/Results'
stats = [np.concatenate(x, 0) for x in zip(*stats)]
names=['pedestrian','rider','car','truck','bus','train','motorcycle','bicycle','traffic light','traffic sign']
plot_curve = True
if len(stats) and stats[0].any():
    p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plot_curve, save_dir=save_dir, names=('pedestrian','rider','car','truck','bus','train','motorcycle','bicycle','traffic light','traffic sign'))
    #for ci, c in enumerate(ap_class):
      #print(f'Class {c} - Precision: {p[ci]}, Recall: {r[ci]}, F1: {f1[ci]}, AP: {ap[ci]}')
    #print(f'p = {p} r = {r} ap = {ap} f1 = {f1} ap_class = {ap_class}')

    AP50_F1_max_idx = len(f1.mean(0)) - f1.mean(0)[::-1].argmax() - 1
    LOGGER.info(f"IOU 50 best mF1 threshold near {AP50_F1_max_idx/1000.0}.")

    ap50, ap = ap[:, 0], ap.mean(1)
    mp, mr, map50, map = p[:, AP50_F1_max_idx].mean(), r[:, AP50_F1_max_idx].mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(np.int64), minlength=6)

    # Print results
    s = ('%-16s' + '%12s' * 7) % ('Class', 'Images', 'Labels', 'P@.5iou', 'R@.5iou', 'F1@.5iou', 'mAP@.5', 'mAP@.5:.95')
    LOGGER.info(s)
    print(s)
    pf = '%-16s' + '%12i' * 2 + '%12.3g' * 5  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, f1.mean(0)[AP50_F1_max_idx], map50, map))
    print(pf % ('all', seen, nt.sum(), mp, mr, f1.mean(0)[AP50_F1_max_idx], map50, map))

# Assuming you have metrics for each class in lists: class_images, class_labels, etc.
#for class_name, img_count, label_count, p, r, f, map_05, map_095 in zip(class_names, class_images, class_labels, class_p, class_r, class_f1, class_map50, class_map95):
#    LOGGER.info(pf % (class_name, img_count, label_count, p, r, f, map_05, map_095))
#    print(pf % (class_name, img_count, label_count, p, r, f, map_05, map_095))
#    pr_metric_result = (map50, map)
    confusion_matrix.plot(save_dir=save_dir, names=names)
else:
    LOGGER.info("Calculate metric failed, might check dataset.")
    pr_metric_result = (0.0, 0.0)
    print("failed")
print(f"validation Loss = {running_loss / len(train_loader.dataset)}")
