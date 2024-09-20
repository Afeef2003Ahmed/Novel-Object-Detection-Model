import torch
import os
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from Model import Model,backbone
from Dataset import load_checkpoint
from Utilities import non_max_suppression
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_test = Model(num_classes=11,backbone=backbone(),channels_list = [256,512,1024]).to(device)
model_test.eval()
loaded_checkpoint = load_checkpoint()
if loaded_checkpoint is not None:
    model_test.load_state_dict(loaded_checkpoint["model_state_dict"])
    
images = ["1478020423696984256_jpg.rf.37aaedf0811997aecc704936adba8650.jpg","1478020391703290567_jpg.rf.n4p5a85D1rlZLRWMO1TZ.jpg","1478020325197357739_jpg.rf.hEKMGh4al8OZhgV5gZ5M.jpg","1478020285197265564_jpg.rf.3e2c41f8eaa6331d686aab33fcbacb19.jpg","1478020240698496203_jpg.rf.7c4e1d6a29ba4fdc6643a9f40f408df9.jpg","1478020214191388441_jpg.rf.ZnLVhyDYzLqDXd1U2j80.jpg","1478019977680527341_jpg.rf.6toJe4DIevJ9USzTojlk.jpg","1478019976185898081_jpg.rf.3c442aec304a7ecdd8f3d84823f5e792.jpg","1478019957687018435_jpg.rf.f4d1d5ec89c63e083d0930dd819271d4.jpg","1478020223200419014_jpg.rf.PXIapHVCa4yFaXCv01vV.jpg","1478020237697303809_jpg.rf.EPPlgdEXaTfMZ0EDXM2Z.jpg","1478020256192066071_jpg.rf.96e11e77e038d32881587a54b162c8fd.jpg"]
for image in images:
  
  save_path = os.path.join("/raid/cs21resch15003/Afeef_Intern/CustomModel_Self_DrivingCars/inference",image)
  image_path = os.path.join("/raid/cs21resch15003/Afeef_Intern/CustomModel_Self_DrivingCars/Self_Driving_Car/export",image)
  #label_path = os.path.join("/content/VOCDataset/VOCdevkit/VOC2007/Annotations",image.replace('.jpg','.txt'))
  #json_file = "/raid/cs21resch15003/Afeef_Intern/CustomModel_BDD100K/BDD100K_Dataset/bdd100k_labels_train(xywh).json"
  #label = pd.read_csv(label_path)
 
  print(f'The Target Labels for {image} is: /n')
  #print(label)
  #image_path = "/content/pedestrians.jpg"
# Load the image
  image = Image.open(image_path).convert("RGB")
  new_dimensions = (416, 416)  # specify desired width and height
  resized_image = image.resize(new_dimensions, Image.ANTIALIAS)

  transform = transforms.Compose([
      transforms.Resize((416,416)),

      transforms.ToTensor(),
      transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
  ])


  tensor_image = transform(image)
  tensor_image = tensor_image.unsqueeze(0).to(device)


  outputs = model_test.forward(tensor_image)

  conf_thres = 0.5
  iou_thres = 0.5
  #outputs = torch.tensor(outputs)
  #print(outputs)
  predictions = non_max_suppression(outputs, conf_thres, iou_thres, multi_label=False)
  class_names = ["car","pedestrian","trafficLight-Red","trafficLight-Green","truck","trafficLight","biker","trafficLight-RedLeft","trafficLight-GreenLeft","trafficLight-Yellow","trafficLight-YellowLeft"]
  fig, ax = plt.subplots(1, figsize=(12,9))
  ax.imshow(resized_image)


  pred_tensor = predictions[0]


  for i in range(len(pred_tensor)):
    pred = pred_tensor[i]


    x1, y1, x2, y2, conf, class_idx = pred.detach().numpy()
    class_idx = int(class_idx.item())
    box_width = x2 - x1
    box_height = y2 - y1


    bbox = patches.Rectangle((x1, y1), box_width, box_height, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(bbox)
    plt.text(x1, y1, s=f'{class_names[class_idx]} {conf:.2f}', color='black',
            verticalalignment='top', bbox={'color': 'white', 'pad': 0})
    
  
  plt.axis('off')  # Turn off axis
  plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
  plt.close()
  
