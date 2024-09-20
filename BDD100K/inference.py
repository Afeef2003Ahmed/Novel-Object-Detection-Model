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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_test = Model(num_classes=10,backbone=backbone(),channels_list = [256,512,1024]).to(device)
model_test.eval()
loaded_checkpoint = load_checkpoint()
if loaded_checkpoint is not None:
    model_test.load_state_dict(loaded_checkpoint["model_state_dict"])

images = ["f4566c1f-c00a52fd.jpg","cd2dbf3d-78c22773.jpg","ce53c349-27a39a23.jpg","ce12452d-d6ea0b64.jpg","d1b4734f-7ea0afda.jpg","ce3d5a1d-0faa6716.jpg","dc1bb31e-3cb2e900.jpg","ce480c21-65967e6b.jpg","cea99cfc-68704523.jpg","d98457bb-6e42ca93.jpg","d473563b-d9204d14.jpg"]
for image in images:
  
  save_path = os.path.join("/raid/cs21resch15003/Afeef_Intern/CustomModel_BDD100K/inference",image)
  image_path = os.path.join("/raid/cs21resch15003/Afeef_Intern/CustomModel_BDD100K/BDD100K_Dataset/bdd100k/bdd100k/images/100k/test",image)
  #label_path = os.path.join("/content/VOCDataset/VOCdevkit/VOC2007/Annotations",image.replace('.jpg','.txt'))
  #label = pd.read_csv(label_path)
  print(f'The Target Labels for {image} is: /n')
  #print(label)
  #image_path = "/content/pedestrians.jpg"
# Load the image
  image = Image.open(image_path).convert("RGB")
  new_dimensions = (416, 416)  # specify desired width and height
  resized_image = image.resize(new_dimensions, Image.LANCZOS)
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
  class_names = ['pedestrian','rider','car','truck','bus','train','motorcycle','bicycle','traffic light','traffic sign']
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
    plt.text(x1, y1, s=f'{class_names[class_idx-1]} {conf:.2f}', color='black',
            verticalalignment='top', bbox={'color': 'white', 'pad': 0})
    
  plt.axis('off')  # Turn off axis
  plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
  plt.close()
