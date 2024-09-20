import torch
import numpy as np
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
from PIL import Image, ImageDraw




import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
# "b11496d7-cc5e8ff8.jpg", "fdf465ad-3b5d188e.jpg", "b06e051d-8d849b20.jpg"
images = ["0a0c3694-54476ae3.jpg","00a0f008-3c67908e.jpg","0a3e70d1-a515ffaf.jpg","0a4d38c3-aba5cdcd.jpg"]

json_file = "/raid/cs21resch15003/Afeef_Intern/CustomModel_BDD100K/BDD100K_Dataset/bdd100k_labels_train(xywh).json"
with open(json_file, 'r') as f:
    data = json.load(f)


def draw_bounding_boxes(image_path, annotation_data):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    #dpi = 100
    fig, ax = plt.subplots(1,figsize=(12,9))
    #ax.set_position([0, 0, 1, 1])
    ax.imshow(image, extent=[0, width, height, 0])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    classes = ['pedestrian','rider','car','truck','bus','train','motorcycle','bicycle','traffic light','traffic sign']
    for label in annotation_data['labels']:
        if 'box2d' in label and label['category'] in classes:
            box = label['box2d']
            x1 = box['x'] * width
            y1 = box['y'] * height
            w = box['w'] * width
            h = box['h'] * height
            rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            category = label['category']
            ax.text(x1, y1 - 10, category, color='red', fontsize=12, backgroundcolor='white')
    
    ax.axis('off')
    return fig

output_dir = "/raid/cs21resch15003/Afeef_Intern/CustomModel_BDD100K/inference_2"
os.makedirs(output_dir, exist_ok=True)

for image_name in images:
    annotation_data = next((item for item in data if item["name"] == image_name), None)
    if annotation_data:
        image_path = os.path.join("/raid/cs21resch15003/Afeef_Intern/CustomModel_BDD100K/BDD100K_Dataset/bdd100k/bdd100k/images/100k/train", image_name)
        plt_figure = draw_bounding_boxes(image_path, annotation_data)

        # Save the annotated image
        save_path = os.path.join(output_dir, image_name)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()  # Close the current figure to free up memory
        print(f"Saved annotated image: {save_path}")

        # Optionally display the image
        #plt.figure(figsize=(12, 8))
        #plt.imshow(plt.imread(save_path))
        #plt.title(image_name)
        #plt.axis('off')
        #plt.show()
    else:
        print(f"No annotations found for {image_name}")
