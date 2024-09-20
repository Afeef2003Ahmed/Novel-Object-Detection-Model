import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# List of images to process
images = ["0a0c3694-54476ae3.jpg", "00a0f008-3c67908e.jpg", "0a3e70d1-a515ffaf.jpg", "0a4d38c3-aba5cdcd.jpg"]

# Path to the JSON file containing annotations
json_file = "/raid/cs21resch15003/Afeef_Intern/CustomModel_BDD100K/BDD100K_Dataset/bdd100k_labels_train(xywh).json"
with open(json_file, 'r') as f:
    data = json.load(f)

def draw_bounding_boxes(image_path, annotation_data):
    # Open the image
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    
    # Create a plot to display the image
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    
    # Classes to include in the visualization
    classes = ['pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign']
    
    # Draw the bounding boxes
    for label in annotation_data['labels']:
        if 'box2d' in label and label['category'] in classes:
            box = label['box2d']
            x1 = box['x']
            y1 = box['y']
            w = box['w']
            h = box['h']
            
            # Scale the bounding box coordinates to the image size
            x1_scaled = x1 * width
            y1_scaled = y1 * height
            w_scaled = w * width
            h_scaled = h * height
            
            # Create a rectangle patch for the bounding box
            rect = patches.Rectangle((x1_scaled, y1_scaled), w_scaled, h_scaled, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            # Add the category label above the bounding box
            category = label['category']
            ax.text(x1_scaled, y1_scaled - 10, category, color='red', fontsize=12, backgroundcolor='white')
    
    ax.axis('off')  # Hide the axes
    return fig

# Output directory for the annotated images
output_dir = "/raid/cs21resch15003/Afeef_Intern/CustomModel_BDD100K/inference_2"
os.makedirs(output_dir, exist_ok=True)

# Process each image
for image_name in images:
    annotation_data = next((item for item in data if item["name"] == image_name), None)
    if annotation_data:
        image_path = os.path.join("/raid/cs21resch15003/Afeef_Intern/CustomModel_BDD100K/BDD100K_Dataset/bdd100k/bdd100k/images/100k/train", image_name)
        plt_figure = draw_bounding_boxes(image_path, annotation_data)

        # Save the annotated image
        save_path = os.path.join(output_dir, image_name)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()  # Close the figure to free up memory
        print(f"Saved annotated image: {save_path}")
    else:
        print(f"No annotations found for {image_name}")
