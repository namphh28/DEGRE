from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import numpy as np
from PIL import Image

def generate_gradcam_visualization(model, input_tensor, image, cfg, save_path):
    grad_cam = GradCAMPlusPlus(model=model, target_layers=[model.target_layer])
    targets = [ClassifierOutputTarget(1)]  # Assuming binary classification, target COVID class
    grayscale_cam = grad_cam(input_tensor.unsqueeze(0))[0, :]
    rgb_img = np.array(image.convert('RGB')) / 255.0
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    Image.fromarray((visualization * 255).astype(np.uint8)).save(save_path)