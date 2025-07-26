import argparse
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from src.utils.config import Config
from src.models.base_classifier import BaseClassifier
from src.utils.xai import generate_gradcam_visualization

def visualize(cfg, model_path, image_path, output_dir):
    """
    Generate GradCAM++ visualization for a given image using a trained model.

    Args:
        cfg: Configuration object with hyperparameters and paths.
        model_path: Path to the saved model weights.
        image_path: Path to the input image.
        output_dir: Directory to save the visualization.
        
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    device = cfg.DEVICE
    model = BaseClassifier(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(cfg.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Generate visualization
    output_path = os.path.join(output_dir, f'gradcam_{os.path.basename(image_path)}')
    generate_gradcam_visualization(model, input_tensor, image, cfg, output_path)
    print(f"GradCAM++ visualization saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate GradCAM++ visualization for a CT scan image.")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved model weights (e.g., models/saved/best_model.pth)')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the input image (e.g., data/raw/sample_image.png)')
    parser.add_argument('--output_dir', type=str, default='xai_visualizations',
                        help='Directory to save the visualization')
    args = parser.parse_args()

    # Initialize configuration
    cfg = Config()

    # Run visualization
    visualize(cfg, args.model_path, args.image_path, args.output_dir)

if __name__ == "__main__":
    main()
