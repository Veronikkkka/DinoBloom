import torch
import argparse
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pathlib import Path
from dinov2.models import build_model  # Assuming this is the correct import path
from dinov2.data.datasets.my_dataset import ADK20Dataset  # Adjust if needed

# Argument Parser
parser = argparse.ArgumentParser(description="Evaluate DINOv2 Model")
parser.add_argument("--model-path", type=str, default="DinoBloom-S.pth", help="Path to trained model checkpoint")
parser.add_argument("--data-path", type=str, default="/home/paperspace/Documents/nika_space/tiny-imagenet-200/test/images", help="Path to evaluation dataset")
parser.add_argument("--batch-size", type=int, default=16, help="Batch size for evaluation")
parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
parser.add_argument("--arch", type=str, default="vit_small", help="Model architecture (e.g., vit_base_patch16)")
parser.add_argument("--patch-size", type=int, default=16, help="Patch size")
parser.add_argument("--layerscale", type=float, default=1e-6, help="Layer scale")
parser.add_argument("--ffn-layer", type=str, default="mlp", help="FFN layer type")
parser.add_argument("--block-chunks", type=int, default=1, help="Block chunks")
parser.add_argument("--qkv-bias", action="store_true", help="Use QKV bias")
parser.add_argument("--proj-bias", action="store_true", help="Use projection bias")
parser.add_argument("--ffn-bias", action="store_true", help="Use FFN bias")
parser.add_argument("--num-register-tokens", type=int, default=0, help="Number of register tokens")
parser.add_argument("--interpolate-offset", type=float, default=0.0, help="Interpolate offset")
parser.add_argument("--interpolate-antialias", action="store_true", help="Use anti-aliasing interpolation")
parser.add_argument("--drop-path-rate", type=float, default=0.0, help="Drop path rate")
parser.add_argument("--drop-path-uniform", action="store_true", help="Use uniform drop path")
args = parser.parse_args()

# Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load Dataset
dataset = ADK20Dataset(root=args.data_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

# Build Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
student, teacher, embed_dim = build_model(args, only_teacher=False, img_size=224)
checkpoint = torch.load(args.model_path, map_location=device)
print(checkpoint.keys())
student.load_state_dict(checkpoint["teacher"], strict=False)
student.to(device)
student.eval()

# Evaluation Loop
correct, total = 0, 0
with torch.no_grad():
    for images, labels, _ in dataloader:
        # print(images)
        images, labels = images.to(device), labels.to(device)
        outputs = student(images)
        _, preds = torch.max(outputs, 1)
        print(preds)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

# Compute Accuracy
accuracy = 100 * correct / total
print(f"Model Accuracy: {accuracy:.2f}%")
