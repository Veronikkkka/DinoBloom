import numpy as np
import cv2
import argparse
from pathlib import Path
import torch


def rgb_to_raw(image_path="", img=None):
    if img is None:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    else:
        img = np.array(img)
    if img is None:
        raise ValueError("Error loading image. Please check the path.")
   
    if len(img.shape) == 3:
        img_raw = img[:, :, 1]  # Extract green channel as a naive RAW simulation
    else:
        img_raw = img
    

    if img_raw.dtype != np.uint16:
        img_raw = (img_raw.astype(np.float32) / 255.0 * 65535).astype(np.uint16)
    
    return img_raw

# def rgb_to_raw(image_path, local_crops_number=6):
#     """
#     Reads an image from disk, simulates a RAW image by extracting (for example)
#     the green channel, and returns a dictionary formatted like the output
#     of your DataAugmentationDINO pipeline.
#     """
#     # Read the image using OpenCV (unchanged mode)
#     img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
#     if img is None:
#         raise ValueError("Error loading image. Please check the path.")
    
#     # If the image has three channels, simulate RAW by taking the green channel.
#     if len(img.shape) == 3:
#         img_raw = img[:, :, 1]  # Using the green channel as a naive RAW simulation
#     else:
#         img_raw = img
    
#     # Convert to uint16 if needed.
#     if img_raw.dtype != np.uint16:
#         img_raw = (img_raw.astype(np.float32) / 255.0 * 65535).astype(np.uint16)
    
#     # Normalize the raw image to [0, 1] (as float32)
#     img_raw = img_raw.astype(np.float32) / 65535.0
    
#     # Convert the raw image to a torch tensor.
#     # Assuming the raw image is single channel, add a channel dimension.
#     raw_tensor = torch.from_numpy(img_raw).unsqueeze(0)  # Shape: [1, H, W]
    
#     # For consistency, we simulate two global crops (for student and teacher)
#     # and several local crops. Here we simply use the same raw tensor for each crop.
#     output = {
#         "global_crops": [raw_tensor, raw_tensor],  # Two global crops
#         "global_crops_teacher": [raw_tensor, raw_tensor],
#         "local_crops": [raw_tensor for _ in range(local_crops_number)],
#         "offsets": ()  # Keeping offsets empty as before
#     }
#     # print("Type: ", type(rgb_to_raw))
#     return output


from PIL import Image

def raw_to_rgb(raw_input, normalize: bool = True) -> Image.Image:

    if isinstance(raw_input, torch.Tensor):
        raw_np = raw_input.detach().cpu().numpy()
    else:
        raw_np = raw_input

    if raw_np.ndim == 3 and raw_np.shape[0] == 1:
        raw_np = raw_np[0]

    if normalize:
        raw_np = (raw_np - raw_np.min()) / (raw_np.max() - raw_np.min() + 1e-8) * 255.0
        raw_np = raw_np.astype(np.uint8)

    rgb_image = cv2.cvtColor(raw_np, cv2.COLOR_GRAY2RGB)

    return Image.fromarray(rgb_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RGB image to RAW format.")
    parser.add_argument("image_path", type=str, help="Path to the input image")

    args = parser.parse_args()
    
    raw = rgb_to_raw(args.image_path)
    print(raw, type(raw), raw.shape)
    rgb = raw_to_rgb(raw)
    print(rgb)
