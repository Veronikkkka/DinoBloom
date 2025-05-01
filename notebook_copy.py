import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
# PCA for feature inferred
from sklearn.decomposition import PCA
from skimage.filters import threshold_otsu
import os
import random
from glob import glob


embed_sizes={"dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitg14": 1536}

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

patch_num_h=16
patch_num_w=16
img_size=224
eval_model="dinov2_vits14"


def get_dino_bloom(modelpath="DinoBloom-S.pth",modelname="dinov2_vits14"):
    # load the original DINOv2 model with the correct architecture and parameters.
    model=torch.hub.load('facebookresearch/dinov2', modelname)
    # load finetuned weights
    pretrained = torch.load(modelpath, map_location=torch.device('cpu'))
    # make correct state dict for loading
    new_state_dict = {}
    for key, value in pretrained['teacher'].items():
        if 'dino_head' in key or "ibot_head" in key:
            pass
        else:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = value

    #corresponds to 224x224 image. patch size=14x14 => 16*16 patches
    pos_embed = nn.Parameter(torch.zeros(1, 257, embed_sizes["dinov2_vits14"]))
    model.pos_embed = pos_embed

    model.load_state_dict(new_state_dict, strict=True)
    return model

model=get_dino_bloom()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

paths=np.random.choice(list(Path("segmentation_WBC/Dataset_2").glob("*.bmp")),4)
images_for_plotting = [Image.open(img_path).convert('RGB').resize((img_size, img_size)) for img_path in paths]
if torch.cuda.is_available():
  model.cuda()
  imgs_tensor = torch.stack([transform(Image.open(img_path).convert('RGB').resize((img_size,img_size))).cuda() for img_path in paths])
else:
  imgs_tensor = torch.stack([transform(Image.open(img_path).convert('RGB').resize((img_size,img_size)))for img_path in paths])

with torch.no_grad():
    # Ensure the input tensor is on GPU by calling .cuda() on it
    features_dict = model.forward_features(imgs_tensor)
    features = features_dict['x_norm_patchtokens']

features = features.reshape(len(paths) * patch_num_h*patch_num_w, embed_sizes[eval_model]).cpu().numpy()
pca = PCA(n_components=3)
pca.fit(features)
pca_features = pca.transform(features)

for i in range(3):
    pca_features[:, i] = (pca_features[:, i] - pca_features[:, i].min()) / (pca_features[:, i].max() - pca_features[:, i].min())

pca_features_rgb = pca_features.copy()
pca_features_rgb = pca_features_rgb.reshape(len(paths), patch_num_h, patch_num_w, 3)

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
for i, ax in enumerate(axs.flat):
    #print(Path(images_for_plotting[i]).stem)
    ax.imshow(pca_features_rgb[i][..., ::-1])
    ax.axis('off')  # Remove axis
plt.savefig('features.png')
plt.show()
# plt.close()

