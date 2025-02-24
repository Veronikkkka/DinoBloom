import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution with no bias."""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_channels, out_channels, stride=1, padding=1):
    """3x3 convolution with no bias."""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False)

class Merge_block(BaseModule):
    def __init__(self, fea_c, ada_c, mid_c, return_ada=True):
        """
        Args:
            fea_c (int): Number of channels in the feature map (e.g. patch tokens).
            ada_c (int): Number of channels in the adapter feature map.
            mid_c (int): Intermediate channel number for fusion.
            return_ada (bool): Whether to output an updated adapter for cascading.
        """
        super(Merge_block, self).__init__()
        self.conv_1 = conv1x1(fea_c + ada_c, mid_c, 1)
        self.conv_2 = conv1x1(mid_c, fea_c, 1)
        self.return_ada = return_ada
        if self.return_ada:
            self.conv_3 = conv3x3(mid_c, ada_c * 2, stride=2)
        
    def forward(self, fea, adapter, ratio=1.0):
        """
        Args:
            fea (Tensor): Feature map from the backbone. Shape (B, fea_c, H, W).
            adapter (Tensor): Adapter feature map. Shape (B, ada_c, H, W).
            ratio (float): Merge ratio.
        Returns:
            If return_ada is True:
                (fea_out, new_adapter)
            Otherwise:
                (fea_out, None)
        """
        res = fea
        # Concatenate along the channel dimension.
        fusion = torch.cat([fea, adapter], dim=1)
        fusion = self.conv_1(fusion)
        ada = self.conv_2(fusion)
        fea_out = ratio * ada + res
        if self.return_ada:
            new_adapter = self.conv_3(fusion)
            return fea_out, new_adapter
        else:
            return fea_out, None

class Model_level_Adapeter(nn.Module):
    def __init__(self, in_c, in_dim, w_lut=True):
        """
        A basic adapter that takes an input feature map (e.g. from a RAW pre-encoder)
        and outputs an adaptive feature map with a desired number of channels.
        
        Args:
            in_c (int): Number of input channels (e.g., typically 3 for RGB).
            in_dim (int): Desired number of output channels for the adapter features.
            w_lut (bool): Whether the adapter uses LUT-related processing (placeholder here).
        """
        super(Model_level_Adapeter, self).__init__()
        # A simple three-layer convolutional block.
        self.conv1 = nn.Conv2d(in_c, in_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_dim)
        
        # You can add more layers or a LUT branch if needed.
        self.w_lut = w_lut

    def forward(self, x):
        """
        Args:
            x (Tensor): Input feature map from the pre-encoder. Shape (B, in_c, H, W).
        Returns:
            Tensor: Adapted feature map with shape (B, in_dim, H, W).
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


if __name__ == '__main__':
    # Assume feature map from patch embed is of shape (B, embed_dim, H, W)
    B, embed_dim, H, W = 2, 768, 14, 14
    fea = torch.randn(B, embed_dim, H, W)
    
    # Assume adapter feature map from Model_level_Adapeter is of shape (B, ada_dim, H, W)
    ada_dim = 32
    adapter = torch.randn(B, ada_dim, H, W)
    
    # Create a merge block: merging fea (channels=embed_dim) with adapter (channels=ada_dim)
    mid_c = embed_dim  # for example, you can set the intermediate channel to embed_dim
    merge_block = Merge_block(fea_c=embed_dim, ada_c=ada_dim, mid_c=mid_c, return_ada=True)
    
    fea_out, new_adapter = merge_block(fea, adapter, ratio=1.0)
    print("Output feature map shape:", fea_out.shape)
    if new_adapter is not None:
        print("New adapter shape:", new_adapter.shape)
    
    # Example Model_level_Adapeter usage:
    model_adapter = Model_level_Adapeter(in_c=3, in_dim=ada_dim, w_lut=True)
    raw_input = torch.randn(B, 3, H, W)
    adapted_feat = model_adapter(raw_input)
    print("Adapted feature shape:", adapted_feat.shape)


import torch
import torch.nn as nn
import torch.nn.functional as F

class Input_level_Adapeter(nn.Module):
    def __init__(self, mode='normal', lut_dim=32, k_size=3, w_lut=True, in_channels=3):
        """
        Args:
            mode (str): Operating mode. Can be 'normal' or another mode if you extend this module.
            lut_dim (int): The output channel dimension if using the LUT branch.
            k_size (int): Kernel size for the convolutional layers.
            w_lut (bool): Whether to use the LUT branch.
            in_channels (int): Number of input channels. Typically 3 for RGB/RAW images.
        """
        super(Input_level_Adapeter, self).__init__()
        self.mode = mode
        self.lut_dim = lut_dim
        self.k_size = k_size
        self.w_lut = w_lut

        # First convolutional block.
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        # Second convolutional block.
        self.conv2 = nn.Conv2d(16, 32, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        
        # If using LUT processing, map the 32-channel features to lut_dim channels.
        if self.w_lut:
            self.lut_conv = nn.Conv2d(32, lut_dim, kernel_size=1, bias=False)
        
        # Create two downsampling layers for multi-scale outputs.
        self.down1 = nn.Conv2d(32 if not w_lut else lut_dim, 
                               32 if not w_lut else lut_dim, 
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.down2 = nn.Conv2d(32 if not w_lut else lut_dim, 
                               32 if not w_lut else lut_dim, 
                               kernel_size=3, stride=2, padding=1, bias=False)
        
    def forward(self, x):
        """
        Forward pass for the input-level adapter.
        Args:
            x (Tensor): Input image tensor of shape (B, in_channels, H, W).
        Returns:
            List[Tensor]: A list of feature maps at multiple scales. For example:
                          [feat_full, feat_down1, feat_down2]
                          where feat_down2 is the most downsampled feature used for adaptation.
        """
        # Initial conv block.
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        # If enabled, adjust features via LUT branch.
        if self.w_lut:
            out = self.lut_conv(out)
        
        # Compute multi-scale features.
        feat_full = out                   # Original resolution feature.
        feat_down1 = self.relu(self.down1(feat_full))  # Downsampled by a factor of 2.
        feat_down2 = self.relu(self.down2(feat_down1))   # Downsampled further.
        
        # Return a list of features. In your transformer, you can pick the desired scale.
        return [feat_full, feat_down1, feat_down2]

