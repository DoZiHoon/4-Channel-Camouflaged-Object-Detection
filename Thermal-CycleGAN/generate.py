import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from models import networks
import torchvision.transforms as transforms

def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from state_dict keys."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_key = k[7:]
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict

def add_module_prefix(state_dict):
    """Add 'module.' prefix to state_dict keys."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if not k.startswith('module.'):
            new_key = 'module.' + k
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict

def main(opt):
    # Setup device
    if opt.gpu_ids:
        device = torch.device(f"cuda:{opt.gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # Create output directory if not exists
    os.makedirs(opt.output_dir, exist_ok=True)

    # Initialize Generator network (RGB+Segmentation -> IR)
    netG = networks.define_G(
        input_nc=4,  # 3 channels RGB + 1 channel segmentation = 4 channels input
        output_nc=1, # 1 channel thermal output
        ngf=64,
        netG='resnet_9blocks',
        norm='instance',
        use_dropout=False,
        init_type='normal',
        init_gain=0.02,
        gpu_ids=opt.gpu_ids if opt.gpu_ids else []
    )

    # If multiple GPUs specified, wrap model with DataParallel
    if device.type == 'cuda' and len(opt.gpu_ids) > 1:
        netG = torch.nn.DataParallel(netG, device_ids=opt.gpu_ids)

    netG.to(device)
    netG.eval()

    # Load checkpoint
    checkpoint = torch.load(opt.model_path, map_location=device)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Fix state_dict keys to match model
    model_keys = list(netG.state_dict().keys())
    ckpt_keys = list(state_dict.keys())

    # Check if model keys start with 'module.' (DataParallel)
    model_has_module = model_keys[0].startswith('module.')
    ckpt_has_module = ckpt_keys[0].startswith('module.')

    if model_has_module and not ckpt_has_module:
        # Model expects 'module.' prefix but checkpoint keys don't have it
        state_dict = add_module_prefix(state_dict)
    elif not model_has_module and ckpt_has_module:
        # Model does NOT expect 'module.' prefix but checkpoint keys have it
        state_dict = remove_module_prefix(state_dict)
    # else keys match, no change needed

    # Load weights
    netG.load_state_dict(state_dict)
    print(f"Model loaded from: {opt.model_path}")

    # Define preprocessing (resize, tensor conversion, normalization)
    preprocess_rgb = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    preprocess_seg = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load and sort input image file paths
    rgb_paths = sorted([
        os.path.join(opt.rgb_dir, f)
        for f in os.listdir(opt.rgb_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    seg_paths = sorted([
        os.path.join(opt.seg_dir, f)
        for f in os.listdir(opt.seg_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    assert len(rgb_paths) == len(seg_paths), "Number of RGB and segmentation images must match."

    for rgb_path, seg_path in tqdm(zip(rgb_paths, seg_paths), total=len(rgb_paths), desc="Generating IR images"):
        # Load RGB and segmentation images
        rgb = Image.open(rgb_path).convert('RGB')
        seg = Image.open(seg_path).convert('L')

        # Apply preprocessing transforms
        rgb_tensor = preprocess_rgb(rgb)
        seg_tensor = preprocess_seg(seg)

        # Concatenate into 4-channel tensor (RGB + Segmentation)
        input_tensor = torch.cat([rgb_tensor, seg_tensor], dim=0).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = netG(input_tensor)

        # Convert output tensor to numpy image scaled [0, 255]
        output_image = output_tensor.squeeze().cpu().numpy()
        output_image = ((output_image + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

        # Save output image
        filename = os.path.basename(rgb_path)
        save_path = os.path.join(opt.output_dir, filename)
        Image.fromarray(output_image).save(save_path)

    print(f"All images saved to: {opt.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb_dir', type=str, required=True, help='Path to RGB images')
    parser.add_argument('--seg_dir', type=str, required=True, help='Path to segmentation masks')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained generator model')
    parser.add_argument('--output_dir', type=str, default='./generated_IR', help='Directory to save generated IR images')
    parser.add_argument('--gpu_ids', nargs='*', type=int, default=[], help='GPU ids to use, e.g. --gpu_ids 0 1')
    opt = parser.parse_args()

    main(opt)
