import torch
from ultralytics import YOLO
import os

# 0. Define paths
YAML_PATH = 'ultralytics/cfg/models/v8/yolov8x.yaml'  # Must contain `in_channels: 4`
OUTPUT_PATH = 'model/yolov8x_4ch_pretrained.pt'

# 1. Automatically download pretrained yolov8x.pt
print("⬇️ Downloading pretrained YOLOv8x model using Ultralytics...")
yolo_model = YOLO('yolov8x.pt')  # This triggers download and loads the model
PRETRAINED_PATH = yolo_model.ckpt_path if hasattr(yolo_model, "ckpt_path") else 'yolov8x.pt'
print(f"✅ Download complete: {PRETRAINED_PATH}")

# 2. Load pretrained model state_dict
print("🔄 Loading pretrained weights...")
pretrained_sd = yolo_model.model.state_dict()  # ✅ this is a DetectionModel

# 3. Initialize your custom 4-channel model
print("📐 Initializing YOLOv8x model with 4-channel input...")
model_4ch = YOLO(YAML_PATH)

# 4. Replace first Conv2d layer
original_conv = model_4ch.model.model[0].conv
new_conv = torch.nn.Conv2d(
    in_channels=4,
    out_channels=original_conv.out_channels,
    kernel_size=original_conv.kernel_size,
    stride=original_conv.stride,
    padding=original_conv.padding,
    bias=original_conv.bias is not None
)
torch.nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
model_4ch.model.model[0].conv = new_conv
print("✅ First Conv2d layer updated to 4-channel input.")

# 5. Transfer weights (except for conv1)
new_sd = model_4ch.model.state_dict()
num_loaded = 0

for k in new_sd.keys():
    if "model.0.conv.weight" in k:
        continue
    if k in pretrained_sd and pretrained_sd[k].shape == new_sd[k].shape:
        new_sd[k] = pretrained_sd[k]
        num_loaded += 1

model_4ch.model.load_state_dict(new_sd)
print(f"🔁 {num_loaded} layers loaded from pretrained weights (Conv1 skipped).")

# 6. Save final model
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
model_4ch.save(OUTPUT_PATH)
print(f"✅ Final 4-channel YOLOv8x model saved to: {OUTPUT_PATH}")