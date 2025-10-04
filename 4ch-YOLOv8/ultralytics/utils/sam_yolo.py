import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# 🔧 경로 설정
image_dir = "/home/ricky/Data/Animal/images"
output_dir = "/home/ricky/Data/Animal/masks"
os.makedirs(output_dir, exist_ok=True)

# ✅ 모델 로드
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

# 🎯 프롬프트 설정
prompt = "a deer"

# 🖼 이미지 순회
for fname in sorted(os.listdir(image_dir)):
    if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    image_path = os.path.join(image_dir, fname)
    image = Image.open(image_path).convert("RGB")

    # ✂️ 입력 전처리
    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    # ✅ logits → 마스크 생성
    mask = torch.sigmoid(outputs.logits)[0][0].cpu().numpy()
    mask = (mask * 255).astype(np.uint8)
    mask_resized = cv2.resize(mask, image.size)

    # ⚠️ 마스크 유효성 검사
    if mask_resized.shape[0] < 10 or mask_resized.shape[1] < 10:
        print(f"⚠️ 마스크 이상: {fname}")
        continue

    out_path = os.path.join(output_dir, f"{os.path.splitext(fname)[0]}_mask.png")
    cv2.imwrite(out_path, mask_resized)
    print(f"✅ 저장됨: {out_path}")
