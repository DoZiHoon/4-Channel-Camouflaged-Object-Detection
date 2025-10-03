import os
import cv2
import numpy as np
from tqdm import tqdm

# ===== 경로 설정 =====
img_dir = "/home/ricky/Data/drone_video/test/images"
label_dir = "/home/ricky/Data/drone_video/test/labels"
output_img_dir = "/home/ricky/Data/drone_video/test/padded_images"
output_mask_dir = "/home/ricky/Data/drone_video/test/padded_masks"
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# ===== 패딩 설정 =====
PAD = 30

# ===== 이미지 반복 =====
img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg") or f.endswith(".png")])

for img_file in tqdm(img_files, desc="패딩 내부 객체 생성 중"):
    label_path = os.path.join(label_dir, img_file.replace(".jpg", ".txt").replace(".png", ".txt"))

    if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        new_w = w + PAD

        # (1) 왼쪽 패딩 추가
        padded_img = cv2.copyMakeBorder(img, 0, 0, PAD, 0, borderType=cv2.BORDER_REFLECT)

        # (2) 마스크 생성
        mask = np.zeros((h, new_w), dtype=np.uint8)

        # (3) 사각형 객체 삽입 (패딩 내부에만)
        # 예: (x=5~25), y는 이미지 중간
        x1, x2 = 5, 25
        y1, y2 = h // 2 - 15, h // 2 + 15
        cv2.rectangle(mask, (x1, y1), (x2, y2), color=255, thickness=-1)

        # (4) 저장
        out_img_path = os.path.join(output_img_dir, img_file.replace(".jpg", ".png").replace(".png", ".png"))
        out_mask_path = os.path.join(output_mask_dir, img_file.replace(".jpg", ".png").replace(".png", ".png"))
        cv2.imwrite(out_img_path, padded_img)
        cv2.imwrite(out_mask_path, mask)
