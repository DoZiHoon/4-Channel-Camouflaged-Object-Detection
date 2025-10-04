

































import os
import cv2
import numpy as np
from PIL import Image

def imread_unicode(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return np.array(img.convert('RGB'))

def parse_label_line(line):
    try:
        return list(map(float, line.strip().split()))
    except ValueError:
        return None

def is_segmentation_format(values):
    return len(values) > 5

def create_mask_from_bbox(mask, values, img_size):
    _, x, y, w, h = values
    H, W = img_size
    x1 = int((x - w / 2) * W)
    y1 = int((y - h / 2) * H)
    x2 = int((x + w / 2) * W)
    y2 = int((y + h / 2) * H)
    x1 = max(0, min(x1, W - 1))
    y1 = max(0, min(y1, H - 1))
    x2 = max(0, min(x2, W - 1))
    y2 = max(0, min(y2, H - 1))
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

def create_mask_from_segmentation(mask, values, img_size):
    cls = int(values[0])
    H, W = img_size
    points = np.array([
        (float(values[i]) * W, float(values[i + 1]) * H)
        for i in range(1, len(values), 2)
    ], dtype=np.int32)
    if len(points) >= 3:
        cv2.fillPoly(mask, [points], 255)

def safe_remove(file_path):
    try:
        os.remove(file_path)
        if os.path.exists(file_path):
            print(f"[삭제 실패] {file_path}")
        else:
            print(f"[삭제 완료] {file_path}")
    except Exception as e:
        print(f"[에러] {file_path} 삭제 중 오류: {e}")

def convert_yolo_and_segmentation(rgb_dir, label_dir, mask_dir, target_class_id=0):
    os.makedirs(mask_dir, exist_ok=True)
    files = [f for f in os.listdir(rgb_dir) if f.lower().endswith('.jpg')]

    for file in files:
        img_path = os.path.join(rgb_dir, file)
        name = os.path.splitext(file)[0]
        label_path = os.path.join(label_dir, f"{name}.txt")

        if not os.path.exists(label_path):
            print(f"[스킵] 라벨 없음: {label_path}")
            continue

        with open(label_path, 'r') as f:
            lines = f.readlines()

        if not lines:
            print(f"[삭제] 비어있는 라벨: {label_path}")
            safe_remove(label_path)
            safe_remove(img_path)
            continue

        img = imread_unicode(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)

        valid_label_found = False
        for line in lines:
            values = parse_label_line(line)
            if values is None:
                print(f"[경고] 잘못된 라벨 형식: {line.strip()}")
                continue

            cls = int(values[0])
            if cls != target_class_id:
                continue

            valid_label_found = True

            if is_segmentation_format(values):
                create_mask_from_segmentation(mask, values, (H, W))
            else:
                create_mask_from_bbox(mask, values, (H, W))

        if not valid_label_found:
            print(f"[삭제] 대상 클래스 없음: {label_path}")
            safe_remove(label_path)
            safe_remove(img_path)
            continue

        out_path = os.path.join(mask_dir, file)
        cv2.imwrite(out_path, mask)


rgb_dir = r'C:\Users\KSH\Downloads\per.v1i.yolov8\TrainA'
label_dir = r'C:\Users\KSH\Downloads\per.v1i.yolov8\Total_labels'
mask_dir = r'C:\Users\KSH\Downloads\per.v1i.yolov8\TrainSegA'
target_class_id = 0  # 군인 클래스 ID

convert_yolo_and_segmentation(rgb_dir, label_dir, mask_dir, target_class_id)
