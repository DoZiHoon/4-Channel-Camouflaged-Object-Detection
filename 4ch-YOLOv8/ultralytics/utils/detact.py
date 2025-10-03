import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.plotting import colors

# ✅ YOLO 스타일 유지 + 조절된 라벨 박스
class CustomAnnotator:
    def __init__(self, im, line_width=2, font_size=0.5):
        self.im = im
        self.line_width = line_width
        self.font_size = font_size
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(self.im, (x1, y1), (x2, y2), color, thickness=self.line_width)

        if label:
            (tw, th), baseline = cv2.getTextSize(label, self.font, self.font_size, thickness=self.line_width)
            th += baseline
            padding = int(th * 0.2)
            shift = int(padding * 0.4)  # ✅ 상자 위로 이동

            top_left = (x1, y1 - th - padding - shift)
            bottom_right = (x1 + tw, y1 + baseline - shift)
            top_left = (max(top_left[0], 0), max(top_left[1], 0))

            cv2.rectangle(self.im, top_left, bottom_right, color, -1)
            text_thickness = 3
            cv2.putText(self.im, label, (x1, y1 - baseline), self.font, self.font_size,
                        txt_color, text_thickness, lineType=cv2.LINE_AA)

    def result(self):
        return self.im

# ✅ 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("/home/ricky/ultralytics-main/runs/detect/train89/weights/best.pt")
model.to(device)

# ✅ 경로 설정
rgb_dir = Path("/home/ricky/Data/Animal/images")
thermal_dir = Path("/home/ricky/Data/Animal/output")
base_save_dir = Path("/home/ricky/ultralytics-main/runs/Animal_95")

# # ✅ 경로 설정
# rgb_dir = Path("/home/ricky/다운로드/ppt_samples/images")
# thermal_dir = Path("/home/ricky/다운로드/ppt_samples/output")
# base_save_dir = Path("/home/ricky/ultralytics-main/runs/ppt_samples")

labels_dir = base_save_dir / "labels"
rgb_vis_dir = base_save_dir / "save_rgb_vis"
thermal_vis_dir = base_save_dir / "save_thermal_vis"
rgb_video_dir = base_save_dir / "videos/rgb"
thermal_video_dir = base_save_dir / "videos/thermal"

for d in [labels_dir, rgb_vis_dir, thermal_vis_dir, rgb_video_dir, thermal_video_dir]:
    d.mkdir(parents=True, exist_ok=True)

rgb_video_path = rgb_video_dir / "rgb_output.mp4"
thermal_video_path = thermal_video_dir / "thermal_output.mp4"
rgb_out = None
thermal_out = None
video_writer_initialized = False
fps = 1244 / 41

image_list = sorted(rgb_dir.glob("*.jpg"))
total = len(image_list)

for idx, rgb_path in enumerate(image_list, 1):
    filename = rgb_path.name
    print(f"\n🔄 [{idx}/{total}] 처리 중: {filename}")

    thermal_path = thermal_dir / filename
    txt_path = labels_dir / filename.replace(".jpg", ".txt")

    rgb = cv2.imread(str(rgb_path))
    thermal = cv2.imread(str(thermal_path), cv2.IMREAD_UNCHANGED)

    if rgb is None or thermal is None:
        print(f"❌ 이미지 누락: {filename}")
        continue

    if rgb.shape[:2] != thermal.shape[:2]:
        thermal = cv2.resize(thermal, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

    if len(thermal.shape) == 3:
        if thermal.shape[2] == 3:
            thermal = cv2.cvtColor(thermal, cv2.COLOR_BGR2GRAY)
        elif thermal.shape[2] == 4:
            thermal = cv2.cvtColor(thermal, cv2.COLOR_BGRA2GRAY)

    thermal_norm = cv2.normalize(thermal, None, 0, 255, cv2.NORM_MINMAX)
    if len(thermal_norm.shape) == 2:
        thermal_norm = thermal_norm[:, :, None]

    img4ch = np.concatenate([rgb, thermal_norm], axis=2)

    orig_h, orig_w = thermal.shape[:2]
    img_resized = cv2.resize(img4ch, (640, 640))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
     results = model.predict(img_tensor, conf=0.5, verbose=False, device=device, iou=0.5, max_det=100)[0]
   
    boxes = results.boxes
    print(f"📦 탐지된 객체 수: {len(boxes)}")

    scale_x = orig_w / 640
    scale_y = orig_h / 640

    rgb_annotated = rgb.copy()
    thermal_annotated = cv2.cvtColor(thermal.copy(), cv2.COLOR_GRAY2BGR)

    # ✅ 해상도 기반 조절 (바운딩박스 두께 1.5배)
    scale = min(orig_w, orig_h) / 640
    line_width = max(2, int(scale * 2.5 * 1.5))  # ✅ 1.5배 확대
    font_size = max(0.5, scale * 0.5)

    rgb_annotator = CustomAnnotator(rgb_annotated, line_width=line_width, font_size=font_size)
    thermal_annotator = CustomAnnotator(thermal_annotated, line_width=line_width, font_size=font_size)

    label_lines = []

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf.item()
        cls = int(box.cls.item())
        label = f"{model.names[cls]} {conf:.2f}"

        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        rgb_annotator.box_label([x1, y1, x2, y2], label, color=colors(cls, bgr=True))
        thermal_annotator.box_label([x1, y1, x2, y2], label, color=colors(cls, bgr=True))

        cx = ((x1 + x2) / 2) / orig_w
        cy = ((y1 + y2) / 2) / orig_h
        w = (x2 - x1) / orig_w
        h = (y2 - y1) / orig_h

        label_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.4f}")

    rgb_result = rgb_annotator.result()
    thermal_result = thermal_annotator.result()
    cv2.imwrite(str(rgb_vis_dir / filename), rgb_result)
    cv2.imwrite(str(thermal_vis_dir / filename), thermal_result)
    print(f"🖼️ 저장 완료 - RGB: {filename}, Thermal: {filename}")

    if not video_writer_initialized:
        h, w = rgb_result.shape[:2]
        rgb_out = cv2.VideoWriter(str(rgb_video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        thermal_out = cv2.VideoWriter(str(thermal_video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        video_writer_initialized = True

    rgb_out.write(rgb_result)
    thermal_out.write(thermal_result)

    with open(txt_path, "w") as f:
        f.write("\n".join(label_lines))
    print(f"📝 라벨 저장 완료: {txt_path.name}")

if rgb_out:
    rgb_out.release()
    thermal_out.release()
    print(f"\n🎞️ 영상 저장 완료: {rgb_video_path.name}, {thermal_video_path.name}")
