import argparse
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.plotting import colors
from tqdm import tqdm

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
            shift = int(padding * 0.4)

            top_left = (x1, y1 - th - padding - shift)
            bottom_right = (x1 + tw, y1 + baseline - shift)
            top_left = (max(top_left[0], 0), max(top_left[1], 0))

            cv2.rectangle(self.im, top_left, bottom_right, color, -1)
            text_thickness = 3
            cv2.putText(self.im, label, (x1, y1 - baseline), self.font, self.font_size,
                        txt_color, text_thickness, lineType=cv2.LINE_AA)

    def result(self):
        return self.im

def load_4ch_image(rgb_path, ir_path, target_size):
    rgb = cv2.imread(str(rgb_path))
    ir = cv2.imread(str(ir_path), cv2.IMREAD_UNCHANGED)

    if rgb is None or ir is None:
        raise FileNotFoundError(f"Missing image: {rgb_path} or {ir_path}")

    if rgb.shape[:2] != ir.shape[:2]:
        ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

    # IR to grayscale if needed
    if len(ir.shape) == 3:
        if ir.shape[2] == 3:
            ir = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
        elif ir.shape[2] == 4:
            ir = cv2.cvtColor(ir, cv2.COLOR_BGRA2GRAY)

    ir_norm = cv2.normalize(ir, None, 0, 255, cv2.NORM_MINMAX)
    ir_norm = ir_norm[:, :, None]  # Add channel dim

    img4ch = np.concatenate([rgb, ir_norm], axis=2)
    img_resized = cv2.resize(img4ch, (target_size, target_size))

    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return img_tensor, rgb.shape[:2], rgb, ir_norm

def run_inference(
    weights, source_rgb, source_ir, project, name,
    imgsz=640, conf=0.25, iou=0.45, max_det=100, device='cuda'
):
    save_dir = Path(project) / name
    labels_dir = save_dir / "labels"
    rgb_vis_dir = save_dir / "save_rgb_vis"
    thermal_vis_dir = save_dir / "save_thermal_vis"

    for d in [labels_dir, rgb_vis_dir, thermal_vis_dir]:
        d.mkdir(parents=True, exist_ok=True)

    model = YOLO(weights)
    model.to(device)
    model.fuse()

    rgb_paths = sorted(Path(source_rgb).glob("*"))
    ir_paths = sorted(Path(source_ir).glob("*"))

    ir_filenames = {p.name for p in ir_paths}
    rgb_paths = [p for p in rgb_paths if p.name in ir_filenames]

    for idx, rgb_path in enumerate(tqdm(rgb_paths, desc="Processing images"), 1):
        try:
            ir_path = Path(source_ir) / rgb_path.name

            img_tensor, (orig_h, orig_w), rgb_img, ir_norm = load_4ch_image(rgb_path, ir_path, imgsz)
            img_tensor = img_tensor.to(device)

            with torch.no_grad():
                results = model.predict(img_tensor, conf=conf, iou=iou, max_det=max_det, verbose=False)[0]

            boxes = results.boxes

            scale_x = orig_w / imgsz
            scale_y = orig_h / imgsz

            rgb_annotator = CustomAnnotator(rgb_img.copy(), line_width=2, font_size=0.5)
            thermal_img = cv2.cvtColor(ir_norm.squeeze(), cv2.COLOR_GRAY2BGR)
            thermal_annotator = CustomAnnotator(thermal_img.copy(), line_width=2, font_size=0.5)

            label_lines = []

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf_score = box.conf.item()
                cls = int(box.cls.item())
                label = f"{model.names[cls]} {conf_score:.2f}"

                x1_o = int(x1 * scale_x)
                y1_o = int(y1 * scale_y)
                x2_o = int(x2 * scale_x)
                y2_o = int(y2 * scale_y)

                c = colors(cls, bgr=True)
                rgb_annotator.box_label([x1_o, y1_o, x2_o, y2_o], label, color=c)
                thermal_annotator.box_label([x1_o, y1_o, x2_o, y2_o], label, color=c)

                cx = ((x1 + x2) / 2) / imgsz
                cy = ((y1 + y2) / 2) / imgsz
                w = (x2 - x1) / imgsz
                h = (y2 - y1) / imgsz

                label_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf_score:.4f}")

            cv2.imwrite(str(rgb_vis_dir / rgb_path.name), rgb_annotator.result())
            cv2.imwrite(str(thermal_vis_dir / rgb_path.name), thermal_annotator.result())

            with open(labels_dir / rgb_path.name.replace(".jpg", ".txt"), "w") as f:
                f.write("\n".join(label_lines))

        except Exception as e:
            print(f"Error processing {rgb_path.name}: {e}")

    print(f"\nAll images visualizations and labels saved: {rgb_vis_dir}, {thermal_vis_dir}, {labels_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="YOLOv8 4-channel inference with image save")
    parser.add_argument("--weights", type=str, required=True, help="Model weights path")
    parser.add_argument("--source_rgb", type=str, required=True, help="RGB images folder")
    parser.add_argument("--source_ir", type=str, required=True, help="IR images folder")
    parser.add_argument("--project", type=str, default="runs/predict", help="Output project folder")
    parser.add_argument("--name", type=str, default="exp", help="Run name")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for model input")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--max_det", type=int, default=100, help="Max detections per image")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    args = parser.parse_args()

    run_inference(
        weights=args.weights,
        source_rgb=args.source_rgb,
        source_ir=args.source_ir,
        project=args.project,
        name=args.name,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device
    )
