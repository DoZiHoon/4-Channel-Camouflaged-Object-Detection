from ultralytics.data.dataset import YOLODataset
from pathlib import Path
import os
import cv2
import numpy as np
import yaml  # ✅ PyYAML 사용


# class YOLO4ChannelDataset(YOLODataset):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.buffer = list(range(len(self.im_files)))

#         # 열화상 이미지 경로 생성: RGB 경로에서 thermal로 폴더명만 바꾸기
#         self.thermal_files = [
#             str(Path(p).as_posix()).replace("/images/rgb/", "/images/thermal/").replace("/images/rgb_val/", "/images/thermal/")
#             for p in self.im_files
#         ]

#         # 디버깅용 출력 (한 번만 출력되게 조절할 수 있음)
#         print("샘플 RGB 이미지 경로:", self.im_files[0])
#         print("매칭된 열화상 이미지 경로:", self.thermal_files[0])
        
#     def load_image(self, index):
#         rgb_path = self.im_files[index]
#         thermal_path = self.thermal_files[index]
    
#         rgb = cv2.imread(rgb_path)  # (H, W, 3)
#         thermal = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)  # (H, W)
    
#         if rgb is None or thermal is None:
#             raise FileNotFoundError(f"RGB or Thermal image not found:\n{rgb_path}\n{thermal_path}")
    
#         # 열화상 크기를 RGB에 맞춤
#         if rgb.shape[:2] != thermal.shape[:2]:
#             thermal = cv2.resize(thermal, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    
#         thermal = thermal[:, :, None]  # (H, W, 1)
#         img4ch = np.concatenate([rgb, thermal], axis=2)  # (H, W, 4)
    
#         return img4ch, rgb.shape[:2], rgb.shape[:2]


# class YOLO4ChannelDataset(YOLODataset):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.buffer = list(range(len(self.im_files)))

#         # 열화상 이미지 경로 생성
#         self.thermal_files = [
#             str(Path(p).as_posix()).replace("/images/rgb/", "/images/thermal/").replace("/images/rgb_val/", "/images/thermal/")
#             for p in self.im_files
#         ]

#         print("샘플 RGB 이미지 경로:", self.im_files[0])
#         print("매칭된 열화상 이미지 경로:", self.thermal_files[0])

#     def load_image(self, index):
#         rgb_path = self.im_files[index]
#         thermal_path = self.thermal_files[index]
    
#         rgb = cv2.imread(rgb_path)  # (H, W, 3)
#         thermal = cv2.imread(thermal_path, cv2.IMREAD_UNCHANGED)
    
#         if rgb is None or thermal is None:
#             raise FileNotFoundError(f"RGB or Thermal image not found:\n{rgb_path}\n{thermal_path}")
    
#         # 해상도 맞추기
#         if rgb.shape[:2] != thermal.shape[:2]:
#             thermal = cv2.resize(thermal, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    
#         # ✅ 채널 확인 및 안정화
#         if len(thermal.shape) == 3:
#             if thermal.shape[2] == 3:
#                 thermal = cv2.cvtColor(thermal, cv2.COLOR_BGR2GRAY)
#             elif thermal.shape[2] == 4:
#                 thermal = cv2.cvtColor(thermal, cv2.COLOR_BGRA2GRAY)
    
#         # ✅ 여기가 핵심!
#         # (H, W) → (H, W, 1)
#         if len(thermal.shape) == 2:
#             thermal = thermal[:, :, None]
#         elif len(thermal.shape) == 4:
#             thermal = thermal.squeeze()
#             if len(thermal.shape) == 2:
#                 thermal = thermal[:, :, None]
    
#         # 병합
#         img4ch = np.concatenate([rgb, thermal], axis=2)
    
#         return img4ch, rgb.shape[:2], rgb.shape[:2]

class YOLO4ChannelDataset(YOLODataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = list(range(len(self.im_files)))

        # 열화상 이미지 경로 생성
        self.thermal_files = [
            str(Path(p).as_posix()).replace("/images/rgb/", "/images/thermal/").replace("/images/rgb_val/", "/images/thermal/")
            for p in self.im_files
        ]

        print("샘플 RGB 이미지 경로:", self.im_files[0])
        print("매칭된 열화상 이미지 경로:", self.thermal_files[0])

    def load_image(self, index):
        rgb_path = self.im_files[index]
        thermal_path = self.thermal_files[index]

        rgb = cv2.imread(rgb_path)  # (H, W, 3)
        thermal = cv2.imread(thermal_path, cv2.IMREAD_UNCHANGED)

        if rgb is None or thermal is None:
            raise FileNotFoundError(f"RGB or Thermal image not found:\n{rgb_path}\n{thermal_path}")

        if rgb.shape[:2] != thermal.shape[:2]:
            thermal = cv2.resize(thermal, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

        # 열화상 채널 안정화
        if len(thermal.shape) == 3:
            if thermal.shape[2] == 3:
                thermal = cv2.cvtColor(thermal, cv2.COLOR_BGR2GRAY)
            elif thermal.shape[2] == 4:
                thermal = cv2.cvtColor(thermal, cv2.COLOR_BGRA2GRAY)

        # ✅ 열화상 대비 정규화 (핵심)
        thermal = cv2.normalize(thermal, None, 0, 255, cv2.NORM_MINMAX)

        # (H, W) → (H, W, 1)
        if len(thermal.shape) == 2:
            thermal = thermal[:, :, None]
        elif len(thermal.shape) == 4:
            thermal = thermal.squeeze()
            if len(thermal.shape) == 2:
                thermal = thermal[:, :, None]

        # 병합: RGB (0~255) + 열화상 (0~255) → YOLO 내부에서 /255.0 적용됨
        img4ch = np.concatenate([rgb, thermal], axis=2)

        return img4ch, rgb.shape[:2], rgb.shape[:2]


