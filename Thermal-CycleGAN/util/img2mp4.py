import cv2
import os
from natsort import natsorted

# 경로 설정
image_folder = r"/home/ricky/thermal"
output_dir = r"/home/ricky/"
os.makedirs(output_dir, exist_ok=True)

output_video_path = os.path.join(output_dir, "thermal_output.mp4")
target_duration_sec = 7  # :흰색_확인_표시: 목표 영상 길이 (초)# 이미지 목록 정렬
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images = natsorted(images)
num_frames = len(images)# :흰색_확인_표시: FPS 자동 계산
fps = num_frames / target_duration_sec# 영상 사이즈
first_frame_path = os.path.join(image_folder, images[0])
frame = cv2.imread(first_frame_path)
height, width, _ = frame.shape
frame_size = (width, height)# 비디오 저장 객체 생성
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)# 영상 생성
for image_name in images:
    img_path = os.path.join(image_folder, image_name)
    frame = cv2.imread(img_path)
    out.write(frame)
out.release()
print(f":흰색_확인_표시: 7초짜리 영상 저장 완료: {output_video_path} (FPS: {fps:.2f})")











































































































































































on_sec = 21  # ✅ 목표 영상 길이 (초)

# 이미지 목록 정렬
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images = natsorted(images)
num_frames = len(images)

# ✅ FPS 자동 계산
fps = num_frames / target_duration_sec

# 영상 사이즈
first_frame_path = os.path.join(image_folder, images[0])
frame = cv2.imread(first_frame_path)
height, width, _ = frame.shape
frame_size = (width, height)

# 비디오 저장 객체 생성
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

# 영상 생성
for image_name in images:
    img_path = os.path.join(image_folder, image_name)
    frame = cv2.imread(img_path)
    out.write(frame)

out.release()
print(f"✅ 21초짜리 영상 저장 완료: {output_video_path} (FPS: {fps:.2f})")