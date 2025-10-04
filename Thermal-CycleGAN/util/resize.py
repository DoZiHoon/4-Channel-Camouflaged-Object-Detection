import os
from PIL import Image

# 입력 폴더와 출력 폴더 경로 설정
input_dir = r'C:\Users\KSH\Documents\Final_project\pytorch-CycleGAN-and-pix2pix-master\zett_thermal'     # 예: 'inputs'
output_dir = r'C:\Users\KSH\Documents\Final_project\pytorch-CycleGAN-and-pix2pix-master\zett_thermal_resize'   # 예: 'resized_inputs'

# 출력 폴더 없으면 생성
os.makedirs(output_dir, exist_ok=True)

# 원하는 리사이즈 크기 (width, height)
resize_size = (720, 1280)

# 이미지 확장자 목록
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

# 폴더 내 모든 파일 순회
for filename in os.listdir(input_dir):
    if any(filename.lower().endswith(ext) for ext in image_extensions):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            with Image.open(input_path) as img:
                resized_img = img.resize(resize_size, Image.LANCZOS)
                resized_img.save(output_path)
                print(f'Resized and saved: {output_path}')
        except Exception as e:
            print(f'Error processing {filename}: {e}')