import cv2
import os

def extract_frames_from_video(video_path, output_folder):
    # 비디오 파일 열기
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("비디오를 열 수 없습니다.")
        return

    # 출력 폴더 생성 (없으면)
    os.makedirs(output_folder, exist_ok=True)

    frame_count = 0
    while True:
        ret, frame = cap.read()  # 프레임 읽기
        if not ret:
            break  # 더 이상 프레임이 없으면 종료

        # JPG 파일로 저장
        filename = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(filename, frame)
        frame_count += 1

    cap.release()
    
    print(f"총 {frame_count}개의 프레임을 저장했습니다.")

# 사용 예시
video_path = "/home/ricky/Data/Animal/video/animal.mp4"            # 불러올 비디오 파일 경로
output_folder = "/home/ricky/Data/Animal/images"    # JPG 저장할 폴더
extract_frames_from_video(video_path, output_folder)