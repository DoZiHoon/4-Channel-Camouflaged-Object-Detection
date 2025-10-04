# detector.py (업데이트된 핵심 구조)

import os
import cv2
import time
import numpy as np
from threading import Thread
from datetime import datetime
from sklearn.cluster import DBSCAN
from collections import defaultdict

video_paths = {
    "cam1": {"rgb": "static/videos/cam1_rgb.mp4", "thermal": "static/videos/cam1_thermal.mp4"},
    "cam2": {"rgb": "static/videos/cam2_rgb.mp4", "thermal": "static/videos/cam2_thermal.mp4"},
    "cam3": {"rgb": "static/videos/cam3_rgb.mp4", "thermal": "static/videos/cam3_thermal.mp4"},
    "cam4": {"rgb": "static/videos/cam4_rgb.mp4", "thermal": "static/videos/cam4_thermal.mp4"},
}

log_callback_fn = None
record_tasks = defaultdict(list)      # 스트리밍 기반 클립 저장용
warn_judged_flags = defaultdict(bool) 
velocity_flags = defaultdict(bool)
record_flags = defaultdict(bool)
current_frame_idx = defaultdict(int)
warning_flags = defaultdict(bool)
current_warnings = dict()

def set_log_callback(fn):
    global log_callback_fn
    log_callback_fn = fn

def parse_labels(label_path):
    data = []
    for fname in sorted(os.listdir(label_path)):
        if fname.endswith(".txt"):
            frame_id = int(fname.replace("frame_", "").replace(".txt", ""))
            with open(os.path.join(label_path, fname)) as f:
                for line in f:
                    parts = line.strip().split()
                    cls_id = parts[0]
                    conf = float(parts[5]) if len(parts) >= 6 else 1.0
                    data.append((frame_id, cls_id, conf))
    return data

def cluster_frames(data, eps=400, min_samples=1):
    if not data:
        return {}
    frame_ids = np.array([[d[0]] for d in data])
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(frame_ids)
    clusters = defaultdict(list)
    for (frame_id, cls_id, conf), label in zip(data, db.labels_):
        clusters[label].append((frame_id, cls_id, conf))
    return clusters


def get_warning_label_from_initial_frames(cam, mode, start_frame, num_frames=100):
    label_dir = f"static/labels/{cam}_{mode}"
    class_conf_sums = defaultdict(float)

    for i in range(start_frame, start_frame + num_frames):
        label_path = os.path.join(label_dir, f"frame_{i:05d}.txt")
        if not os.path.exists(label_path):
            continue

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    cls_id = parts[0]
                    conf = float(parts[5])
                    class_conf_sums[cls_id] += conf

    if not class_conf_sums:
        return None  # 판단 불가

    dominant_cls = max(class_conf_sums.items(), key=lambda x: x[1])[0]
    return "person" if dominant_cls == '0' else "animal"



def prepare_record_tasks():
    for cam in video_paths:
        for mode in ["rgb", "thermal"]:
            label_path = f"static/labels/{cam}_{mode}"
            if not os.path.exists(label_path):
                continue
            data = parse_labels(label_path)
            clusters = cluster_frames(data)
            for cluster_id, frames in clusters.items():
                start = min(f[0] for f in frames)
                end = max(f[0] for f in frames)

                class_scores = defaultdict(float)
                class_confidences = defaultdict(list)

                for _, cls_id, conf in frames:
                    class_scores[cls_id] += conf
                    class_confidences[cls_id].append(conf)

                # 📌 신뢰도 총합 기준으로 대표 클래스 결정
                label = max(class_scores.items(), key=lambda x: x[1])[0]
                
                # 📌 선택된 label의 평균 confidence 계산
                avg_conf = sum(class_confidences[label]) / len(class_confidences[label])

                # 📌 record_tasks에 평균 신뢰도도 포함해서 저장
                record_tasks[f"{cam}_{mode}"].append((start, end, label, avg_conf))



def cluster_positions_for_velocity(pos_dir, start, end, eps=20, min_samples=1):
    frames = []
    coords = []

    for i in range(start, end + 1):
        pos_path = os.path.join(pos_dir, f"frame_{i:05d}.txt")
        if os.path.exists(pos_path):
            with open(pos_path) as f:
                content = f.read().strip()
                if content:  # 파일 내용이 있을 때만 처리
                    try:
                        vals = list(map(float, content.split(',')))
                        coords.append(vals)
                        frames.append(i)
                    except ValueError:
                        continue

    if not coords:
        return []

    coords = np.array(coords)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    clusters = defaultdict(list)
    for frame_id, label in zip(frames, db.labels_):
        clusters[label].append(frame_id)

    return list(clusters.values())


def cluster_positions_by_file_content(pos_dir, start, end, eps=400, min_samples=1):
    frames = []
    for i in range(start, end + 1):
        pos_path = os.path.join(pos_dir, f"frame_{i:05d}.txt")
        if os.path.exists(pos_path):
            with open(pos_path) as f:
                content = f.read().strip()
                if content:  # 내용이 비어있지 않으면 포함
                    frames.append(i)

    if not frames:
        return []

    frame_arr = np.array([[f] for f in frames])
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(frame_arr)

    clusters = defaultdict(list)
    for frame_id, label in zip(frames, db.labels_):
        clusters[label].append(frame_id)

    return list(clusters.values())



def compute_velocity_log(cam, mode, start, end):
    pos_dir = f"static/positions/{cam}_{mode}"
    start_path = os.path.join(pos_dir, f"frame_{start:05d}.txt")
    end_paths = [os.path.join(pos_dir, f"frame_{i:05d}.txt") for i in range(end-2, end+1)]

    if not os.path.exists(start_path):
        print(f"[❌ 시작 위치 파일 없음] {start_path}")
        return "속도 정보 없음"
    if not all(os.path.exists(p) for p in end_paths):
        print(f"[❌ 종료 위치 파일 일부 없음] {[p for p in end_paths if not os.path.exists(p)]}")
        return "속도 정보 없음"

    def read_pos(path):
        with open(path) as f:
            vals = list(map(float, f.read().strip().split()))
        return np.array(vals)

    p0 = read_pos(start_path)
    pend = read_pos(end_paths[-1]) 
    delta = pend - p0
    duration = (end - start + 1) / 30
    v = delta / duration

    x_dir = "동" if v[0] > 0 else "서"
    z_dir = "남" if v[2] > 0 else "북"
    y_dir = "상승중" if v[1] > 0 else "하강중"

    print(f"[🧮 속도 계산] Δx={delta[0]:.2f}, Δy={delta[1]:.2f}, Δz={delta[2]:.2f} | duration={duration:.2f}")

    return f"{x_dir}쪽으로 {abs(v[0]):.1f}m/s, {z_dir}쪽으로 {abs(v[2]):.1f}m/s, {abs(v[1]):.1f}m/s 속력으로 {y_dir}"



def save_clip_buffer(cam, mode, label, buffer, fps, size,avg_conf=None):
    now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    clip_dir = f"static/clips/{cam}_{mode}"
    os.makedirs(clip_dir, exist_ok=True)
    temp_path = os.path.join(clip_dir, f"{now}_temp.mp4")
    final_path = temp_path.replace("_temp", "")

    # 🎞 영상 저장
    writer = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for frame in buffer:
        writer.write(frame)
    writer.release()
    os.system(f"ffmpeg -y -i {temp_path} -c:v libx264 -preset fast -crf 23 -movflags +faststart {final_path}")
    os.remove(temp_path)

    # ⏱ 프레임 범위 계산
    frame_end = current_frame_idx[cam] - 1
    frame_start = frame_end - len(buffer) + 1
    print(f"[🎞 클립 저장 완료] {cam}_{mode} | 클립: {final_path} | 프레임: {frame_start}~{frame_end}")

    # 🔑 로그용 카메라 이름 (예: cam1_rgb 또는 cam1_thermal)
    log_cam = f"{cam}_{mode}" 

    if avg_conf is not None:
        label_str = "person" if label == '0' else "animal"
        print(f"[🔔 감지 로그] {log_cam} | {label_str} (avg conf {avg_conf:.2f})")


    # 🪪 감지 로그 콜백 (사람/동물 감지)
    if log_callback_fn:
        label_str = "person" if label == '0' else "animal"
        message = f"avg conf {avg_conf:.2f}" if avg_conf is not None else None
        print(f"[DEBUG] Calling log_callback with: {label_str}, {final_path}, {message}")
        log_callback_fn(log_cam, label_str, final_path, message=message)
    # 🧠 고정 위치파일 전체 대상 군집화
    pos_dir = f"static/positions/{cam}_{mode}"
    clusters = cluster_positions_by_file_content(pos_dir, 0, 99999)

    if not clusters:
        print(f"[🚨 속도 분석 실패] 위치 군집 없음")
        return

    # 🔍 현재 클립 범위와 겹치는 군집 찾기
    matched_cluster = None
    for cluster in clusters:
        if any(f in cluster for f in range(frame_start, frame_end + 1)):
            matched_cluster = cluster
            break

    if not matched_cluster:
        print(f"[🚨 속도 분석 실패] 클립과 일치하는 군집 없음")
        return

    cluster_start = min(matched_cluster)
    cluster_end = max(matched_cluster)
    print(f"[📌 위치 기반 군집 속도 분석] {cam}_{mode} | 프레임: {cluster_start}~{cluster_end}")

    try:
        velocity_log = compute_velocity_log(cam, mode, cluster_start, cluster_end)
        print(f"[✅ 속도 분석 결과] {velocity_log}")

        # ✅ 중복 방지를 위한 클러스터 고유 키
        cluster_key = f"{cam}_{mode}_{cluster_start}_{cluster_end}"
        if not velocity_flags[cluster_key] and mode != "thermal" and log_callback_fn:
            log_callback_fn(cam, "velocity", velocity_log)
            velocity_flags[cluster_key] = True

    except Exception as e:
        print(f"[🚨 속도 분석 실패] {e}")








def generate_stream(cam, mode):
    cap = cv2.VideoCapture(video_paths[cam][mode])
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_idx = current_frame_idx[cam]
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    clip_buffer = []

    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            current_frame_idx[cam] = 0
            continue

        task_key = f"{cam}_{mode}"
        for start, end, cluster_label, avg_conf in record_tasks[task_key]:
            key = f"{task_key}_{start}_{end}"
            warn_key = f"{task_key}_{start}"

            # 🔸 워닝 배너 표시 조건 (단 한 번만)
            if frame_idx == start and not warning_flags[warn_key]:
                warning_flags[warn_key] = True
                print(f"[🔔 워닝 플래그 활성화] {warn_key}")

            # 🔸 워닝 판단 로직: start+100프레임에 딱 한 번만 라벨 판단
            if frame_idx == start + 150 and not warn_judged_flags[warn_key]:
                warn_judged_flags[warn_key] = True

                if mode != "thermal":
                    label = get_warning_label_from_initial_frames(cam, mode, start, num_frames=100)
                    if label:
                        current_warnings[cam] = {"label": label}
                        print(f"[✅ 워닝 판단 완료] {warn_key} → {label}")

            # 클립 저장 로직
            if start <= frame_idx <= end:
                clip_buffer.append(frame)
                record_flags[key] = True

            elif frame_idx > end and record_flags.get(key):
                if clip_buffer:
                    save_clip_buffer(cam, mode, cluster_label, clip_buffer, fps, (width, height), avg_conf=avg_conf)
                record_flags[key] = False
                clip_buffer = []

        _, buffer = cv2.imencode('.jpg', frame)
        frame_idx += 1
        current_frame_idx[cam] = frame_idx

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        time.sleep(1 / fps)






def start_all_detections():
    prepare_record_tasks()
    
    for cam in video_paths:
        for mode in ["rgb", "thermal"]:
            Thread(
                target=lambda: list(generate_stream(cam, mode)),
                daemon=True
            ).start()

def clear_warning_by_cam(cam_mode):
    if cam_mode in current_warnings:
        del current_warnings[cam_mode]

def analyze_all_velocity_clusters(cam, mode, eps=20, min_samples=3):
    """
    static/positions/{cam}_{mode} 안의 모든 3D 위치 파일을 클러스터링 후 속도 분석 수행
    감지(start~end)와 무관하게 완전히 독립적으로 처리
    """
    pos_dir = f"static/positions/{cam}_{mode}"
    coords = []
    frames = []

    for fname in sorted(os.listdir(pos_dir)):
        if fname.endswith(".txt") and fname.startswith("frame_"):
            try:
                frame_id = int(fname.replace("frame_", "").replace(".txt", ""))
                with open(os.path.join(pos_dir, fname)) as f:
                    vals = list(map(float, f.read().strip().split()))
                    if len(vals) == 3:
                        coords.append(vals)
                        frames.append(frame_id)
            except:
                continue


__all__ = ["start_all_detections", "generate_stream", "set_log_callback", "current_warnings", "clear_warning_by_cam", "analyze_all_velocity_clusters"]
