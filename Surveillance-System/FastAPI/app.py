import os
import json
import asyncio
import logging

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from detector import start_all_detections, generate_stream, set_log_callback, current_warnings, clear_warning_by_cam
from datetime import datetime
from collections import deque
from threading import Lock, Thread
from detector import analyze_all_velocity_clusters

# 📝 로그 파일 설정
logging.basicConfig(filename='logs/server.log', level=logging.INFO, format='%(asctime)s - %(message)s')

app = FastAPI()

# CORS 설정: 모든 origin 허용 (개발용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📁 정적 파일 mount
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/clips", StaticFiles(directory="static/clips"), name="clips")
templates = Jinja2Templates(directory="templates")

# 🧠 로그 저장소
MAX_LOGS = 100
log_buffer = deque(maxlen=MAX_LOGS)
log_lock = Lock()

# 🌐 웹소켓 연결 추적
websocket_connections = set()

# 📤 웹소켓 전송 비동기 함수
async def send_to_websocket(cam, label, content, message=None):
    print(f"[DEBUG] 📨 send_to_websocket() called")
    print(f"[DEBUG] cam={cam}, label={label}, content={content}, message={message}")

    log = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cam": cam,
        "label": label
    }

    if message:
        log["message"] = message

    if label == "velocity":
        log["message"] = content
    elif isinstance(content, str) and (content.endswith(".mp4") or content.startswith("static/")):
        log["clip"] = "/" + content.replace("static/", "").replace("\\", "/")

    data = json.dumps(log)
    to_remove = set()
    for ws in websocket_connections:
        try:
            await ws.send_text(data)
        except:
            to_remove.add(ws)
    websocket_connections.difference_update(to_remove)


# 📝 로그 콜백 함수
def log_callback(cam, label, clip_path, message=None):
    log = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cam": cam,
        "label": label
    }

    if message:
        log["message"] = message

    if isinstance(clip_path, str) and (clip_path.endswith(".mp4") or clip_path.startswith("static/")):
        log["clip"] = "/" + clip_path.replace("static/", "").replace("\\", "/")

    print(f"[DEBUG] 📤 WebSocket 로그 전송 준비: {log}")

    with log_lock:
        log_buffer.appendleft(log)

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(send_to_websocket(cam, label, clip_path, message=message))
    except RuntimeError:
        asyncio.run(send_to_websocket(cam, label, clip_path, message=message))


# 🔧 웹소켓 등록
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    websocket_connections.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)

# 🧠 감지 스레드 시작
@app.on_event("startup")
def startup_event():
    set_log_callback(log_callback)
    Thread(target=start_all_detections, daemon=True).start()

# 메인 페이지
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 스트리밍
@app.get("/stream/{cam}")
def stream(cam: str, mode: str = "rgb"):
    return StreamingResponse(generate_stream(cam, mode), media_type="multipart/x-mixed-replace; boundary=frame")

# 전체 로그
@app.get("/logs")
def get_logs():
    with log_lock:
        return list(log_buffer)

# 워닝 상태 (cam별 단 1개)
@app.get("/current-warning")
def get_current_warning():
    return {"warnings": current_warnings}

# 워닝 해제
@app.post("/clear_warning/{cam}")
def clear_warning(cam: str):
    clear_warning_by_cam(cam)
    return {"status": "cleared"}

@app.get("/analyze_velocity/{cam}/{mode}")
def run_velocity_analysis(cam: str, mode: str):
    analyze_all_velocity_clusters(cam, mode)
    return {"status": "done"}
