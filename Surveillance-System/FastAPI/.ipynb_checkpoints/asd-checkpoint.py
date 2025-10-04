from flask import Flask, render_template, Response
from detector import detect_from_video

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # 스트리밍 페이지

@app.route('/video')
def video():
    return Response(detect_from_video('static/videos/test.mp4'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)