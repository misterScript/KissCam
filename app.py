import cv2
import socket
import os
import threading
import numpy as np
import base64
import time
import random
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from datetime import datetime

class KissCamServer:
    def __init__(self, host='0.0.0.0', port=5001):
        self.app = Flask(__name__, static_folder='.', static_url_path='')
        self.host = host
        self.port = port
        self.video_frame = None
        self.overlay_active = False
        self.custom_text = '💋 KISS CAM! 💋'
        self.lock = threading.Lock()
        self.last_capture_time = 0
        self.zoom_level = 1.0
        self.active_effect = None
        self.effect_time_start = 0
        self.effect_duration = 10
        self.sound_to_play = None
        self.overlay_image = None
        
        self.load_overlays()
        self.setup_routes()

        if not os.path.exists('capturas'):
            os.makedirs('capturas')
        
        if not os.path.exists('static/sounds'):
            os.makedirs('static/sounds')

    def load_overlays(self):
        self.overlays = {}
        frames_path = os.path.join('static', 'frames')
        if os.path.exists(frames_path):
            for filename in os.listdir(frames_path):
                if filename.endswith('.png'):
                    img_path = os.path.join(frames_path, filename)
                    self.overlays[filename.split('.')[0]] = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
    def setup_routes(self):
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/mobile', 'mobile', self.mobile)
        self.app.add_url_rule('/video_feed', 'video_feed', self.video_feed)
        self.app.add_url_rule('/receive_frame', 'receive_frame', self.receive_frame, methods=['POST'])
        self.app.add_url_rule('/toggle_overlay', 'toggle_overlay', self.toggle_overlay)
        self.app.add_url_rule('/set_text', 'set_text', self.set_text, methods=['POST'])
        self.app.add_url_rule('/capture', 'capture', self.capture)
        self.app.add_url_rule('/set_zoom', 'set_zoom', self.set_zoom, methods=['POST'])
        self.app.add_url_rule('/get_photos', 'get_photos', self.get_photos)
        self.app.add_url_rule('/capturas/<path:filename>', 'get_photo', self.get_photo)
        self.app.add_url_rule('/set_effect', 'set_effect', self.set_effect, methods=['POST'])
        self.app.add_url_rule('/get_effect_status', 'get_effect_status', self.get_effect_status)

    def index(self):
        return render_template('index.html', server_ip=self.get_local_ip(), port=self.port)

    def mobile(self):
        return render_template('mobile.html')

    def get_photos(self):
        photos = [f for f in os.listdir('capturas') if f.endswith('.jpg')]
        photos.sort(key=lambda f: os.path.getmtime(os.path.join('capturas', f)), reverse=True)
        return jsonify(photos[:5])

    def get_photo(self, filename):
        return send_from_directory('capturas', filename)

    def receive_frame(self):
        try:
            data = request.json
            if 'image' in data:
                image_data = data['image'].split(',')[1]
                frame_bytes = base64.b64decode(image_data)
                with self.lock:
                    self.video_frame = frame_bytes
                return jsonify({'status': 'success'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 400

    def set_zoom(self):
        try:
            data = request.json
            new_zoom = data.get('zoom')
            if new_zoom:
                with self.lock:
                    self.zoom_level = float(new_zoom)
                return jsonify({'status': 'success', 'zoom': self.zoom_level})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 400

    def get_sound_files(self):
        sound_path = os.path.join('static', 'sounds')
        if os.path.exists(sound_path):
            return [f for f in os.listdir(sound_path) if f.endswith('.mp3')]
        return []

    def set_effect(self):
        data = request.json
        effect_name = data.get('effect')
        
        with self.lock:
            self.active_effect = effect_name
            self.effect_time_start = time.time()
            sound_files = self.get_sound_files()
            self.sound_to_play = random.choice(sound_files) if sound_files else None
            
        return jsonify({'status': 'success', 'effect': effect_name})

    def get_effect_status(self):
        with self.lock:
            if self.active_effect and (time.time() - self.effect_time_start > self.effect_duration):
                self.active_effect = None
                self.sound_to_play = None
            
            response = {
                'active_effect': self.active_effect,
                'sound_to_play': self.sound_to_play
            }
            # Clear sound_to_play after sending to avoid replaying
            self.sound_to_play = None
            
        return jsonify(response)

    def apply_effect(self, frame, effect_name):
        h, w, _ = frame.shape
        if effect_name in self.overlays:
            overlay_img = self.overlays[effect_name]
            overlay_img_resized = cv2.resize(overlay_img, (w, h), interpolation=cv2.INTER_LINEAR)
            
            alpha_s = overlay_img_resized[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            
            for c in range(0, 3):
                frame[:, :, c] = (alpha_s * overlay_img_resized[:, :, c] +
                                  alpha_l * frame[:, :, c])
        elif effect_name == 'sepia':
            frame = cv2.transform(frame, np.matrix([[0.272, 0.534, 0.131],
                                                    [0.349, 0.686, 0.168],
                                                    [0.393, 0.769, 0.189]]))
        
        return frame

    def apply_zoom(self, frame, zoom_level):
        if zoom_level <= 1.0:
            return frame
        
        h, w, _ = frame.shape
        zoom_w = int(w / zoom_level)
        zoom_h = int(h / zoom_level)
        
        start_x = (w - zoom_w) // 2
        start_y = (h - zoom_h) // 2
        
        cropped_frame = frame[start_y:start_y + zoom_h, start_x:start_x + zoom_w]
        resized_frame = cv2.resize(cropped_frame, (w, h), interpolation=cv2.INTER_LINEAR)
        return resized_frame

    def generate_frames(self):
        while True:
            time.sleep(0.03)
            with self.lock:
                if self.video_frame is not None:
                    frame_bytes = self.video_frame
                else:
                    continue

            try:
                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                zoomed_frame = self.apply_zoom(frame, self.zoom_level)

                if self.overlay_active:
                    self.draw_overlay(zoomed_frame)
                
                if self.active_effect:
                    zoomed_frame = self.apply_effect(zoomed_frame, self.active_effect)

                ret, buffer = cv2.imencode('.jpg', zoomed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                print(f"Error procesando frame: {e}")
                continue

    def video_feed(self):
        return Response(self.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def draw_overlay(self, frame):
        h, w, _ = frame.shape
        center_x, center_y = w // 2, h // 2
        heart_color = (0, 0, 255)
        cv2.putText(frame, '❤', (center_x - 100, center_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 3, heart_color, 10, cv2.LINE_AA)

        text_size = cv2.getTextSize(self.custom_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
        text_x = int((w - text_size[0]) / 2)
        text_y = int(h - text_size[1] - 50)
        cv2.putText(frame, self.custom_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

    def toggle_overlay(self):
        self.overlay_active = not self.overlay_active
        return jsonify({'overlay_active': self.overlay_active})

    def set_text(self):
        data = request.json
        if 'text' in data:
            with self.lock:
                self.custom_text = data['text']
            return jsonify({'status': 'success'})
        return jsonify({'status': 'error', 'message': 'No text provided'}), 400

    def capture(self):
        current_time = time.time()
        if current_time - self.last_capture_time < 3:
            return jsonify({'status': 'error', 'message': 'Espera 3 segundos entre capturas'})

        with self.lock:
            if self.video_frame is None:
                return jsonify({'status': 'error', 'message': 'No hay video para capturar'})
            frame_bytes = self.video_frame

        try:
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"kisscam_{timestamp}.jpg"
            filepath = os.path.join('capturas', filename)
            cv2.imwrite(filepath, frame)

            self.last_capture_time = current_time
            return jsonify({'status': 'success', 'filename': filename})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})

    def get_local_ip(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        except Exception:
            ip = '127.0.0.1'
        finally:
            s.close()
        return ip

    def run(self):
        print(f"🎥 Kiss Cam Server iniciado")
        print(f"📱 Acceso móvil: http://{self.get_local_ip()}:{self.port}/mobile")
        print(f"💻 Acceso PC: http://{self.get_local_ip()}:{self.port}/")
        print("=" * 50)
        self.app.run(host='0.0.0.0', port=self.port, debug=False)

if __name__ == '__main__':
    server = KissCamServer()
    server.run()