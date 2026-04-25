import cv2
import numpy as np
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import io

app = Flask(__name__)
# Crucial for live hosting
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

def apply_double_shield(img_bytes, wm_bytes):
    nparr_img = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr_img, cv2.IMREAD_COLOR)
    
    nparr_wm = np.frombuffer(wm_bytes, np.uint8)
    # Changed to IMREAD_UNCHANGED to handle transparency
    watermark = cv2.imdecode(nparr_wm, cv2.IMREAD_UNCHANGED)

    h, w = img.shape[:2]
    if h % 2 != 0: img = img[:-1, :, :]
    if w % 2 != 0: img = img[:, :-1, :]
    h, w = img.shape[:2]

    # --- LAYER 1: VISIBLE WATERMARK ---
    wm_visible_w = int(w * 0.15)
    scale_ratio = wm_visible_w / watermark.shape[1]
    wm_visible_h = int(watermark.shape[0] * scale_ratio)
    wm_visible = cv2.resize(watermark, (wm_visible_w, wm_visible_h))

    margin = 20
    start_y = h - wm_visible_h - margin
    start_x = w - wm_visible_w - margin
    
    overlay = img[start_y:h-margin, start_x:w-margin]
    if wm_visible.shape[2] == 4:
        alpha = wm_visible[:, :, 3] / 255.0
        for c in range(0, 3):
            overlay[:, :, c] = overlay[:, :, c] * (1 - alpha) + wm_visible[:, :, c] * alpha
    else:
        cv2.addWeighted(overlay, 0.7, wm_visible, 0.3, 0, overlay)
    
    img[start_y:h-margin, start_x:w-margin] = overlay

    # --- LAYER 2: INVISIBLE DCT ---
    wm_gray = cv2.cvtColor(wm_visible, cv2.COLOR_BGR2GRAY) if wm_visible.shape[2] >= 3 else wm_visible
    wm_hidden_h, wm_hidden_w = h // 8, w // 8
    wm_hidden = cv2.resize(wm_gray, (wm_hidden_w, wm_hidden_h)) / 255.0

    b_channel = np.float32(img[:, :, 0]) / 255.0
    img_dct = cv2.dct(b_channel)
    img_dct[wm_hidden_h:wm_hidden_h*2, wm_hidden_w:wm_hidden_w*2] += (wm_hidden * 0.08)
    
    img[:, :, 0] = cv2.idct(img_dct) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)

    _, buffer = cv2.imencode('.png', img)
    return io.BytesIO(buffer)

@app.route('/', methods=['GET'])
def home():
    return "LuminaMark Engine is Online", 200

@app.route('/process', methods=['POST', 'OPTIONS'])
def process_image():
    if request.method == 'OPTIONS':
        return '', 200
    try:
        if 'image' not in request.files or 'watermark' not in request.files:
            return jsonify({"error": "Missing files"}), 400
            
        img_file = request.files['image'].read()
        wm_file = request.files['watermark'].read()
        result_io = apply_double_shield(img_file, wm_file)
        result_io.seek(0)
        return send_file(result_io, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Render uses port 10000 by default
    app.run(host='0.0.0.0', port=10000)
