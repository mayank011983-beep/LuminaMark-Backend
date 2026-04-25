import cv2
import numpy as np
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import io

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

def apply_double_shield(img_bytes, wm_bytes):
    # 1. Decode images
    nparr_img = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr_img, cv2.IMREAD_COLOR)
    
    nparr_wm = np.frombuffer(wm_bytes, np.uint8)
    watermark = cv2.imdecode(nparr_wm, cv2.IMREAD_UNCHANGED)

    if img is None or watermark is None:
        raise ValueError("Could not decode images. Check file formats.")

    # 2. Force even dimensions for DCT math
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
    
    # Safety check for coordinates
    if start_y < 0 or start_x < 0:
        start_y, start_x = 0, 0

    overlay = img[start_y:start_y+wm_visible_h, start_x:start_x+wm_visible_w]
    
    # Flexible Overlay Logic (Fixes the Tuple Error)
    if len(wm_visible.shape) == 3 and wm_visible.shape[2] == 4:
        # Logo has Alpha channel (Transparency)
        alpha = wm_visible[:, :, 3] / 255.0
        for c in range(0, 3):
            overlay[:, :, c] = overlay[:, :, c] * (1 - alpha) + wm_visible[:, :, c] * alpha
    else:
        # Standard blending for JPG or 3-channel PNG
        wm_to_blend = wm_visible[:, :, :3] if len(wm_visible.shape) == 3 else cv2.cvtColor(wm_visible, cv2.COLOR_GRAY2BGR)
        cv2.addWeighted(overlay, 0.7, wm_to_blend, 0.3, 0, overlay)
    
    img[start_y:start_y+wm_visible_h, start_x:start_x+wm_visible_w] = overlay

    # --- LAYER 2: INVISIBLE DCT ---
    # Convert watermark to grayscale for hidden math
    if len(wm_visible.shape) == 3:
        wm_gray = cv2.cvtColor(wm_visible, cv2.COLOR_BGR2GRAY)
    elif len(wm_visible.shape) == 2:
        wm_gray = wm_visible
    else: # 4 channels
        wm_gray = cv2.cvtColor(wm_visible[:,:,:3], cv2.COLOR_BGR2GRAY)

    wm_hidden_h, wm_hidden_w = h // 8, w // 8
    wm_hidden = cv2.resize(wm_gray, (wm_hidden_w, wm_hidden_h)) / 255.0

    # Process Blue channel
    b_channel = np.float32(img[:, :, 0]) / 255.0
    img_dct = cv2.dct(b_channel)
    
    # Embed hidden logo in mid-frequencies
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
            return jsonify({"error": "Please upload both house photo and logo"}), 400
            
        img_file = request.files['image'].read()
        wm_file = request.files['watermark'].read()
        
        result_io = apply_double_shield(img_file, wm_file)
        result_io.seek(0)
        return send_file(result_io, mimetype='image/png')
    except Exception as e:
        # This will now catch and report the specific Python error
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
