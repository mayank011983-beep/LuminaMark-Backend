import cv2
import numpy as np
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import io

app = Flask(__name__)

# --- PRODUCTION CORS SETUP ---
# This allows your Netlify frontend to talk to this Render backend
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

def embed_deep_burn_dct(img_bytes, wm_bytes, strength=0.08):
    # 1. Decode the uploaded files into image matrices
    nparr_img = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr_img, cv2.IMREAD_COLOR)
    
    nparr_wm = np.frombuffer(wm_bytes, np.uint8)
    watermark = cv2.imdecode(nparr_wm, cv2.IMREAD_GRAYSCALE)

    # 2. Safety Check: DCT requires EVEN pixel dimensions
    h, w = img.shape[:2]
    if h % 2 != 0: img = img[:-1, :, :]
    if w % 2 != 0: img = img[:, :-1, :]
    h, w = img.shape[:2]

    # 3. Prepare the Watermark
    # We resize it to a fixed ratio so the decoder always knows where it is
    wm_h, wm_w = h // 8, w // 8
    watermark = cv2.resize(watermark, (wm_w, wm_h))
    watermark_norm = watermark / 255.0 
    
    # 4. Extract the Blue Channel (Least visible to human eye)
    b_channel = np.float32(img[:, :, 0]) / 255.0
    img_dct = cv2.dct(b_channel)
    
    # 5. DEEP BURN EMBEDDING
    # Targeting Mid-Frequencies to survive AI compression and resizing
    offset_y, offset_x = wm_h, wm_w 
    
    # Add mathematical noise to break AI "Remove" algorithms
    noise = np.random.normal(0, 0.01, (wm_h, wm_w))
    
    # Apply the watermark and noise to the frequency map
    img_dct[offset_y:offset_y+wm_h, offset_x:offset_x+wm_w] += (watermark_norm + noise) * strength
    
    # 6. Rebuild the Image (Inverse DCT)
    b_inv = cv2.idct(img_dct) * 255.0
    b_inv = np.clip(b_inv, 0, 255)
    img[:, :, 0] = b_inv.astype(np.uint8)
    
    # 7. Convert to PNG (Lossless) to keep the hidden data safe
    _, buffer = cv2.imencode('.png', img)
    return io.BytesIO(buffer)

@app.route('/', methods=['GET'])
def health_check():
    return "LuminaMark Engine is Online", 200

@app.route('/process', methods=['POST', 'OPTIONS'])
def process_image():
    # Handle the browser's "pre-flight" security check
    if request.method == 'OPTIONS':
        return '', 200

    if 'image' not in request.files or 'watermark' not in request.files:
        return jsonify({"error": "Missing image or watermark file"}), 400
    
    try:
        img_file = request.files['image'].read()
        wm_file = request.files['watermark'].read()
        
        result_io = embed_deep_burn_dct(img_file, wm_file)
        result_io.seek(0)
        
        return send_file(
            result_io, 
            mimetype='image/png', 
            as_attachment=True, 
            download_name="LuminaMark_Protected.png"
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Default port for local testing; Render will override this automatically
    app.run(host='0.0.0.0', port=5000)
