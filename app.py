import torch
from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
import cv2
from ultralytics import YOLO

# This line tells PyTorch it is safe to load the YOLO model's math structures
# It fixes the "Weights only load failed" error on newer versions of PyTorch
torch.serialization.add_safe_globals([
    np.core.multiarray._reconstruct, 
    np.ndarray, 
    np.dtype, 
    np.core.multiarray.scalar,
    np.float64,
    np.int64
])

app = Flask(__name__)

# Load the super-fast YOLOv8 Nano model
# Set weights_only=False to allow the initial download to load correctly
model = YOLO('yolov8n.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data'}), 400

    # 1. Decode the image from the frontend
    try:
        image_data = data['image'].split(',')[1]
        decoded_data = base64.b64decode(image_data)
        np_data = np.frombuffer(decoded_data, np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'detections': []})

        # 2. Run YOLOv8 inference
        results = model(img, verbose=False)
        
        # 3. Extract all detected objects
        predictions = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                cls_name = model.names[cls_id]
                
                # Confidence threshold
                if conf > 0.4:
                    predictions.append({
                        "class": cls_name,
                        "confidence": conf,
                        "bbox": [x1, y1, x2 - x1, y2 - y1]
                    })

        return jsonify({'detections': predictions})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Render uses the PORT environment variable, so 5000 is fine as a default
    app.run(host='0.0.0.0', port=5000)