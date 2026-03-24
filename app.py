import torch
from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
import cv2
from ultralytics import YOLO
import os
import gc

app = Flask(__name__)

# Load the smallest model (YOLOv8 Nano)
model = YOLO('yolov8n.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'detections': []})

        # Decode the base64 image from the browser
        image_str = data['image'].split(',')[1]
        decoded = base64.b64decode(image_str)
        nparr = np.frombuffer(decoded, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'detections': []})

        # Inference: imgsz=320 is fast and light on RAM
        results = model.predict(img, imgsz=320, conf=0.4, verbose=False)
        
        predictions = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                predictions.append({
                    "class": model.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": [x1, y1, x2 - x1, y2 - y1]
                })

        gc.collect() # Clean up memory
        return jsonify({'detections': predictions})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Use port 5000 for local testing
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)