import torch
from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
import cv2
from ultralytics import YOLO
import os
import gc

app = Flask(__name__)

# Load the smallest model
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

        # Decode image
        image_str = data['image'].split(',')[1]
        decoded = base64.b64decode(image_str)
        nparr = np.frombuffer(decoded, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'detections': []})

        # Run inference
        # imgsz=640 is standard, but we use a compromise for speed vs accuracy
        results = model.predict(img, imgsz=480, conf=0.25, verbose=False)
        
        predictions = []
        for r in results:
            for box in r.boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = model.names[cls]
                
                # Normalize names for better UI display
                display_name = name.replace('_', ' ')
                
                predictions.append({
                    "class": display_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2 - x1, y2 - y1] # [x, y, w, h]
                })

        gc.collect() 
        return jsonify({'detections': predictions})

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)