from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load the super-fast YOLOv8 Nano model
# (It will download a small ~6MB file the first time you run this)
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
    image_data = data['image'].split(',')[1]
    decoded_data = base64.b64decode(image_data)
    np_data = np.frombuffer(decoded_data, np.uint8)
    img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

    # 2. Run YOLOv8 inference (verbose=False keeps your terminal clean)
    results = model(img, verbose=False)
    
    # 3. Extract all detected objects
    predictions = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # Get confidence score and class name
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            cls_name = model.names[cls_id]
            
            # Only include objects with > 40% confidence to reduce noise
            if conf > 0.4:
                predictions.append({
                    "class": cls_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2 - x1, y2 - y1] # x, y, width, height
                })

    return jsonify({'detections': predictions})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)