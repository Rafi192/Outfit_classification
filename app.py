from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import os
import io


app = Flask(__name__)

# Loading my YOLOv11 trained  model
model_path = os.path.join(os.path.dirname(__file__), 'models', 'best.pt')
model = YOLO(model_path)
print("Model loaded successfully.", model_path)
print("Model details:", model)

@app.route('/api/classify_outfit', methods=['GET','POST'])
def classify_outfit():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read())).convert("RGB")

    # Predciton using YOLO model
    results = model(image)
    probs = results[0].probs  # class probabilities

    probs = results[0].probs  
    if probs is None:
        return jsonify({"error": "Model did not return classification probabilities"}), 500


    top_class_id = probs.top1
    top_class_name = results[0].names[top_class_id]
    confidence = float(probs.top1conf)


    return jsonify({
        "status": "success",
        "top_prediction": {
            "class": top_class_name,
            "confidence": round(confidence, 3)
        }
    }), 200



if __name__ == '__main__':
    app.run(debug=True)



