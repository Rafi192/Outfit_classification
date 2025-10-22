from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import os
import io


app = Flask(__name__)

# Loading my YOLOv11 trained  model
model_path = os.path.join(os.path.dirname(__file__), 'models', 'best.pt')
model = YOLO(model_path, device='cpu')

# print("Model loaded successfully.", modelgit st_path)
# print("Model details:", model)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "API is running! Use POST /api/classify_outfit to classify images."})

@app.route('/api/classify_outfit', methods=['POST'])
def classify_outfit():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['image']
    try:
        image = Image.open(io.BytesIO(file.read()))
    except Exception as e:
        return jsonify({"error": f"Cannot read image: {str(e)}"}), 400

    try:
        results = model.predict(image, verbose=False)  # safe for classification
        probs = results[0].probs

        if probs is None:
            return jsonify({"error": "Model did not return probabilities"}), 500

        top_class_id = probs.top1
        top_class_name = results[0].names[top_class_id]
        confidence = float(probs.top1conf)

        return jsonify({
            "status": "success",
            "top_prediction": {
                "class": top_class_name,
                "confidence": round(confidence, 3)
            }
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
