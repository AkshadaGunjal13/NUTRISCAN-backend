from flask import Flask, request, jsonify
from ocr_utils import extract_text_from_image
from ml_model import predict_safety
from flask_cors import CORS
import os
import firebase_admin
from firebase_admin import credentials, firestore

app = Flask(__name__)
CORS(app)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Firebase Admin
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/analyze', methods=['POST'])
def analyze_food():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    try:
        file.save(file_path)
        if os.path.getsize(file_path) == 0:
            return jsonify({"error": "Uploaded file is empty"}), 400

        # Extract text using OCR
        ingredients_text = extract_text_from_image(file_path)

        # Optional: get user UID for profile-based checks
        uid = request.form.get('uid')
        user_data = {}
        if uid:
            doc = db.collection("users").document(uid).get()
            if doc.exists:
                user_data = doc.to_dict()

        # Predict safety using ML model
        health_result = predict_safety(ingredients_text)

        # Personalized alert logic
        alert_messages = []

        # Diet preference alert
        diet = user_data.get("diet", "").lower()
        non_veg_keywords = ["chicken", "meat", "fish", "egg", "pork", "beef"]
        if diet == "vegetarian" and any(k in ingredients_text.lower() for k in non_veg_keywords):
            alert_messages.append("Contains non-vegetarian ingredients — not suitable for your diet.")

        # Allergies alert
        allergies = user_data.get("allergies", "").lower().split(",")
        for allergy in allergies:
            if allergy.strip() and allergy.strip() in ingredients_text.lower():
                alert_messages.append(f"Contains allergen: {allergy.strip()}.")

        # Health condition alerts (example: diabetes, hypertension)
        conditions = user_data.get("conditions", "").lower().split(",")
        if "diabetes" in conditions and "sugar" in ingredients_text.lower():
            alert_messages.append("High sugar content — not recommended for diabetes.")

        # Return response
        return jsonify({
            "ingredients": ingredients_text,
            "health_result": health_result,
            "alert": " | ".join(alert_messages) if alert_messages else "No issues detected"
        })

    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
