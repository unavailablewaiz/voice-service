import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Lazy-loaded model (important for Render free)
model = None

def get_model():
    global model
    if model is None:
        model = pipeline(
            "image-to-text",
            model="nlpconnect/vit-gpt2-image-captioning",
            device=-1  # CPU only
        )
    return model


# -------------------------
# Health check (NO 404)
# -------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "vision",
        "model_loaded": model is not None
    })


# -------------------------
# Image caption endpoint
# -------------------------
@app.route("/caption", methods=["POST"])
def caption():
    data = request.get_json(force=True)

    if not data or "image_url" not in data:
        return jsonify({"error": "image_url is required"}), 400

    pipe = get_model()
    result = pipe(data["image_url"])

    return jsonify({
        "caption": result[0]["generated_text"]
    })


# -------------------------
# Render-compatible runner
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
