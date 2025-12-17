from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

pipe = pipeline(
    "image-text-to-text",
    model="HuggingFaceTB/SmolVLM-256M-Instruct",
    device="cpu"
)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": data["image_url"]},
                {"type": "text", "text": data["question"]}
            ]
        }
    ]

    result = pipe(text=messages)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
