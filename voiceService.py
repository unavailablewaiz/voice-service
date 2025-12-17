import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

pipe = None

def get_model():
    global pipe
    if pipe is None:
        pipe = pipeline(
            "image-to-text",
            model="nlpconnect/vit-gpt2-image-captioning",
            device=-1
        )
    return pipe

@app.route("/caption", methods=["POST"])
def caption():
    data = request.json
    image_url = data["image_url"]

    pipe = get_model()
    result = pipe(image_url)

    return jsonify({
        "caption": result[0]["generated_text"]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
