from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch

from flask import Flask, request, jsonify
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio
import tempfile
import os

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

model.eval()

app = Flask(__name__)

@app.route("/ping", methods=["GET"])
def ping():
    return "pong", 200


@app.route("/asr", methods=["POST"])
def transcribe_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    audio_file = request.files["file"]

    # save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        audio_path = tmp.name
        audio_file.save(audio_path)

    try:
        # load and convert to 16kHz mono
        waveform, sample_rate = torchaudio.load(audio_path)

        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
            sample_rate = 16000

        # get duration
        duration_sec = waveform.shape[1] / sample_rate

        # get inputs
        inputs = processor(waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt", padding=True)

        # perform inference
        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])

        return jsonify({
            "transcription": transcription,
            "duration": f"{duration_sec:.1f}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(audio_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)