import os
import pandas as pd
import requests
from tqdm import tqdm

# Constants
API_URL = "http://localhost:8001/asr"
AUDIO_DIR = "../data/common_voice/cv-valid-dev"
INPUT_CSV = "../data/common_voice/cv-valid-dev.csv"
CSV_PATH = "./cv-valid-dev.csv"

# Load CSV
df = pd.read_csv(CSV_PATH)

# Add a new column for the generated text
generated = []

print(f"Processing {len(df)} files...")

# Loop through each row
for idx, row in tqdm(df.iterrows(), total=len(df)):
    filename = row["filename"]
    audio_path = os.path.join(AUDIO_DIR, filename)

    try:
        with open(audio_path, "rb") as f:
            response = requests.post(API_URL, files={"file": f})

        if response.status_code == 200:
            transcription = response.json().get("transcription", "")
        else:
            transcription = f"[ERROR: Bad response {response.status_code}]"
    except Exception as e:
        transcription = f"[ERROR: {str(e)}]"

    generated.append(transcription)

# Add to DataFrame and save
df["generated_text"] = generated
df.to_csv(CSV_PATH, index=False)

print(f"\nDone. Transcriptions saved to: {CSV_PATH}")
