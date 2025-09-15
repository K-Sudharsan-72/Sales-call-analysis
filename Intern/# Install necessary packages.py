# Install necessary packages
import yt_dlp
import whisper
from pydub import AudioSegment
import torch
from transformers import pipeline
import re
import numpy as np

# URL of sales call
url = "https://www.youtube.com/watch?v=4ostqJD3Psc"

# Download audio
ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': 'call.%(ext)s',
    'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav', 'preferredquality': '192'}],
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])

audio_path = "call.wav"

# Load tiny/ base model (fast enough for Colab free)
model = whisper.load_model("base")

# Transcribe with word timestamps
result = model.transcribe(audio_path, word_timestamps=True)

# Store segments
segments = result["segments"]

# Build transcript with speaker turns
transcript = []
current_speaker = "SalesRep"

for seg in segments:
    text = seg["text"].strip()
    start = seg["start"]
    end = seg["end"]
    duration = end - start

    transcript.append({
        "speaker": current_speaker,
        "text": text,
        "start": start,
        "end": end,
        "duration": duration
    })

    # Alternate speakers (rough diarization)
    current_speaker = "Customer" if current_speaker == "SalesRep" else "SalesRep"

durations = {}
for seg in transcript:
    durations[seg["speaker"]] = durations.get(seg["speaker"], 0) + seg["duration"]

total_time = sum(durations.values())
talk_ratio = {k: round((v/total_time)*100, 2) for k,v in durations.items()}

question_count = sum([seg["text"].count("?") for seg in transcript])

longest_monologue = max([seg["duration"] for seg in transcript])

sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

all_text = " ".join([seg["text"] for seg in transcript])
sentiment = sentiment_analyzer(all_text[:4000])[0]  # keep under token limit

insight = ""
if talk_ratio.get("SalesRep",0) > 70:
    insight = "Sales rep dominated the conversation. Encourage more customer engagement."
elif question_count < 3:
    insight = "Few questions were asked. Train sales rep to ask more discovery questions."
else:
    insight = "Good balance of talk-time and engagement detected."

print("🔹 Talk-time ratio:", talk_ratio)
print("🔹 Number of questions:", question_count)
print("🔹 Longest monologue duration (sec):", round(longest_monologue,2))
print("🔹 Call sentiment:", sentiment)
print("🔹 Insight:", insight)
