import os

# Add ffmpeg to PATH for the current script execution
os.environ["PATH"] += r";C:\Users\User\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin"

import yt_dlp
import whisper
from pydub import AudioSegment
import torch
from transformers import pipeline
import re
import numpy as np

# URL of sales call
url = "https://www.youtube.com/watch?v=4ostqJD3Psc"

# yt-dlp options with explicit ffmpeg location
ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': 'call.%(ext)s',
    'ffmpeg_location': r"C:\Users\User\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin\ffmpeg.exe",
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        'preferredquality': '192',
    }],
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])

audio_path = "call.wav"

# Load Whisper model
model = whisper.load_model("base")

# Transcribe audio with word timestamps
result = model.transcribe(audio_path, word_timestamps=True)

segments = result["segments"]

# Build transcript with alternating speakers
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

    # Alternate speakers
    current_speaker = "Customer" if current_speaker == "SalesRep" else "SalesRep"

# Calculate talk durations
durations = {}
for seg in transcript:
    durations[seg["speaker"]] = durations.get(seg["speaker"], 0) + seg["duration"]

total_time = sum(durations.values())
talk_ratio = {k: round((v / total_time) * 100, 2) for k, v in durations.items()}

# Count number of questions
question_count = sum([seg["text"].count("?") for seg in transcript])

# Find longest monologue
longest_monologue = max([seg["duration"] for seg in transcript])

# Sentiment analysis
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

all_text = " ".join([seg["text"] for seg in transcript])
sentiment = sentiment_analyzer(all_text[:4000])[0]

# Generate insight
insight = ""
if talk_ratio.get("SalesRep", 0) > 70:
    insight = "Sales rep dominated the conversation. Encourage more customer engagement."
elif question_count < 3:
    insight = "Few questions were asked. Train sales rep to ask more discovery questions."
else:
    insight = "Good balance of talk-time and engagement detected."

# Print results
print("🔹 Talk-time ratio:", talk_ratio)
print("🔹 Number of questions:", question_count)
print("🔹 Longest monologue duration (sec):", round(longest_monologue, 2))
print("🔹 Call sentiment:", sentiment)
print("🔹 Insight:", insight)
