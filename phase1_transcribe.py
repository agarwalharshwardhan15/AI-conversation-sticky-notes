# %% Cell 1: Import libraries
import whisper
import os

# %% Cell 2: Load the transcription model
model = whisper.load_model("base")

# %% Cell 3: Build a path that always points to the audio subfolder,
# no matter which folder you're standing in when you run it
script_folder = os.path.dirname(os.path.abspath(__file__))
audio_path = os.path.join(script_folder, "Audio", "conversation_test.mp3")
print("Looking for audio file at:", audio_path)

# %% Cell 4: Transcribe
result = model.transcribe(audio_path)

# %% Cell 5: Print the full transcript
print(result["text"])

# %% Cell 6: Print timestamped segments
for segment in result["segments"]:
    start = round(segment["start"], 1)
    end = round(segment["end"], 1)
    text = segment["text"]
    print(f"[{start}s - {end}s]  {text}")