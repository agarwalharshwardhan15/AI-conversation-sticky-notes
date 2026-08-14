# %% Cell 1: Imports
import whisper
import os
import re

# %% Cell 2: Load Whisper and transcribe
model = whisper.load_model("base")
script_folder = os.path.dirname(os.path.abspath(__file__))
audio_path = os.path.join(script_folder, "Audio", "conversation_test.mp3")
print("Looking for audio file at:", audio_path)
result = model.transcribe(audio_path)

# %% Cell 3: Define your agenda and its related keywords
agenda = "Website Performance Improvement"
agenda_keywords = [
    "website", "site", "performance", "loading", "load",
    "bug", "feature", "database", "api", "deployment", "server",
    "image", "images", "optimize", "optimization", "speed", "slow", "fast",
    "query", "queries", "code", "test", "testing", "cloudflare", "cdn", "homepage"
]

# %% Cell 4: A relevance-checking function that matches WHOLE WORDS only
def is_relevant(text, keywords):
    words_in_text = set(re.findall(r"\b[a-z']+\b", text.lower()))
    matched = [kw for kw in keywords if kw in words_in_text]
    return len(matched) > 0, matched

# %% Cell 5: Run every segment through the filter
relevant_segments = []
irrelevant_segments = []

for segment in result["segments"]:
    text = segment["text"].strip()
    relevant, matched_keywords = is_relevant(text, agenda_keywords)
    if relevant:
        relevant_segments.append((text, matched_keywords))
    else:
        irrelevant_segments.append(text)

# %% Cell 6: Print results
print("=" * 50)
print(f"AGENDA: {agenda}")
print("=" * 50)

print("\n✅ RELEVANT SEGMENTS:")
for text, matched in relevant_segments:
    print(f"  • {text}")
    print(f"    (matched keywords: {matched})")

print("\n❌ FILTERED OUT (irrelevant):")
for text in irrelevant_segments:
    print(f"  • {text}")