# %% Cell 1: Imports
import whisper
import os
import re
from transformers import pipeline

# %% Cell 2: Load Whisper and transcribe
model = whisper.load_model("base")
script_folder = os.path.dirname(os.path.abspath(__file__))
audio_path = os.path.join(script_folder, "Audio", "conversation_test.mp3")
result = model.transcribe(audio_path)

# %% Cell 3: Agenda keywords
agenda = "Website Performance Improvement"
agenda_keywords = [
    "website", "site", "performance", "loading", "load",
    "bug", "feature", "database", "api", "deployment", "server",
    "image", "images", "optimize", "optimization", "speed", "slow", "fast",
    "query", "queries", "code", "test", "testing", "cloudflare", "cdn", "homepage"
]

def is_relevant(text, keywords):
    words_in_text = set(re.findall(r"\b[a-z']+\b", text.lower()))
    matched = [kw for kw in keywords if kw in words_in_text]
    return len(matched) > 0, matched

# %% Cell 4: Load the zero-shot classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# %% Cell 5: Define categories with richer descriptions for better accuracy
category_labels = {
    "an idea or suggestion for improvement": "Idea",
    "a problem or issue being reported": "Problem",
    "a task or action being assigned to someone": "Action Item",
    "a decision that has been made or agreed on": "Decision",
    "a genuine question seeking information": "Question"
}
candidate_labels = list(category_labels.keys())

# %% Cell 6: Rule-based override for patterns zero-shot commonly misses
def rule_based_category(text):
    text_lower = text.lower()

    if re.match(r"^(should we|could we|do we|is it|are we|what if|how do|why do|when do)\b", text_lower):
        return "Question"

    if re.search(r"\bcan you\b.*\b(by|before|today|tomorrow|friday|monday|tuesday|wednesday|thursday|saturday|sunday)\b", text_lower):
        return "Action Item"

    if re.search(r"\b(let's decide|we will|we've decided|agreed|decided to)\b", text_lower):
        return "Decision"

    if re.search(r"\b(maybe we should|what if|we could|i suggest|how about)\b", text_lower):
        return "Idea"

    return None

# %% Cell 7: Filter, then categorize
sticky_notes = []
filtered_out = []

for segment in result["segments"]:
    text = segment["text"].strip()
    relevant, matched_keywords = is_relevant(text, agenda_keywords)

    if relevant:
        rule_result = rule_based_category(text)
        if rule_result:
            sticky_notes.append((text, rule_result, "rule-based"))
        else:
            classification = classifier(
                text, candidate_labels,
                hypothesis_template="This sentence is best described as {}."
            )
            best_label = classification["labels"][0]
            confidence = round(classification["scores"][0] * 100, 1)
            best_category = category_labels[best_label]
            sticky_notes.append((text, best_category, f"{confidence}% AI confidence"))
    else:
        filtered_out.append(text)

# %% Cell 8: Print sticky notes
icons = {
    "Idea": "💡", "Problem": "❗", "Action Item": "🎯",
    "Decision": "📌", "Question": "❓"
}

print("=" * 60)
print(f"AGENDA: {agenda}")
print("=" * 60)
print("\nSTICKY NOTES:\n")
for text, category, source in sticky_notes:
    icon = icons.get(category, "📝")
    print(f"{icon} {category.upper()}  ({source})")
    print(f'   "{text}"\n')

print("=" * 60)
print("FILTERED OUT (small talk / off-topic):")
for text in filtered_out:
    print(f"  ❌ {text}")