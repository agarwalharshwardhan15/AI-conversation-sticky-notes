import streamlit as st
import whisper
import os
import re
import tempfile
from transformers import pipeline

# --- Page setup ---
st.set_page_config(page_title="AI Conversation Assistant", layout="centered")
st.title("🎙️ AI Conversation Assistant")
st.write("Upload a conversation recording, set an agenda, and get filtered sticky notes.")

# --- Load models once, cached so it doesn't reload on every interaction ---
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

whisper_model = load_whisper_model()
classifier = load_classifier()

# --- Category setup ---
category_labels = {
    "an idea or suggestion for improvement": "Idea",
    "a problem or issue being reported": "Problem",
    "a task or action being assigned to someone": "Action Item",
    "a decision that has been made or agreed on": "Decision",
    "a genuine question seeking information": "Question"
}
candidate_labels = list(category_labels.keys())

icons = {
    "Idea": "💡", "Problem": "❗", "Action Item": "🎯",
    "Decision": "📌", "Question": "❓"
}
colors = {
    "Idea": "#FFF9C4", "Problem": "#FFCDD2", "Action Item": "#BBDEFB",
    "Decision": "#C8E6C9", "Question": "#E1BEE7"
}

def is_relevant(text, keywords):
    words_in_text = set(re.findall(r"\b[a-z']+\b", text.lower()))
    matched = [kw for kw in keywords if kw in words_in_text]
    return len(matched) > 0, matched

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

# --- UI inputs ---
agenda = st.text_input("📌 Agenda / Topic", value="Website Performance Improvement")
keywords_input = st.text_area(
    "🔑 Agenda Keywords (comma-separated)",
    value="website, site, performance, loading, load, bug, feature, database, api, deployment, server, image, images, optimize, optimization, speed, slow, fast, query, queries, code, test, testing, cloudflare, cdn, homepage"
)
uploaded_file = st.file_uploader("🎙️ Upload conversation audio", type=["mp3", "wav", "m4a"])

process_button = st.button("PROCESS")

# --- Processing ---
if process_button:
    if uploaded_file is None:
        st.error("Please upload an audio file first.")
    else:
        agenda_keywords = [kw.strip().lower() for kw in keywords_input.split(",") if kw.strip()]

        # Save uploaded file to a temp location so Whisper can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        with st.spinner("Transcribing audio..."):
            result = whisper_model.transcribe(tmp_path)

        os.remove(tmp_path)

        sticky_notes = []
        filtered_out = []

        with st.spinner("Filtering and categorizing..."):
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

        # --- Display results ---
        st.subheader("📌 Important Points")
        if sticky_notes:
            for text, category, source in sticky_notes:
                color = colors.get(category, "#EEEEEE")
                icon = icons.get(category, "📝")
                st.markdown(
                    f"""
                    <div style="background-color:{color}; padding:12px; border-radius:10px; margin-bottom:10px;">
                        <b>{icon} {category.upper()}</b> <span style="font-size:12px; color:#555;">({source})</span><br>
                        {text}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("No relevant content found for this agenda.")

        st.subheader("🗑️ Filtered Content (small talk / off-topic)")
        for text in filtered_out:
            st.markdown(f"~~{text}~~")