import streamlit as st
import whisper
import os
import re
import tempfile
from transformers import pipeline

# --- Page setup ---
st.set_page_config(page_title="AI Conversation Assistant", page_icon="🎙️", layout="wide")

# --- Load models once, cached ---
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# --- Category setup ---
category_labels = {
    "an idea or suggestion for improvement": "Idea",
    "a problem or issue being reported": "Problem",
    "a task or action being assigned to someone": "Action Item",
    "a decision that has been made or agreed on": "Decision",
    "a genuine question seeking information": "Question"
}
candidate_labels = list(category_labels.keys())
category_order = ["Problem", "Idea", "Question", "Action Item", "Decision"]

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

# --- Sidebar: inputs ---
with st.sidebar:
    st.title("🎙️ AI Conversation Assistant")
    st.caption("Upload a conversation, set an agenda, get filtered sticky notes.")

    st.markdown("### 📌 Agenda / Topic")
    agenda = st.text_input("Agenda", value="Website Performance Improvement", label_visibility="collapsed")

    st.markdown("### 🔑 Agenda Keywords")
    keywords_input = st.text_area(
        "Keywords", label_visibility="collapsed", height=110,
        value="website, site, performance, loading, load, bug, feature, database, api, deployment, server, image, images, optimize, optimization, speed, slow, fast, query, queries, code, test, testing, cloudflare, cdn, homepage"
    )

    st.markdown("### 🎙️ Upload Audio")
    uploaded_file = st.file_uploader("Audio", type=["mp3", "wav", "m4a"], label_visibility="collapsed")

    col1, col2 = st.columns(2)
    with col1:
        process_button = st.button("PROCESS", use_container_width=True, type="primary")
    with col2:
        example_button = st.button("Try Example", use_container_width=True)

    st.markdown("---")
    st.markdown("#### Legend")
    for cat in category_order:
        st.markdown(f"{icons[cat]} **{cat}**")

# --- Main area ---
st.markdown("## Important Points")

def run_pipeline(audio_path_or_bytes, is_temp_file, agenda_keywords):
    whisper_model = load_whisper_model()
    classifier = load_classifier()

    with st.spinner("Transcribing audio..."):
        result = whisper_model.transcribe(audio_path_or_bytes)

    if is_temp_file:
        os.remove(audio_path_or_bytes)

    sticky_notes = []
    filtered_out = []

    with st.spinner("Filtering and categorizing..."):
        for segment in result["segments"]:
            text = segment["text"].strip()
            relevant, matched_keywords = is_relevant(text, agenda_keywords)

            if relevant:
                rule_result = rule_based_category(text)
                if rule_result:
                    sticky_notes.append((text, rule_result, "Rule-based", 100))
                else:
                    classification = classifier(
                        text, candidate_labels,
                        hypothesis_template="This sentence is best described as {}."
                    )
                    best_label = classification["labels"][0]
                    confidence = round(classification["scores"][0] * 100, 1)
                    best_category = category_labels[best_label]
                    sticky_notes.append((text, best_category, "AI classifier", confidence))
            else:
                filtered_out.append(text)

    return sticky_notes, filtered_out, result

# Determine which audio source to use
audio_source = None
is_temp = False

if example_button:
    example_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Audio", "conversation_test.mp3")
    if os.path.exists(example_path):
        audio_source = example_path
        is_temp = False
    else:
        st.error("Example file not found. Make sure Audio/conversation_test.mp3 exists.")

elif process_button:
    if uploaded_file is None:
        st.error("Please upload an audio file first.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(uploaded_file.read())
            audio_source = tmp_file.name
        is_temp = True

if audio_source:
    agenda_keywords = [kw.strip().lower() for kw in keywords_input.split(",") if kw.strip()]
    sticky_notes, filtered_out, result = run_pipeline(audio_source, is_temp, agenda_keywords)

    total_segments = len(sticky_notes) + len(filtered_out)
    relevant_pct = round((len(sticky_notes) / total_segments) * 100, 1) if total_segments else 0

    # --- Metrics row ---
    m1, m2, m3 = st.columns(3)
    m1.metric("Segments Processed", total_segments)
    m2.metric("Relevant Points Found", len(sticky_notes))
    m3.metric("Filtered as Noise", f"{len(filtered_out)} ({100 - relevant_pct:.0f}%)")

    st.markdown("---")

    if sticky_notes:
        # --- Group sticky notes by category ---
        grouped = {cat: [] for cat in category_order}
        for text, category, source, confidence in sticky_notes:
            grouped.setdefault(category, []).append((text, source, confidence))

        for cat in category_order:
            items = grouped.get(cat, [])
            if not items:
                continue
            st.markdown(f"### {icons[cat]} {cat}")
            for text, source, confidence in items:
                bar_color = "#4CAF50" if confidence >= 70 else "#FFA726" if confidence >= 50 else "#EF5350"
                st.markdown(
                    f"""
                    <div style="background-color:{colors.get(cat, '#EEEEEE')}; padding:14px 16px; border-radius:12px;
                                margin-bottom:12px; box-shadow: 0 1px 4px rgba(0,0,0,0.15);">
                        <div style="color:#1a1a1a; font-size:15px;">{text}</div>
                        <div style="margin-top:8px; display:flex; align-items:center; gap:8px;">
                            <span style="font-size:11px; color:#444;">{source}{f' · {confidence}%' if source == 'AI classifier' else ''}</span>
                            <div style="flex:1; background:rgba(0,0,0,0.1); border-radius:4px; height:5px; max-width:120px;">
                                <div style="width:{confidence}%; background:{bar_color}; height:5px; border-radius:4px;"></div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.info("No relevant content found for this agenda.")

    # --- Download button ---
    if sticky_notes:
        export_lines = [f"AGENDA: {agenda}", "=" * 50, "", "STICKY NOTES:", ""]
        for text, category, source, confidence in sticky_notes:
            export_lines.append(f"{icons[category]} {category.upper()} ({source})")
            export_lines.append(f'  "{text}"')
            export_lines.append("")
        export_lines.append("FILTERED OUT:")
        for text in filtered_out:
            export_lines.append(f"  - {text}")
        export_text = "\n".join(export_lines)

        st.download_button(
            "📥 Download Sticky Notes (.txt)",
            data=export_text,
            file_name="sticky_notes_output.txt",
            mime="text/plain"
        )

    # --- Filtered content ---
    st.markdown("### 🗑️ Filtered Content (small talk / off-topic)")
    for text in filtered_out:
        st.markdown(f"<span style='color:#888;'>~~{text}~~</span>", unsafe_allow_html=True)

    # --- Full transcript, collapsed ---
    with st.expander("📄 Show full raw transcript"):
        st.write(result["text"])

else:
    st.info("Upload a conversation recording (or click **Try Example**) and press **PROCESS** to get started.")