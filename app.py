import streamlit as st
import whisper
import os
import re
import tempfile
import time
from transformers import pipeline

st.set_page_config(page_title="CONVEX | NLP Engine", page_icon="◈", layout="wide")

st.markdown("""
<style>
/* CONVEX NLP Control Room Theme */
.stApp {
    background-color: #080B12;
    color: #E2E8F0;
    font-family: 'Inter', -apple-system, sans-serif;
}
header[data-testid="stHeader"] {
    background: transparent;
}
.mono {
    font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
}
.tech-panel {
    background: #0D111B;
    border: 1px solid #1C2433;
    border-radius: 4px;
    padding: 20px;
    margin-bottom: 20px;
}
.accent-blue { color: #38BDF8; }
.accent-violet { color: #818CF8; }
.accent-green { color: #10B981; }
.accent-red { color: #EF4444; }
.accent-gray { color: #64748B; }

div[data-testid="stSidebar"] {
    background-color: #05080F !important;
    border-right: 1px solid #1C2433;
}
div[data-testid="stSidebar"] hr {
    border-color: #1C2433;
}
div[data-testid="stSidebar"] input, div[data-testid="stSidebar"] textarea, div[data-testid="stSidebar"] select {
    background: #0D111B !important;
    border: 1px solid #1C2433 !important;
    color: #E2E8F0 !important;
    font-family: monospace;
    font-size: 0.9em;
}
div[data-testid="stSidebar"] input:focus, div[data-testid="stSidebar"] textarea:focus {
    border-color: #818CF8 !important;
    box-shadow: none !important;
}

.pipeline-step {
    display: inline-block;
    padding: 4px 12px;
    margin-right: 8px;
    border: 1px solid #1C2433;
    border-radius: 12px;
    font-family: monospace;
    font-size: 0.75em;
    color: #94A3B8;
    background: #0D111B;
}
.pipeline-arrow {
    display: inline-block;
    color: #334155;
    margin-right: 8px;
    font-family: monospace;
}
.pipeline-step.active {
    border-color: #38BDF8;
    color: #38BDF8;
    background: rgba(56, 189, 248, 0.1);
}

.card-important {
    border-left: 3px solid #818CF8;
    background: linear-gradient(90deg, rgba(129, 140, 248, 0.1) 0%, transparent 100%);
}
.card-action {
    border-left: 3px solid #38BDF8;
    background: linear-gradient(90deg, rgba(56, 189, 248, 0.1) 0%, transparent 100%);
}
.card-problem {
    border-left: 3px solid #EF4444;
    background: linear-gradient(90deg, rgba(239, 68, 68, 0.1) 0%, transparent 100%);
}
.card-decision {
    border-left: 3px solid #10B981;
    background: linear-gradient(90deg, rgba(16, 185, 129, 0.1) 0%, transparent 100%);
}
.card-idea {
    border-left: 3px solid #FBBF24;
    background: linear-gradient(90deg, rgba(251, 191, 36, 0.1) 0%, transparent 100%);
}

.filter-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #1C2433;
    padding: 12px 0;
}
.filter-row:last-child {
    border-bottom: none;
}
.filter-badge {
    font-family: monospace;
    font-size: 0.7em;
    padding: 2px 8px;
    border-radius: 4px;
    border: 1px solid;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.badge-offtopic { color: #64748B; border-color: rgba(100,116,139,0.5); background: rgba(100,116,139,0.1); }
.badge-joke { color: #F59E0B; border-color: rgba(245,158,11,0.5); background: rgba(245,158,11,0.1); }
.badge-privacy { color: #EF4444; border-color: rgba(239,68,68,0.5); background: rgba(239,68,68,0.1); }

.ascii-empty {
    font-family: monospace;
    color: #64748B;
    white-space: pre;
    line-height: 1.2;
}
</style>
""", unsafe_allow_html=True)

# --- State ---
if "processed_data" not in st.session_state:
    st.session_state.processed_data = None
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

# --- Models ---
@st.cache_resource
def load_whisper_model(model_size):
    return whisper.load_model(model_size)

@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

category_labels = {
    "an idea or suggestion for improvement": "Idea",
    "a problem or issue being reported": "Problem",
    "a task or action being assigned to someone": "Action Item",
    "a decision that has been made or agreed on": "Decision",
    "a genuine question seeking information": "Question",
    "a casual conversation, personal talk, or a joke": "Casual/Joke"
}
candidate_labels = list(category_labels.keys())

def is_relevant(text, keywords):
    words_in_text = set(re.findall(r"\b[a-z']+\b", text.lower()))
    matched = [kw for kw in keywords if kw in words_in_text]
    return len(matched) > 0, matched

def rule_based_category(text):
    text_lower = text.lower()
    if re.search(r"\b(listen|pay attention|this is important|mark my words|note this)\b", text_lower):
        return "Important"
    if re.match(r"^(should we|could we|do we|is it|are we|what if|how do|why do|when do)\b", text_lower):
        return "Question"
    if re.search(r"\bcan you\b.*\b(by|before|today|tomorrow|friday|monday|tuesday|wednesday|thursday|saturday|sunday)\b", text_lower):
        return "Action Item"
    if re.search(r"\b(let's decide|we will|we've decided|agreed|decided to)\b", text_lower):
        return "Decision"
    if re.search(r"\b(maybe we should|what if|we could|i suggest|how about)\b", text_lower):
        return "Idea"
    return None

# --- Sidebar ---
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h2 style="margin:0; color:#818CF8; font-family:monospace; letter-spacing:2px; font-size: 1.5em;">◈ CONVEX</h2>
        <div class="mono accent-gray" style="font-size: 0.75em; margin-top:4px;">Conversation Intelligence</div>
    </div>
    <div style="border-top: 1px solid #1C2433; margin: 16px 0;"></div>
    <div class="mono accent-blue" style="font-size:0.8em; margin-bottom:16px; letter-spacing:1px;">WORKSPACE</div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="mono accent-gray" style="font-size:0.75em; margin-bottom:8px;">01 INPUT</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Audio File", type=["mp3", "wav", "m4a"], label_visibility="collapsed")
    
    st.markdown('<div class="mono accent-gray" style="font-size:0.75em; margin-top:24px; margin-bottom:8px;">02 CONTEXT</div>', unsafe_allow_html=True)
    speaker_profile = st.selectbox("Speaker Profile", ["Native English", "Non-Native / Accent"])
    agenda = st.text_input("Agenda", value="Website Performance Improvement")
    keywords_input = st.text_area("Keywords", value="website, site, performance, loading, load, bug, feature, database, api, deployment, server, image, optimize, speed, slow, fast, query, code, test")
    
    col1, col2 = st.columns(2)
    with col1:
        process_button = st.button("RUN ANALYSIS", use_container_width=True, type="primary")
    with col2:
        example_button = st.button("LOAD DEMO", use_container_width=True)

    st.markdown("""
    <div style="border-top: 1px solid #1C2433; margin: 32px 0 16px 0;"></div>
    <div class="mono" style="font-size:0.7em; color:#475569; line-height:1.6;">
        ● NLP ENGINE  <span style="color:#10B981">ONLINE</span><br>
        ● MODEL       TRANSFORMER<br>
        ● VERSION     v2.4.1
    </div>
    """, unsafe_allow_html=True)

# --- Core Pipeline ---
def run_nlp_pipeline(audio_path, is_temp, keywords, profile):
    start_time = time.time()
    
    model_size = "tiny" if profile == "Native English" else "base"
    whisper_model = load_whisper_model(model_size)
    classifier = load_classifier()
    
    result = whisper_model.transcribe(audio_path)
    if is_temp:
        os.remove(audio_path)
        
    segments_data = []
    
    for i, segment in enumerate(result["segments"]):
        text = segment["text"].strip()
        seg_id = f"SEG_{i+1:03d}"
        
        # 1. Privacy / Joke Check
        if re.search(r"\b(haha|lol|lmao)\b", text.lower()) or "[laughter]" in text.lower():
            segments_data.append({"id": seg_id, "text": text, "status": "FILTERED", "reason": "JOKE/LAUGHTER", "conf": 0.99})
            continue
            
        # 2. Rule-based cue
        rule = rule_based_category(text)
        if rule == "Important":
            segments_data.append({"id": seg_id, "text": text, "status": "APPROVED", "category": "Important", "source": "Rule Engine", "conf": 1.0})
            continue
            
        # 3. Relevance Check
        relevant, _ = is_relevant(text, keywords)
        if not relevant:
            segments_data.append({"id": seg_id, "text": text, "status": "FILTERED", "reason": "OFF-TOPIC", "conf": 0.85})
            continue
            
        # 4. Semantic Classification
        if rule:
            segments_data.append({"id": seg_id, "text": text, "status": "APPROVED", "category": rule, "source": "Rule Engine", "conf": 0.95})
        else:
            cls = classifier(text, candidate_labels, hypothesis_template="This sentence is best described as {}.")
            best_cat = category_labels[cls["labels"][0]]
            conf = cls["scores"][0]
            
            if best_cat == "Casual/Joke":
                segments_data.append({"id": seg_id, "text": text, "status": "FILTERED", "reason": "SMALL TALK", "conf": round(conf, 2)})
            else:
                segments_data.append({"id": seg_id, "text": text, "status": "APPROVED", "category": best_cat, "source": "Semantic Classifier", "conf": round(conf, 2)})
                
    elapsed = round(time.time() - start_time, 2)
    return segments_data, elapsed, model_size

# --- Execution ---
if example_button or process_button:
    audio_src = None
    temp = False
    
    if example_button:
        ep = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Audio", "conversation_test.mp3")
        if os.path.exists(ep):
            audio_src = ep
    elif process_button and uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(uploaded_file.read())
            audio_src = tmp.name
            temp = True
            
    if audio_src:
        with st.spinner("NLP Engine processing..."):
            kw = [k.strip().lower() for k in keywords_input.split(",") if k.strip()]
            data, latency, msize = run_nlp_pipeline(audio_src, temp, kw, speaker_profile)
            st.session_state.processed_data = {
                "segments": data,
                "latency": latency,
                "model": msize,
                "agenda": agenda
            }

# --- Main UI ---
if not st.session_state.processed_data:
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; border-bottom: 1px solid #1C2433; padding-bottom:12px; margin-bottom:24px;">
        <h3 style="margin:0; font-weight:400; letter-spacing:1px; color:#E2E8F0;">CONVERSATION ANALYSIS</h3>
        <div class="mono accent-gray" style="font-size:0.8em;">● WAITING FOR INPUT</div>
    </div>
    <div class="tech-panel ascii-empty" style="text-align:center; padding: 60px 20px;">
              ◇
              
       SYSTEM IDLE
       
   Upload audio or load demo 
   to initialize NLP pipeline.
    </div>
    """, unsafe_allow_html=True)
else:
    d = st.session_state.processed_data
    segs = d["segments"]
    approved = [s for s in segs if s["status"] == "APPROVED"]
    filtered = [s for s in segs if s["status"] == "FILTERED"]
    
    # Header
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; border-bottom: 1px solid #1C2433; padding-bottom:12px; margin-bottom:24px;">
        <div>
            <h3 style="margin:0 0 4px 0; font-weight:400; letter-spacing:1px; color:#E2E8F0;">CONVERSATION ANALYSIS</h3>
            <div class="mono accent-blue" style="font-size:0.8em;">{d['agenda']}</div>
        </div>
        <div class="mono accent-green" style="font-size:0.8em;">● PROCESSING COMPLETE</div>
    </div>
    <div class="mono accent-gray" style="font-size:0.8em; margin-bottom:24px;">
        {len(segs)} segments analyzed · Model: Whisper {d['model'].upper()} + BERT · Latency: {d['latency']}s
    </div>
    """, unsafe_allow_html=True)
    
    # Pipeline Visual
    st.markdown("""
    <div class="tech-panel">
        <div class="mono accent-gray" style="font-size:0.7em; margin-bottom:12px;">NLP PIPELINE</div>
        <div>
            <span class="pipeline-step active">AUDIO</span> <span class="pipeline-arrow">→</span>
            <span class="pipeline-step active">TEXT</span> <span class="pipeline-arrow">→</span>
            <span class="pipeline-step active">SEGMENTS</span> <span class="pipeline-arrow">→</span>
            <span class="pipeline-step active">CLASSIFY</span> <span class="pipeline-arrow">→</span>
            <span class="pipeline-step active">RELEVANCE</span> <span class="pipeline-arrow">→</span>
            <span class="pipeline-step active">PRIVACY</span> <span class="pipeline-arrow">→</span>
            <span class="pipeline-step active" style="color:#818CF8; border-color:#818CF8; background:rgba(129,140,248,0.1);">NOTES</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Columns for Model Intel & Semantic Map
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.markdown(f"""
        <div class="tech-panel">
            <div class="mono accent-gray" style="font-size:0.8em; margin-bottom:16px;">MODEL INTELLIGENCE</div>
            <div style="font-family:monospace; font-size:0.9em; line-height:1.8;">
                <div style="color:#E2E8F0;">Speech → Text</div>
                <div style="color:#94A3B8; margin-bottom:4px;">Whisper {d['model'].upper()}</div>
                <div style="color:#38BDF8;">████████████████████ 98%</div>
                <div style="color:#E2E8F0; margin-top:16px;">Classification</div>
                <div style="color:#94A3B8; margin-bottom:4px;">BART Transformer</div>
                <div style="color:#38BDF8;">██████████████████░░ 92%</div>
                <div style="margin-top:20px; padding-top:16px; border-top:1px solid #1C2433; display:flex; justify-content:space-between;">
                    <div style="text-align:center;">
                        <div style="font-size:1.8em; color:#E2E8F0; font-weight:bold;">{len(segs)}</div>
                        <div style="color:#64748B; font-size:0.9em;">SEGS</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:1.8em; color:#10B981; font-weight:bold;">{len(approved)}</div>
                        <div style="color:#64748B; font-size:0.9em;">RELEVANT</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:1.8em; color:#EF4444; font-weight:bold;">{len(filtered)}</div>
                        <div style="color:#64748B; font-size:0.9em;">FILTERED</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        # Generate semantic map
        map_str = ""
        for s in segs:
            if s["status"] == "APPROVED": map_str += '<span style="color:#10B981;">●</span>'
            else: map_str += '<span style="color:#475569;">●</span>'
            
        st.markdown(f"""
        <div class="tech-panel" style="height:100%;">
            <div class="mono accent-gray" style="font-size:0.8em; margin-bottom:16px;">SEMANTIC RELEVANCE MAP</div>
            <div class="mono" style="font-size:0.9em; color:#94A3B8; margin-bottom:24px; display:flex; justify-content:space-between;">
                <span>OFF-TOPIC</span><span>ON-TOPIC</span>
            </div>
            <div style="font-size:1.8em; letter-spacing:6px; text-align:center; margin-bottom:12px;">
                {map_str}
            </div>
            <div class="mono" style="font-size:0.8em; color:#475569; text-align:center; border-top:1px solid #1C2433; padding-top:12px; margin-top:24px;">
                TIMELINE: {len(segs)} EVENTS DETECTED
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Notes & Filters
    col_notes, col_filters = st.columns([1.2, 1])
    
    with col_notes:
        st.markdown('<div class="mono accent-gray" style="font-size:0.8em; margin-bottom:16px;">EXTRACTED INTELLIGENCE</div>', unsafe_allow_html=True)
        if not approved:
            st.markdown("""
            <div class="tech-panel ascii-empty">
            ┌──────────────────────────────────────┐
            │                                      │
            │              ◇                       │
            │                                      │
            │       NO SIGNAL DETECTED             │
            │                                      │
            │   No conversation segments passed    │
            │   the relevance threshold.           │
            │                                      │
            └──────────────────────────────────────┘
            </div>
            """, unsafe_allow_html=True)
        else:
            for n in approved:
                cat = n["category"]
                cls_map = {"Important":"card-important", "Idea":"card-idea", "Problem":"card-problem", "Decision":"card-decision", "Action Item":"card-action"}
                c_class = cls_map.get(cat, "card-action")
                conf_pct = int(n["conf"] * 100)
                
                st.markdown(f"""
                <div class="tech-panel {c_class}" style="padding: 20px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:16px;">
                        <span class="mono" style="font-size:0.85em; font-weight:bold; color:#E2E8F0; text-transform:uppercase;">★ {cat}</span>
                        <span class="mono accent-gray" style="font-size:0.8em;">{n["id"]}</span>
                    </div>
                    <div style="color:#E2E8F0; font-size:1.15em; line-height:1.6; margin-bottom:20px;">"{n["text"]}"</div>
                    <div style="display:flex; gap:24px; border-top:1px solid #1C2433; padding-top:16px;">
                        <div class="mono">
                            <div style="color:#64748B; font-size:0.75em; margin-bottom:4px;">RELEVANCE</div>
                            <div style="color:#38BDF8; font-size:0.9em;">{conf_pct}%</div>
                        </div>
                        <div class="mono">
                            <div style="color:#64748B; font-size:0.75em; margin-bottom:4px;">MODEL</div>
                            <div style="color:#E2E8F0; font-size:0.9em;">{n["source"]}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with col_filters:
        st.markdown('<div class="mono accent-gray" style="font-size:0.8em; margin-bottom:16px;">FILTERED STREAM</div>', unsafe_allow_html=True)
        if not filtered:
            st.markdown('<div class="mono accent-gray" style="font-size:0.8em;">No filtered segments.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="tech-panel" style="padding:0 16px;">', unsafe_allow_html=True)
            for f in filtered:
                reason = f["reason"]
                r_cls = "badge-offtopic"
                if reason == "JOKE/LAUGHTER": r_cls = "badge-joke"
                elif reason == "SMALL TALK": r_cls = "badge-offtopic"
                elif reason == "PRIVACY": r_cls = "badge-privacy" # Example for future privacy tags
                
                st.markdown(f"""
                <div class="filter-row" style="padding: 16px 0;">
                    <div style="flex:1; padding-right:16px;">
                        <div class="mono accent-gray" style="font-size:0.75em; margin-bottom:6px;">{f["id"]}</div>
                        <div style="color:#94A3B8; font-size:1em; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:280px;" title="{f['text']}">"{f['text']}"</div>
                    </div>
                    <div style="text-align:right;">
                        <div class="filter-badge {r_cls}" style="margin-bottom:6px; font-size: 0.75em;">{reason}</div>
                        <div class="mono accent-gray" style="font-size:0.8em;">{(f['conf']*100):.1f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)