# CONVEX: Conversation Intelligence NLP Engine ◈

CONVEX is an advanced, Explainable AI pipeline designed to act as a "Meeting Intelligence Control Room." It ingests casual meeting audio, transcribes the conversation, filters out off-topic chatter and jokes to protect privacy, and automatically extracts critical information into concise, categorized **Sticky Notes** (Ideas, Decisions, Problems, Action Items).

Unlike traditional generative AI tools that hallucinate meeting summaries, CONVEX uses **Zero-Shot Semantic Classification** and **Deterministic Rule-Based Heuristics** to extract the *exact* factual sentences spoken, achieving a **100% ROUGE-1 Precision score**.

---

## 🚀 Features

*   **Zero-Shot Semantic Classification:** Uses `BART-Large-MNLI` to understand conversation semantics without requiring labeled training data.
*   **Rule-Based Cue Engine:** Guarantees 100% capture of explicit commands (e.g., *"Listen, this is important"*, *"Can you do this by tomorrow"*).
*   **Dynamic Domain Optimization (Speaker Profile):** Automatically switches backend ASR architecture (`Whisper Tiny` vs `Whisper Base`) based on the speaker's accent to balance processing time and accuracy.
*   **Explainable AI Dashboard:** 
    *   **Filtered Stream:** Visually explains *why* data was discarded (e.g., Tagged as `JOKE/LAUGHTER` or `OFF-TOPIC`).
    *   **Semantic Relevance Map:** A visual timeline showing exactly when the meeting went off-topic.
*   **Privacy-First:** Aggressively filters small talk and personal conversations based on user-defined Agenda Keywords.

---

## 🧠 The NLP Pipeline Architecture

1.  **Audio Ingestion:** MP3/WAV files are uploaded securely.
2.  **ASR Transcription:** OpenAI's Whisper model converts audio to segmented text arrays.
3.  **Heuristic Noise Filter:** Regex patterns instantly discard laughter and explicit jokes.
4.  **Lexical Relevance Check:** Bag-of-words tokenization ensures the segment aligns with the defined Agenda keywords.
5.  **Hybrid Classification:**
    *   *Rules Engine* captures deterministic cues.
    *   *Zero-Shot AI* calculates semantic probabilities for remaining sentences (Idea, Problem, Decision).
6.  **UI Rendering:** Approved segments are rendered as HTML/CSS Sticky Notes; discarded segments populate the Filtered Stream.

---

## 📊 Academic Study: Native vs. Non-Native Processing

A core component of CONVEX is its dynamic algorithm selector, born from an empirical study across 20 audio samples (`study_results.csv`). 

**Findings:**
*   **Native Speakers:** Lightweight models (`Tiny`) process in ~2s with ~94% accuracy.
*   **Non-Native Speakers:** Lightweight models suffer a 30% accuracy penalty on heavy accents. Heavy models (`Base`) are required to maintain ~89% accuracy (at the cost of longer processing times).

**Result:** CONVEX allows users to select a "Speaker Profile" in the UI, dynamically loading the mathematically optimal model for the situation.

---

## 🛠️ Installation & Setup

### Prerequisites
*   Python 3.8+
*   FFmpeg (must be installed and added to system PATH for Whisper to decode audio)

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ConvAI-Project.git
cd ConvAI-Project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Core dependencies: `streamlit`, `openai-whisper`, `transformers`, `torch`, `torchaudio`)*

### 3. Run the Control Room
```bash
streamlit run app.py
```
The CONVEX dashboard will launch locally at `http://localhost:8501`.

---

## 📈 Performance Metrics
Evaluated manually against a ground-truth dataset of 100 meeting segments:
*   **Precision:** 90.4%
*   **Recall:** 86.3%
*   **F1-Score:** 88.3%
*   **ROUGE-1 Precision:** 100%

---
*Developed as part of an academic NLP implementation study.*