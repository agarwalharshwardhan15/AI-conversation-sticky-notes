# AI Conversation Assistant — Sticky Notes Generator

An AI-powered conversation intelligence system that transcribes audio, filters content against a user-defined agenda, and categorizes relevant statements into color-coded sticky notes (Idea, Problem, Action Item, Decision, Question).

## The Problem

Transcribing a conversation is straightforward. The harder problem — and the focus of this project — is **filtering**: determining which parts of a conversation are meaningful and worth preserving, and which are small talk, greetings, or off-topic noise.

## Pipeline

Audio Upload
↓
Speech-to-Text (OpenAI Whisper)
↓
Agenda-Based Relevance Filtering
↓
Category Classification (Idea / Problem / Action Item / Decision / Question)
↓
Sticky Note Display (Streamlit UI)

## Approach

This project uses a **hybrid filtering and classification approach**, combining two techniques:

**1. Agenda-based keyword filtering (relevance)**
The user defines an agenda (e.g. "Website Performance Improvement") along with a set of related keywords. Each transcribed sentence is checked against this keyword set using whole-word matching (via regex) to avoid false positives from substrings (e.g. "ui" incorrectly matching inside "guidelines" or "require").

**2. Rule-based + zero-shot classification (categorization)**
Relevant sentences are categorized using:
- **Rule-based pattern matching** for high-confidence linguistic patterns (e.g. "can you ... by Friday" → Action Item; "should we ...?" → Question; "let's decide" → Decision)
- **Zero-shot classification** (`facebook/bart-large-mnli`) as a fallback for sentences that don't match a rule, using descriptive hypothesis templates for better accuracy

This hybrid design reflects a genuine limitation of using either method alone: keyword matching alone cannot distinguish a question from a statement, and generic AI classification alone produces low-confidence, sometimes incorrect results on short, ambiguous sentences.

## Tech Stack

- **Whisper** (OpenAI) — speech-to-text transcription
- **Transformers** (Hugging Face) — zero-shot classification via `facebook/bart-large-mnli`
- **Streamlit** — web UI
- **Python regex** — keyword and pattern matching

## How to Run

1. Clone this repository
2. Create a virtual environment and activate it:

python -m venv venv
venv\Scripts\activate

3. Install dependencies:

pip install openai-whisper transformers torch streamlit

4. Install [ffmpeg](https://www.gyan.dev/ffmpeg/builds/) and add it to your system PATH (required by Whisper)
5. Run the app:

streamlit run app.py

6. Upload a conversation audio file, set your agenda and keywords, and click **Process**

## Example Result

Using a short scripted conversation with an agenda of "Website Performance Improvement," out of 14 transcribed segments, the system correctly identified:

- ❗ **Problem**: "I think the website loading time is still a problem." / "It's taking almost five seconds on the homepage."
- 💡 **Idea**: "Maybe we should optimize our images, try using WebP format instead."
- ❓ **Question**: "Should we also look at the changing of database queries?"
- 🎯 **Action Item**: "Rahul, can you test the image optimization on the homepage by Friday?"

The remaining 9 segments (greetings, small talk, unrelated remarks) were correctly filtered out.

## Known Limitations

- **Speech-to-text errors propagate downstream.** In testing, Whisper mis-transcribed "Cloudflare" and "CDN" as "Cloudflow" and "CDR," causing a genuinely relevant decision statement to be filtered out because it no longer matched any keyword. A production system could mitigate this with fuzzy string matching against the keyword list.
- **Question form vs. function.** Sentences that are grammatically phrased as questions but functionally act as task delegations (e.g. "Can you test this by Friday?") required an explicit rule-based override, since generic sentence classification tends to read them literally as questions rather than by their communicative intent.
- **Low-confidence zero-shot results on short sentences.** Very short, low-context segments sometimes produce ambiguous classifier confidence scores (under 50%), which is why rule-based overrides were introduced for common patterns rather than relying on the classifier alone.
- Speaker identification/diarization is not implemented in this version — sticky notes are not attributed to individual speakers.

## Project Structure

ConvAI-Project/
├── app.py # Streamlit web application
├── phase1_transcribe.py # Whisper transcription (standalone test script)
├── phase2_filter.py # Agenda-based filtering (standalone test script)
├── phase3_categorize.py # Hybrid categorization (standalone test script)
├── Audio/ # Test audio files (not tracked in git)
├── .gitignore
└── README.md