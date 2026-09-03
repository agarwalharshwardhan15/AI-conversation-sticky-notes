import whisper
import os
import time
import csv

# --- Setup ---
script_folder = os.path.dirname(os.path.abspath(__file__))
study_folder = os.path.join(script_folder, "StudyData")

categories = ["native", "non_native"]
model_sizes = ["tiny", "base"]

results = []

# --- Load each model once, reuse across all files (much faster than reloading per file) ---
loaded_models = {}
for size in model_sizes:
    print(f"Loading Whisper '{size}' model...")
    loaded_models[size] = whisper.load_model(size)

# --- Run every file through every model size ---
for category in categories:
    category_folder = os.path.join(study_folder, category)
    if not os.path.exists(category_folder):
        print(f"WARNING: folder not found: {category_folder}")
        continue

    audio_files = [f for f in os.listdir(category_folder) if f.lower().endswith((".mp3", ".wav", ".m4a"))]

    for filename in audio_files:
        file_path = os.path.join(category_folder, filename)
        print(f"\nProcessing {category}/{filename} ...")

        for size in model_sizes:
            model = loaded_models[size]

            start_time = time.time()
            result = model.transcribe(file_path)
            elapsed = round(time.time() - start_time, 2)

            segments = result["segments"]
            if segments:
                avg_confidence = round(
                    sum(s.get("avg_logprob", -1) for s in segments) / len(segments), 3
                )
            else:
                avg_confidence = None

            word_count = len(result["text"].split())

            print(f"  [{size:>6}] time: {elapsed:>6}s | confidence: {avg_confidence} | words: {word_count}")

            results.append({
                "category": category,
                "filename": filename,
                "model_size": size,
                "processing_time_sec": elapsed,
                "avg_confidence": avg_confidence,
                "word_count": word_count,
                "transcript_preview": result["text"][:150]
            })

# --- Save results to CSV ---
output_path = os.path.join(script_folder, "study_results.csv")
with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\n\nDone! Results saved to: {output_path}")
print(f"Total combinations tested: {len(results)}")