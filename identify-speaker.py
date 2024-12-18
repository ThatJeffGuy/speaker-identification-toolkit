import os
import json
import pandas as pd
import tempfile
from pydub import AudioSegment
from playsound import playsound
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.prompt import Prompt

# Initialize Rich Console
console = Console()
console.clear()

# Title Screen
console.print("""[bold cyan]
=====================================
     SPEAKER IDENTIFICATION TOOL
=====================================
[/bold cyan]""")
console.print("This tool identifies target speakers in audio files using JSON metadata and user input.")

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_DIR = os.path.join(SCRIPT_DIR, "jsons")
AUDIO_DIR = os.path.join(SCRIPT_DIR, "wavs")
MAPPING_FILE = os.path.join(SCRIPT_DIR, "mappings.csv")

# Ensure Directories Exist
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

# Validate Mapping File
def validate_mapping_file():
    if not os.path.isfile(MAPPING_FILE):
        return pd.DataFrame(columns=['json_file', 'target_speaker_label'])
    return pd.read_csv(MAPPING_FILE)

# Load Existing Mapping File
df = validate_mapping_file()

# Helper Function to Process Single File
def process_json_file(json_file, wav_file, existing_mapping):
    if json_file in existing_mapping['json_file'].values:
        console.print(f"[yellow]Skipping {json_file}: Already mapped.[/yellow]")
        return None

    with open(json_file, 'r') as f:
        segments = json.load(f)
    unique_speakers = list(set(seg["speaker"] for seg in segments))

    for speaker_label in unique_speakers:
        console.print(f"[cyan]Checking speaker: {speaker_label}[/cyan]")
        speaker_segments = [seg for seg in segments if seg["speaker"] == speaker_label]

        # Play segments to identify speaker
        for segment in speaker_segments:
            start_time, end_time = segment["start"] * 1000, segment["end"] * 1000
            try:
                audio = AudioSegment.from_wav(wav_file)[start_time:end_time]
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                audio.export(temp_file.name, format="wav")
                playsound(temp_file.name)
            except Exception as e:
                console.print(f"[red]Error playing segment: {e}[/red]")
                continue
            finally:
                temp_file.close()
                os.unlink(temp_file.name)

            user_input = Prompt.ask("Is this the target speaker?", choices=["y", "n", "t"], default="n")
            if user_input.lower() == "y":
                return {"json_file": os.path.basename(json_file), "target_speaker_label": speaker_label}
            elif user_input.lower() == "t":
                continue
        console.print(f"[yellow]Skipping speaker: {speaker_label}[/yellow]")

    return {"json_file": os.path.basename(json_file), "target_speaker_label": "not-target-speaker"}

# Multi-threaded Speaker Identification
console.print("[bold cyan]Starting speaker identification...[/bold cyan]")
json_files = [os.path.join(JSON_DIR, f) for f in os.listdir(JSON_DIR) if f.endswith(".json")]
existing_mapping = validate_mapping_file()
new_mappings = []

with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as executor:
    futures = []
    for json_file in json_files:
        wav_file = os.path.join(AUDIO_DIR, os.path.splitext(os.path.basename(json_file))[0] + ".wav")
        if os.path.exists(wav_file):
            futures.append(executor.submit(process_json_file, json_file, wav_file, existing_mapping))

    for future in as_completed(futures):
        result = future.result()
        if result:
            new_mappings.append(result)

# Update and Save Mapping File
if new_mappings:
    df = pd.concat([existing_mapping, pd.DataFrame(new_mappings)], ignore_index=True)
    df.to_csv(MAPPING_FILE, index=False)
    console.print("[bold green]Mapping updated successfully.[/bold green]")
else:
    console.print("[yellow]No new mappings were added.[/yellow]")

console.print("[bold green]Speaker identification complete.[/bold green]")

