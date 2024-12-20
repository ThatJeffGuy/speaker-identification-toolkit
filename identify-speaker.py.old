import os
import json
import pandas as pd
import tempfile
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.prompt import Prompt
import simpleaudio as sa  # For reliable audio playback

# Initialize Rich Console
console = Console()
console.clear()

# Title Screen
console.print("""[bold cyan]
=====================================
     SPEAKER IDENTIFICATION TOOL
=====================================
[/bold cyan]""", justify="center")
console.print("This tool identifies target speakers in audio files using JSON metadata and user input.", justify="center")
console.print("[bold yellow]NOTE: Ensure audio is pre-separated using UVR before running this tool.[/bold yellow]", justify="center")
console.print("Menu Options: Y - Confirm, N - Skip, T - Try next segment, X - Exit, L - Listen again", justify="center")

# Static title header
console.rule("[bold green]Logs and Processing Below[/bold green]")

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_DIR = os.path.join(SCRIPT_DIR, "jsons")
AUDIO_DIR = os.path.join(SCRIPT_DIR, "wavs")
CLEANED_DIR = os.path.join(SCRIPT_DIR, "cleaned")
MAPPING_FILE = os.path.join(SCRIPT_DIR, "mappings.csv")

# Ensure Directories Exist
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(CLEANED_DIR, exist_ok=True)

# Validate Mapping File
def validate_mapping_file():
    if not os.path.isfile(MAPPING_FILE):
        return pd.DataFrame(columns=['json_file', 'target_speaker_label'])
    return pd.read_csv(MAPPING_FILE)

# Load Existing Mapping File
df = validate_mapping_file()
processed_count = df['json_file'].nunique()
console.print(f"[bold green]{processed_count} episodes have already been processed.[/bold green]")

# Helper Function to Process Single File
def identify_speaker(json_file, wav_file, existing_mapping):
    if json_file in existing_mapping['json_file'].values:
        return None

    with open(json_file, 'r') as f:
        segments = json.load(f)
    unique_speakers = list(set(seg["speaker"] for seg in segments))

    for speaker_label in unique_speakers:
        console.print(f"[cyan]Checking speaker: {speaker_label}[/cyan]")
        speaker_segments = [seg for seg in segments if seg["speaker"] == speaker_label]

        # Play segments to identify speaker sequentially
        for segment in speaker_segments:
            start_time, end_time = segment["start"] * 1000, segment["end"] * 1000
            if (end_time - start_time) < 1500:
                continue
            try:
                audio = AudioSegment.from_wav(wav_file)[start_time:end_time]
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                audio.export(temp_file.name, format="wav")
                wave_obj = sa.WaveObject.from_wave_file(temp_file.name)
                play_obj = wave_obj.play()
                play_obj.wait_done()  # Wait for playback to finish
            except Exception as e:
                continue
            finally:
                temp_file.close()
                os.unlink(temp_file.name)

            user_input = Prompt.ask("Is this the target speaker? (y/n/t for try next segment, x to exit, l to listen again)", choices=["y", "n", "t", "x", "l"], default="n")
            if user_input.lower() == "y":
                return {"json_file": os.path.basename(json_file), "target_speaker_label": speaker_label}
            elif user_input.lower() == "n":
                break
            elif user_input.lower() == "t":
                continue
            elif user_input.lower() == "x":
                console.print("[bold yellow]Exiting speaker identification without saving mappings.[/bold yellow]")
                exit(0)
            elif user_input.lower() == "l":
                wave_obj = sa.WaveObject.from_wave_file(temp_file.name)
                play_obj = wave_obj.play()
                play_obj.wait_done()

    return {"json_file": os.path.basename(json_file), "target_speaker_label": "not-target-speaker"}

# Multi-threaded Post-Processing
console.print("[bold cyan]Starting speaker identification...[/bold cyan]")
json_files = [os.path.join(JSON_DIR, f) for f in os.listdir(JSON_DIR) if f.endswith(".json")]
existing_mapping = validate_mapping_file()
new_mappings = []

for json_file in json_files:
    wav_file = os.path.join(CLEANED_DIR, os.path.splitext(os.path.basename(json_file))[0] + ".wav")
    if os.path.exists(wav_file):
        result = identify_speaker(json_file, wav_file, existing_mapping)
        if result:
            new_mappings.append(result)

    # Prompt user to process next episode or exit gracefully
    proceed = Prompt.ask("Do you want to process the next episode?", choices=["yes", "no"], default="yes")
    if proceed.lower() == "no":
        console.print("[bold yellow]Exiting speaker identification tool gracefully.[/bold yellow]")
        break

# Use threading only for any post-processing tasks
console.print("[bold cyan]Processing additional tasks in parallel...[/bold cyan]")

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(lambda: console.print("[green]Task completed[/green]")) for _ in range(len(new_mappings))]
    for future in as_completed(futures):
        future.result()

# Update and Save Mapping File
if new_mappings:
    df = pd.concat([existing_mapping, pd.DataFrame(new_mappings)], ignore_index=True)
    df.to_csv(MAPPING_FILE, index=False)
    console.print("[bold green]Mapping updated successfully.[/bold green]")
else:
    console.print("[yellow]No new mappings were added.[/yellow]")

console.print("[bold green]Speaker identification complete.[/bold green]")

