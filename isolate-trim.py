import os
import json
import pandas as pd
from rich.console import Console
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize Rich Console
console = Console()
console.clear()

# Title Screen
console.print("""[bold cyan]
=====================================
      SPEAKER ISOLATION TOOL
=====================================
[/bold cyan]""")

# Define Paths Relative to Script Location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_DIR = os.path.join(SCRIPT_DIR, "jsons")
AUDIO_DIR = os.path.join(SCRIPT_DIR, "wavs")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "targeted")
MAPPING_FILE = os.path.join(SCRIPT_DIR, "mappings.csv")

# Ensure Directories Exist
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Speaker Mapping
if not os.path.exists(MAPPING_FILE):
    console.print(f"[bold red]Error:[/bold red] Speaker mapping file not found: {MAPPING_FILE}", style="red")
    exit(1)

try:
    df_mapping = pd.read_csv(MAPPING_FILE)
except Exception as e:
    console.print(f"[red]Error: Failed to read {MAPPING_FILE}. {e}[/red]")
    exit(1)
console.print(f"[bold green]Loaded speaker mapping for {len(df_mapping)} files.[/bold green]")

# Sort mapping to ensure files like '1.wav' are processed before '1000.wav'
df_mapping['numeric_sort'] = df_mapping['json_file'].str.extract(r'(\d+)').astype(float)
df_mapping = df_mapping.sort_values(by='numeric_sort').drop(columns=['numeric_sort'])

# Maximum length for trimmed clips (in seconds)
MAX_CLIP_LENGTH = 5
OVERLAP_DURATION = 0.5  # Overlap duration in seconds

def process_file(json_file, target_speaker_label):
    """Processes a single file for the target speaker."""
    json_path = os.path.join(JSON_DIR, json_file)
    wav_file = os.path.join(AUDIO_DIR, os.path.splitext(json_file)[0] + ".wav")

    if not os.path.exists(json_path) or not os.path.exists(wav_file):
        console.print(f"[red]Missing JSON or WAV file for {json_file}. Skipping.[/red]")
        return 0

    # Load JSON and filter target speaker segments
    with open(json_path, "r") as f:
        segments = json.load(f)

    seen_segments = {}
    target_segments = [
        seg for seg in segments 
        if seg['speaker'] == target_speaker_label and not seen_segments.setdefault((seg['start'], seg['end']), False)
    ]

    if not target_segments:
        console.print(f"[yellow]No segments for target speaker {target_speaker_label} in {json_file}.[/yellow]")
        return 0

    # Load the entire audio file once
    try:
        audio = AudioSegment.from_wav(wav_file)
    except Exception as e:
        console.print(f"[red]Error: Failed to load audio file {wav_file}. {e}[/red]")
        return 0

    segment_count = 0
    for segment in target_segments:
        start_time, end_time = segment["start"] * 1000, segment["end"] * 1000
        if (end_time - start_time) < 1000:
            continue

        segment_audio = audio[start_time:end_time]
        if len(segment_audio) == 0:
            continue

        output_file = os.path.join(
            OUTPUT_DIR, f"{os.path.splitext(json_file)[0]}_{target_speaker_label}_{start_time / 1000:.2f}-{end_time / 1000:.2f}.wav"
        )
        segment_audio.export(output_file, format="wav")
        segment_count += 1

    return segment_count

# Multi-threaded Processing
console.print("[bold cyan]Starting multi-threaded processing...[/bold cyan]")
total_segments = 0

with ThreadPoolExecutor() as executor:
    futures = []
    for _, row in df_mapping.iterrows():
        json_file = row['json_file']
        target_speaker_label = row['target_speaker_label']
        if target_speaker_label == "not-target-speaker" or pd.isna(target_speaker_label):
            console.print(f"[yellow]Skipping {json_file}: No target speaker identified.[/yellow]")
            continue

        futures.append(executor.submit(process_file, json_file, target_speaker_label))

    for future in as_completed(futures):
        total_segments += future.result()

console.print(f"[bold green]Processing complete. Total segments saved: {total_segments}[/bold green]")
console.print(f"[bold green]Check the 'targeted' directory for results.[/bold green]")

