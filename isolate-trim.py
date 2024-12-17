import os
import json
import subprocess
import pandas as pd
from rich.console import Console

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

def crop_and_trim_silence(input_file, output_prefix, max_length, overlap):
    """Crops audio to chunks of max_length seconds with overlap and trims silence."""
    probe_cmd = ["ffprobe", "-i", input_file, "-show_entries", "format=duration", "-v", "error", "-of", "csv=p=0"]
    result = subprocess.run(probe_cmd, stdout=subprocess.PIPE, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        console.print(f"[red]Error: Unable to determine duration for {input_file}. Skipping.[/red]")
        return

    duration = float(result.stdout.strip())

    start_time = 0
    chunk_index = 1
    while start_time < duration:
        end_time = min(start_time + max_length, duration)
        chunk_output = f"{output_prefix}_chunk{chunk_index:02d}_raw.wav"

        console.print(f"[cyan]Extracting chunk {chunk_index} to: {chunk_output}[/cyan]")

        crop_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", input_file, "-ss", str(start_time), "-to", str(end_time),
            "-c", "copy", chunk_output
        ]
        result = subprocess.run(crop_cmd, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0 or not os.path.exists(chunk_output):
            console.print(f"[red]Error: Failed to crop audio chunk {chunk_output}. stderr: {result.stderr}[/red]")
            start_time += (max_length - overlap)
            continue

        # Validate file size before running ffprobe
        if os.stat(chunk_output).st_size == 0:
            console.print(f"[yellow]Warning: Cropped file {chunk_output} is empty. Skipping.[/yellow]")
            os.remove(chunk_output)
            start_time += (max_length - overlap)
            continue

        trimmed_output = f"{output_prefix}_chunk{chunk_index:02d}.wav"
        console.print(f"[cyan]Trimming output: {trimmed_output}[/cyan]")
        trim_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", chunk_output,
            "-af", "silenceremove=start_periods=1:start_duration=0.5:start_threshold=-30dB:stop_periods=1:stop_duration=0.5:stop_threshold=-30dB",
            trimmed_output
        ]
        trim_result = subprocess.run(trim_cmd, stderr=subprocess.PIPE, text=True)
        if trim_result.returncode != 0:
            console.print(f"[red]Error trimming {chunk_output}: {trim_result.stderr}[/red]")
            os.remove(chunk_output)
            start_time += (max_length - overlap)
            continue

        # Validate that the trimmed file is not empty
        if os.stat(trimmed_output).st_size == 0:
            console.print(f"[yellow]Warning: Trimmed file {trimmed_output} is empty. Skipping.[/yellow]")
            os.remove(trimmed_output)
            os.remove(chunk_output)
            start_time += (max_length - overlap)
            continue

        console.print(f"[green]Segment processed and saved: {trimmed_output}[/green]")
        os.remove(chunk_output)
        start_time += (max_length - overlap)
        chunk_index += 1

# Process Each File (limited to first 10 files for debugging)
for _, row in df_mapping.head(10).reset_index(drop=True).iterrows():
    json_file = row['json_file']
    target_speaker_label = row['target_speaker_label']
    if target_speaker_label == "not-target-speaker" or pd.isna(target_speaker_label):
        console.print(f"[yellow]Skipping {json_file}: No target speaker identified.[/yellow]")
        continue

    json_path = os.path.join(JSON_DIR, json_file)
    wav_file = os.path.join(AUDIO_DIR, os.path.splitext(json_file)[0] + ".wav")

    # Check input WAV file duration before processing
    probe_wav_cmd = [
        "ffprobe", "-i", wav_file, "-show_entries", "format=duration", "-v", "error", "-of", "csv=p=0"
    ]
    wav_duration_result = subprocess.run(probe_wav_cmd, stdout=subprocess.PIPE, text=True)
    if not wav_duration_result.stdout.strip() or float(wav_duration_result.stdout.strip()) == 0.0:
        console.print(f"[red]Input WAV file {wav_file} is silent or corrupt. Skipping.[/red]")
        continue

    if not os.path.exists(json_path) or not os.path.exists(wav_file):
        console.print(f"[red]Missing JSON or WAV file for {json_file}. Skipping.[/red]")
        continue

    # Load JSON and filter target speaker segments
    with open(json_path, "r") as f:
        segments = json.load(f)
    console.print(f"[cyan]Loaded {len(segments)} segments from {json_file}[/cyan]")
    console.print(f"[cyan]Filtering for target speaker: {target_speaker_label}[/cyan]")

    seen_segments = {}
    target_segments = []
    for seg in segments:
        key = (seg['start'], seg['end'])
        if seg['speaker'] == target_speaker_label and key not in seen_segments:
            seen_segments[key] = seg
            target_segments.append(seg)

    console.print(f"[cyan]Found {len(target_segments)} segments for {target_speaker_label}[/cyan]")
    if not target_segments:
        console.print(f"[yellow]No segments for target speaker {target_speaker_label} in {json_file}.[/yellow]")
        continue

    # Process target speaker segments
    for segment in target_segments:
        start_time, end_time = segment["start"], segment["end"]
        console.print(f"[cyan]Processing segment {start_time}-{end_time} for {target_speaker_label}[/cyan]")

        if end_time - start_time < 1.0:
            console.print(f"[yellow]Skipping short segment {start_time}-{end_time}.[/yellow]")
            continue

        output_prefix = os.path.join(
            OUTPUT_DIR, f"{os.path.splitext(json_file)[0]}_{target_speaker_label}_{start_time:.2f}-{end_time:.2f}"
        )
        temp_segment = f"{output_prefix}_raw.wav"

        extract_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", wav_file, "-ss", str(start_time), "-to", str(end_time),
            "-c", "copy", temp_segment
        ]
        if subprocess.run(extract_cmd).returncode != 0 or not os.path.exists(temp_segment):
            console.print(f"[red]Error: Failed to extract segment {start_time}-{end_time} for {json_file}. Skipping.[/red]")
            continue

        crop_and_trim_silence(temp_segment, output_prefix, MAX_CLIP_LENGTH, OVERLAP_DURATION)
        os.remove(temp_segment)

console.print("[bold green]Processing complete. Check the 'targeted' directory for results.[/bold green]")

