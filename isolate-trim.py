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

df_mapping = pd.read_csv(MAPPING_FILE)
console.print(f"[bold green]Loaded speaker mapping for {len(df_mapping)} files.[/bold green]")

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

        crop_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", input_file, "-ss", str(start_time), "-to", str(end_time),
            "-c", "copy", chunk_output
        ]
        if subprocess.run(crop_cmd).returncode != 0 or not os.path.exists(chunk_output):
            console.print(f"[red]Error: Failed to crop audio chunk {chunk_output}. Skipping.[/red]")
            start_time += (max_length - overlap)
            continue

        # Validate file size before running ffprobe
        if os.stat(chunk_output).st_size == 0:
            console.print(f"[yellow]Warning: Cropped file {chunk_output} is empty. Skipping.[/yellow]")
            os.remove(chunk_output)
            start_time += (max_length - overlap)
            continue

        trimmed_output = f"{output_prefix}_chunk{chunk_index:02d}.wav"
        trim_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", chunk_output,
            "-af", "silenceremove=start_periods=1:start_duration=0.5:start_threshold=-30dB:stop_periods=1:stop_duration=0.5:stop_threshold=-30dB",
            trimmed_output
        ]
        subprocess.run(trim_cmd)

        # Validate that the trimmed file is not empty using ffprobe
        probe_trimmed_cmd = [
            "ffprobe", "-i", trimmed_output, "-show_entries", "format=duration", "-v", "error", "-of", "csv=p=0"
        ]
        probe_result = subprocess.run(probe_trimmed_cmd, stdout=subprocess.PIPE, text=True)
        if probe_result.returncode != 0 or not probe_result.stdout.strip().replace('N/A', '').isdigit() or float(probe_result.stdout.strip().replace('N/A', '0')) == 0.0:
            console.print(f"[yellow]Warning: Trimmed file {trimmed_output} is silent or empty. Skipping.[/yellow]")
            os.remove(trimmed_output)
        os.remove(chunk_output)
        start_time += (max_length - overlap)
        chunk_index += 1

# Process Each File
for _, row in df_mapping.iterrows():
    json_file = row['json_file']
    target_speaker_label = row['target_speaker_label']
    if target_speaker_label == "not-target-speaker" or pd.isna(target_speaker_label):
        console.print(f"[yellow]Skipping {json_file}: No target speaker identified.[/yellow]")
        continue

    json_path = os.path.join(JSON_DIR, json_file)
    wav_file = os.path.join(AUDIO_DIR, os.path.splitext(json_file)[0] + ".wav")
    if not os.path.exists(json_path) or not os.path.exists(wav_file):
        console.print(f"[red]Missing JSON or WAV file for {json_file}. Skipping.[/red]")
        continue

    # Load JSON and filter target speaker segments
    with open(json_path, "r") as f:
        segments = json.load(f)
    seen_segments = set()
    target_segments = [
        seg for seg in segments 
        if seg["speaker"] == target_speaker_label and (seg["start"], seg["end"]) not in seen_segments and not seen_segments.add((seg["start"], seg["end"]))
    ]

    if not target_segments:
        console.print(f"[yellow]No segments for target speaker {target_speaker_label} in {json_file}.[/yellow]")
        continue

    # Process target speaker segments
    for segment in target_segments:
        start_time, end_time = segment["start"], segment["end"]
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

