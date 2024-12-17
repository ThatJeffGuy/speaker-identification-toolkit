import os
import json
import subprocess
import pandas as pd
from rich.console import Console
from rich.prompt import Prompt

# Initialize Rich Console
console = Console()
console.clear()

# Title Screen
console.print("""[bold cyan]
=====================================
      SPEAKER ISOLATION TOOL
=====================================
[/bold cyan]""")
console.print("This tool isolates and trims audio segments based on speaker metadata.")
console.print("Input directories for WAV and JSON files, and an output directory, will be created in the same folder as this script.")

# Define Paths Relative to Script Location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_DIR = os.path.join(SCRIPT_DIR, "jsons")
AUDIO_DIR = os.path.join(SCRIPT_DIR, "wavs")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "targeted")

# Ensure Directories Exist
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pause to allow user to populate directories
console.print(f"[bold yellow]Please add JSON metadata files to: {JSON_DIR}[/bold yellow]")
console.print(f"[bold yellow]Please add corresponding WAV audio files to: {AUDIO_DIR}[/bold yellow]")
input("\nPress Enter to continue...")

# Load and Validate Files
json_files = [f for f in os.listdir(JSON_DIR) if f.endswith(".json")]
audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]

if not json_files:
    console.print(f"[bold red]Error:[/bold red] No JSON files found in: {JSON_DIR}", style="red")
    exit(1)
if not audio_files:
    console.print(f"[bold red]Error:[/bold red] No WAV files found in: {AUDIO_DIR}", style="red")
    exit(1)

console.print(f"[bold green]Found {len(json_files)} JSON file(s) and {len(audio_files)} WAV file(s). Starting processing...[/bold green]")

# Maximum length for trimmed clips (in seconds)
MAX_CLIP_LENGTH = 5
OVERLAP_DURATION = 0.5  # Overlap duration in seconds

def crop_and_trim_silence(input_file, output_prefix, max_length, overlap):
    """Crops audio to chunks of max_length seconds with overlap and trims silence."""
    # Step 1: Get the duration of the input audio
    probe_cmd = [
        "ffprobe", "-i", input_file, "-show_entries", "format=duration",
        "-v", "error", "-of", "csv=p=0"
    ]
    result = subprocess.run(probe_cmd, stdout=subprocess.PIPE, text=True)
    duration = float(result.stdout.strip())

    # Step 2: Split into chunks of max_length seconds with overlap
    start_time = 0
    chunk_index = 1
    while start_time < duration:
        end_time = min(start_time + max_length, duration)
        chunk_output = f"{output_prefix}_chunk{chunk_index:02d}_raw.wav"

        # Crop the audio to the current chunk
        crop_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", input_file, "-ss", str(start_time), "-to", str(end_time),
            "-c", "copy", chunk_output
        ]
        subprocess.run(crop_cmd)

        # Step 3: Trim silence from the cropped chunk
        trimmed_output = f"{output_prefix}_chunk{chunk_index:02d}.wav"
        trim_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", chunk_output,
            "-af", "silenceremove=start_periods=1:start_duration=0.5:start_threshold=-30dB:stop_periods=1:stop_duration=0.5:stop_threshold=-30dB",
            trimmed_output
        ]
        subprocess.run(trim_cmd)

        console.print(f"[green]Processed chunk:[/green] {trimmed_output}")

        # Cleanup the raw chunk
        os.remove(chunk_output)
        start_time += (max_length - overlap)  # Apply overlap
        chunk_index += 1

def rename_files_sequentially(directory):
    """Renames all WAV files in the directory to sequential numbers starting at 1.wav."""
    wav_files = sorted([f for f in os.listdir(directory) if f.endswith(".wav")])
    for index, file_name in enumerate(wav_files, start=1):
        old_path = os.path.join(directory, file_name)
        new_path = os.path.join(directory, f"{index}.wav")
        os.rename(old_path, new_path)
    console.print(f"[bold cyan]Renamed {len(wav_files)} files sequentially in {directory}.[/bold cyan]")

# Process Each File
for json_file in json_files:
    json_path = os.path.join(JSON_DIR, json_file)
    wav_file = os.path.join(AUDIO_DIR, os.path.splitext(json_file)[0] + ".wav")

    if not os.path.exists(wav_file):
        console.print(f"[bold red]Error:[/bold red] Corresponding WAV file not found for {json_file}", style="red")
        continue

    # Load JSON segments
    with open(json_path, "r") as f:
        try:
            segments = json.load(f)
        except json.JSONDecodeError:
            console.print(f"[bold red]Error:[/bold red] JSON file {json_file} is not properly formatted. Skipping.", style="red")
            continue

    # Process each segment
    for segment in segments:
        speaker = segment.get("speaker")
        start_time = segment.get("start")
        end_time = segment.get("end")

        if not (speaker and start_time is not None and end_time is not None):
            console.print(f"[bold yellow]Skipping invalid segment in {json_file}: {segment}[/bold yellow]", style="yellow")
            continue

        if end_time - start_time < 1.0:
            console.print(f"[bold yellow]Skipping short segment ({start_time}-{end_time}) for speaker {speaker}.[/bold yellow]", style="yellow")
            continue

        output_prefix = os.path.join(OUTPUT_DIR, f"{os.path.splitext(json_file)[0]}_{speaker}_{start_time:.2f}-{end_time:.2f}")

        # Extract the segment
        temp_segment = f"{output_prefix}_raw.wav"
        extract_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", wav_file, "-ss", str(start_time), "-to", str(end_time),
            "-c", "copy", temp_segment
        ]
        subprocess.run(extract_cmd)

        # Crop and trim silence
        crop_and_trim_silence(temp_segment, output_prefix, MAX_CLIP_LENGTH, OVERLAP_DURATION)

        # Cleanup the raw segment
        os.remove(temp_segment)

# Rename files sequentially after processing
rename_files_sequentially(OUTPUT_DIR)

console.print("[bold green]All files processed. Check the 'targeted' directory for results.[/bold green]")

