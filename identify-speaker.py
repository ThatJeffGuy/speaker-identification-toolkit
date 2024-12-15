import os
import json
import pandas as pd
import tempfile
import subprocess
from playsound import playsound
from rich.console import Console
from rich.prompt import Prompt
from rich.text import Text

# Initialize Rich Console
console = Console()
console.clear()

# Title Screen
console.print("""[bold cyan]
=====================================
     SPEAKER IDENTIFICATION TOOL
=====================================
[/bold cyan]""")
console.print("This tool will identify target speakers in audio files based on provided JSON metadata.")
console.print("Ensure the paths to JSON and audio files are correct before proceeding.\n")

# Paths
JSON_DIR = "./jsons"  # JSON files with all speakers
AUDIO_DIR = "./wavs"  # Original WAV files
MAPPING_FILE = "./mappings.csv"  # Speaker mapping CSV

# Helper Function to Validate Paths
def validate_path(path, path_type="directory", should_exist=True):
    """Validate a path and optionally create or exit on failure."""
    if path_type == "directory":
        if should_exist and not os.path.isdir(path):
            console.print(f"[bold red]Error:[/bold red] {path}", style="red")
            return False
        elif not should_exist:
            os.makedirs(path, exist_ok=True)
    elif path_type == "file":
        if should_exist and not os.path.isfile(path):
            console.print(f"[bold yellow]Warning:[/bold yellow] File does not exist: {path}", style="yellow")
            return False
    return True

# Validate Paths
paths_valid = True
paths_valid &= validate_path(JSON_DIR, "directory")
paths_valid &= validate_path(AUDIO_DIR, "directory")
paths_valid &= validate_path(MAPPING_FILE, "file", should_exist=False)

if not paths_valid:
    console.print("\n[bold red]One or more paths are invalid. Please fix the issues and restart the tool.[/bold red]", style="red")
    exit(1)

console.print("\n[bold green]All paths validated successfully![/bold green]\n")

# Confirm Start
start_tool = Prompt.ask("[bold yellow]Are you ready to start identifying speakers?[/bold yellow]", choices=["y", "n"], default="y")
if start_tool.lower() != "y":
    console.print("\n[bold red]Exiting tool as per user request.[/bold red]", style="red")
    exit(0)

# Load or Initialize Mapping File
df = pd.DataFrame(columns=['json_file', 'target_speaker_label'])
if os.path.isfile(MAPPING_FILE):
    df = pd.read_csv(MAPPING_FILE)

# Pre-validate Files
json_files = [f for f in os.listdir(JSON_DIR) if f.endswith(".json")]
valid_files = []
for json_file in json_files:
    json_path = os.path.join(JSON_DIR, json_file)
    wav_file = os.path.join(AUDIO_DIR, os.path.splitext(json_file)[0] + ".wav")
    if not os.path.exists(json_path) or not os.path.exists(wav_file):
        console.print(f"[bold yellow]Skipping {json_file}:[/bold yellow] Missing JSON or WAV file.", style="yellow")
    else:
        valid_files.append((json_file, json_path, wav_file))

# Display Initial Counts
console.print(f"[bold green]Found {len(valid_files)} valid file(s) to process.[/bold green]")
already_mapped_count = 0
valid_files_to_process = []

# Filter Out Already Mapped Files
for json_file, json_path, wav_file in valid_files:
    if json_file in df['json_file'].values and pd.notna(df.loc[df['json_file'] == json_file, 'target_speaker_label'].values[0]):
        already_mapped_count += 1
    else:
        valid_files_to_process.append((json_file, json_path, wav_file))

# Display Count of Already Mapped Files
if already_mapped_count > 0:
    console.print(f"[bold blue]Skipped {already_mapped_count} already mapped file(s).[/bold blue]", style="blue")

# Process Files
for json_index, (json_file, json_path, wav_file) in enumerate(valid_files_to_process, start=1):
    with open(json_path, "r") as f:
        try:
            segments = json.load(f)
        except json.JSONDecodeError:
            console.print(f"[bold red]Error:[/bold red] JSON file {json_file} is not properly formatted. Skipping.", style="red")
            continue

    if not segments:
        console.print(f"[bold yellow]No segments found in {json_file}. Skipping.[/bold yellow]", style="yellow")
        continue

    console.print(f"Processing Episode: [bold green]{json_file}[/bold green]", style="bold blue")

    unique_speakers = list(set(seg["speaker"] for seg in segments))
    target_speaker_label = None

    for speaker_index, speaker_label in enumerate(unique_speakers, start=1):
        console.print(f"Speaker Number/Total: {speaker_index}/{len(unique_speakers)}", style="cyan")

        speaker_segments = [seg for seg in segments if seg["speaker"] == speaker_label]

        for segment_index, segment in enumerate(speaker_segments, start=1):
            start_time, end_time = segment["start"], segment["end"]

            if not (2.0 <= end_time - start_time <= 10.0):
                continue

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_segment:
                segment_path = temp_segment.name
                cmd = [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", wav_file, "-ss", str(start_time), "-to", str(end_time), "-c", "copy",
                    segment_path
                ]
                subprocess_run = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if subprocess_run.returncode != 0:
                    console.print(f"[bold red]Error extracting segment {start_time}-{end_time} for speaker {speaker_label}.[/bold red]", style="red")
                    continue

            try:
                playsound(segment_path)
            except Exception as e:
                console.print(f"[bold red]Error playing segment for speaker {speaker_label}:[/bold red] {e}", style="red")
                os.remove(segment_path)
                continue

            os.remove(segment_path)

            user_input = Prompt.ask(f"Is this the target speaker? [Y]es, [N]o, [T]ry Another Segment")

            if user_input.lower() == "y":
                target_speaker_label = speaker_label
                break
            elif user_input.lower() == "n":
                console.print("[bold magenta]Skipping this speaker and moving to the next one immediately.[/bold magenta]", style="magenta")
                break
            elif user_input.lower() == "t":
                console.print("[bold cyan]Trying another segment for the same speaker.[/bold cyan]", style="cyan")
                continue

        if target_speaker_label:
            break

    if target_speaker_label:
        df = pd.concat([df, pd.DataFrame({"json_file": [json_file], "target_speaker_label": [target_speaker_label]})], ignore_index=True)
        console.print(f"[bold green]Mapped {json_file} to speaker {target_speaker_label}.[/bold green]", style="green")
    else:
        df = pd.concat([df, pd.DataFrame({"json_file": [json_file], "target_speaker_label": ["not-target-speaker"]})], ignore_index=True)
        console.print(f"[bold yellow]No target speaker identified for {json_file}.[/bold yellow]", style="yellow")

    console.print(f"[bold blue]Processed entries: {json_index}/{len(valid_files_to_process)}[/bold blue]", style="blue")
    console.print(f"[bold blue]Unprocessed remaining: {len(valid_files_to_process) - json_index}[/bold blue]", style="blue")

    df.to_csv(MAPPING_FILE, index=False)
    console.print(f"[bold cyan]Mapping saved to {MAPPING_FILE}.[/bold cyan]", style="cyan")

    # Ask user whether to process the next episode
    continue_processing = Prompt.ask("[bold yellow]Do you want to process the next episode? Default: yes[/bold yellow]", choices=["y", "n"], default="y")
    if continue_processing.lower() != "y":
        console.print("[bold red]Stopping processing as per user request.[/bold red]", style="red")
        break

console.print("[bold green]Processing complete.[/bold green]")
