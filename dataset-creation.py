import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeElapsedColumn
import multiprocessing
import json

# Initialize Console
console = Console()
console.clear()

# Title Screen
console.print("""[bold cyan]
=====================================
     ENG AUDIO EXTRACTION TOOL
=====================================
[/bold cyan]""")
console.print("This tool extracts only English audio tracks from video files using FFmpeg with essential metadata passthrough.\n")

# Define Directory Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(SCRIPT_DIR, "videos")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "wavs")

# Ensure Directories Exist
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List Video Files
video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith((".mkv", ".mp4", ".avi"))]
if not video_files:
    console.print(f"[bold yellow]No video files found in '{VIDEO_DIR}'.[/bold yellow]")
    exit(1)

console.print(f"[bold green]Found {len(video_files)} video file(s). Starting extraction...[/bold green]")

# Function to extract English audio with minimal metadata
def extract_eng_audio(file):
    input_path = os.path.join(VIDEO_DIR, file)
    output_path = os.path.join(OUTPUT_DIR, os.path.splitext(file)[0] + ".wav")

    try:
        # Use ffprobe to parse JSON output for stream info
        ffprobe_command = [
            "ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
            "stream=index:stream_tags=language", "-of", "json", input_path
        ]
        result = subprocess.run(ffprobe_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stream_data = json.loads(result.stdout)

        # Find English audio stream index
        stream_index = None
        for stream in stream_data.get("streams", []):
            if stream.get("tags", {}).get("language", "") == "eng":
                stream_index = stream["index"]
                break

        if stream_index is None:
            return file, "No English audio track found"

        # Use FFmpeg to clean metadata and extract the English audio track
        command = [
            "ffmpeg", "-i", input_path,
            "-map", f"0:{stream_index}",  # Map the identified English audio stream
            "-acodec", "pcm_s16le",         # 16-bit PCM
            "-ar", "44100",                # 44.1kHz
            "-ac", "1",                    # Mono audio
            "-map_metadata", "-1",         # Strip all metadata
            "-y", output_path               # Overwrite existing file
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return file, "Success"
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        return file, f"Error: {e}"

# Multithreaded Processing with Progress Bar
MAX_THREADS = min(8, max(1, multiprocessing.cpu_count() - 1))
console.print(f"[bold cyan]Using up to {MAX_THREADS} threads for processing...[/bold cyan]")

results = []
with Progress(
    "[progress.percentage]{task.percentage:>3.0f}%",
    BarColumn(),
    TimeElapsedColumn(),
    console=console
) as progress:
    task = progress.add_task("[cyan]Extracting English audio...", total=len(video_files))

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {executor.submit(extract_eng_audio, file): file for file in video_files}
        for future in as_completed(futures):
            file = futures[future]
            try:
                original, status = future.result()
                if status == "Success":
                    progress.console.log(f"[green]Extracted:[/green] {original}")
                elif "No English audio track found" in status:
                    progress.console.log(f"[yellow]{status}:[/yellow] {original}")
                else:
                    progress.console.log(f"[red]{status}:[/red] {original}")
                results.append((original, status))
            except Exception as e:
                progress.console.log(f"[red]Error processing {file}:[/red] {e}")
            progress.update(task, advance=1)

# Summary
console.print("\n[bold green]Processing complete![/bold green]")
for original, status in results:
    console.print(f"[white]{original}[/white] -> [cyan]{status}[/cyan]")

console.print(f"\n[bold green]Check the '{OUTPUT_DIR}' directory for extracted audio files.[/bold green]")

