import os
import re
import subprocess
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeElapsedColumn

# Initialize Console
console = Console()
console.clear()

# Title Screen
console.print("""[bold cyan]
=====================================
     ENG AUDIO EXTRACTION TOOL
=====================================
[/bold cyan]""")
console.print("This tool extracts only English audio tracks from video files using FFmpeg.\n")

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

# Function to extract English audio
def extract_eng_audio(file):
    input_path = os.path.join(VIDEO_DIR, file)
    output_path = os.path.join(OUTPUT_DIR, os.path.splitext(file)[0] + "_eng.wav")

    try:
        # Use FFmpeg to find and extract the English audio track
        command = [
            "ffmpeg", "-i", input_path,
            "-map", "0:m:language:eng",  # Map English audio track
            "-acodec", "pcm_s16le",      # 16-bit PCM
            "-ar", "44100",             # 44.1kHz
            "-ac", "1",                 # Mono audio
            "-y", output_path            # Overwrite existing file
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return file, "Success"
    except subprocess.CalledProcessError as e:
        return file, f"Error: {e}"

# Sequential Processing with Progress Bar
results = []
with Progress(
    "[progress.percentage]{task.percentage:>3.0f}%",
    BarColumn(),
    TimeElapsedColumn(),
    console=console
) as progress:
    task = progress.add_task("[cyan]Extracting English audio...", total=len(video_files))

    for file in video_files:
        original, status = extract_eng_audio(file)
        progress.update(task, advance=1)

        if status == "Success":
            progress.console.log(f"[green]Extracted:[/green] {original}")
        else:
            progress.console.log(f"[yellow]{status}:[/yellow] {original}")

        results.append((original, status))

# Summary
console.print("\n[bold green]Processing complete![/bold green]")
for original, status in results:
    console.print(f"[white]{original}[/white] -> [cyan]{status}[/cyan]")

console.print(f"\n[bold green]Check the '{OUTPUT_DIR}' directory for extracted audio files.[/bold green]")

