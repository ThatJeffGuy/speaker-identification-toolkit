import os
import subprocess
from rich.console import Console

# Initialize Rich Console
console = Console()
console.clear()

# Title Screen
console.print("""[bold cyan]
=====================================
     VIDEO TO AUDIO EXTRACTION TOOL
=====================================
[/bold cyan]""")
console.print("This tool extracts audio tracks from video files using FFmpeg.")
console.print("Input and output directories will be created in the same folder as this script.")

# Define Paths Relative to Script Location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "videos")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "wavs")

# Ensure Directories Exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pause to allow user to populate videos directory
console.print(f"[bold yellow]Please add video files to the directory: {INPUT_DIR}[/bold yellow]")
input("\nPress Enter to continue...")

# Process Video Files
video_files = [f for f in sorted(os.listdir(INPUT_DIR)) if f.endswith((".mkv", ".mp4"))]
if not video_files:
    console.print(f"[bold red]Error:[/bold red] No video files found in: {INPUT_DIR}", style="red")
    exit(1)

console.print(f"[bold green]Found {len(video_files)} video file(s). Starting processing...[/bold green]")
for index, file_name in enumerate(video_files, start=1):
    input_path = os.path.join(INPUT_DIR, file_name)
    output_file = os.path.join(OUTPUT_DIR, f"{os.path.splitext(file_name)[0]}.wav")

    # Construct FFmpeg command
    ffmpeg_command = [
        "ffmpeg",
        "-i", input_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_file
    ]

    console.print(f"Processing: [bold blue]{file_name}[/bold blue] ({index}/{len(video_files)})")

    # Execute the FFmpeg command
    try:
        subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        console.print(f"[bold green]Success:[/bold green] Extracted audio to: {output_file}", style="green")
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Error processing {file_name}:[/bold red] {e}", style="red")

console.print("[bold green]All files processed successfully.[/bold green]")

