import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydub import AudioSegment
from rich.console import Console
from rich.prompt import Prompt
import multiprocessing

# Initialize Rich Console
console = Console()
console.clear()

# Title Screen
console.print("""[bold cyan]
=====================================
     VIDEO TO AUDIO EXTRACTION TOOL
=====================================
[/bold cyan]""")
console.print("This tool extracts audio tracks from video files and saves them as WAV files.")
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
video_files = [f for f in sorted(os.listdir(INPUT_DIR)) if f.endswith(('.mp4', '.mkv', '.avi'))]
if not video_files:
    console.print(f"[bold red]Error:[/bold red] No video files found in: {INPUT_DIR}", style="red")
    exit(1)

console.print(f"[bold green]Found {len(video_files)} video file(s). Starting processing...[/bold green]")

# Function to process a single file using pydub
def process_video(file_name):
    input_path = os.path.join(INPUT_DIR, file_name)
    output_file = os.path.join(OUTPUT_DIR, f"{os.path.splitext(file_name)[0]}.wav")

    try:
        console.print(f"[cyan]Processing:[/cyan] {file_name}")
        start_time = time.time()
        audio = AudioSegment.from_file(input_path)
        
        # Check if the audio file already matches the desired settings
        if audio.frame_rate != 44100 or audio.channels != 1:
            audio = audio.set_frame_rate(44100).set_channels(1)  # Set to 44.1kHz mono
        
        audio.export(output_file, format="wav")  # Export WAV
        end_time = time.time()
        console.print(f"[bold green]Success:[/bold green] Extracted to {output_file} in {end_time - start_time:.2f}s")
    except Exception as e:
        console.print(f"[bold red]Error processing {file_name}:[/bold red] {e}")

# Dynamically set thread count based on CPU cores
MAX_THREADS = min(8, max(1, multiprocessing.cpu_count() - 1))
console.print(f"[bold cyan]Using up to {MAX_THREADS} threads for processing...[/bold cyan]")

# Batch Processing
total_files = len(video_files)
batch_size = max(1, total_files // MAX_THREADS)
console.print(f"[bold cyan]Processing files in batches of {batch_size}...[/bold cyan]")

def batch_process(files_batch):
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [executor.submit(process_video, file_name) for file_name in files_batch]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                console.print(f"[bold red]Error during processing:[/bold red] {e}")

# Split files into batches
for i in range(0, total_files, batch_size):
    batch = video_files[i:i + batch_size]
    console.print(f"[bold blue]Processing batch {i // batch_size + 1}/{(total_files + batch_size - 1) // batch_size}...[/bold blue]")
    batch_process(batch)

console.print("\n[bold green]All files processed successfully.[/bold green]")
console.print(f"[bold green]Check the '{OUTPUT_DIR}' directory for extracted WAV files.[/bold green]")

