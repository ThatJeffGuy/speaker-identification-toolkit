import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydub import AudioSegment
from rich.console import Console
from rich.prompt import Prompt
import multiprocessing
from pyannote.audio import Pipeline
import torch

# Initialize Rich Console
console = Console()
console.clear()

# Title Screen
console.print("""[bold cyan]
=====================================
      SPEAKER DIARIZATION TOOL
=====================================
[/bold cyan]""")
console.print("This tool processes WAV audio files and generates speaker diarization metadata.")
console.print("Input and output directories will be created in the same folder as this script.")

# Define Paths Relative to Script Location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "wavs")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "jsons")

# Ensure Directories Exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pause to allow user to populate audio directory
console.print(f"[bold yellow]Please add WAV files to the directory: {INPUT_DIR}[/bold yellow]")
input("\nPress Enter to continue...")

# Validate Audio Files
audio_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".wav")]
if not audio_files:
    console.print(f"[bold red]Error:[/bold red] No WAV files found in: {INPUT_DIR}", style="red")
    exit(1)

console.print(f"[bold green]Found {len(audio_files)} WAV file(s). Starting processing...[/bold green]")

# Prompt User for Hugging Face Token
HF_TOKEN = Prompt.ask("[bold cyan]Enter your Hugging Face token[/bold cyan]", default="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")
if not HF_TOKEN or not HF_TOKEN.startswith("hf_"):
    console.print("[bold red]Error:[/bold red] Invalid Hugging Face token.", style="red")
    exit(1)

# Initialize Pyannote Pipeline with GPU Support
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold cyan]Using device: {device}[/bold cyan]")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HF_TOKEN)
    pipeline.to(device)
    console.print("[bold green]Hugging Face token validated. Pipeline initialized successfully on GPU.[/bold green]")
except Exception as e:
    console.print(f"[bold red]Failed to initialize pipeline:[/bold red] {e}")
    exit(1)

# Function to process a single audio file
def process_audio(file_name):
    input_path = os.path.join(INPUT_DIR, file_name)
    output_file = os.path.join(OUTPUT_DIR, f"{os.path.splitext(file_name)[0]}.json")

    try:
        console.print(f"[cyan]Processing:[/cyan] {file_name}")
        start_time = time.time()

        # Load and validate audio using pydub
        audio = AudioSegment.from_wav(input_path)
        if audio.frame_rate != 44100 or audio.channels != 1:
            audio = audio.set_frame_rate(44100).set_channels(1)
            console.print(f"[yellow]Converted:[/yellow] {file_name} to 44.1kHz mono")
            audio.export(input_path, format="wav")

        # Perform diarization with pipeline
        diarization = pipeline(input_path)
        diarization_data = [
            {"speaker": turn.label, "start": turn.start, "end": turn.end}
            for turn in diarization.itertracks(yield_label=True)
        ]

        # Save output to JSON
        with open(output_file, "w") as f:
            json.dump(diarization_data, f, indent=2)

        end_time = time.time()
        console.print(f"[bold green]Success:[/bold green] Processed {file_name} in {end_time - start_time:.2f}s")
    except Exception as e:
        console.print(f"[bold red]Error processing {file_name}:[/bold red] {e}")

# Dynamically set thread count based on CPU cores
MAX_THREADS = min(8, max(1, multiprocessing.cpu_count() - 1))
console.print(f"[bold cyan]Using up to {MAX_THREADS} threads for processing...[/bold cyan]")

# Multithreaded Processing
with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    futures = [executor.submit(process_audio, file_name) for file_name in audio_files]
    for future in as_completed(futures):
        try:
            future.result()
        except Exception as e:
            console.print(f"[bold red]Error during processing:[/bold red] {e}")

console.print("\n[bold green]All files processed successfully.[/bold green]")
console.print(f"[bold green]Check the '{OUTPUT_DIR}' directory for diarization JSON files.[/bold green]")
