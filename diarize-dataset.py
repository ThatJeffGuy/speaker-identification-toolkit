import os
import json
import time
import getpass
from rich.console import Console
import logging
import sys

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

# Try to import pyannote.audio
try:
    from pyannote.audio import Pipeline
    import torch
    from pydub import AudioSegment
except ImportError:
    console.print("\n[bold red]ERROR: Required package pyannote.audio not found![/bold red]")
    console.print("\n[bold yellow]Installation Instructions:[/bold yellow]")
    console.print("1. [white]Install CMake (required for building dependencies):[/white]")
    console.print("   - Windows: Download from https://cmake.org/download/")
    console.print("   - Or use conda: [cyan]conda install -c conda-forge cmake[/cyan]")
    console.print("\n2. [white]Install PyTorch with audio support:[/white]")
    console.print("   [cyan]pip install torch torchaudio[/cyan]")
    console.print("\n3. [white]Install pyannote.audio:[/white]")
    console.print("   [cyan]pip install pyannote.audio[/cyan]")
    console.print("\n[bold yellow]For more information, visit:[/bold yellow]")
    console.print("https://github.com/pyannote/pyannote-audio")
    sys.exit(1)

# Define Paths Relative to Script Location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "wavs")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "jsons")
CONFIG_FILE = os.path.join(SCRIPT_DIR, ".env")

# Try to load token from environment
HF_TOKEN = os.getenv("HF_TOKEN")

# If token not found, prompt the user
if not HF_TOKEN:
    console.print("[bold yellow]Hugging Face token not found in environment[/bold yellow]")
    console.print("[cyan]You need a Hugging Face token to access the speaker diarization model.[/cyan]")
    console.print("[cyan]You can get it from https://huggingface.co/settings/tokens[/cyan]")
    
    # Ask if user wants to save the token
    save_token = console.input("[yellow]Do you want to save the token for future use? (y/n): [/yellow]").strip().lower() == 'y'
    
    # Get the token (using getpass to hide input)
    HF_TOKEN = getpass.getpass("Enter your Hugging Face token: ")
    
    # Save the token if requested
    if save_token and HF_TOKEN:
        try:
            with open(CONFIG_FILE, 'w') as f:
                f.write(f"HF_TOKEN={HF_TOKEN}\n")
            console.print(f"[green]Token saved to {CONFIG_FILE}[/green]")
            console.print("[yellow]Note: This file contains sensitive information. Keep it secure.[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Could not save token: {e}[/yellow]")

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
        diarization_data = []
        for entry in diarization.itertracks(yield_label=True):
            if isinstance(entry, tuple) and len(entry) >= 2:
                turn = entry[0]  # The segment
                label = entry[-1]  # The speaker label
                diarization_data.append({
                    "speaker": label,
                    "start": turn.start,
                    "end": turn.end
                })
            else:
                console.print(f"[bold yellow]Unexpected diarization entry format: {entry}[/bold yellow]")

        # Save output to JSON
        with open(output_file, "w") as f:
            json.dump(diarization_data, f, indent=2)

        end_time = time.time()
        console.print(f"[bold green]Success:[/bold green] Processed {file_name} in {end_time - start_time:.2f}s")
    except Exception as e:
        console.print(f"[bold red]Error processing {file_name}:[/bold red] {e}")
        proceed = input("Do you want to proceed with the next file? (y/n): ").strip().lower()
        if proceed != 'y':
            console.print("[bold red]Exiting processing.[/bold red]")
            exit(1)

# Sequential Processing
for file_name in audio_files:
    process_audio(file_name)

console.print("\n[bold green]All files processed successfully.[/bold green]")
console.print(f"[bold green]Check the '{OUTPUT_DIR}' directory for diarization JSON files.[/bold green]")