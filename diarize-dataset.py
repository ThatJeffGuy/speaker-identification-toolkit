import os
import json
import time
from pydub import AudioSegment
from rich.console import Console
from pyannote.audio import Pipeline
import torch
import logging

# Suppress INFO logs and reproducibility warnings
logging.getLogger("pyannote.audio").setLevel(logging.WARNING)
logging.getLogger("pyannote.core").setLevel(logging.WARNING)
logging.getLogger("speechbrain").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Initialize Rich Console with a fixed width
CONSOLE_WIDTH = 50
console = Console(width=CONSOLE_WIDTH)

# Constants
SEPARATOR = "=" * CONSOLE_WIDTH

# Clear the terminal screen
def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

clear_console()

# Title Screen
console.print(
    f"{SEPARATOR}\n[bold cyan]DATASET DIARIZATION TOOL[/bold cyan]\n{SEPARATOR}\n",
    justify="center"
)
console.print(
    "This tool processes WAV audio files and generates speaker diarization metadata as JSON files.",
    justify="center"
)

# Define Paths Relative to Script Location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "wavs")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "jsons")
TOKEN_FILE = os.path.join(SCRIPT_DIR, ".hf_token")

# Ensure Directories Exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to get the Hugging Face token
def get_hugging_face_token():
    # Check if token exists in the token file
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r") as f:
            token = f.read().strip()
            console.print("[bold green]Found existing Hugging Face token.[/bold green]")
            
            # Ask if user wants to use the existing token
            use_existing = console.input(
                f"{SEPARATOR}\n[bold yellow]Use existing token? (y/n):\n{SEPARATOR}[/bold yellow]\n"
            ).strip().lower()
            if use_existing == 'y':
                return token
    
    # If no token or user wants a new one, prompt for input
    clear_console()
    console.print(
        f"{SEPARATOR}\n[bold cyan] HUGGING FACE TOKEN SETUP [/bold cyan]\n{SEPARATOR}\n",
        justify="left"
    )
    console.print("[bold yellow]Please enter your Hugging Face token.[/bold yellow]")
    console.print("[italic]Note: Your token will be visible as you type.[/italic]")
    console.print("[italic]You can get a token from:[/italic]")
    console.print("[italic]https://huggingface.co/settings/tokens[/italic]\n")
    
    # Use regular input for visible token
    token = input("Hugging Face Token: ").strip()
    
    # Save token to file for future use
    with open(TOKEN_FILE, "w") as f:
        f.write(token)
    
    console.print("\n[bold green]Token saved for future use.[/bold green]")
    return token

# Pause to allow user to populate audio directory
console.print(
    f"{SEPARATOR}\n[bold yellow]Please add WAV files to the directory:\n{INPUT_DIR}[/bold yellow]\n{SEPARATOR}",
    justify="left"
)
input("\nPress Enter to continue...")

# Validate Audio Files
audio_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".wav")]
if not audio_files:
    console.print(
        f"{SEPARATOR}\n[bold red]ERROR: No WAV files found in:\n{INPUT_DIR}[/bold red]\n{SEPARATOR}",
        justify="left"
    )
    exit(1)

console.print(
    f"{SEPARATOR}\n[bold green]Found {len(audio_files)} WAV file(s).\nStarting processing...[/bold green]\n{SEPARATOR}",
    justify="left"
)

# Get Hugging Face token
HF_TOKEN = get_hugging_face_token()

# Initialize Pyannote Pipeline with GPU Support
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold cyan]Using device: {device}[/bold cyan]")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HF_TOKEN)
    pipeline.to(device)
    console.print(
        f"[bold green]Hugging Face token validated.\nPipeline initialized successfully on {device}.[/bold green]"
    )
except Exception as e:
    console.print(
        f"{SEPARATOR}\n[bold red]ERROR: Failed to initialize pipeline:[/bold red]\n{e}\n{SEPARATOR}",
        justify="left"
    )
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
        console.print(
            f"{SEPARATOR}\n[bold red]ERROR processing {file_name}:[/bold red]\n{e}\n{SEPARATOR}",
            justify="left"
        )
        proceed = console.input(
            f"{SEPARATOR}\nDo you want to proceed with the next file? (y/n):\n{SEPARATOR}\n"
        ).strip().lower()
        if proceed != 'y':
            console.print("[bold red]Exiting processing.[/bold red]")
            exit(1)

# Sequential Processing
for file_name in audio_files:
    process_audio(file_name)

console.print(
    f"{SEPARATOR}\n[bold green]All files processed successfully![/bold green]\n{SEPARATOR}",
    justify="left"
)
console.print(
    f"[bold green]Check the directory below for results:\n{OUTPUT_DIR}[/bold green]",
    justify="left"
)