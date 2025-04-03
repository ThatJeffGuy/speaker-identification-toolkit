import os
import json
import time
from pydub import AudioSegment
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TextColumn
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

# Clear the terminal screen
def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

# Constants
SEPARATOR = "=" * CONSOLE_WIDTH

# Title Screen
def print_title():
    title = Text("SPEAKER DIARIZATION TOOL", style="bold cyan")
    console.print(Panel(title, border_style="cyan", expand=False, padding=(1, 2)))
    console.print("This tool processes WAV audio files and generates speaker diarization metadata.")
    console.print("[italic yellow]Speaker diarization identifies who spoke when in an audio recording.[/italic yellow]\n")

# Define Paths Relative to Script Location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "wavs")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "jsons")
HF_TOKEN = ""  # Replace with actual token

def main():
    # Display title
    print_title()
    
    # Ensure Directories Exist
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Validate HF Token
    if HF_TOKEN.startswith("hf_"):
        console.print("[cyan]Using Hugging Face token for authentication.[/cyan]")
    else:
        console.print("[bold red]Warning: Invalid Hugging Face token format.[/bold red]")
        console.print("[yellow]You may need to update the HF_TOKEN variable with a valid token.[/yellow]")
        update_token = console.input("[bold yellow]Do you want to provide a token now? (y/n): [/bold yellow]").strip().lower()
        if update_token == 'y':
            new_token = console.input("[bold cyan]Enter your Hugging Face token: [/bold cyan]").strip()
            if new_token and new_token.startswith("hf_"):
                console.print("[green]Token accepted.[/green]")
                global HF_TOKEN
                HF_TOKEN = new_token
            else:
                console.print("[bold red]Invalid token format. Exiting.[/bold red]")
                return
    
    # Pause to allow user to populate audio directory
    console.print(f"[bold yellow]Please add WAV files to the directory: {INPUT_DIR}[/bold yellow]")
    input("\nPress Enter to continue...")
    
    # Validate Audio Files
    audio_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".wav")]
    if not audio_files:
        console.print(f"[bold red]Error:[/bold red] No WAV files found in: {INPUT_DIR}", style="red")
        return
    
    console.print(f"[bold green]Found {len(audio_files)} WAV file(s). Starting processing...[/bold green]")
    
    # Initialize Pyannote Pipeline with GPU Support
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        console.print(f"[bold cyan]Using device: {device}[/bold cyan]")
        
        with console.status("[bold cyan]Initializing diarization pipeline...[/bold cyan]"):
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HF_TOKEN)
            pipeline.to(device)
        
        console.print("[bold green]Hugging Face token validated. Pipeline initialized successfully.[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Failed to initialize pipeline:[/bold red] {e}")
        return
    
    # Process each audio file sequentially with progress updates
    for file_index, file_name in enumerate(audio_files, 1):
        input_path = os.path.join(INPUT_DIR, file_name)
        output_file = os.path.join(OUTPUT_DIR, f"{os.path.splitext(file_name)[0]}.json")
        
        # Skip if output already exists
        if os.path.exists(output_file):
            console.print(f"[yellow]Skipping:[/yellow] {file_name} (output already exists)")
            continue
        
        try:
            console.print(f"\n[cyan]Processing ({file_index}/{len(audio_files)}):[/cyan] {file_name}")
            start_time = time.time()
            
            # Load and validate audio using pydub
            with console.status("[cyan]Validating audio format...[/cyan]"):
                audio = AudioSegment.from_wav(input_path)
                if audio.frame_rate != 44100 or audio.channels != 1:
                    console.print(f"[yellow]Converting:[/yellow] {file_name} to 44.1kHz mono")
                    audio = audio.set_frame_rate(44100).set_channels(1)
                    audio.export(input_path, format="wav")
            
            # Perform diarization with pipeline
            with console.status("[cyan]Performing speaker diarization...[/cyan]"):
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
            
            # Count speakers
            speakers = set(item["speaker"] for item in diarization_data)
            
            # Save output to JSON
            with open(output_file, "w") as f:
                json.dump(diarization_data, f, indent=2)
            
            end_time = time.time()
            console.print(f"[bold green]Success:[/bold green] Processed {file_name} in {end_time - start_time:.2f}s")
            console.print(f"[green]Identified {len(speakers)} unique speakers with {len(diarization_data)} segments[/green]")
            
        except Exception as e:
            console.print(f"[bold red]Error processing {file_name}:[/bold red] {e}")
            proceed = console.input("[bold yellow]Do you want to proceed with the next file? (y/n): [/bold yellow]").strip().lower()
            if proceed != 'y':
                console.print("[bold red]Exiting processing.[/bold red]")
                break
    
    # Calculate statistics
    processed_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".json")]
    
    console.print(f"\n{SEPARATOR}")
    console.print("[bold green]Processing complete![/bold green]")
    console.print(f"[bold green]Successfully processed: {len(processed_files)}/{len(audio_files)} files[/bold green]")
    console.print(f"[bold green]Check the '{OUTPUT_DIR}' directory for diarization JSON files.[/bold green]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold red]Program interrupted by user. Exiting...[/bold red]")
    except Exception as e:
        console.print(f"\n[bold red]An unexpected error occurred: {e}[/bold red]")