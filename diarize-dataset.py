import os
import json
from rich.console import Console
from rich.prompt import Prompt

# Initialize Rich Console
console = Console()
console.clear()

# Title Screen
console.print("""[bold cyan]
=====================================
      SPEAKER DIARIZATION TOOL
=====================================
[/bold cyan]""")
console.print("This tool processes audio files and generates speaker diarization metadata.")
console.print("Input and output directories will be created in the same folder as this script.")

# Define Paths Relative to Script Location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "wavs")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "jsons")

# Ensure Directories Exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Prompt User for Hugging Face Token
HF_TOKEN = Prompt.ask("[bold cyan]Enter your Hugging Face token[/bold cyan]", default="")

# Validate Token
if not HF_TOKEN or not HF_TOKEN.startswith("hf_"):
    console.print("[bold red]Error:[/bold red] Invalid Hugging Face token.", style="red")
    exit(1)

# Pause to allow user to populate audio directory
console.print(f"[bold yellow]Please add audio files (WAV format) to the directory: {INPUT_DIR}[/bold yellow]")
input("\nPress Enter to continue...")

# Validate Audio Files
audio_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".wav")]
if not audio_files:
    console.print(f"[bold red]Error:[/bold red] No WAV files found in: {INPUT_DIR}", style="red")
    exit(1)

# Mocked pipeline for diarization (Replace with actual pipeline initialization)
def mock_pipeline(file_path):
    """Mocked diarization function, replace with actual implementation."""
    return [
        {"speaker": "SPEAKER_1", "start": 0.0, "end": 10.0},
        {"speaker": "SPEAKER_2", "start": 10.0, "end": 20.0}
    ]

# Process Audio Files
console.print(f"[bold green]Found {len(audio_files)} audio file(s). Starting processing...[/bold green]")
for index, file_name in enumerate(audio_files, start=1):
    input_path = os.path.join(INPUT_DIR, file_name)
    output_file = os.path.join(OUTPUT_DIR, f"{os.path.splitext(file_name)[0]}.json")

    console.print(f"Processing: [bold blue]{file_name}[/bold blue] ({index}/{len(audio_files)})")

    try:
        # Perform diarization (Replace `mock_pipeline` with actual diarization pipeline call)
        diarization_data = mock_pipeline(input_path)

        # Save output to JSON
        with open(output_file, "w") as f:
            json.dump(diarization_data, f, indent=2)

        console.print(f"[bold green]Success:[/bold green] Diarization output saved to: {output_file}", style="green")
    except Exception as e:
        console.print(f"[bold red]Error processing {file_name}:[/bold red] {e}", style="red")

console.print("[bold green]All files processed successfully.[/bold green]")

