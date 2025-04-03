import os
import json
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TextColumn
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Initialize Rich Console
console = Console()
console.clear()

# Constants
SEPARATOR = "=" * 50

# Title Screen
def print_title():
    title = Text("SPEAKER ISOLATION TOOL", style="bold cyan")
    console.print(Panel(title, border_style="cyan", expand=False, padding=(1, 2)))
    console.print("This tool extracts audio segments from identified speakers.")
    console.print("[italic yellow]Audio clips are saved with speaker label, start and end timestamps.[/italic yellow]\n")

# Define Paths Relative to Script Location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_DIR = os.path.join(SCRIPT_DIR, "jsons")
AUDIO_DIR = os.path.join(SCRIPT_DIR, "wavs")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "targeted")
MAPPING_FILE = os.path.join(SCRIPT_DIR, "mappings.csv")

# Settings for segment extraction
MAX_CLIP_LENGTH = 5  # Maximum length for trimmed clips (in seconds)
OVERLAP_DURATION = 0.5  # Overlap duration in seconds

def process_file(file_data):
    """Processes a single file for the target speaker."""
    json_file, target_speaker_label = file_data
    json_path = os.path.join(JSON_DIR, json_file)
    wav_file = os.path.splitext(json_file)[0] + ".wav"
    wav_path = os.path.join(AUDIO_DIR, wav_file)

    if not os.path.exists(json_path) or not os.path.exists(wav_path):
        return 0, json_file, f"Missing JSON or WAV file"

    # Load JSON and filter target speaker segments
    try:
        with open(json_path, "r") as f:
            segments = json.load(f)
    except Exception as e:
        return 0, json_file, f"Failed to read JSON: {e}"

    # Remove duplicates and get only target speaker segments
    seen_segments = {}
    target_segments = [
        seg for seg in segments 
        if seg['speaker'] == target_speaker_label and not seen_segments.setdefault((seg['start'], seg['end']), True)
    ]

    if not target_segments:
        return 0, json_file, f"No segments for target speaker '{target_speaker_label}'"

    # Load the entire audio file once
    try:
        audio = AudioSegment.from_wav(wav_path)
    except Exception as e:
        return 0, json_file, f"Failed to load audio: {e}"

    # Process segments
    segment_count = 0
    base_filename = os.path.splitext(json_file)[0]
    
    for segment in target_segments:
        start_time, end_time = segment["start"] * 1000, segment["end"] * 1000  # Convert to milliseconds
        segment_duration = end_time - start_time
        
        # Skip very short segments
        if segment_duration < 1000:  # Shorter than 1 second
            continue

        # Extract segment audio
        segment_audio = audio[start_time:end_time]
        if len(segment_audio) == 0:  # Skip empty segments
            continue

        # Format output filename with speaker and timestamps
        output_file = os.path.join(
            OUTPUT_DIR, 
            f"{base_filename}_{target_speaker_label}_{start_time/1000:.2f}-{end_time/1000:.2f}.wav"
        )
        
        # Export audio segment
        segment_audio.export(output_file, format="wav")
        segment_count += 1

    return segment_count, json_file, "Success"

def main():
    # Display title
    print_title()
    
    # Ensure Directories Exist
    os.makedirs(JSON_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load Speaker Mapping
    if not os.path.exists(MAPPING_FILE):
        console.print(f"[bold red]Error:[/bold red] Speaker mapping file not found: {MAPPING_FILE}", style="red")
        console.print(f"[yellow]Please run the identify-speaker.py script first to create speaker mappings.[/yellow]")
        return
    
    try:
        df_mapping = pd.read_csv(MAPPING_FILE)
        if 'wav_file' not in df_mapping.columns or 'speaker' not in df_mapping.columns:
            console.print(f"[bold red]Error:[/bold red] Invalid mapping file format. Expected columns 'wav_file' and 'speaker'.", style="red")
            return
    except Exception as e:
        console.print(f"[red]Error: Failed to read {MAPPING_FILE}. {e}[/red]")
        return
    
    console.print(f"[bold green]Loaded speaker mapping for {len(df_mapping)} files.[/bold green]")
    
    # Create file processing list
    files_to_process = []
    skipped_files = []
    
    for _, row in df_mapping.iterrows():
        wav_file = row['wav_file']
        json_file = os.path.splitext(wav_file)[0] + ".json"
        target_speaker = row['speaker']
        
        if pd.isna(target_speaker) or target_speaker == "":
            skipped_files.append((json_file, "No target speaker specified"))
            continue
            
        files_to_process.append((json_file, target_speaker))
    
    if not files_to_process:
        console.print("[bold yellow]No files to process. All mappings are either empty or already processed.[/bold yellow]")
        return
    
    # Ask for confirmation
    console.print(f"[bold green]Found {len(files_to_process)} files to process.[/bold green]")
    proceed = console.input("[bold yellow]Do you want to proceed with speaker isolation? (y/n): [/bold yellow]").strip().lower()
    if proceed != 'y':
        console.print("[yellow]Operation cancelled by user.[/yellow]")
        return
    
    console.print("[bold cyan]Starting multi-threaded processing...[/bold cyan]")
    
    # Multi-threaded Processing
    MAX_THREADS = min(8, max(1, multiprocessing.cpu_count() - 1))
    console.print(f"[cyan]Using up to {MAX_THREADS} threads for processing...[/cyan]")
    
    total_segments = 0
    successful_files = []
    failed_files = []
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Isolating speakers...", total=len(files_to_process))
        
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            futures = {executor.submit(process_file, file_data): file_data for file_data in files_to_process}
            
            for future in as_completed(futures):
                file_data = futures[future]
                try:
                    segments, json_file, status = future.result()
                    
                    if status == "Success":
                        total_segments += segments
                        successful_files.append((json_file, segments))
                        progress.console.log(f"[green]Processed:[/green] {json_file} ({segments} segments)")
                    else:
                        failed_files.append((json_file, status))
                        progress.console.log(f"[red]Failed:[/red] {json_file} - {status}")
                except Exception as e:
                    failed_files.append((file_data[0], str(e)))
                    progress.console.log(f"[red]Error:[/red] {file_data[0]} - {e}")
                
                progress.update(task, advance=1)
    
    # Final summary
    console.print(f"\n{SEPARATOR}")
    console.print(f"[bold green]Processing complete![/bold green]")
    console.print(f"[bold cyan]Summary:[/bold cyan]")
    console.print(f"[green]Successfully processed: {len(successful_files)} files[/green]")
    console.print(f"[green]Total segments saved: {total_segments}[/green]")
    console.print(f"[yellow]Failed: {len(failed_files)} files[/yellow]")
    
    if successful_files:
        console.print("\n[bold green]Top files by segment count:[/bold green]")
        # Sort by segment count and show top 5
        for json_file, count in sorted(successful_files, key=lambda x: x[1], reverse=True)[:5]:
            console.print(f"[white]{json_file}:[/white] [cyan]{count} segments[/cyan]")
    
    console.print(f"\n[bold green]Check the '{OUTPUT_DIR}' directory for isolated audio segments.[/bold green]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold red]Program interrupted by user. Exiting...[/bold red]")
    except Exception as e:
        console.print(f"\n[bold red]An unexpected error occurred: {e}[/bold red]")