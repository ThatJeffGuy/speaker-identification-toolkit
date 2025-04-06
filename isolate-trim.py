import os
import json
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TextColumn
from rich.table import Table
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import re

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
GLOBAL_MAPPING_FILE = os.path.join(SCRIPT_DIR, "global_mappings.csv")

# Settings for segment extraction
MAX_CLIP_LENGTH = 5  # Maximum length for trimmed clips (in seconds)
OVERLAP_DURATION = 0.5  # Overlap duration in seconds

def find_matching_speaker(target_speaker, available_speakers):
    """
    Find the best matching speaker from available speakers based on the target speaker.
    Uses multiple strategies to match speaker labels.
    """
    # Direct match
    if target_speaker in available_speakers:
        return target_speaker
    
    # Case insensitive match
    lower_target = target_speaker.lower()
    for speaker in available_speakers:
        if speaker.lower() == lower_target:
            return speaker
    
    # Extract number from target (for Speaker_X, REVIEW_Speaker_X patterns)
    number_match = re.search(r'(\d+)', target_speaker)
    if number_match:
        speaker_number = number_match.group(1)
        
        # Try matching just the number
        if speaker_number in available_speakers:
            return speaker_number
        
        # Try matching Speaker_X
        speaker_x = f"Speaker_{speaker_number}"
        if speaker_x in available_speakers:
            return speaker_x
        
        # Try matching just "X" where X is a number
        for speaker in available_speakers:
            if speaker == speaker_number:
                return speaker
    
    # If the target is REVIEW_Speaker_X, try Speaker_X
    if target_speaker.startswith("REVIEW_Speaker_"):
        speaker_suffix = target_speaker[len("REVIEW_"):]
        if speaker_suffix in available_speakers:
            return speaker_suffix
    
    # If there's only one speaker, use it
    if len(available_speakers) == 1:
        return list(available_speakers)[0]
    
    # Prefer Speaker_1 or 1 if available
    if "Speaker_1" in available_speakers:
        return "Speaker_1"
    if "1" in available_speakers:
        return "1"
    
    # No match found
    return None

def process_file(file_data):
    """Processes a single file for the target speaker."""
    try:
        # Handle None or improperly formatted file_data
        if not file_data or not isinstance(file_data, tuple) or len(file_data) < 3:
            return 0, "Unknown file", "Invalid file data format"
            
        # Unpack tuple safely - now includes the json_speaker to look for
        json_file, target_speaker, json_speaker = file_data
        
        # Convert json_file to string if needed
        if not isinstance(json_file, str):
            json_file = str(json_file)
        
        # Fix extension if needed
        if not json_file.endswith('.json'):
            json_file = os.path.splitext(json_file)[0] + '.json'
            
        wav_file = os.path.splitext(json_file)[0] + '.wav'
        
        json_path = os.path.join(JSON_DIR, json_file)
        wav_path = os.path.join(AUDIO_DIR, wav_file)

        # Check if files exist
        if not os.path.exists(json_path):
            return 0, json_file, f"JSON file not found: {json_path}"
        if not os.path.exists(wav_path):
            return 0, json_file, f"WAV file not found: {wav_path}"

        # Load JSON data
        try:
            with open(json_path, "r") as f:
                segments = json.load(f)
        except Exception as e:
            return 0, json_file, f"Failed to read JSON: {e}"

        # Get all available speakers
        available_speakers = set()
        for segment in segments:
            if 'speaker' in segment:
                available_speakers.add(segment['speaker'])
        
        if not available_speakers:
            return 0, json_file, "No speaker information found in JSON"
        
        # Find the matching speaker in the JSON file
        matched_speaker = find_matching_speaker(json_speaker, available_speakers)
        
        if not matched_speaker:
            return 0, json_file, f"Could not match JSON speaker '{json_speaker}' with available speakers: {available_speakers}"
        
        # If we had to use a different speaker, log it
        if matched_speaker != json_speaker:
            console.print(f"[yellow]{json_file}: Using JSON speaker '{matched_speaker}' to match '{json_speaker}'[/yellow]")
        
        # Filter segments for the matched speaker
        target_segments = []
        seen_segments = set()  # Track unique segments
        
        for segment in segments:
            # Skip if we don't have required fields
            if 'speaker' not in segment or 'start' not in segment or 'end' not in segment:
                continue
                
            # Check if this is our matched speaker
            if segment['speaker'] == matched_speaker:
                # Create a unique identifier for this segment
                segment_id = (segment['start'], segment['end'])
                
                # Skip if we've seen this segment before
                if segment_id in seen_segments:
                    continue
                    
                seen_segments.add(segment_id)
                target_segments.append(segment)
        
        # Check if we found any segments
        if not target_segments:
            return 0, json_file, f"No segments found for matched speaker '{matched_speaker}'"

        # Load the audio file
        try:
            audio = AudioSegment.from_wav(wav_path)
        except Exception as e:
            return 0, json_file, f"Failed to load audio: {e}"

        # Process and save segments
        segment_count = 0
        base_filename = os.path.splitext(json_file)[0]
        
        for segment in target_segments:
            start_time = segment["start"] * 1000  # Convert to milliseconds
            end_time = segment["end"] * 1000
            segment_duration = end_time - start_time
            
            # Skip very short segments
            if segment_duration < 1000:  # Less than 1 second
                continue

            # Extract the segment
            try:
                segment_audio = audio[start_time:end_time]
                
                # Skip empty segments
                if len(segment_audio) == 0:
                    continue
                
                # Format output filename - use the target speaker (global speaker label) for naming
                output_file = os.path.join(
                    OUTPUT_DIR, 
                    f"{base_filename}_{target_speaker}_{start_time/1000:.2f}-{end_time/1000:.2f}.wav"
                )
                
                # Export audio segment
                segment_audio.export(output_file, format="wav")
                segment_count += 1
            except Exception as e:
                console.print(f"[yellow]Error processing segment {start_time/1000:.2f}-{end_time/1000:.2f} in {json_file}: {e}[/yellow]")
                continue

        return segment_count, json_file, "Success"
    except Exception as e:
        # Return a valid tuple with the error
        file_name = str(file_data[0]) if isinstance(file_data, tuple) and len(file_data) > 0 else "Unknown file"
        return 0, file_name, f"Process error: {str(e)}"

def display_global_speakers(global_df):
    """Display a table of all global speakers and their file counts."""
    if global_df is None or global_df.empty:
        return
    
    # Count files per global speaker
    speaker_counts = global_df.groupby('global_speaker')['file'].nunique().reset_index()
    speaker_counts.columns = ['Global Speaker', 'File Count']
    speaker_counts = speaker_counts.sort_values('File Count', ascending=False)
    
    # Create and display table
    table = Table(title="Available Global Speakers")
    table.add_column("Speaker", style="cyan")
    table.add_column("Files", style="green", justify="right")
    
    for _, row in speaker_counts.iterrows():
        table.add_row(row['Global Speaker'], str(row['File Count']))
    
    console.print(table)
    return speaker_counts['Global Speaker'].tolist()

def main():
    # Display title
    print_title()
    
    # Ensure Directories Exist
    os.makedirs(JSON_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load global mapping file
    if not os.path.exists(GLOBAL_MAPPING_FILE):
        console.print(f"[bold red]Error:[/bold red] Global mapping file not found: {GLOBAL_MAPPING_FILE}", style="red")
        console.print(f"[yellow]Please run the cross-reference.py script first to create global speaker mappings.[/yellow]")
        return
    
    try:
        global_df = pd.read_csv(GLOBAL_MAPPING_FILE)
        required_cols = ['file', 'original_speaker', 'global_speaker']
        missing_cols = [col for col in required_cols if col not in global_df.columns]
        
        if missing_cols:
            console.print(f"[bold red]Error:[/bold red] Global mapping file is missing columns: {', '.join(missing_cols)}", style="red")
            return
            
        if global_df.empty:
            console.print(f"[bold red]Error:[/bold red] Global mapping file is empty.", style="red")
            return
            
        console.print(f"[bold green]Loaded global speaker mapping for {global_df['file'].nunique()} files.[/bold green]")
        console.print(f"[green]Found {global_df['global_speaker'].nunique()} unique global speakers.[/green]")
    except Exception as e:
        console.print(f"[red]Error: Failed to read {GLOBAL_MAPPING_FILE}. {e}[/red]")
        return
    
    # Display available global speakers and get user selection
    console.print(f"[bold cyan]Available Global Speakers:[/bold cyan]")
    global_speakers = display_global_speakers(global_df)
    
    if not global_speakers:
        console.print(f"[bold red]Error:[/bold red] No global speakers found in the mapping file.", style="red")
        return
    
    # Ask user which speaker to isolate
    console.print(f"[bold yellow]Which global speaker do you want to isolate?[/bold yellow]")
    target_speaker = console.input("Enter the exact speaker name: ").strip()
    
    if target_speaker not in global_speakers:
        console.print(f"[yellow]Warning: '{target_speaker}' not found in global speakers list.[/yellow]")
        proceed = console.input(f"[bold yellow]Proceed anyway? (y/n): [/bold yellow]").strip().lower()
        if proceed != 'y':
            console.print("[yellow]Operation cancelled by user.[/yellow]")
            return
    
    # Filter global_df for just the files with our target speaker
    target_files = global_df[global_df['global_speaker'] == target_speaker]
    
    if target_files.empty:
        console.print(f"[bold red]No files found with global speaker '{target_speaker}'.[/bold red]")
        return
    
    console.print(f"[bold green]Found {len(target_files)} segments with global speaker '{target_speaker}'.[/bold green]")
    console.print(f"[green]These segments are from {target_files['file'].nunique()} unique files.[/green]")
    
    # Group by file and original speaker to create our processing list
    files_to_process = []
    
    # Get unique file/original_speaker combinations
    file_speaker_groups = target_files.groupby(['file', 'original_speaker'])
    
    for (file, original_speaker), group in file_speaker_groups:
        json_file = os.path.splitext(file)[0] + ".json"
        files_to_process.append((json_file, target_speaker, original_speaker))
    
    if not files_to_process:
        console.print("[bold yellow]No files to process.[/bold yellow]")
        return
    
    # Ask for confirmation
    console.print(f"[bold green]Ready to process {len(files_to_process)} file/speaker combinations.[/bold green]")
    proceed = console.input(f"[bold yellow]Do you want to proceed with isolating speaker '{target_speaker}'? (y/n): [/bold yellow]").strip().lower()
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
        task = progress.add_task(f"[cyan]Isolating speaker '{target_speaker}'...", total=len(files_to_process))
        
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
                    file_name = file_data[0] if isinstance(file_data, tuple) and len(file_data) > 0 else "Unknown file"
                    failed_files.append((file_name, str(e)))
                    progress.console.log(f"[red]Exception:[/red] {file_name} - {e}")
                
                progress.update(task, advance=1)
    
    # Final summary
    console.print(f"\n{SEPARATOR}")
    console.print(f"[bold green]Processing complete![/bold green]")
    console.print(f"[bold cyan]Summary for speaker '{target_speaker}':[/bold cyan]")
    console.print(f"[green]Successfully processed: {len(successful_files)} files[/green]")
    console.print(f"[green]Total segments saved: {total_segments}[/green]")
    console.print(f"[yellow]Failed: {len(failed_files)} files[/yellow]")
    
    if successful_files:
        console.print("\n[bold green]Top files by segment count:[/bold green]")
        # Sort by segment count and show top 5
        for json_file, count in sorted(successful_files, key=lambda x: x[1], reverse=True)[:5]:
            console.print(f"[white]{json_file}:[/white] [cyan]{count} segments[/cyan]")
    
    console.print(f"\n[bold green]Check the '{OUTPUT_DIR}' directory for isolated audio segments.[/bold green]")
    console.print(f"[green]Filename format: filename_{target_speaker}_start-end.wav[/green]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold red]Program interrupted by user. Exiting...[/bold red]")
    except Exception as e:
        console.print(f"\n[bold red]An unexpected error occurred: {e}[/bold red]")